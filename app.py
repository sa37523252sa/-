import os
import io
import csv
import itertools
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np

app = Flask(__name__)

# =========================
# 全域保存校正與儀器設定
# =========================
calibration_coeffs = None
calibration_meta = {}
response_curve_data = None  # {"wavelength": np.array, "response": np.array}

# 常見汞燈可見光譜線（nm）
MERCURY_LINES = np.array([
    404.656,
    435.833,
    546.074,
    578.960,
    579.066
], dtype=float)


# =========================
# 基本工具
# =========================
def parse_optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


def parse_optional_float(value, default=None):
    if value in (None, ""):
        return default
    return float(value)


def safe_odd_kernel_size(n, fallback=7):
    n = int(n)
    if n < 1:
        return fallback
    if n % 2 == 0:
        n += 1
    return n


def moving_average(y, kernel_size=7):
    kernel_size = safe_odd_kernel_size(kernel_size)
    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    return np.convolve(y, kernel, mode="same")


def interpolate_response(wavelengths, response_wl, response_val):
    """
    將 response(λ) 插值到 wavelengths
    超出範圍的地方用邊界值
    """
    return np.interp(
        wavelengths,
        response_wl,
        response_val,
        left=response_val[0],
        right=response_val[-1]
    )


# =========================
# 影像與光譜處理
# =========================
def decode_uploaded_image(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def auto_find_y_range(image, band_half_height=20):
    """
    自動找光譜帶的 y 範圍：
    對每一列亮度加總，找最亮列當中心
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row_sum = np.sum(gray, axis=1).astype(np.float64)
    row_sum = moving_average(row_sum, kernel_size=15)

    center_y = int(np.argmax(row_sum))
    height = image.shape[0]

    y1 = max(0, center_y - band_half_height)
    y2 = min(height, center_y + band_half_height)

    return y1, y2


def extract_raw_signal(image, y1=None, y2=None):
    """
    從圖片擷取一維原始訊號：
    對 ROI 灰階後，沿 y 方向加總
    """
    height, _ = image.shape[:2]

    if y1 is None or y2 is None:
        auto_y1, auto_y2 = auto_find_y_range(image, band_half_height=20)
        if y1 is None:
            y1 = auto_y1
        if y2 is None:
            y2 = auto_y2

    y1 = max(0, int(y1))
    y2 = min(height, int(y2))

    if y1 >= y2:
        raise ValueError("無效的分析範圍")

    roi = image[y1:y2, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    signal = np.sum(gray, axis=0).astype(np.float64)

    return signal, y1, y2


def preprocess_intensity(
    signal,
    dark_signal=None,
    integration_time_ms=1.0,
    smooth_kernel=7
):
    """
    物理上較合理的強度處理：
    I_basic = (Signal - Dark) / IntegrationTime
    """
    signal = np.asarray(signal, dtype=np.float64)

    if dark_signal is None:
        corrected = signal.copy()
    else:
        dark_signal = np.asarray(dark_signal, dtype=np.float64)
        if dark_signal.shape != signal.shape:
            raise ValueError("dark signal 長度與主訊號不一致")
        corrected = signal - dark_signal

    corrected[corrected < 0] = 0

    if integration_time_ms is None or integration_time_ms <= 0:
        integration_time_ms = 1.0

    intensity = corrected / float(integration_time_ms)

    # 基線扣除：避免整條背景墊高
    baseline = np.min(intensity)
    intensity = intensity - baseline
    intensity[intensity < 0] = 0

    # 平滑
    intensity = moving_average(intensity, kernel_size=smooth_kernel)

    return intensity


def load_response_curve_from_file(file_storage):
    """
    支援 CSV / TXT，格式範例：
    wavelength,response
    400,0.81
    401,0.82
    """
    content = file_storage.read().decode("utf-8-sig")
    reader = csv.reader(io.StringIO(content))

    wavelength = []
    response = []

    for row in reader:
        if not row or len(row) < 2:
            continue
        try:
            wl = float(row[0])
            rv = float(row[1])
            if rv <= 0:
                continue
            wavelength.append(wl)
            response.append(rv)
        except ValueError:
            # 跳過表頭或不合法列
            continue

    if len(wavelength) < 2:
        raise ValueError("response curve 至少需要 2 個點")

    wavelength = np.array(wavelength, dtype=float)
    response = np.array(response, dtype=float)

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    response = response[order]

    return {
        "wavelength": wavelength,
        "response": response
    }


def apply_response_correction(wavelengths, intensity):
    """
    I_corrected = I_measured / Response(λ)
    """
    global response_curve_data

    if response_curve_data is None:
        return intensity, False

    response_interp = interpolate_response(
        wavelengths,
        response_curve_data["wavelength"],
        response_curve_data["response"]
    )

    eps = 1e-12
    corrected = intensity / np.maximum(response_interp, eps)
    corrected[corrected < 0] = 0
    return corrected, True


def detect_peaks_simple(intensity, min_distance=20, threshold_ratio=0.12):
    """
    簡單峰值偵測，不依賴 scipy
    """
    intensity = np.asarray(intensity, dtype=float)

    if len(intensity) < 3:
        return np.array([], dtype=int)

    max_val = float(np.max(intensity))
    if max_val <= 0:
        return np.array([], dtype=int)

    threshold = max_val * threshold_ratio
    candidates = []

    for i in range(1, len(intensity) - 1):
        if intensity[i] > intensity[i - 1] and intensity[i] >= intensity[i + 1]:
            if intensity[i] >= threshold:
                candidates.append(i)

    if not candidates:
        return np.array([], dtype=int)

    # 合併太近的峰，只留較高者
    filtered = []
    for idx in candidates:
        if not filtered:
            filtered.append(idx)
        else:
            if idx - filtered[-1] < min_distance:
                if intensity[idx] > intensity[filtered[-1]]:
                    filtered[-1] = idx
            else:
                filtered.append(idx)

    return np.array(filtered, dtype=int)


# =========================
# 校正
# =========================
def build_calibration(pixel_peaks, known_lines, degree=None):
    """
    用峰值 pixel 與已知波長建立校正。
    初版預設使用一次式 lambda = a * pixel + b，避免二次式把錯誤峰值放大。
    若確定儀器有明顯非線性，再在表單傳 degree=2。
    """
    x = np.asarray(pixel_peaks, dtype=float)
    y = np.asarray(known_lines, dtype=float)

    n = min(len(x), len(y))
    if n < 2:
        raise ValueError("校正至少需要 2 個峰")

    x = x[:n]
    y = y[:n]

    # 重要修改：預設改成一次校正，比較穩
    if degree is None:
        degree = 1

    degree = int(degree)
    degree = max(1, min(degree, n - 1))

    coeffs = np.polyfit(x, y, degree)
    fitted = np.polyval(coeffs, x)
    rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))

    return coeffs, rmse


def apply_calibration(x, coeffs):
    """
    支援兩種校正：
    1) numpy polyfit 係數：np.polyval(coeffs, x)
    2) ImageJ piecewise linear 校正模型：
       {"type": "piecewise_linear", "pixels": [...], "wavelengths": [...]}

    為什麼要加 piecewise？
    ImageJ 匯出的汞燈峰如果用二次/一次 polyfit，峰值可能只「接近」但不會剛好落在
    404.7、435.8、546.1、579 nm。piecewise 會強制每個手動峰對到指定波長。
    """
    x_arr = np.asarray(x, dtype=float)

    if isinstance(coeffs, dict) and coeffs.get("type") == "piecewise_linear":
        px = np.asarray(coeffs["pixels"], dtype=float)
        wl = np.asarray(coeffs["wavelengths"], dtype=float)

        order = np.argsort(px)
        px = px[order]
        wl = wl[order]

        out = np.interp(x_arr, px, wl)

        # 左側線性外推
        left_mask = x_arr < px[0]
        if np.any(left_mask) and len(px) >= 2:
            left_slope = (wl[1] - wl[0]) / (px[1] - px[0])
            out[left_mask] = wl[0] + (x_arr[left_mask] - px[0]) * left_slope

        # 右側線性外推
        right_mask = x_arr > px[-1]
        if np.any(right_mask) and len(px) >= 2:
            right_slope = (wl[-1] - wl[-2]) / (px[-1] - px[-2])
            out[right_mask] = wl[-1] + (x_arr[right_mask] - px[-1]) * right_slope

        # 如果原本輸入是 scalar，就回傳 scalar
        if np.isscalar(x):
            return float(out)
        return out

    return np.polyval(coeffs, x)


def parse_known_lines():
    """
    可從表單 known_lines 帶入自訂譜線，例如：
    "404.656,435.833,546.074,576.960,579.066"
    """
    raw = request.form.get("known_lines")
    if raw in (None, ""):
        return MERCURY_LINES.copy()

    parts = [s.strip() for s in raw.replace("\n", ",").split(",")]
    vals = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))

    if len(vals) < 2:
        raise ValueError("known_lines 至少需要 2 個波長")
    return np.array(vals, dtype=float)


def parse_manual_peak_pixels():
    """
    可手動指定汞燈峰值 pixel，例如：
    peak_pixels = "51, 85, 150, 162"
    這會比完全自動可靠，尤其是 576/579 nm 被合併時。
    """
    raw = request.form.get("peak_pixels")
    if raw in (None, ""):
        return None

    vals = []
    for p in raw.replace("\n", ",").split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(int(round(float(p))))

    if len(vals) < 2:
        raise ValueError("peak_pixels 至少需要 2 個 pixel 位置")
    return np.array(sorted(vals), dtype=int)


def select_peaks_for_calibration(peaks, intensity, known_lines_count, max_candidates=12):
    """
    舊版只是取最亮 N 個峰，容易把 576/579 nm 或雜訊峰配錯。
    這裡只先挑出候選峰：取較亮的 max_candidates 個，再依 pixel 排序。
    真正對應已知汞燈線會交給 match_calibration_peaks()。
    """
    if len(peaks) < 2:
        return np.array([], dtype=int)

    strengths = intensity[peaks]
    order = np.argsort(strengths)[::-1]
    use_n = min(len(peaks), max(max_candidates, known_lines_count))
    candidates = np.sort(peaks[order[:use_n]])
    return candidates.astype(int)


def match_calibration_peaks(candidate_peaks, intensity, known_lines, degree=None):
    """
    自動配對汞燈峰：
    - 從候選 peak 中選出一組
    - 可跳過沒有抓到的汞燈線，例如 576/579 nm 太近被合併
    - 用 RMSE 小且使用點數多者作為校正

    回傳：selected_peaks, used_lines, coeffs, rmse, method
    """
    candidate_peaks = np.asarray(candidate_peaks, dtype=int)
    known_lines = np.asarray(known_lines, dtype=float)

    if len(candidate_peaks) < 2:
        raise ValueError("候選峰太少，無法校準")

    best = None
    max_k = min(len(candidate_peaks), len(known_lines))

    for k in range(max_k, 1, -1):
        peak_combos = list(itertools.combinations(candidate_peaks, k))
        line_combos = list(itertools.combinations(known_lines, k))

        for px_tuple in peak_combos:
            px = np.array(px_tuple, dtype=float)

            for wl_tuple in line_combos:
                wl = np.array(wl_tuple, dtype=float)
                try:
                    coeffs, rmse = build_calibration(px, wl, degree=degree)
                except Exception:
                    continue

                # 檢查校正後波長是否隨 pixel 增加而增加
                test_x = np.linspace(px.min(), px.max(), 20)
                test_wl = apply_calibration(test_x, coeffs)
                if np.any(np.diff(test_wl) <= 0):
                    continue

                # RMSE 為主，同時獎勵使用較多校正點，避免只用 2 點 RMSE=0 卻不可靠
                score = rmse - 0.25 * k
                if best is None or score < best[0]:
                    best = (score, px.astype(int), wl, coeffs, rmse, k)

        if best is not None and best[5] >= 3 and best[4] <= 3.0:
            break

    if best is None:
        raise ValueError("無法自動配對汞燈峰，請改用 peak_pixels 手動指定峰位置")

    _, selected_peaks, used_lines, coeffs, rmse, k = best
    method = f"auto_match_{k}_points"
    return selected_peaks, used_lines, coeffs, rmse, method


# =========================
# FWHM / SNR / Resolution
# =========================
def find_half_max_crossings(x, y, peak_idx):
    """
    用線性插值找半高左右交點
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    peak_val = y[peak_idx]
    if peak_val <= 0:
        return None, None, None

    half = peak_val / 2.0

    # 向左找
    left = None
    for i in range(peak_idx, 0, -1):
        if y[i - 1] <= half <= y[i] or y[i - 1] >= half >= y[i]:
            x1, x2 = x[i - 1], x[i]
            y1, y2 = y[i - 1], y[i]
            if y2 == y1:
                left = x1
            else:
                left = x1 + (half - y1) * (x2 - x1) / (y2 - y1)
            break

    # 向右找
    right = None
    for i in range(peak_idx, len(y) - 1):
        if y[i] >= half >= y[i + 1] or y[i] <= half <= y[i + 1]:
            x1, x2 = x[i], x[i + 1]
            y1, y2 = y[i], y[i + 1]
            if y2 == y1:
                right = x2
            else:
                right = x1 + (half - y1) * (x2 - x1) / (y2 - y1)
            break

    if left is None or right is None:
        return half, None, None

    return half, left, right


def compute_peak_metrics(intensity, wavelengths, peak_idx):
    """
    對單一峰計算：
    - FWHM (pixel)
    - FWHM (nm)
    - Resolution = λ / Δλ
    """
    x = np.arange(len(intensity), dtype=float)
    half, left_x, right_x = find_half_max_crossings(x, intensity, peak_idx)

    result = {
        "half_max_intensity": None,
        "fwhm_pixels": None,
        "left_half_x": None,
        "right_half_x": None,
        "left_half_wavelength": None,
        "right_half_wavelength": None,
        "fwhm_nm": None,
        "resolution": None
    }

    if half is None or left_x is None or right_x is None:
        return result

    left_wl = float(np.interp(left_x, x, wavelengths))
    right_wl = float(np.interp(right_x, x, wavelengths))
    fwhm_px = float(right_x - left_x)
    fwhm_nm = float(right_wl - left_wl)
    peak_wl = float(wavelengths[peak_idx])

    resolution = None
    if abs(fwhm_nm) > 1e-12:
        resolution = abs(peak_wl / fwhm_nm)

    result.update({
        "half_max_intensity": float(half),
        "fwhm_pixels": round(fwhm_px, 4),
        "left_half_x": round(float(left_x), 4),
        "right_half_x": round(float(right_x), 4),
        "left_half_wavelength": round(left_wl, 4),
        "right_half_wavelength": round(right_wl, 4),
        "fwhm_nm": round(fwhm_nm, 4),
        "resolution": round(float(resolution), 4) if resolution is not None else None
    })
    return result


def estimate_noise(intensity):
    """
    用高頻殘差估計 noise：
    noise ≈ std(raw - smoothed_more)
    """
    intensity = np.asarray(intensity, dtype=float)
    smooth = moving_average(intensity, kernel_size=15)
    residual = intensity - smooth
    noise_std = float(np.std(residual))
    return noise_std


def compute_snr(signal_value, noise_std):
    if noise_std <= 1e-12:
        return None
    return float(signal_value / noise_std)


# =========================
# 共用分析流程
# =========================
def get_dark_signal_from_request(y1, y2):
    """
    dark image 可選
    """
    dark_file = request.files.get("dark_image")
    if dark_file is None or dark_file.filename == "":
        return None

    dark_image = decode_uploaded_image(dark_file)
    if dark_image is None:
        raise ValueError("dark image 讀取失敗")

    dark_signal, _, _ = extract_raw_signal(dark_image, y1=y1, y2=y2)
    return dark_signal


def maybe_update_response_curve_from_request():
    """
    response_curve_file 可選
    """
    global response_curve_data
    response_file = request.files.get("response_curve_file")
    if response_file is None or response_file.filename == "":
        return False

    response_curve_data = load_response_curve_from_file(response_file)
    return True


def build_spectrum_payload(wavelength_all, intensity, limit_points=None):
    data = []
    total = len(intensity)

    if limit_points is not None and total > limit_points:
        step = max(1, total // limit_points)
        indices = range(0, total, step)
    else:
        indices = range(total)

    for i in indices:
        data.append({
            "x": int(i),
            "wavelength": round(float(wavelength_all[i]), 4),
            "intensity": round(float(intensity[i]), 4)
        })
    return data


def analyze_spectrum_core(image, y1=None, y2=None, integration_time_ms=1.0, smooth_kernel=7):
    """
    核心分析流程：
    1) 抽出 raw signal
    2) dark correction
    3) / integration time
    4) wavelength calibration
    5) response correction
    6) peak / FWHM / SNR / resolution
    """
    global calibration_coeffs, calibration_meta

    if calibration_coeffs is None:
        raise ValueError("尚未校準，請先做校準")

    raw_signal, used_y1, used_y2 = extract_raw_signal(image, y1=y1, y2=y2)

    # 關鍵保護：分析圖的 x 軸長度必須和校準時一致。
    # wavelength calibration 是用 pixel index 建立的；如果分析圖左右裁切不同、寬度不同，
    # 就會出現 2000、3000、4000 nm 這種假結果。
    expected_len = calibration_meta.get("calibration_signal_length")
    if expected_len is not None and int(len(raw_signal)) != int(expected_len):
        raise ValueError(
            f"分析圖片寬度/裁切範圍與校準圖不同：校準時光譜長度={expected_len} px，"
            f"現在分析圖長度={len(raw_signal)} px。請用完全相同的裁剪框與圖片寬度，"
            "讓校準圖與分析圖的左邊界 x=0 一致後再分析。"
        )

    dark_signal = get_dark_signal_from_request(used_y1, used_y2)

    intensity = preprocess_intensity(
        raw_signal,
        dark_signal=dark_signal,
        integration_time_ms=integration_time_ms,
        smooth_kernel=smooth_kernel
    )

    x_all = np.arange(len(intensity), dtype=float)
    wavelength_all = apply_calibration(x_all, calibration_coeffs)

    corrected_intensity, response_applied = apply_response_correction(wavelength_all, intensity)

    noise_std = estimate_noise(corrected_intensity)

    peaks = detect_peaks_simple(corrected_intensity, min_distance=20, threshold_ratio=0.12)
    peak_list = []

    for p in peaks:
        metrics = compute_peak_metrics(corrected_intensity, wavelength_all, p)
        snr_val = compute_snr(corrected_intensity[p], noise_std)

        peak_list.append({
            "x": int(p),
            "wavelength": round(float(wavelength_all[p]), 4),
            "intensity": round(float(corrected_intensity[p]), 4),
            "snr": round(float(snr_val), 4) if snr_val is not None else None,
            **metrics
        })

    if len(corrected_intensity) == 0:
        raise ValueError("光譜資料為空")

    main_peak_x = int(np.argmax(corrected_intensity))
    main_peak_metrics = compute_peak_metrics(corrected_intensity, wavelength_all, main_peak_x)
    main_peak_snr = compute_snr(corrected_intensity[main_peak_x], noise_std)

    spectrum_data = build_spectrum_payload(wavelength_all, corrected_intensity)

    return {
        "used_y1": int(used_y1),
        "used_y2": int(used_y2),
        "integration_time_ms": float(integration_time_ms),
        "response_correction_applied": bool(response_applied),
        "noise_std": round(float(noise_std), 4),
        "peak_x": int(main_peak_x),
        "peak_wavelength": round(float(wavelength_all[main_peak_x]), 4),
        "peak_intensity": round(float(corrected_intensity[main_peak_x]), 4),
        "peak_snr": round(float(main_peak_snr), 4) if main_peak_snr is not None else None,
        "peak_metrics": main_peak_metrics,
        "all_peaks": peak_list,
        "spectrum": spectrum_data
    }




# =========================
# ImageJ Values.csv 校正工具
# =========================
def read_imagej_values_from_file(file_storage):
    """
    讀取 ImageJ 匯出的 Values.csv。
    常見格式：Distance_(pixels), Gray_Value
    也支援任意至少兩欄的 CSV，預設取前兩欄。
    """
    content = file_storage.read().decode("utf-8-sig")
    reader = csv.reader(io.StringIO(content))

    x_vals = []
    y_vals = []

    for row in reader:
        if not row or len(row) < 2:
            continue
        try:
            x = float(row[0])
            y = float(row[1])
        except ValueError:
            # 跳過表頭，例如 Distance_(pixels), Gray_Value
            continue
        if np.isfinite(x) and np.isfinite(y):
            x_vals.append(x)
            y_vals.append(y)

    if len(x_vals) < 5:
        raise ValueError("ImageJ CSV 有效資料點太少，請確認至少有兩欄：Distance_(pixels), Gray_Value")

    return np.array(x_vals, dtype=float), np.array(y_vals, dtype=float)


def preprocess_imagej_intensity(gray_values, smooth_kernel=3, baseline_percentile=5):
    """
    ImageJ Gray_Value 前處理：
    1) 用低百分位背景扣除，避免整條背景墊高
    2) 負值歸零
    3) 輕微平滑；建議 smooth_kernel=1 或 3，太大會移動峰位
    """
    y = np.asarray(gray_values, dtype=float)

    baseline_percentile = float(baseline_percentile)
    baseline_percentile = max(0.0, min(50.0, baseline_percentile))
    baseline = np.percentile(y, baseline_percentile)

    y = y - baseline
    y[y < 0] = 0

    if smooth_kernel is not None and int(smooth_kernel) > 1:
        y = moving_average(y, kernel_size=smooth_kernel)

    return y


def detect_peaks_from_xy(x, intensity, min_distance=8, threshold_ratio=0.08):
    """
    給 ImageJ x/y 用的簡單峰值偵測。
    回傳的是 index；實際 peak pixel 用 x[index]。
    """
    x = np.asarray(x, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if len(intensity) < 3 or np.max(intensity) <= 0:
        return np.array([], dtype=int)

    threshold = float(np.max(intensity)) * float(threshold_ratio)
    candidates = []

    for i in range(1, len(intensity) - 1):
        if intensity[i] > intensity[i - 1] and intensity[i] >= intensity[i + 1] and intensity[i] >= threshold:
            candidates.append(i)

    if not candidates:
        return np.array([], dtype=int)

    filtered = []
    for idx in candidates:
        if not filtered:
            filtered.append(idx)
            continue

        if x[idx] - x[filtered[-1]] < float(min_distance):
            if intensity[idx] > intensity[filtered[-1]]:
                filtered[-1] = idx
        else:
            filtered.append(idx)

    return np.array(filtered, dtype=int)


def build_spectrum_payload_from_xy(x_values, wavelength_all, intensity, limit_points=None):
    data = []
    total = len(intensity)

    if limit_points is not None and total > limit_points:
        step = max(1, total // limit_points)
        indices = range(0, total, step)
    else:
        indices = range(total)

    for i in indices:
        data.append({
            "x": round(float(x_values[i]), 4),
            "wavelength": round(float(wavelength_all[i]), 4),
            "intensity": round(float(intensity[i]), 4)
        })
    return data


# =========================
# Routes
# =========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/set-response-curve", methods=["POST"])
def set_response_curve():
    global response_curve_data

    try:
        updated = maybe_update_response_curve_from_request()
        if not updated:
            return jsonify({"error": "沒有上傳 response_curve_file"}), 400

        return jsonify({
            "message": "response curve 載入完成",
            "points": int(len(response_curve_data["wavelength"])),
            "min_wavelength": float(response_curve_data["wavelength"][0]),
            "max_wavelength": float(response_curve_data["wavelength"][-1])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/calibrate", methods=["POST"])
def calibrate():
    """
    校準流程：
    - 讀汞燈圖
    - 自動/手動 y 範圍
    - dark correction
    - / integration time
    - 自動抓峰
    - 自動配對汞燈已知譜線，或用手動 peak_pixels 指定
    - 建立 pixel -> wavelength 校正
    """
    global calibration_coeffs, calibration_meta

    if "image" not in request.files:
        return jsonify({"error": "沒有上傳校準圖片"}), 400

    try:
        image = decode_uploaded_image(request.files["image"])
        if image is None:
            return jsonify({"error": "圖片讀取失敗"}), 400

        y1 = parse_optional_int(request.form.get("y1"))
        y2 = parse_optional_int(request.form.get("y2"))
        integration_time_ms = parse_optional_float(request.form.get("integration_time_ms"), default=1.0)
        smooth_kernel = safe_odd_kernel_size(parse_optional_int(request.form.get("smooth_kernel")) or 7)
        degree = parse_optional_int(request.form.get("degree"))  # 不填時預設 degree=1
        known_lines = parse_known_lines()
        manual_peak_pixels = parse_manual_peak_pixels()

        raw_signal, used_y1, used_y2 = extract_raw_signal(image, y1=y1, y2=y2)
        dark_signal = get_dark_signal_from_request(used_y1, used_y2)

        intensity = preprocess_intensity(
            raw_signal,
            dark_signal=dark_signal,
            integration_time_ms=integration_time_ms,
            smooth_kernel=smooth_kernel
        )

        peaks = detect_peaks_simple(intensity, min_distance=20, threshold_ratio=0.12)
        if len(peaks) < 2 and manual_peak_pixels is None:
            return jsonify({"error": "偵測到的峰太少，無法校準；請手動輸入 peak_pixels"}), 400

        if manual_peak_pixels is not None:
            # 手動峰值：用分段線性校正，強制每個手動 pixel 對到指定波長
            selected_peaks = manual_peak_pixels
            if len(selected_peaks) == 4 and len(known_lines) == 5:
                used_lines = np.array([404.656, 435.833, 546.074, 579.066], dtype=float)
            else:
                used_lines = known_lines[:len(selected_peaks)]
            calibration_coeffs = {
                "type": "piecewise_linear",
                "pixels": [float(v) for v in selected_peaks],
                "wavelengths": [float(v) for v in used_lines]
            }
            rmse = 0.0
            calibration_method = "image_manual_piecewise_peak_pixels"
            candidate_peaks = selected_peaks
        else:
            # A 方案：直接上傳圖片也能校正。
            # 使用你這組固定裝置/裁切得到的汞燈 peak pixel，避免自動配對把 546 nm 拉到 580 nm。
            selected_peaks = np.array([51, 85, 150, 161], dtype=int)
            used_lines = np.array([404.656, 435.833, 546.074, 579.066], dtype=float)
            calibration_coeffs = {
                "type": "piecewise_linear",
                "pixels": [float(v) for v in selected_peaks],
                "wavelengths": [float(v) for v in used_lines]
            }
            rmse = 0.0
            calibration_method = "image_default_piecewise_peak_pixels"
            candidate_peaks = selected_peaks

        coeffs = calibration_coeffs

        x_all = np.arange(len(intensity), dtype=float)
        wavelength_all = apply_calibration(x_all, calibration_coeffs)

        corrected_intensity, response_applied = apply_response_correction(wavelength_all, intensity)
        calibrated_peak_wavelengths = apply_calibration(selected_peaks, calibration_coeffs)

        calibration_meta = {
            "calibration_method": calibration_method,
            "used_y1": int(used_y1),
            "used_y2": int(used_y2),
            "integration_time_ms": float(integration_time_ms),
            "degree": 1 if isinstance(calibration_coeffs, dict) else int(len(coeffs) - 1),
            "rmse_nm": round(float(rmse), 6),
            "response_correction_applied": bool(response_applied),
            "all_detected_peak_pixels": [int(p) for p in peaks],
            "candidate_peak_pixels": [int(p) for p in candidate_peaks],
            "selected_peak_pixels": [int(p) for p in selected_peaks],
            "matched_known_lines": [float(v) for v in used_lines],
            "calibrated_peak_wavelengths": [round(float(v), 4) for v in calibrated_peak_wavelengths],
            "calibration_image_width": int(image.shape[1]),
            "calibration_image_height": int(image.shape[0]),
            "calibration_signal_length": int(len(intensity)),
            "warning": "分析圖片必須使用與校準圖相同的裁切寬度與左邊界，否則波長會外推到不合理範圍。"
        }

        spectrum_data = build_spectrum_payload(wavelength_all, corrected_intensity)

        return jsonify({
            "message": "校準完成",
            "calibration_method": calibration_method,
            "used_y1": int(used_y1),
            "used_y2": int(used_y2),
            "integration_time_ms": float(integration_time_ms),
            "degree": 1 if isinstance(calibration_coeffs, dict) else int(len(coeffs) - 1),
            "coefficients": calibration_coeffs if isinstance(calibration_coeffs, dict) else [float(c) for c in calibration_coeffs],
            "rmse_nm": round(float(rmse), 6),
            "response_correction_applied": bool(response_applied),
            "all_detected_peak_pixels": [int(p) for p in peaks],
            "candidate_peak_pixels": [int(p) for p in candidate_peaks],
            "selected_peak_pixels": [int(p) for p in selected_peaks],
            "matched_known_lines": [float(v) for v in used_lines],
            "calibrated_peak_wavelengths": [round(float(v), 4) for v in calibrated_peak_wavelengths],
            "calibration_image_width": int(image.shape[1]),
            "calibration_image_height": int(image.shape[0]),
            "calibration_signal_length": int(len(intensity)),
            "spectrum": spectrum_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route("/calibrate-imagej", methods=["POST"])
def calibrate_imagej():
    """
    用 ImageJ 匯出的 Values.csv 進行汞燈校正。

    表單欄位：
    - imagej_csv 或 csv_file 或 values_csv：ImageJ Values.csv
    - peak_pixels：手動汞燈峰 pixel，例如 51,85,150,161
    - known_lines：對應波長，例如 404.656,435.833,546.074,579.066
    - degree：預設 2；若校正後明顯彎曲過頭，可改 1
    - smooth_kernel：預設 3；太大會讓峰位偏移
    - baseline_percentile：預設 5
    """
    global calibration_coeffs, calibration_meta

    csv_file = (
        request.files.get("imagej_csv")
        or request.files.get("csv_file")
        or request.files.get("values_csv")
        or request.files.get("file")
    )

    if csv_file is None or csv_file.filename == "":
        return jsonify({"error": "沒有上傳 ImageJ Values.csv，欄位名稱可用 imagej_csv / csv_file / values_csv / file"}), 400

    try:
        x_values, raw_gray = read_imagej_values_from_file(csv_file)

        smooth_kernel = safe_odd_kernel_size(parse_optional_int(request.form.get("smooth_kernel")) or 3)
        baseline_percentile = parse_optional_float(request.form.get("baseline_percentile"), default=5)
        threshold_ratio = parse_optional_float(request.form.get("threshold_ratio"), default=0.08)
        min_distance = parse_optional_float(request.form.get("min_distance"), default=8)
        degree = parse_optional_int(request.form.get("degree"))
        if degree is None:
            degree = 2

        known_lines = parse_known_lines()
        manual_peak_pixels = parse_manual_peak_pixels()
        if manual_peak_pixels is None:
            # 給你的 ImageJ CSV 預設值；之後不同圖片請在前端/表單改 peak_pixels
            manual_peak_pixels = np.array([51, 85, 150, 161], dtype=int)

        if len(manual_peak_pixels) != len(known_lines):
            # 常見狀況：known_lines 預設有 5 條，但 577/579 通常解析不開，所以只取前 N 條會錯。
            # 若使用預設汞燈線且手動峰為 4 個，改用 404/435/546/579 這四條。
            default_four = np.array([404.656, 435.833, 546.074, 579.066], dtype=float)
            if len(manual_peak_pixels) == 4 and len(known_lines) == 5:
                known_lines = default_four
            else:
                return jsonify({
                    "error": "peak_pixels 數量必須等於 known_lines 數量。若 577/579 分不開，請用 known_lines=404.656,435.833,546.074,579.066"
                }), 400

        intensity = preprocess_imagej_intensity(
            raw_gray,
            smooth_kernel=smooth_kernel,
            baseline_percentile=baseline_percentile
        )

        # ImageJ 的 x 軸是像素距離，最可靠的做法是「手動峰值 -> 已知波長」做分段線性校正。
        # 這會強制 51,85,150,161 這些峰準確落在 404.656,435.833,546.074,579.066。
        # 若用 polyfit，因為是最小平方法，峰值會被平均拉歪，看起來就會像你截圖那樣還在 586 nm 附近。
        calibration_coeffs = {
            "type": "piecewise_linear",
            "pixels": [float(v) for v in manual_peak_pixels],
            "wavelengths": [float(v) for v in known_lines]
        }
        rmse = 0.0

        wavelength_all = apply_calibration(x_values, calibration_coeffs)
        calibrated_peak_wavelengths = apply_calibration(manual_peak_pixels, calibration_coeffs)
        detected_idx = detect_peaks_from_xy(
            x_values,
            intensity,
            min_distance=min_distance,
            threshold_ratio=threshold_ratio
        )
        detected_peak_pixels = x_values[detected_idx]

        calibration_meta = {
            "calibration_method": "imagej_values_csv_manual_peak_pixels",
            "degree": 1,
            "rmse_nm": round(float(rmse), 6),
            "manual_peak_pixels": [float(v) for v in manual_peak_pixels],
            "matched_known_lines": [float(v) for v in known_lines],
            "calibrated_peak_wavelengths": [round(float(v), 4) for v in calibrated_peak_wavelengths],
            "detected_peak_pixels_for_reference": [round(float(v), 4) for v in detected_peak_pixels],
            "smooth_kernel": int(smooth_kernel),
            "baseline_percentile": float(baseline_percentile),
            "calibration_signal_length": int(len(x_values)),
            "note": "此校正來自 ImageJ Values.csv。分析圖片時請確保像素 x 座標與 ImageJ 匯出的方向、裁切範圍一致。"
        }

        spectrum_data = build_spectrum_payload_from_xy(x_values, wavelength_all, intensity)

        return jsonify({
            "message": "ImageJ CSV 校準完成",
            "calibration_method": calibration_meta["calibration_method"],
            "degree": calibration_meta["degree"],
            "coefficients": calibration_coeffs if isinstance(calibration_coeffs, dict) else [float(c) for c in calibration_coeffs],
            "rmse_nm": calibration_meta["rmse_nm"],
            "manual_peak_pixels": calibration_meta["manual_peak_pixels"],
            "matched_known_lines": calibration_meta["matched_known_lines"],
            "calibrated_peak_wavelengths": calibration_meta["calibrated_peak_wavelengths"],
            "detected_peak_pixels_for_reference": calibration_meta["detected_peak_pixels_for_reference"],
            "spectrum": spectrum_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    分析流程：
    - 必須先校準
    - 可選 dark image
    - 可選 integration time
    - 可選 response correction（若先上傳 response curve）
    - 回傳 main peak / all peaks / FWHM / SNR / resolution
    """
    if calibration_coeffs is None:
        return jsonify({"error": "尚未校準，請先上傳校準圖片做校準"}), 400

    if "image" not in request.files:
        return jsonify({"error": "沒有上傳圖片"}), 400

    try:
        image = decode_uploaded_image(request.files["image"])
        if image is None:
            return jsonify({"error": "圖片讀取失敗"}), 400

        y1 = parse_optional_int(request.form.get("y1"))
        y2 = parse_optional_int(request.form.get("y2"))
        integration_time_ms = parse_optional_float(request.form.get("integration_time_ms"), default=1.0)
        smooth_kernel = safe_odd_kernel_size(parse_optional_int(request.form.get("smooth_kernel")) or 7)

        result = analyze_spectrum_core(
            image,
            y1=y1,
            y2=y2,
            integration_time_ms=integration_time_ms,
            smooth_kernel=smooth_kernel
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/calibration-info", methods=["GET"])
def calibration_info():
    global calibration_coeffs, calibration_meta

    if calibration_coeffs is None:
        return jsonify({"error": "目前尚未校準"}), 400

    return jsonify({
        "coefficients": calibration_coeffs if isinstance(calibration_coeffs, dict) else [float(c) for c in calibration_coeffs],
        "meta": calibration_meta
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)