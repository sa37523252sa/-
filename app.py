import os
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np

app = Flask(__name__)

# 全域保存校正參數
calibration_coeffs = None

# 常見汞燈可見光譜線（nm）
MERCURY_LINES = np.array([
    404.656,
    435.833,
    546.074,
    576.960,
    579.066
], dtype=float)


def auto_find_y_range(image, band_half_height=20):
    """
    自動找光譜帶的 y 範圍
    做法：對每一列的亮度加總，找最亮那一列當中心
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 每一列亮度總和
    row_sum = np.sum(gray, axis=1).astype(np.float64)

    # 平滑，避免雜訊影響
    kernel = np.ones(15) / 15
    row_sum = np.convolve(row_sum, kernel, mode="same")

    center_y = int(np.argmax(row_sum))

    height = image.shape[0]
    y1 = max(0, center_y - band_half_height)
    y2 = min(height, center_y + band_half_height)

    return y1, y2


def extract_spectrum(image, y1=None, y2=None):
    """
    從圖片擷取一維光譜：
    對 ROI 灰階後，沿 y 方向加總，得到每個 x 的強度
    若 y1/y2 沒給，則自動找光譜帶
    """
    height, width = image.shape[:2]

    # 沒填就自動找
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

    # 對每個 x 欄位做亮度總和
    intensity = np.sum(gray, axis=0).astype(np.float64)

    # 背景扣除
    intensity -= np.min(intensity)
    intensity[intensity < 0] = 0

    # 平滑
    kernel = np.ones(7) / 7
    intensity = np.convolve(intensity, kernel, mode="same")

    return intensity, y1, y2


def detect_peaks_simple(intensity, min_distance=20, threshold_ratio=0.12):
    """
    簡單峰值偵測，不依賴 scipy
    """
    if len(intensity) < 3:
        return np.array([], dtype=int)

    threshold = np.max(intensity) * threshold_ratio
    candidates = []

    for i in range(1, len(intensity) - 1):
        if intensity[i] > intensity[i - 1] and intensity[i] >= intensity[i + 1]:
            if intensity[i] >= threshold:
                candidates.append(i)

    if not candidates:
        return np.array([], dtype=int)

    # 合併距離太近的峰，只保留較高的
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


def build_calibration(pixel_peaks, known_lines):
    """
    用峰值 pixel 與已知波長建立校正
    2點用一次，3點以上用二次
    """
    n = min(len(pixel_peaks), len(known_lines))
    if n < 2:
        raise ValueError("校正至少需要 2 個峰")

    x = np.array(pixel_peaks[:n], dtype=float)
    y = np.array(known_lines[:n], dtype=float)

    degree = 1 if n < 3 else 2
    coeffs = np.polyfit(x, y, degree)
    return coeffs


def apply_calibration(x, coeffs):
    return np.polyval(coeffs, x)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/calibrate", methods=["POST"])
def calibrate():
    global calibration_coeffs

    if "image" not in request.files:
        return jsonify({"error": "沒有上傳汞燈圖片"}), 400

    file = request.files["image"]

    y1 = request.form.get("y1")
    y2 = request.form.get("y2")

    try:
        y1 = int(y1) if y1 not in (None, "") else None
        y2 = int(y2) if y2 not in (None, "") else None
    except ValueError:
        return jsonify({"error": "y1 或 y2 格式錯誤"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "圖片讀取失敗"}), 400

    try:
        intensity, used_y1, used_y2 = extract_spectrum(image, y1=y1, y2=y2)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    peaks = detect_peaks_simple(intensity, min_distance=20, threshold_ratio=0.12)

    if len(peaks) < 2:
        return jsonify({"error": "偵測到的峰太少，無法校正"}), 400

    # 取最亮的幾個峰
    peak_strengths = intensity[peaks]
    sorted_idx = np.argsort(peak_strengths)[::-1]
    peaks = peaks[sorted_idx]

    use_n = min(len(peaks), len(MERCURY_LINES))
    selected_peaks = np.sort(peaks[:use_n])

    # 假設由左到右是短波到長波
    known_lines = MERCURY_LINES[:use_n]

    try:
        calibration_coeffs = build_calibration(selected_peaks, known_lines)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    x_all = np.arange(len(intensity), dtype=float)
    wavelength_all = apply_calibration(x_all, calibration_coeffs)

    spectrum_data = []
    for x, wl, inten in zip(x_all, wavelength_all, intensity):
        spectrum_data.append({
            "x": int(x),
            "wavelength": round(float(wl), 3),
            "intensity": round(float(inten), 3)
        })

    calibrated_peak_wavelengths = apply_calibration(selected_peaks, calibration_coeffs)

    return jsonify({
        "message": "校準完成",
        "used_y1": int(used_y1),
        "used_y2": int(used_y2),
        "coefficients": [float(c) for c in calibration_coeffs],
        "detected_peak_pixels": [int(p) for p in selected_peaks],
        "matched_mercury_lines": [float(v) for v in known_lines],
        "calibrated_peak_wavelengths": [round(float(v), 3) for v in calibrated_peak_wavelengths],
        "spectrum": spectrum_data
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    global calibration_coeffs

    if calibration_coeffs is None:
        return jsonify({"error": "尚未校準，請先上傳汞燈圖片做校準"}), 400

    if "image" not in request.files:
        return jsonify({"error": "沒有上傳圖片"}), 400

    file = request.files["image"]

    y1 = request.form.get("y1")
    y2 = request.form.get("y2")

    try:
        y1 = int(y1) if y1 not in (None, "") else None
        y2 = int(y2) if y2 not in (None, "") else None
    except ValueError:
        return jsonify({"error": "y1 或 y2 格式錯誤"}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "圖片讀取失敗"}), 400

    try:
        intensity, used_y1, used_y2 = extract_spectrum(image, y1=y1, y2=y2)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    peak_x = int(np.argmax(intensity))
    peak_intensity = float(intensity[peak_x])
    peak_wavelength = float(apply_calibration(np.array([peak_x], dtype=float), calibration_coeffs)[0])

    x_all = np.arange(len(intensity), dtype=float)
    wavelength_all = apply_calibration(x_all, calibration_coeffs)

    spectrum_data = []
    for x, wl, inten in zip(x_all, wavelength_all, intensity):
        spectrum_data.append({
            "x": int(x),
            "wavelength": round(float(wl), 3),
            "intensity": round(float(inten), 3)
        })

    peaks = detect_peaks_simple(intensity, min_distance=20, threshold_ratio=0.12)
    multi_peaks = []
    for p in peaks:
        multi_peaks.append({
            "x": int(p),
            "wavelength": round(float(apply_calibration(np.array([p], dtype=float), calibration_coeffs)[0]), 3),
            "intensity": round(float(intensity[p]), 3)
        })

    return jsonify({
        "used_y1": int(used_y1),
        "used_y2": int(used_y2),
        "peak_x": peak_x,
        "peak_wavelength": round(peak_wavelength, 3),
        "peak_intensity": round(peak_intensity, 3),
        "all_peaks": multi_peaks,
        "spectrum": spectrum_data
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
