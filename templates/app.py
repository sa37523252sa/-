from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

def pixel_to_wavelength(x, width, wl_min, wl_max):
    """把 x 像素位置轉成波長"""
    return wl_min + (x / (width - 1)) * (wl_max - wl_min)

def analyze_spectrum(image, wl_min=400, wl_max=700, y1=None, y2=None):
    """
    分析光譜圖片
    image: OpenCV 讀進來的 BGR 圖
    wl_min, wl_max: 左右邊界波長
    y1, y2: 要分析的縱向範圍，若不給就抓整張
    """

    height, width = image.shape[:2]

    # 若沒指定分析區域，就用整張
    if y1 is None:
        y1 = 0
    if y2 is None:
        y2 = height

    roi = image[y1:y2, :]

    # 轉灰階
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 每個 x 欄位做亮度總和，得到一維光譜
    intensity = np.sum(gray, axis=0)

    # 找主峰
    peak_x = int(np.argmax(intensity))
    peak_intensity = float(intensity[peak_x])
    peak_wavelength = float(pixel_to_wavelength(peak_x, width, wl_min, wl_max))

    # 全部資料一起回傳
    spectrum_data = []
    for x in range(width):
        wavelength = pixel_to_wavelength(x, width, wl_min, wl_max)
        spectrum_data.append({
            "x": x,
            "wavelength": round(float(wavelength), 2),
            "intensity": float(intensity[x])
        })

    return {
        "peak_x": peak_x,
        "peak_wavelength": round(peak_wavelength, 2),
        "peak_intensity": round(peak_intensity, 2),
        "spectrum": spectrum_data
    }

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "沒有上傳圖片"}), 400

    file = request.files["image"]

    # 波長範圍，可由前端傳入
    wl_min = float(request.form.get("wl_min", 400))
    wl_max = float(request.form.get("wl_max", 700))

    # 可選：指定分析區域
    y1 = request.form.get("y1")
    y2 = request.form.get("y2")
    y1 = int(y1) if y1 is not None and y1 != "" else None
    y2 = int(y2) if y2 is not None and y2 != "" else None

    # 讀圖片
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "圖片讀取失敗"}), 400

    result = analyze_spectrum(image, wl_min=wl_min, wl_max=wl_max, y1=y1, y2=y2)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
