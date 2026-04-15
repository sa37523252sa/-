import os
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np

app = Flask(__name__)

def pixel_to_wavelength(x, width, wl_min, wl_max):
    if width <= 1:
        return wl_min
    return wl_min + (x / (width - 1)) * (wl_max - wl_min)

def analyze_spectrum(image, wl_min=400, wl_max=700, y1=None, y2=None):
    height, width = image.shape[:2]

    if y1 is None:
        y1 = 0
    if y2 is None:
        y2 = height

    y1 = max(0, y1)
    y2 = min(height, y2)

    if y1 >= y2:
        raise ValueError("無效的分析範圍")

    roi = image[y1:y2, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    intensity = np.sum(gray, axis=0).astype(float)

    peak_x = int(np.argmax(intensity))
    peak_intensity = float(intensity[peak_x])
    peak_wavelength = float(pixel_to_wavelength(peak_x, width, wl_min, wl_max))

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "沒有上傳圖片"}), 400

    file = request.files["image"]

    try:
        wl_min = float(request.form.get("wl_min", 400))
        wl_max = float(request.form.get("wl_max", 700))
    except ValueError:
        return jsonify({"error": "波長範圍格式錯誤"}), 400

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
        result = analyze_spectrum(image, wl_min=wl_min, wl_max=wl_max, y1=y1, y2=y2)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "分析失敗"}), 500

    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
