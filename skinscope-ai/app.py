from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
from skin_logic import detect_face, extract_facial_zones, analyze_skin, recommend_products

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)

        # Convert to OpenCV format
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        landmarks = detect_face(image_cv2)
        if not landmarks:
            return jsonify({"error": "No face detected"}), 400

        results = extract_facial_zones(image_cv2, landmarks)
        analysis = analyze_skin(results["t_zone"], results["u_zone"])
        recommendations = recommend_products(analysis)
        print("🧠 Skin analysis:", analysis)

        recommendations = recommend_products(analysis)
        print("🛍️ Recommended:", recommendations)
        return jsonify({
            "skin_type": analysis["skin_type"],
            "t_oiliness": round(analysis["moisture"]["t_oiliness"], 2),
            "u_dryness": round(analysis["moisture"]["u_dryness"], 2),
            "concerns": analysis["concerns"],
            "products": recommendations["recommendations"]
        })

    except Exception as e:
        print("❌ Backend error:", str(e))
        return jsonify({"error": "Server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
