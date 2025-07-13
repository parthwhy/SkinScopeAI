import cv2
import numpy as np
import mediapipe as mp
import os
import json
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

# MediaPipe Initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Zone indices
T_ZONE_INDICES = [10, 67, 103, 297, 332, 151, 168, 197, 5, 4, 1, 2, 6, 8]
U_ZONE_INDICES = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]


def detect_face(image):
    preprocessed = cv2.medianBlur(image, 3)
    lab = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    lab = cv2.merge((l, a, b))
    preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    results = face_mesh.process(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))

    print("🧠 Face Detection:", "Detected" if results.multi_face_landmarks else "❌ No Face Found")
    return results.multi_face_landmarks[0] if results.multi_face_landmarks else None


def extract_facial_zones(image, landmarks):
    h, w = image.shape[:2]

    def get_coords(indices):
        return np.array([(int(landmarks.landmark[i].x * w),
                          int(landmarks.landmark[i].y * h)) for i in indices])

    t_coords = cv2.convexHull(get_coords(T_ZONE_INDICES))
    u_coords = cv2.convexHull(get_coords(U_ZONE_INDICES))

    t_mask = np.zeros((h, w), dtype=np.uint8)
    u_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(t_mask, [t_coords], 255)
    cv2.fillPoly(u_mask, [u_coords], 255)

    k = max(3, min(h, w) // 150)
    k = k + 1 if k % 2 == 0 else k
    kernel = np.ones((k, k), np.uint8)

    t_mask = cv2.morphologyEx(t_mask, cv2.MORPH_CLOSE, kernel)
    u_mask = cv2.morphologyEx(u_mask, cv2.MORPH_CLOSE, kernel)
    t_mask = cv2.GaussianBlur(t_mask, (k, k), 0)
    u_mask = cv2.GaussianBlur(u_mask, (k, k), 0)

    def crop_to_bbox(img, mask):
        if mask.sum() == 0:
            return np.array([]), (0, 0, 0, 0)
        x, y, w, h = cv2.boundingRect(mask)
        return img[y:y + h, x:x + w], (x, y, w, h)

    t_zone, t_rect = crop_to_bbox(image, t_mask)
    u_zone, u_rect = crop_to_bbox(image, u_mask)

    print("🧠 T-zone shape:", t_zone.shape)
    print("🧠 U-zone shape:", u_zone.shape)

    return {
        't_zone': t_zone,
        'u_zone': u_zone,
        't_mask': t_mask,
        'u_mask': u_mask,
        't_rect': t_rect,
        'u_rect': u_rect
    }


def analyze_skin(t_zone, u_zone):
    if t_zone.size == 0 or u_zone.size == 0:
        print("⚠️ No facial region found!")
        return {
            "skin_type": "Normal",
            "moisture": {"t_oiliness": 0.5, "u_dryness": 0.5},
            "concerns": ["Insufficient Data"]
        }

    t_lab = cv2.cvtColor(t_zone, cv2.COLOR_BGR2LAB)
    u_lab = cv2.cvtColor(u_zone, cv2.COLOR_BGR2LAB)
    t_l = t_lab[:, :, 0]
    u_l = u_lab[:, :, 0]

    def detect_skin_type(t_l, u_l):
        t_oiliness = np.mean(t_l) / 255 if t_l.size > 0 else 0.5
        u_dryness = 1 - (np.mean(u_l) / 255) if u_l.size > 0 else 0.5

        if t_oiliness > 0.7 and u_dryness < 0.3:
            return "Oily"
        elif t_oiliness > 0.6 and u_dryness > 0.4:
            return "Combination"
        elif t_oiliness < 0.4 and u_dryness > 0.6:
            return "Dry"
        return "Normal"

    def analyze_moisture(t_l, u_l):
        return {
            "t_oiliness": min(1.0, max(0, np.mean(t_l) / 255 * 1.2)),
            "u_dryness": min(1.0, max(0, 1 - np.mean(u_l) / 255 * 1.1))
        }

    def detect_concerns(t_zone, u_zone):
        concerns = []
        if np.mean(t_lab[:, :, 1]) > 140:
            concerns.append("Redness")
        if np.var(cv2.cvtColor(u_zone, cv2.COLOR_BGR2GRAY)) > 300:
            concerns.append("Uneven Texture")
        if np.mean(cv2.Canny(cv2.cvtColor(u_zone, cv2.COLOR_BGR2GRAY), 50, 150)) > 25:
            concerns.append("Visible Pores")
        return concerns or ["No Major Concerns"]

    analysis = {
        "skin_type": detect_skin_type(t_l, u_l),
        "moisture": analyze_moisture(t_l, u_l),
        "concerns": detect_concerns(t_zone, u_zone)
    }

    print("🧠 Skin analysis:", analysis)
    return analysis


def recommend_products(analysis):
    skin_type = analysis.get("skin_type", "").lower()
    concerns = [c.lower() for c in analysis.get("concerns", [])]

    all_products = [
        {
            "name": "CeraVe Hydrating Cleanser",
            "reason": "Best for dry sensitive skin",
            "url": "https://walmart.com/cerave",
            "img": "/skinscope-ai/static/products/cerave.png",
            "tags": ["dry", "sensitive", "hydration"]
        },
        {
            "name": "The Ordinary Niacinamide Serum",
            "reason": "Reduces redness & oiliness",
            "url": "https://walmart.com/ordinary",
            "img": "/skinscope-ai/static/products/ordinary.png",
            "tags": ["redness", "oiliness", "texture"]
        },
        {
            "name": "Cetaphil Daily Hydrating Lotion",
            "reason": "Lightweight for dry or flaky skin",
            "url": "https://walmart.com/cetaphil",
            "img": "/skinscope-ai/static/products/cetaphil.png",
            "tags": ["dry", "hydration", "sensitive"]
        },
        {
            "name": "Neutrogena Oil-Free Acne Wash",
            "reason": "Great for oily & acne-prone skin",
            "url": "https://walmart.com/neutrogena",
            "img": "/skinscope-ai/static/products/neutrogena.png",
            "tags": ["oiliness", "acne"]
        }
    ]

    # Simple match based on tags
    recommended = []
    for p in all_products:
        if skin_type in p["tags"] or any(concern in p["tags"] for concern in concerns):
            recommended.append(p)

    # Fallback if empty
    if not recommended:
        recommended = all_products[:3]

    return {"recommendations": recommended[:3]}
