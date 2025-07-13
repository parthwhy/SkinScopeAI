import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt

# Initialize MediaPipe FaceMesh with enhanced settings
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Corrected landmark indices for T-zone and U-zone
T_ZONE_INDICES = [
    10, 67, 103, 297, 332,       # Central forehead
    151, 168, 197, 5, 4, 1, 2,   # Nose bridge
    6, 8                         # Nose tip
]

U_ZONE_INDICES = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323
]

def detect_face(image):
    """Detect face landmarks using MediaPipe with refined landmarks"""
    # Apply preprocessing
    preprocessed = cv2.medianBlur(image, 3)
    lab = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    results = face_mesh.process(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
    return results.multi_face_landmarks[0] if results.multi_face_landmarks else None

def apply_masked_overlay(base, color, mask, alpha=0.3):
    """Blend a color overlay onto base image where mask is set"""
    overlay = base.copy()
    color_layer = np.full_like(base, color)
    blended = cv2.addWeighted(base, 1 - alpha, color_layer, alpha, 0)
    overlay[mask > 0] = blended[mask > 0]
    return overlay

def extract_facial_zones(image, landmarks):
    """Extract refined T-zone and U-zone with adaptive morphological filtering"""
    h, w = image.shape[:2]
    
    def get_coords(indices):
        return np.array([(int(landmarks.landmark[i].x * w), 
                         int(landmarks.landmark[i].y * h)) for i in indices])
    
    # Get convex hull of points for each zone
    t_coords = cv2.convexHull(get_coords(T_ZONE_INDICES))
    u_coords = cv2.convexHull(get_coords(U_ZONE_INDICES))
    
    # Create masks
    t_mask = np.zeros((h, w), dtype=np.uint8)
    u_mask = np.zeros((h, w), dtype=np.uint8)
    
    cv2.fillPoly(t_mask, [t_coords], 255)
    cv2.fillPoly(u_mask, [u_coords], 255)
    
    # Ensure kernel size is odd and at least 3
    kernel_size = max(3, min(h, w) // 150)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply morphological operations
    t_mask = cv2.morphologyEx(t_mask, cv2.MORPH_CLOSE, kernel)
    u_mask = cv2.morphologyEx(u_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply Gaussian blur
    t_mask = cv2.GaussianBlur(t_mask, (kernel_size, kernel_size), 0)
    u_mask = cv2.GaussianBlur(u_mask, (kernel_size, kernel_size), 0)
    
    # Extract regions with bounding box cropping
    def crop_to_bbox(image, mask):
        if mask.sum() == 0:
            return np.array([]), (0,0,0,0)
        x, y, w, h = cv2.boundingRect(mask)
        return image[y:y+h, x:x+w], (x, y, w, h)
    
    t_zone, t_rect = crop_to_bbox(image, t_mask)
    u_zone, u_rect = crop_to_bbox(image, u_mask)
    
    return {
        't_zone': t_zone,
        'u_zone': u_zone,
        't_mask': t_mask,
        'u_mask': u_mask,
        't_rect': t_rect,
        'u_rect': u_rect
    }

def analyze_skin(t_zone, u_zone):
    """Enhanced skin analysis with adaptive techniques"""
    if t_zone.size == 0 or u_zone.size == 0:
        return {
            "skin_type": "Normal",
            "moisture": {"t_oiliness": 0.5, "u_dryness": 0.5},
            "concerns": ["Insufficient Data"]
        }
    
    # Convert to LAB color space
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
        t_oiliness = np.mean(t_l) / 255 if t_l.size > 0 else 0.5
        u_dryness = 1 - (np.mean(u_l) / 255) if u_l.size > 0 else 0.5
        return {
            "t_oiliness": min(1.0, max(0, t_oiliness * 1.2)),
            "u_dryness": min(1.0, max(0, u_dryness * 1.1))
        }
    
    def detect_concerns(t_zone, u_zone):
        concerns = []
        
        # Redness detection
        a_channel = t_lab[:, :, 1]
        if a_channel.size > 0 and np.mean(a_channel) > 140:
            concerns.append("Redness")
        
        # Texture analysis
        if u_zone.size > 0:
            gray_u = cv2.cvtColor(u_zone, cv2.COLOR_BGR2GRAY)
            if np.var(gray_u) > 300:
                concerns.append("Uneven Texture")
        
        # Pore visibility
        if u_zone.size > 0:
            gray_u = cv2.cvtColor(u_zone, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_u, 50, 150)
            if np.mean(edges) > 25:
                concerns.append("Visible Pores")
        
        return concerns or ["No Major Concerns"]
    
    return {
        "skin_type": detect_skin_type(t_l, u_l),
        "moisture": analyze_moisture(t_l, u_l),
        "concerns": detect_concerns(t_zone, u_zone)
    }

def visualize_analysis(image, skin_analysis):
    """Create visualization of analysis results"""
    viz = image.copy()
    h, w = image.shape[:2]
    
    font_scale = max(0.5, min(1.0, w / 1000))
    thickness = max(1, int(w / 500))
    
    cv2.putText(viz, f"Skin Type: {skin_analysis['skin_type']}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    cv2.putText(viz, f"T-Zone Oiliness: {skin_analysis['moisture']['t_oiliness']*100:.1f}%", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
    cv2.putText(viz, f"U-Zone Dryness: {skin_analysis['moisture']['u_dryness']*100:.1f}%", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
    cv2.putText(viz, f"Concerns: {', '.join(skin_analysis['concerns'])}", 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
    
    # Create moisture meter
    moisture_height = 100
    moisture_img = np.zeros((moisture_height, w, 3), dtype=np.uint8)
    
    t_width = int(skin_analysis['moisture']['t_oiliness'] * w)
    cv2.rectangle(moisture_img, (0, 0), (t_width, moisture_height//2), (0, 255, 255), -1)
    
    u_width = int(skin_analysis['moisture']['u_dryness'] * w)
    cv2.rectangle(moisture_img, (0, moisture_height//2), (u_width, moisture_height), (255, 255, 0), -1)
    
    return np.vstack([viz, moisture_img])

def visualize_results(image, results, skin_analysis=None):
    """Create labeled visualization of zones and analysis"""
    viz = image.copy()
    
    viz = apply_masked_overlay(viz, (0, 255, 0), results['t_mask'])
    viz = apply_masked_overlay(viz, (255, 0, 0), results['u_mask'])
    
    cv2.putText(viz, "T-Zone (Forehead/Nose/Chin)", (10,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(viz, "U-Zone (Cheeks)", (10,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    
    plt.figure(figsize=(12,8))
    
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
    plt.title("Zone Visualization")
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(results['t_zone'], cv2.COLOR_BGR2RGB))
    plt.title("T-Zone Extraction")
    plt.axis('off')
    
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(results['u_zone'], cv2.COLOR_BGR2RGB))
    plt.title("U-Zone Extraction")
    plt.axis('off')
    
    if skin_analysis:
        plt.subplot(224)
        analysis_viz = visualize_analysis(image, skin_analysis)
        plt.imshow(cv2.cvtColor(analysis_viz, cv2.COLOR_BGR2RGB))
        plt.title("Skin Analysis Results")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Main execution
image = cv2.imread("selfie.jpg")
if image is None:
    print("Error: Image not found or could not be loaded")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
brightness = np.mean(gray)
if brightness < 80 or brightness > 200:
    print("⚠️ Warning: Suboptimal lighting conditions detected")

landmarks = detect_face(image)

if landmarks:
    print(f"✅ Found {len(landmarks.landmark)} landmarks")
    
    results = extract_facial_zones(image, landmarks)
    skin_analysis = analyze_skin(results['t_zone'], results['u_zone'])
    
    print("\n🔬 Skin Analysis Results:")
    print(f"Skin Type: {skin_analysis['skin_type']}")
    print(f"T-Zone Oiliness: {skin_analysis['moisture']['t_oiliness']*100:.1f}%")
    print(f"U-Zone Dryness: {skin_analysis['moisture']['u_dryness']*100:.1f}%")
    print(f"Key Concerns: {', '.join(skin_analysis['concerns'])}")
    
    visualize_results(image, results, skin_analysis)
    
    cv2.imwrite("t_zone.jpg", results['t_zone'])
    cv2.imwrite("u_zone.jpg", results['u_zone'])
else:
    print("❌ No face detected in the image")

















##open ai integration
import os
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load Hugging Face token
load_dotenv()
client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

# ✅ Use only Hugging Face-hosted models (no `provider=...`)

import re

def recommend_products(skin_analysis):
    products = [
        {"name": "CeraVe Hydrating Cleanser", "type": "cleanser", "skin_type": "dry", "concerns": "dryness", "price": 12.99, "url": "https://walmart.com/cerave"},
        {"name": "Neutrogena Oil-Free Acne Wash", "type": "cleanser", "skin_type": "oily", "concerns": "acne", "price": 8.99, "url": "https://walmart.com/neutrogena"},
        {"name": "La Roche-Posay Effaclar Duo", "type": "treatment", "skin_type": "combination", "concerns": "texture", "price": 19.99, "url": "https://walmart.com/laroche"},
        {"name": "Cetaphil Daily Hydrating Lotion", "type": "moisturizer", "skin_type": "normal", "concerns": "hydration", "price": 14.99, "url": "https://walmart.com/cetaphil"},
        {"name": "The Ordinary Niacinamide Serum", "type": "serum", "skin_type": "all", "concerns": "texture, pores", "price": 6.99, "url": "https://walmart.com/ordinary"},
        {"name": "Paula's Choice BHA Exfoliant", "type": "treatment", "skin_type": "combination", "concerns": "texture, acne", "price": 32.99, "url": "https://walmart.com/paulaschoice"}
    ]

    prompt = f"""
You are a skincare expert recommending Walmart products based on the user's skin profile.

Skin profile:
- Skin Type: {skin_analysis['skin_type']}
- Key Concerns: {', '.join(skin_analysis['concerns'])}
- T-Zone Oiliness: {skin_analysis['moisture']['t_oiliness'] * 100:.1f}%
- U-Zone Dryness: {skin_analysis['moisture']['u_dryness'] * 100:.1f}%

Available products (in JSON):
{json.dumps(products, indent=2)}

Recommend EXACTLY 3 products in this JSON format:
{{
  "recommendations": [
    {{
      "name": "Product Name",
      "reason": "Why it matches in 10 words",
      "url": "Product URL"
    }},
    ...
  ]
}}

Respond only in JSON. No explanations, no markdown.
"""

    # Generate response
    response = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages=[{"role": "user", "content": prompt}]
    )

    reply = response.choices[0].message.content.strip()
    print("🔍 Raw model reply:\n", reply)

    # Extract JSON block using regex
    match = re.search(r'\{[\s\S]+?\}', reply)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            print("⚠️ Couldn't parse cleaned JSON")
    else:
        print("⚠️ No valid JSON found")

    # Fallback
    return {
        "recommendations": [
            {
                "name": "CeraVe Hydrating Cleanser",
                "reason": "Gentle cleanser for normal skin types",
                "url": "https://walmart.com/cerave"
            },
            {
                "name": "The Ordinary Niacinamide Serum",
                "reason": "Improves skin texture and reduces pores",
                "url": "https://walmart.com/ordinary"
            },
            {
                "name": "Cetaphil Daily Hydrating Lotion",
                "reason": "Lightweight moisturizer for balanced hydration",
                "url": "https://walmart.com/cetaphil"
            }
        ]
    }


# ✅ Run the function
if __name__ == "__main__":
    skin_analysis = {
        "skin_type": "Normal",
        "moisture": {"t_oiliness": 0.62, "u_dryness": 0.43},
        "concerns": ["Uneven Texture"]
    }

    recommendations = recommend_products(skin_analysis)

    print("\n✅ Final Recommendations:")
    for i, rec in enumerate(recommendations["recommendations"], 1):
        print(f"{i}. {rec['name']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   URL: {rec['url']}\n")
