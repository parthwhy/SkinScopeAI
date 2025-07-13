# SkinScope AI 🧠✨  
*AI-Powered Skin Analysis & Personalized Skincare Recommender*

SkinScope AI is an end-to-end computer vision solution that analyzes your skin in real-time using your webcam, detects common concerns like dryness, oiliness, and texture irregularities, and recommends tailored skincare products — all powered by AI.

---

## 🖼️ Overview

This project combines **Flask**, **OpenCV**, **MediaPipe**, and **HuggingFace** to deliver:

- 📷 Live facial scanning via webcam  
- 💡 Deep analysis of hydration, oiliness, texture, and redness  
- 🧴 Smart product recommendations based on your skin profile  
- 📊 Radar charts + dynamic UI to visualize skin health  

---

## 📂 Directory Structure

SkinScope-AI/
├── app.py # Main Flask server
├── skin_logic.py # Core skin analysis + AI recommendation
├── requirements.txt # Python dependencies
├── .env # HuggingFace API key (excluded from Git)
│
├── templates/
│ └── index.html # Frontend HTML
├── static/
│ ├── js/
│ │ └── script.js # Camera + analysis JS
│ ├── css/
│ │ └── style.css
│ └── images/ # Icons, product visuals, etc.

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/your-username/skinscope-ai.git
cd skinscope-ai
2️⃣ Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
# Activate it:
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On macOS/Linux
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Add Hugging Face API Key
Create a .env file at the root with your key:

env
Copy
Edit
HUGGINGFACE_API_KEY=your_key_here
Get your API key from: https://huggingface.co/settings/tokens

💻 Run the App
▶️ Start the Backend Server
bash
Copy
Edit
python app.py
It will run locally at:
http://127.0.0.1:5000

🌐 Open the Frontend
Open index.html from the /templates/ directory in your browser.

✅ Camera permissions will be required.

🧠 AI Skin Analysis
The AI will:

Detect facial landmarks using MediaPipe

Segment into T-zone and U-zone

Analyze for:

Hydration

Oiliness

Redness

Texture irregularities

Display:

🔵 Radar chart

📈 Score bars

🧴 Personalized product cards

🧴 Product Recommendations
Automatically matched to your skin type and concerns using rule-based logic and HuggingFace AI.

Each product includes:

Name & description

Skin compatibility

Concern targeting

Link to purchase

⚙️ Technologies Used
Stack	Libraries & Tools
Frontend	HTML, CSS (Bootstrap), JS, Chart.js
Backend	Python, Flask, Flask-CORS
AI & ML	OpenCV, MediaPipe, HuggingFace Inference
Utilities	dotenv, PIL, NumPy, base64, JSON

📦 Requirements
To regenerate or share dependencies:

bash
Copy
Edit
pip freeze > requirements.txt
🛡️ Security
.env is ignored via .gitignore

Image data is processed locally only

HuggingFace API runs in the backend securely

📸 Preview
Optional: Insert screenshots of live demo or radar chart UI

📌 Notes
Tested on Chrome 114+, Firefox 115+

Works best with clear lighting

Camera permissions required for analysis

All product data/images can be replaced or extended

✅ Future Improvements
Deploy to HuggingFace Spaces or Render

Save past scans to local history

Add mobile-first responsive view

Incorporate AI model for diagnosis prediction

📄 License
This project is licensed under the MIT License.
Feel free to use, modify, or contribute!

✨ Author
Parth Patel
Reach out: LinkedIn | GitHub

“Don’t just care for your skin — understand it.”

yaml
Copy
Edit

---

Let me know if you want:
- A `README.pdf` version for offline sharing  
- A downloadable zip bundle  
- GitHub Actions for deployment or linting  
- Or a `setup.bat` script to auto-run the environment  

You're good to ship 🚀
