"""
Wasel v4 Pro — Headless AI Translation API
==========================================
This is a clean, headless REST API designed for integration with external companies.
It takes a base64 image frame and returns the sign language translation as JSON.
It does NOT include any HTML/JS frontend — it is a pure backend engine.

USAGE:
1. Set Gemini API key: export GEMINI_API_KEY="your_key"
2. Run server: python wasel_api.py --port 8000
3. External company sends POST request with frames:

   POST http://your-server:8000/api/v1/translate
   Content-Type: application/json
   {
       "image_base64": "iVBORw0KGgoAAA..."
   }

FORMATTING FOR ARABIC / EGYPTIAN SIGN LANGUAGE (ESL):
To change the language to Arabic or target Egyptian signs, simply edit the PROMPT below!
Gemini understands Arabic natively.
"""

import os, time, base64, io, logging, argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.genai as genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wasel-api")

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--host", default="0.0.0.0")
args = parser.parse_args()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY environment variable. Please set it before running.")

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════
# 🧠 AI ENGINE CONFIGURATION
# ═══════════════════════════════════════
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.0-flash"

# 🌍 HOW TO CUSTOMIZE FOR ARABIC / EGYPTIAN SIGN LANGUAGE
# Gemini supports over 100 languages. To make it translate into Arabic, we just explicitly 
# instruct it to do so in the prompt. To make it look for "Egyptian Sign Language", we name it.
PROMPT_ARABIC_ESL = """
أنت خبير ومترجم للغة الإشارة المصرية (Egyptian Sign Language - ESL).
انظر إلى هذه الصورة لليد والجسم. هل يقوم الشخص بعمل إشارة معينة بلغة الإشارة؟
إذا نعم: أجب فقط بمعنى الإشارة باللغة 'العربية' في كلمة واحدة أو كلمتين كحد أقصى (مثال: شكرا، نعم، لا، مساعدة، سلام).
إذا لم تكن هناك إشارة واضحة: أجب بالضبط بـ: ...
لا تكتب أي شرح، فقط الكلمة.
"""

def analyze_frame(pil_image):
    """Sends frame to Gemini API with the Arabic/ESL prompt."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[PROMPT_ARABIC_ESL, pil_image],
            config=types.GenerateContentConfig(
                max_output_tokens=20,
                temperature=0.1  # Low temp for deterministic, consistent outputs
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return f"Error: {str(e)[:50]}"

# ═══════════════════════════════════════
# 🔌 REST API ENDPOINTS
# ═══════════════════════════════════════

@app.route("/api/v1/translate", methods=["POST"])
def translate_api():
    """
    Primary Endpoint for External Companies.
    Accepts JSON: {"image_base64": "..."}
    Returns JSON: {"translation": "...", "processing_time_ms": 120}
    """
    start_time = time.time()
    
    # 1. Validate request
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.json
    if "image_base64" not in data:
        return jsonify({"error": "Missing 'image_base64' field"}), 400

    try:
        # 2. Decode image
        b64_string = data["image_base64"]
        # Handle cases where data URI scheme is included (e.g., "data:image/jpeg;base64,")
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
            
        img_bytes = base64.b64decode(b64_string)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # 3. Best Practice: Resize image to save bandwidth/API cost before sending to Gemini
        pil_img.thumbnail((512, 512))
        
        # 4. Call AI Core
        result = analyze_frame(pil_img)
        
        # 5. Build response
        ms = int((time.time() - start_time) * 1000)
        logger.info(f"Translation: '{result}' | Time: {ms}ms")
        
        return jsonify({
            "translation": result,
            "processing_time_ms": ms,
            "language": "ar",
            "dialect": "egyptian_sign_language"
        }), 200
        
    except Exception as e:
        logger.error(f"Processing Error: {e}")
        return jsonify({"error": "Failed to process image format"}), 500


@app.route("/api/v1/health", methods=["GET"])
def health_check():
    """Simple endpoint to verify server is running."""
    return jsonify({
        "status": "online",
        "model": MODEL,
        "engine": "wasel-v4-headless",
        "api_version": "1.0"
    }), 200

# ═══════════════════════════════════════
# 🚀 SERVER STARTUP
# ═══════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" 🤟 WASEL V4 PRO — HEADLESS TRANSLATION API")
    print("    Configuration: Arabic / Egyptian Sign Language")
    print("="*50)
    print(f" 📡 Listening on:  http://{args.host}:{args.port}")
    print(f" 🔌 Endpoint:      POST /api/v1/translate")
    print(f" 💓 Health Check:  GET /api/v1/health")
    print("="*50 + "\n")
    
    app.run(host=args.host, port=args.port, debug=False)
