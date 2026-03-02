"""
Wasel v4 Pro — On-Premise Server (Gemma 3)
==========================================
Drop-in replacement for the Colab/Gemini version.
Runs entirely on local GPU — no internet needed after model download.

Requirements:
  pip install transformers torch accelerate flask flask-cors pillow

Usage:
  python server_onprem.py --model google/gemma-3-4b-it --port 5000

Access:
  http://localhost:5000  (same machine)
  http://SERVER_IP:5000  (from any device on the same network)
"""

import argparse, threading, time, base64, io, logging
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wasel")

# ═══════════════════════════════════════
# 1. PARSE ARGS
# ═══════════════════════════════════════
parser = argparse.ArgumentParser(description="Wasel v4 On-Prem Server")
parser.add_argument("--model", default="google/gemma-3-4b-it",
                    help="HuggingFace model ID (default: google/gemma-3-4b-it)")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--host", default="0.0.0.0", help="Bind address (0.0.0.0 = all interfaces)")
parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
parser.add_argument("--max-tokens", type=int, default=30)
args = parser.parse_args()

# ═══════════════════════════════════════
# 2. LOAD GEMMA MODEL
# ═══════════════════════════════════════
logger.info(f"Loading model: {args.model} (dtype={args.dtype})...")
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
model = Gemma3ForConditionalGeneration.from_pretrained(
    args.model, device_map="auto", torch_dtype=DTYPE_MAP[args.dtype]
)
processor = AutoProcessor.from_pretrained(args.model)
DEVICE = next(model.parameters()).device
logger.info(f"✅ Model loaded on {DEVICE} | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ═══════════════════════════════════════
# 3. PROMPT
# ═══════════════════════════════════════
PROMPT = """You are a sign language interpreter.
If you see a hand gesture or sign: reply ONLY the meaning (1-3 words).
If no gesture: reply ...
No explanations. Just the word."""

def ask_ai(pil_image):
    """Send image to local Gemma model and get translation."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": PROMPT}
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
    # Decode only the NEW tokens (skip prompt tokens)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()

# Warm-up inference
logger.info("Warming up model...")
dummy = Image.new("RGB", (100, 100), (0, 0, 0))
_ = ask_ai(dummy)
logger.info("✅ Model warm-up complete")

# ═══════════════════════════════════════
# 4. FLASK APP (identical to Colab version)
# ═══════════════════════════════════════
app = Flask(__name__)
CORS(app)

PAGE = r"""
<!DOCTYPE html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Wasel v4 Pro</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;font-family:'Segoe UI',sans-serif;overflow:hidden;height:100vh}
#c{position:relative;width:100vw;height:100vh}
video{width:100%;height:100%;object-fit:cover;transform:scaleX(-1)}
#top{position:absolute;top:0;left:0;right:0;padding:18px 28px;
background:linear-gradient(180deg,rgba(0,0,0,0.85) 0%,transparent 100%)}
#brand{color:#666;font-size:13px;letter-spacing:3px;text-transform:uppercase}
#txt{color:#00ff88;font-size:42px;font-weight:700;margin-top:6px;
text-shadow:0 2px 20px rgba(0,255,136,0.4);min-height:55px;transition:all .3s}
#bot{position:absolute;bottom:16px;right:24px;color:#444;font-size:12px}
</style></head><body>
<div id='c'>
<video id='v' autoplay playsinline muted></video>
<div id='top'><div id='brand'>WASEL v4 PRO — ON-PREMISE AI TRANSLATOR</div>
<div id='txt'>Starting camera...</div></div>
<div id='bot'>AI Local GPU</div>
</div>
<canvas id='cv' style='display:none'></canvas>
<script>
const v=document.getElementById('v'),cv=document.getElementById('cv'),
cx=cv.getContext('2d'),tx=document.getElementById('txt'),
bt=document.getElementById('bot');
let busy=false;
navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'user'}})
.then(s=>{v.srcObject=s;tx.textContent='Show a sign...';tx.style.color='#555';
setInterval(go,2000)}).catch(e=>{tx.textContent='Camera: '+e.message});
function go(){if(busy)return;busy=true;cv.width=512;cv.height=384;
cx.drawImage(v,0,0,512,384);
const d=cv.toDataURL('image/jpeg',0.6);
bt.textContent='⚡ Analyzing...';
fetch('/translate',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({image:d})})
.then(r=>r.json()).then(d=>{
const t=d.translation||'...';
if(t==='...'||t.length<2){tx.textContent='Show a sign...';tx.style.color='#555'}
else{tx.textContent=t;tx.style.color='#00ff88'}
bt.textContent='⚡ AI Local GPU — '+new Date().toLocaleTimeString();busy=false
}).catch(e=>{bt.textContent='Err: '+e;busy=false})}
</script></body></html>
"""

@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")

@app.route("/translate", methods=["POST"])
def translate():
    t0 = time.time()
    try:
        data = request.json
        img_bytes = base64.b64decode(data["image"].split(",")[1])
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result = ask_ai(pil)
        logger.info(f"Translation: '{result}' ({time.time()-t0:.2f}s)")
        return jsonify({"translation": result})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"translation": f"Error: {str(e)[:40]}"})

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": args.model,
        "device": str(DEVICE),
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0
    })

# ═══════════════════════════════════════
# 5. START
# ═══════════════════════════════════════
if __name__ == "__main__":
    print("\n══════════════════════════════════════")
    print("  🤟 Wasel v4 Pro — ON-PREMISE")
    print("══════════════════════════════════════")
    print(f"  Model: {args.model}")
    print(f"  Device: {DEVICE}")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Health: http://{args.host}:{args.port}/health")
    print("══════════════════════════════════════\n")
    app.run(host=args.host, port=args.port, debug=False)
