import os, io, base64, json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

import torch
import torch.nn as nn
from ultralytics import YOLO


# -------------------- Configuration --------------------
# Paths to saved *state_dict* weights (saved with torch.save(model.state_dict(), ...))
SEVERITY_SD = Path("severity_state.pth")          # required (3 classes)
BINARY_SD   = Path("bin_pretrain_state.pth")      # optional (2 classes)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class order used during training ( model printed: {0:'advanced',1:'early',2:'healthy'})
SEV_CLASSES = ["advanced", "early", "healthy"]
# If you have the binary state dict, we assume {0:'healthy', 1:'peacock'}
BIN_CLASSES = ["healthy", "peacock"]

IMG_SIZE = 448


# -------------------- Leaf crop (HSV) --------------------
def leaf_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = ((h > 25) & (h < 100) & (s > 40) & (v > 40)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m

def crop_leaf(bgr: np.ndarray) -> np.ndarray:
    m = leaf_mask_hsv(bgr)
    ys, xs = np.where(m > 0)
    if len(xs) < 50:  # fallback: center square
        h, w = bgr.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return bgr[y0:y0 + side, x0:x0 + side]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad = int(0.03 * max(bgr.shape[:2]))
    y0 = max(0, y0 - pad); y1 = min(bgr.shape[0] - 1, y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(bgr.shape[1] - 1, x1 + pad)
    return bgr[y0:y1 + 1, x0:x1 + 1]

def preprocess_bgr(bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    crop = crop_leaf(bgr)
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    return out


# -------------------- Model loading --------------------
def load_yolov8_cls_state(state_path: Path, num_classes: int) -> YOLO:
    """
    Build a YOLOv8 classification model with a fresh head and load a plain state_dict safely.
    """
    if not state_path.exists():
        raise FileNotFoundError(f"Missing weights: {state_path}")

    # Build model skeleton
    model = YOLO("yolov8s-cls.yaml", task="classify")
    # Replace head to match num_classes
    head = model.model.model[-1]
    in_feats = head.linear.in_features
    model.model.model[-1].linear = nn.Linear(in_feats, num_classes, bias=True)

    # Load state dict (safe for PyTorch >=2.6 because it's just tensors)
    sd = torch.load(state_path, map_location="cpu")
    if not isinstance(sd, dict):
        raise TypeError(f"Expected a state_dict (dict), got {type(sd)}")

    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    print(f"[load] {state_path.name}: missing={len(missing)} unexpected={len(unexpected)}")

    model.model.to(DEVICE).eval()
    for p in model.model.parameters():
        p.requires_grad = False
    return model


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-9)


# -------------------- Initialize --------------------
print(f"Using device: {DEVICE}")
severity_model = load_yolov8_cls_state(SEVERITY_SD, num_classes=3)

binary_model = None
if BINARY_SD.exists():
    try:
        binary_model = load_yolov8_cls_state(BINARY_SD, num_classes=2)
        print("Binary gate model loaded.")
    except Exception as e:
        print(f"Binary model not loaded (will skip gate): {e}")


# -------------------- Flask app --------------------
app = Flask(__name__)


INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Olive Leaf Classifier</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
    .card { max-width: 920px; margin: auto; padding: 20px; border: 1px solid #e5e7eb; border-radius: 14px; box-shadow: 0 10px 30px rgba(0,0,0,0.06); }
    h1 { margin-top: 0; }
    .row { display: flex; gap: 24px; flex-wrap: wrap; }
    video, canvas, img { width: 360px; max-width: 100%; border-radius: 12px; border: 1px solid #e5e7eb; }
    .btn { padding: 10px 14px; border-radius: 10px; border: 1px solid #ddd; background: #111827; color: white; cursor: pointer; }
    .btn.secondary { background: white; color: #111827; }
    .prob { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .result { margin-top: 10px; padding: 12px; border-radius: 10px; background: #f9fafb; }
    .healthy { color: #059669; font-weight: 600; }
    .warning { color: #b45309; font-weight: 600; }
    .danger  { color: #dc2626; font-weight: 600; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Olive Leaf Classifier</h1>
    <p>Upload a photo or use your camera. The app will crop to the leaf, classify as <b>healthy</b> or <b>not</b>, and if not, report <b>early</b> or <b>advanced</b> peacock spot.</p>

    <div class="row">
      <div>
        <video id="video" autoplay playsinline muted></video>
        <div style="margin-top:8px; display:flex; gap:8px;">
          <button class="btn" id="openCam">Open camera</button>
          <button class="btn secondary" id="snap">Capture</button>
        </div>
      </div>
      <div>
        <canvas id="canvas" width="448" height="448"></canvas>
        <div style="margin-top:8px;">
          <input type="file" id="file" accept="image/*"/>
        </div>
        <div class="result" id="result">No prediction yet.</div>
      </div>
    </div>
  </div>

<script>
async function postImage(blob) {
  const fd = new FormData();
  fd.append("image", blob, "frame.jpg");
  const r = await fetch("/predict", { method: "POST", body: fd });
  const j = await r.json();
  return j;
}

function renderResult(j) {
  const el = document.getElementById("result");
  if (j.error) { el.innerHTML = "<b>Error:</b> " + j.error; return; }

  const p = j.probs || {};
  const fmt = (x) => (x == null ? "-" : (100*x).toFixed(1) + "%");
  let verdict = "";
  if (j.final_label === "healthy") {
    verdict = `<span class="healthy">Healthy</span>`;
  } else if (j.final_label === "early") {
    verdict = `<span class="warning">Not healthy — Early</span>`;
  } else if (j.final_label === "advanced") {
    verdict = `<span class="danger">Not healthy — Advanced</span>`;
  } else {
    verdict = j.final_label;
  }

  el.innerHTML = `
    <div><b>Verdict:</b> ${verdict}</div>
    <div class="prob"><b>Probabilities</b> (advanced / early / healthy):<br>
      ${fmt(p.advanced)} / ${fmt(p.early)} / ${fmt(p.healthy)}
    </div>
    ${j.binary ? `<div class="prob" style="margin-top:6px;">
      <b>Binary gate:</b> ${j.binary.label} (diseased ${fmt(j.binary.prob_diseased)})
    </div>` : ""}
  `;
}

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

document.getElementById("openCam").onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
    video.srcObject = stream;
  } catch (e) {
    alert("Could not open camera: " + e);
  }
};

document.getElementById("snap").onclick = async () => {
  if (!video.srcObject) return alert("Open the camera first.");
  const w = canvas.width, h = canvas.height;
  // draw center crop of the video into canvas for simplicity
  const vw = video.videoWidth, vh = video.videoHeight;
  const side = Math.min(vw, vh);
  const sx = (vw - side) / 2, sy = (vh - side) / 2;
  ctx.drawImage(video, sx, sy, side, side, 0, 0, w, h);
  canvas.toBlob(async (blob) => {
    const j = await postImage(blob);
    renderResult(j);
  }, "image/jpeg", 0.92);
};

document.getElementById("file").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Fit image into canvas
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const nw = img.width * scale, nh = img.height * scale;
    const ox = (canvas.width - nw) / 2, oy = (canvas.height - nh) / 2;
    ctx.drawImage(img, 0, 0, img.width, img.height, ox, oy, nw, nh);
  };
  img.src = URL.createObjectURL(file);

  const j = await postImage(file);
  renderResult(j);
};
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


def read_image_from_request() -> np.ndarray:
    """
    Supports multipart file 'image' or data URL in JSON {image: "data:image/...;base64,...."}
    Returns BGR numpy array.
    """
    if "image" in request.files:
        file = request.files["image"].read()
        img = Image.open(io.BytesIO(file)).convert("RGB")
    else:
        data = request.get_json(silent=True) or {}
        uri = data.get("image")
        if not uri or ";base64," not in uri:
            raise ValueError("No image provided")
        b64 = uri.split(";base64,")[-1]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
    return arr


@torch.inference_mode()
@app.post("/predict")
def predict():
    try:
        bgr = read_image_from_request()
        # training-consistent preprocessing
        pre = preprocess_bgr(bgr, IMG_SIZE)

        # Run severity model (single image as numpy array)
        sev_pred = severity_model.predict([pre], imgsz=IMG_SIZE, conf=0.0, verbose=False)
        probs = sev_pred[0].probs.data.cpu().numpy()  # logits already softmaxed by Ultralytics, but normalize again just in case
        probs = softmax(probs)

        sev = {SEV_CLASSES[i]: float(probs[i]) for i in range(len(SEV_CLASSES))}
        top_idx = int(np.argmax(probs))
        final_label = SEV_CLASSES[top_idx]

        # Optional binary gate if available
        bin_out = None
        if binary_model is not None:
            bin_pred = binary_model.predict([pre], imgsz=IMG_SIZE, conf=0.0, verbose=False)
            p2 = bin_pred[0].probs.data.cpu().numpy()
            p2 = softmax(p2)
            p_diseased = float(p2[1])  # index 1 = 'peacock'
            bin_label = BIN_CLASSES[int(np.argmax(p2))]
            bin_out = {"label": bin_label, "prob_diseased": p_diseased}

            # If binary says "healthy" with very high confidence, prefer that verdict
            if bin_label == "healthy" and p_diseased < 0.25:
                final_label = "healthy"

        return jsonify({
            "final_label": final_label,
            "probs": sev,
            "binary": bin_out
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE})


if __name__ == "__main__":
    # Set this env if you need a specific port: PORT=7860 python app.py
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
