# clearview-backend/app.py
import os
import io
import json
import base64
import traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

# Try to import YOLO (ultralytics). If unavailable, detection is disabled gracefully.
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ----------------- Helpers -----------------

def read_image_from_file_storage(filestorage):
    img_bytes = filestorage.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def read_image_from_dataurl(dataurl):
    header, encoded = dataurl.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def image_to_dataurl_bgr(img_bgr, ext=".jpg", quality=92):
    success, buf = cv2.imencode(ext, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError("Failed to encode image")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def downscale_if_needed(img, max_dim=1400):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def luminance_std(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))

# ----------------- Processing primitives -----------------

def denoise_nl_means(img_bgr, h=8):
    # fast Non-local Means for color images
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)

def median_then_bilateral(img_bgr, ksize=3, d=9, sigmaColor=75, sigmaSpace=75):
    # median to remove salt-and-pepper / speckle then bilateral for edge-preserving smoothing
    median = cv2.medianBlur(img_bgr, ksize)
    bilateral = cv2.bilateralFilter(median, d, sigmaColor, sigmaSpace)
    return bilateral

def apply_clahe_bgr(img_bgr, clip_limit=2.0, tile=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def global_hist_equalization_ycrcb(img_bgr):
    # mild global histogram equalization on Y channel (YCrCb)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

def simple_dehaze(img_bgr, strength=1.0):
    # simplified DCP-like approach (fast and safe)
    img = img_bgr.astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # pick brightest pixel for A
    _, _, _, max_loc = cv2.minMaxLoc(gray)
    A = img[int(max_loc[1]), int(max_loc[0]), :]
    norm = img / (A + 1e-6)
    dark = np.min(norm, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    dark = cv2.erode(dark, kernel)
    t = np.clip((1 - 0.95 * dark) * strength, 0.2, 1.0)
    J = np.empty_like(img)
    for c in range(3):
        J[:,:,c] = (img[:,:,c] - A[c]) / t + A[c]
    return np.clip(J, 0, 255).astype(np.uint8)

def unsharp_mask(img_bgr, strength=0.6):
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=3)
    return cv2.addWeighted(img_bgr, 1.0 + strength, gaussian, -strength, 0)

# ----------------- Detection -----------------

YOLO_MODEL = None
def load_yolo():
    global YOLO_MODEL
    if YOLO_MODEL is None and YOLO_AVAILABLE:
        YOLO_MODEL = YOLO("yolov8n.pt")  # small model; downloads if missing

def run_detection(img_bgr):
    if not YOLO_AVAILABLE:
        return []  # empty if ultralytics not installed
    load_yolo()
    # ultralytics accepts numpy arrays
    results = YOLO_MODEL(img_bgr)
    detections = []
    # results can return multiple result objects; iterate
    for r in results:
        if getattr(r, "boxes", None) is None:
            continue
        for box in r.boxes:
            try:
                score = float(box.conf[0]) if hasattr(box, "conf") else float(box.conf.numpy()[0])
                cls = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls.numpy()[0])
                label = r.names.get(cls, str(cls))
                xyxy = box.xyxy[0].tolist() if hasattr(box, "xyxy") else box.xyxy.numpy().tolist()[0]
                detections.append({"label": label, "confidence": score, "box": xyxy})
            except Exception:
                continue
    return detections

def merge_detections(list_of_lists):
    # merge label confidences: keep max confidence per label
    merged = {}
    for lst in list_of_lists:
        for d in lst:
            label = d.get("label") or d.get("name") or "obj"
            conf = float(d.get("confidence", d.get("score", 0)))
            if label not in merged or conf > merged[label]:
                merged[label] = conf
    return [{"name": k, "confidence": float(v)} for k, v in merged.items()]

# ----------------- Main processing route -----------------

@app.route("/")
def home():
    return jsonify({"status": "online", "service": "ClearView API", "version": "1.0"})

@app.route("/api/enhance", methods=["POST"])
def enhance():
    try:
        # Accept both multipart/form-data and JSON with data URL
        params = {}
        if request.content_type and request.content_type.startswith("multipart/"):
            params_raw = request.form.get("params")
            if params_raw:
                try:
                    params = json.loads(params_raw)
                except:
                    params = {}
            file = request.files.get("file")
            if not file:
                return jsonify({"success": False, "error": "No file in multipart"}), 400
            img = read_image_from_file_storage(file)
        else:
            body = request.get_json(force=True)
            params = body.get("params", {})
            if "image" not in body:
                return jsonify({"success": False, "error": "No image in JSON body"}), 400
            img = read_image_from_dataurl(body["image"])

        # safe param extraction + clamps
        dehaze = bool(params.get("dehaze", True))
        dehaze_strength = float(params.get("dehazeStrength", 1.0))
        dehaze_strength = float(np.clip(dehaze_strength, 0.3, 1.5))
        histogram = bool(params.get("histogram", True))
        brightness = int(np.clip(int(params.get("brightness", 0)), -80, 80))
        contrast = int(np.clip(int(params.get("contrast", 0)), -80, 80))
        saturation = int(np.clip(int(params.get("saturation", 0)), -80, 80))
        sharpness = float(np.clip(float(params.get("sharpness", 0)), 0.0, 3.0))
        denoise_h = float(np.clip(float(params.get("denoise_h", 6.0)), 0.0, 20.0))
        do_detect = bool(params.get("detect", False))

        # core pipeline A (preferred: denoise -> median+bilat -> CLAHE heavy -> adjust -> sharpen)
        img_proc = downscale_if_needed(img, max_dim=1400)
        # initial denoise
        if denoise_h > 0:
            img_proc = denoise_nl_means(img_proc, h=int(denoise_h))
        # remove speckle
        img_proc = median_then_bilateral(img_proc, ksize=3, d=9, sigmaColor=75, sigmaSpace=75)

        # measure contrast (pre)
        pre_contrast = luminance_std(img_proc)

        # apply CLAHE aggressively first (user wanted histogram equalization priority)
        if histogram:
            # adaptive clipLimit: darker / low contrast images -> stronger CLAHE
            clip = 2.0
            if pre_contrast < 18:
                clip = 3.5
            elif pre_contrast < 28:
                clip = 2.8
            img_proc = apply_clahe_bgr(img_proc, clip_limit=clip)

        # optional dehaze (applies after some histogram)
        if dehaze:
            img_proc = simple_dehaze(img_proc, strength=dehaze_strength)

        # mild global hist equalization as additional boost
        img_proc = global_hist_equalization_ycrcb(img_proc)

        # brightness/contrast/saturation adjustments
        img_proc = img_proc.astype(np.float32)
        if brightness != 0:
            img_proc = img_proc + float(brightness)
        if contrast != 0:
            alpha = (contrast + 100.0) / 100.0
            img_proc = (img_proc - 128.0) * alpha + 128.0
        img_proc = np.clip(img_proc, 0, 255).astype(np.uint8)
        if saturation != 0:
            hsv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + saturation / 100.0), 0, 255)
            img_proc = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # final sharpening
        if sharpness > 0:
            img_proc = unsharp_mask(img_proc, strength=min(sharpness, 2.0))

        # safety check: if contrast got worse, fallback to alternate pipeline B
        post_contrast = luminance_std(img_proc)
        if post_contrast < (pre_contrast * 0.9):
            # fallback: milder processing (denoise + moderate CLAHE + gamma correction)
            gamma = 1.1
            fallback = denoise_nl_means(img_proc, h=max(2, int(denoise_h/2)))
            fallback = apply_clahe_bgr(fallback, clip_limit=1.8)
            lut = np.array([((i/255.0)**(1.0/gamma))*255.0 for i in range(256)]).astype("uint8")
            fallback = cv2.LUT(fallback, lut)
            img_proc = fallback

        # final clamp
        img_proc = np.clip(img_proc, 0, 255).astype(np.uint8)

        # detection: run on enhanced image (preferred). If YOLO not available, return empty list.
        detections = []
        if do_detect:
            try:
                detections = run_detection(img_proc)
            except Exception:
                # log but keep running
                traceback.print_exc()
                detections = []

        # encode back to data URL
        dataurl = image_to_dataurl_bgr(img_proc, ext=".jpg", quality=92)

        return jsonify({"success": True, "processed_image": dataurl, "detections": detections})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("Starting ClearView backend on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
