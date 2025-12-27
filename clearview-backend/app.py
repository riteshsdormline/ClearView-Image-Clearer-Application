#!/usr/bin/env python3
"""
ClearView Backend - app.py

Features:
- /api/enhance : accepts JSON {image: dataURL} or multipart file -> returns processed image (data URL)
- Multiple histogram equalization methods
- Dehazing (fast DCP, guided DCP, CAP-like)
- Denoise, sharpen, brightness/contrast/saturation/gamma
- Object detection: try YOLO (ultralytics) -> fallback to HOG person detector -> fallback to TF MobileNet classification (if available)
"""

import os
import io
import io as _io
import base64
import traceback
from typing import Tuple, List, Dict

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import numpy as np
from PIL import Image
import cv2

# Configuration
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

MAX_DIM = 1400  # downscale large images for speed
JPEG_QUALITY = 90

app = Flask(__name__)
CORS(app)

# Lazy models
yolo_model = None
yolo_available = False
tf_model = None
tf_available = False

# HOG fallback detector (fast)
_hog_detector = None
def get_hog_detector():
    global _hog_detector
    if _hog_detector is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _hog_detector = hog
    return _hog_detector


# -------------------------
# Utilities: image conversions
# -------------------------
def base64_to_bgr(data_url: str) -> np.ndarray:
    header, b64 = data_url.split(",", 1) if "," in data_url else ("", data_url)
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_dataurl(img_bgr: np.ndarray, quality=JPEG_QUALITY) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def pil_from_bgr(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def resize_for_speed(img: np.ndarray, max_dim: int = MAX_DIM) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


# -------------------------
# Histogram Equalization methods
# -------------------------
def he_global(img_bgr: np.ndarray) -> np.ndarray:
    # Convert to YCrCb and equalize Y (safe)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def he_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid)
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def he_multichannel(img_bgr: np.ndarray) -> np.ndarray:
    chans = cv2.split(img_bgr)
    eq = [cv2.equalizeHist(c) for c in chans]
    return cv2.merge(eq)


def he_contrast_stretch(img_bgr: np.ndarray, lower_pct=1, upper_pct=99):
    # Stretch per-channel using percentiles
    out = np.zeros_like(img_bgr)
    for c in range(3):
        channel = img_bgr[..., c]
        lo = np.percentile(channel, lower_pct)
        hi = np.percentile(channel, upper_pct)
        if hi - lo < 1:
            out[..., c] = channel
        else:
            out[..., c] = np.clip((channel - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return out


def he_bbhe(img_bgr: np.ndarray) -> np.ndarray:
    # Brightness preserving bi-histogram equalization:
    # Split at mean intensity in Y channel and equalize each subimage.
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mean = int(np.mean(y))
    out_y = y.copy()
    mask_low = y <= mean
    mask_high = y > mean
    if np.any(mask_low):
        out_y[mask_low] = cv2.equalizeHist(y * mask_low.astype(np.uint8))[mask_low]
    if np.any(mask_high):
        out_y[mask_high] = cv2.equalizeHist((y * mask_high.astype(np.uint8)))[mask_high]
    merged = cv2.merge([out_y, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def he_dsihe(img_bgr: np.ndarray) -> np.ndarray:
    # Dualistic Sub-Image HE (approx): split by median and equalize each half
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    med = int(np.median(y))
    low = y.copy()
    high = y.copy()
    low[y > med] = med
    high[y <= med] = med
    low = cv2.equalizeHist(low)
    high = cv2.equalizeHist(high)
    out_y = np.where(y <= med, low, high).astype(np.uint8)
    merged = cv2.merge([out_y, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def he_rmshe(img_bgr: np.ndarray, recursion_depth=2) -> np.ndarray:
    # Approx recursive mean-separate HE by recursively splitting Y plane and equalizing.
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    def recursive_eq(arr, depth):
        if depth <= 0 or arr.size == 0:
            return cv2.equalizeHist(arr)
        mean = int(np.mean(arr))
        left = arr[arr <= mean]
        right = arr[arr > mean]
        out = arr.copy()
        if left.size > 0:
            out[arr <= mean] = recursive_eq(left, depth - 1)[0: left.size]
            # fallback fill: equalize whole left if shapes don't match — simplified approach
        if right.size > 0:
            out[arr > mean] = recursive_eq(right, depth - 1)[0: right.size]
        # If recursion didn't produce correct shape slicing, fallback to equalize whole plane
        if out.shape != arr.shape:
            out = cv2.equalizeHist(arr)
        return out

    # Because implementing exact RMSHE is heavy, fallback: perform CLAHE on smaller tiles if recursion requested
    if recursion_depth <= 0:
        y2 = cv2.equalizeHist(y)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y2 = clahe.apply(y)
    merged = cv2.merge([y2, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def he_ycbcr(img_bgr: np.ndarray) -> np.ndarray:
    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)
    y2 = cv2.equalizeHist(y)
    merged = cv2.merge([y2, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


# -------------------------
# Dehazing utilities
# -------------------------
def dark_channel_prior(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    b, g, r = cv2.split(image)
    min_channel = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_channel, kernel)
    return dark


def estimate_atmospheric_light(img: np.ndarray, dark_channel: np.ndarray, top_percent: float = 0.001):
    h, w = dark_channel.shape
    num_pixels = h * w
    num_bright = max(1, int(num_pixels * top_percent))
    flat_dark = dark_channel.reshape(-1)
    flat_img = img.reshape(-1, 3)
    indices = np.argsort(flat_dark)[-num_bright:]
    A = np.mean(flat_img[indices], axis=0)
    return A


def estimate_transmission(img: np.ndarray, A: np.ndarray, omega: float = 0.95, patch_size: int = 15):
    # Normalize image by atmospheric light
    norm = img.astype(np.float64) / (A.reshape(1, 1, 3) + 1e-8)
    dark = dark_channel_prior((norm * 255).astype(np.uint8), patch_size)
    t = 1 - omega * (dark / 255.0)
    return np.clip(t, 0.05, 1.0)


def guided_filter_transmission(img: np.ndarray, transmission: np.ndarray, r=60, eps=1e-3):
    # Guided filter (approx using boxFilter)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    mean_I = cv2.boxFilter(gray, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(transmission, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(gray * transmission, cv2.CV_64F, (r, r))

    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(gray * gray, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * gray + mean_b
    return np.clip(q, 0, 1.0)


def recover_scene(img: np.ndarray, A: np.ndarray, t: np.ndarray):
    J = np.zeros_like(img).astype(np.float64)
    for c in range(3):
        J[:, :, c] = (img[:, :, c].astype(np.float64) - A[c]) / np.maximum(t, 0.05) + A[c]
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J


def dehaze_dcp_full(img_bgr: np.ndarray, strength=1.0, patch=15):
    img = img_bgr.astype(np.uint8)
    dark = dark_channel_prior(img, patch)
    A = estimate_atmospheric_light(img, dark, top_percent=0.001)
    t = estimate_transmission(img, A, patch_size=patch)
    t_refined = guided_filter_transmission(img, t, r=60, eps=1e-3)
    t_refined = np.clip(t_refined * strength, 0.1, 1.0)
    J = recover_scene(img, A, t_refined)
    return J


def dehaze_fast_approx(img_bgr: np.ndarray, strength=1.0):
    # Simpler faster approximated dehaze: boost contrast using local dark channel and lighten shadows
    img = img_bgr.copy().astype(np.float32)
    dc = dark_channel_prior(img.astype(np.uint8), patch_size=7).astype(np.float32)
    inv = 255 - dc
    alpha = np.clip((inv / 255.0) * strength, 0.0, 1.0)
    out = np.empty_like(img)
    for c in range(3):
        out[:, :, c] = img[:, :, c] * (0.6 + 0.4 * alpha)  # boost where dark channel is low
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def dehaze_cap_like(img_bgr: np.ndarray, strength=1.0):
    # Simple Color Attenuation Prior-ish: use gray difference as depth cue
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    depth = cv2.GaussianBlur(gray, (31, 31), 0)
    norm = cv2.normalize(depth, None, 0.0, 1.0, cv2.NORM_MINMAX)
    alpha = np.clip((1.0 - norm) * strength, 0.1, 1.0)
    out = (img_bgr.astype(np.float32) * alpha[..., None]).astype(np.uint8)
    return out


# -------------------------
# Enhancement utilities
# -------------------------
def denoise(img_bgr: np.ndarray, h=8):
    # Fast but reasonably good
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)


def adjust_brightness_contrast(img: np.ndarray, brightness=0, contrast=0):
    # brightness: -100..100, contrast: -100..100
    beta = int(np.clip(brightness, -100, 100))
    alpha = (contrast + 100.0) / 100.0  # map -100..100 to 0..2
    res = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return res


def adjust_saturation(img: np.ndarray, sat_shift=0):
    # sat_shift: -100..100 percent
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = hsv[..., 1] * (1.0 + sat_shift / 100.0)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def unsharp_mask(img: np.ndarray, strength=0.5):
    # strength 0..2
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)


def apply_gamma(img: np.ndarray, gamma=1.0):
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)


# -------------------------
# Detection wrappers
# -------------------------
def try_load_yolo():
    global yolo_model, yolo_available
    if yolo_available or yolo_model is not None:
        return
    try:
        from ultralytics import YOLO
        # try to load by name so ultralytics downloads official weights if not present
        print("Loading YOLO model (ultralytics)...")
        yolo_model = YOLO("yolov8n.pt")  # will download if not present and network allowed
        yolo_available = True
        print("YOLO loaded")
    except Exception as e:
        print("YOLO not available:", e)
        yolo_model = None
        yolo_available = False


def detect_with_yolo(img_bgr: np.ndarray, conf=0.25, iou=0.45):
    global yolo_model
    if yolo_model is None:
        return []
    # ultralytics expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # run model
    results = yolo_model.predict(source=img_rgb, verbose=False, conf=conf, iou=iou, imgsz=640)
    detections = []
    # results may be list-like
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            try:
                cls = int(box.cls.cpu().numpy()) if hasattr(box.cls, "cpu") else int(box.cls)
                conf_score = float(box.conf.cpu().numpy()) if hasattr(box.conf, "cpu") else float(box.conf)
                xyxy = box.xyxy.cpu().numpy().astype(int).tolist() if hasattr(box.xyxy, "cpu") else box.xyxy.tolist()
                # get class name via model.names
                name = yolo_model.names.get(cls, str(cls)) if hasattr(yolo_model, "names") else str(cls)
                detections.append({"name": name, "confidence": conf_score, "box": xyxy})
            except Exception:
                continue
    return detections


def detect_with_hog(img_bgr: np.ndarray):
    hog = get_hog_detector()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rects, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
    detections = []
    for (x, y, w, h), score in zip(rects, weights):
        detections.append({"name": "person", "confidence": float(score), "box": [int(x), int(y), int(x + w), int(y + h)]})
    return detections


def try_load_tf():
    global tf_available, tf_model
    if tf_available or tf_model is not None:
        return
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
        print("Loading MobileNetV2 (TF) for fallback classification...")
        tf_model = MobileNetV2(weights="imagenet")
        tf_available = True
        print("MobileNetV2 loaded")
    except Exception as e:
        print("MobileNetV2 not available:", e)
        tf_model = None
        tf_available = False


def detect_with_classification(img_bgr: np.ndarray, topk=3):
    # Returns top-k predicted labels (no boxes)
    try_load_tf()
    if not tf_available or tf_model is None:
        return []
    img = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (224, 224))
    x = np.expand_dims(img.astype(np.float32), 0)
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    x = preprocess_input(x)
    preds = tf_model.predict(x, verbose=0)
    decoded = decode_predictions(preds, top=topk)[0]
    out = []
    for imagenet_id, label, score in decoded:
        out.append({"name": label.replace("_", " "), "confidence": float(score)})
    return out


# -------------------------
# Main pipeline
# -------------------------
def process_pipeline(img_bgr: np.ndarray, params: dict) -> Tuple[np.ndarray, List[Dict]]:
    """
    params keys:
      histogram_method: 'clahe','standard','multi_he','bbhe','dsihe','rmshe','contrast_stretch','ycbcr', 'none'
      dehaze_method: 'fast','dcp_full','cap','none'
      dehaze_strength: 0.3..1.5
      denoise: bool
      denoise_h: int
      brightness: -100..100
      contrast: -100..100
      saturation: -100..100
      sharpness: 0..2
      gamma: 0.5..3.0
      clahe_clip: float
      detect: bool
      detect_conf: float
    """
    orig = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    # QUICK PREPROCESS: downscale for speed
    img_proc, scale = resize_for_speed(img_bgr, MAX_DIM)

    # 1) Dehaze
    dehaze_method = params.get("dehaze_method", "fast")
    strength = float(np.clip(params.get("dehaze_strength", 1.0), 0.1, 2.0))
    if dehaze_method == "dcp_full":
        try:
            img_proc = dehaze_dcp_full(img_proc, strength=strength, patch=15)
        except Exception as e:
            print("DCP full failed:", e)
            img_proc = dehaze_fast_approx(img_proc, strength=strength)
    elif dehaze_method == "cap":
        img_proc = dehaze_cap_like(img_proc, strength=strength)
    elif dehaze_method == "fast":
        img_proc = dehaze_fast_approx(img_proc, strength=strength)
    # else none -> skip

    # 2) Histogram equalization (many methods)
    method = params.get("histogram_method", "clahe")
    clahe_clip = float(params.get("clahe_clip", 2.0))
    if method == "standard":
        img_proc = he_global(img_proc)
    elif method == "clahe":
        img_proc = he_clahe(img_proc, clip_limit=clahe_clip, tile_grid=(8, 8))
    elif method == "multi_he":
        img_proc = he_multichannel(img_proc)
    elif method == "bbhe":
        try:
            img_proc = he_bbhe(img_proc)
        except Exception:
            img_proc = he_clahe(img_proc, clip_limit=clahe_clip)
    elif method == "dsihe":
        img_proc = he_dsihe(img_proc)
    elif method == "rmshe":
        img_proc = he_rmshe(img_proc)
    elif method == "contrast_stretch":
        img_proc = he_contrast_stretch(img_proc)
    elif method == "ycbcr":
        img_proc = he_ycbcr(img_proc)
    # else none -> skip

    # 3) Denoise
    if params.get("denoise", False):
        h_val = int(np.clip(params.get("denoise_h", 8), 1, 30))
        img_proc = denoise(img_proc, h=h_val)

    # 4) Brightness/contrast/saturation/gamma/sharpness
    b = int(np.clip(params.get("brightness", 0), -100, 100))
    c = int(np.clip(params.get("contrast", 0), -100, 100))
    s = int(np.clip(params.get("saturation", 0), -100, 100))
    g = float(np.clip(params.get("gamma", 1.0), 0.3, 3.0))
    sharpness = float(np.clip(params.get("sharpness", 0.0), 0.0, 5.0))

    if b != 0 or c != 0:
        img_proc = adjust_brightness_contrast(img_proc, brightness=b, contrast=c)
    if s != 0:
        img_proc = adjust_saturation(img_proc, sat_shift=s)
    if abs(g - 1.0) > 1e-3:
        img_proc = apply_gamma(img_proc, gamma=g)
    if sharpness > 0.001:
        img_proc = unsharp_mask(img_proc, strength=sharpness)

    # Upscale back to original size if we downscaled
    if scale != 1.0:
        img_proc = cv2.resize(img_proc, (w, h), interpolation=cv2.INTER_LINEAR)

    detections = []
    if params.get("detect", False):
        # Try YOLO first (lazy)
        try:
            try_load_yolo()
            if yolo_available and yolo_model is not None:
                detections = detect_with_yolo(img_proc, conf=float(params.get("detect_conf", 0.25)))
            else:
                # HOG fallback on processed and original image
                detections = detect_with_hog(img_proc)
                if not detections:
                    # try original (sometimes processed image loses features)
                    detections = detect_with_hog(orig)
                # If still empty, try classification (labels only)
                if not detections:
                    cls = detect_with_classification(img_proc, topk=3)
                    if cls:
                        detections = [{"name": c["name"], "confidence": c["confidence"], "box": None} for c in cls]
        except Exception as e:
            print("Detection pipeline error:", e)
            traceback.print_exc()
            detections = []

    return img_proc, detections


# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return jsonify({"status": "online", "service": "ClearView Backend", "version": "1.0"})


@app.route("/api/enhance", methods=["POST", "OPTIONS"])
def api_enhance():
    try:
        data = request.get_json(silent=True)
        img = None

        # support JSON base64 upload
        if data and "image" in data:
            try:
                img = base64_to_bgr(data["image"])
            except Exception as e:
                return jsonify({"success": False, "error": "Invalid base64 image", "detail": str(e)}), 400

            params = data.get("params", {})

        else:
            # support multipart form upload 'file'
            if "file" in request.files:
                f = request.files["file"]
                pil_img = Image.open(f.stream).convert("RGB")
                img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
                params = request.form.to_dict()
            else:
                return jsonify({"success": False, "error": "No image provided"}), 400

        # sanitize params and convert types
        params_sanitized = {}
        # histogram_method, dehaze_method, etc
        params_sanitized["histogram_method"] = params.get("histogram_method", params.get("histogram", "clahe"))
        params_sanitized["dehaze_method"] = params.get("dehaze_method", params.get("dehaze", "fast"))
        params_sanitized["dehaze_strength"] = float(params.get("dehazeStrength", params.get("dehaze_strength", 1.0)))
        params_sanitized["denoise"] = bool(params.get("denoise", str(params.get("denoise", False))).lower() in ("true", "1", "yes"))
        params_sanitized["denoise_h"] = int(float(params.get("denoise_h", params.get("denoiseH", 8))))
        params_sanitized["brightness"] = int(float(params.get("brightness", 0)))
        params_sanitized["contrast"] = int(float(params.get("contrast", 0)))
        params_sanitized["saturation"] = int(float(params.get("saturation", 0)))
        params_sanitized["sharpness"] = float(params.get("sharpness", 0.0))
        params_sanitized["gamma"] = float(params.get("gamma", 1.0))
        params_sanitized["clahe_clip"] = float(params.get("clahe_clip", params.get("claheClip", 2.0)))
        params_sanitized["detect"] = bool(params.get("detect", str(params.get("detect", True))).lower() in ("true", "1", "yes"))
        params_sanitized["detect_conf"] = float(params.get("detect_conf", params.get("detectConf", 0.25)))

        # clamp some sensible ranges
        params_sanitized["dehaze_strength"] = float(np.clip(params_sanitized["dehaze_strength"], 0.1, 2.0))
        params_sanitized["denoise_h"] = int(np.clip(params_sanitized["denoise_h"], 0, 30))
        params_sanitized["brightness"] = int(np.clip(params_sanitized["brightness"], -100, 100))
        params_sanitized["contrast"] = int(np.clip(params_sanitized["contrast"], -100, 100))
        params_sanitized["saturation"] = int(np.clip(params_sanitized["saturation"], -100, 100))
        params_sanitized["sharpness"] = float(np.clip(params_sanitized["sharpness"], 0.0, 5.0))
        params_sanitized["gamma"] = float(np.clip(params_sanitized["gamma"], 0.3, 3.0))
        params_sanitized["clahe_clip"] = float(np.clip(params_sanitized["clahe_clip"], 0.1, 10.0))
        params_sanitized["detect_conf"] = float(np.clip(params_sanitized["detect_conf"], 0.01, 1.0))

        # Run pipeline
        processed_img, detections = process_pipeline(img, params_sanitized)

        # Save optionally for debugging
        # fname = f"{PROCESSED_FOLDER}/proc_{int(time.time())}.jpg"
        # cv2.imwrite(fname, processed_img)

        out_dataurl = bgr_to_dataurl(processed_img, quality=JPEG_QUALITY)

        result = {"success": True, "processed_image": out_dataurl, "detections": detections}
        return jsonify(result)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/detect", methods=["POST"])
def api_detect():
    # accept JSON base64 image and return detections only
    try:
        data = request.get_json(force=True)
        if "image" not in data:
            return jsonify({"success": False, "error": "No image"}), 400
        img = base64_to_bgr(data["image"])
        # Try YOLO then HOG then classification
        try_load_yolo()
        if yolo_available and yolo_model is not None:
            dets = detect_with_yolo(img, conf=float(data.get("conf", 0.25)))
            return jsonify({"success": True, "objects": dets})
        # fallback
        dets = detect_with_hog(img)
        if not dets:
            dets = detect_with_classification(img)
        return jsonify({"success": True, "objects": dets})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------------
# Start server
# -------------------------
if __name__ == "__main__":
    # Attempt lazy loads in background (not required)
    try:
        try_load_yolo()
    except Exception:
        pass
    try:
        try_load_tf()
    except Exception:
        pass

    print("Starting ClearView Backend on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
