"""ClearView backend in one file.

Goals:
- Upload image
- Enhance / preprocess for clarity
- Detect/classify objects
- Apply editing controls
- Generate short description + tentative location guess
- Return outputs for download
- Docker-friendly

Notes:
- Some parts are intentionally best-effort. Image enhancement can improve visibility, but it cannot guarantee recognition of every hidden or blurred object.
- For object recognition, YOLO is used first for speed; a fallback Hugging Face image-classification pipeline can be added.
- For image description, a vision-language model can be plugged in later.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Optional heavy imports. Keep app usable even if some are missing.
try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None

APP_DIR = Path(__file__).resolve().parent
WORKDIR = APP_DIR / "workdir"
WORKDIR.mkdir(exist_ok=True)

app = FastAPI(title="ClearView Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons
_yolo_model = None
_img_classifier = None
_desc_model = None


@dataclass
class EditParams:
    brightness: float = 1.0
    saturation: float = 1.0
    warmth: float = 1.0
    gamma: float = 1.0
    sharpness: float = 1.0
    denoise: bool = True
    blur: bool = False
    detail: bool = True
    upscale: bool = False


def _unique_path(prefix: str, suffix: str) -> Path:
    return WORKDIR / f"{prefix}_{uuid.uuid4().hex[:12]}{suffix}"


def _read_upload_image(file: UploadFile) -> np.ndarray:
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    np_buf = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Unsupported or corrupted image")
    return img


def _cv_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    rgb = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _apply_gamma(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.1, float(gamma))
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)


def _apply_warmth(img_bgr: np.ndarray, warmth: float) -> np.ndarray:
    """Warmth > 1 adds warm tone, < 1 cools slightly."""
    warmth = float(warmth)
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    r *= warmth
    b /= max(0.2, warmth)
    merged = cv2.merge([b, g, r])
    return np.clip(merged, 0, 255).astype(np.uint8)


def _denoise_and_sharpen(img_bgr: np.ndarray) -> np.ndarray:
    den = cv2.fastNlMeansDenoisingColored(img_bgr, None, 3, 3, 7, 21)
    blur = cv2.GaussianBlur(den, (0, 0), 1.2)
    sharp = cv2.addWeighted(den, 1.35, blur, -0.35, 0)
    return sharp


def _auto_contrast(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance_image(
    img_bgr: np.ndarray,
    strong: bool = True,
) -> np.ndarray:
    """General image enhancement pipeline for clarity and appearance."""
    out = img_bgr.copy()

    # Baseline clarity improvements
    out = _auto_contrast(out)
    out = cv2.bilateralFilter(out, d=9, sigmaColor=75, sigmaSpace=75)

    if strong:
        out = _denoise_and_sharpen(out)

    # Mild edge/detail recovery
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(out, 0.96, edges_bgr, 0.04, 0)

    return out


def apply_edits(img_bgr: np.ndarray, params: EditParams) -> np.ndarray:
    pil = _cv_to_pil(img_bgr)

    if params.brightness != 1.0:
        pil = ImageEnhance.Brightness(pil).enhance(params.brightness)
    if params.saturation != 1.0:
        pil = ImageEnhance.Color(pil).enhance(params.saturation)
    if params.sharpness != 1.0:
        pil = ImageEnhance.Sharpness(pil).enhance(params.sharpness)
    if params.blur:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=1.2))
    if params.detail:
        pil = pil.filter(ImageFilter.DETAIL)

    out = _pil_to_cv(pil)
    out = _apply_warmth(out, params.warmth)
    out = _apply_gamma(out, params.gamma)

    if params.denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)

    if params.upscale:
        h, w = out.shape[:2]
        out = cv2.resize(out, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    return out


def load_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        # Replace with your own model path if desired
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def detect_and_classify_objects(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Detect visible objects. Uncertain/low-confidence objects can be sorted first."""
    results_out: List[Dict[str, Any]] = []

    try:
        model = load_yolo_model()
        results = model.predict(source=img_bgr, verbose=False)
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                name = model.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                area = max(0.0, (x2 - x1) * (y2 - y1))
                results_out.append({
                    "label": name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "area": round(area, 1),
                    "priority": round((1.0 - conf) * (1.0 + min(1.0, 200000.0 / max(1.0, area))), 4),
                })
    except Exception:
        # Fallback: optional Hugging Face classifier on whole image
        if pipeline is None:
            return results_out
        global _img_classifier
        if _img_classifier is None:
            _img_classifier = pipeline("image-classification")
        pil = _cv_to_pil(img_bgr)
        preds = _img_classifier(pil)
        for p in preds[:5]:
            results_out.append({
                "label": p.get("label", "unknown"),
                "confidence": round(float(p.get("score", 0.0)), 4),
                "bbox": None,
                "area": None,
                "priority": round(1.0 - float(p.get("score", 0.0)), 4),
            })

    results_out.sort(key=lambda x: x.get("priority", 0), reverse=True)
    return results_out


def generate_description(img_bgr: np.ndarray) -> Dict[str, str]:
    """Best-effort image description and tentative location guess."""
    # Placeholder heuristic summary if no VLM is installed.
    h, w = img_bgr.shape[:2]
    desc = f"Image of size {w}x{h}, processed for clarity, with visible objects and scene details analyzed."
    location = "unknown location"
    location_hint = "No reliable location cues found in the image."
    return {
        "description": desc[:300],
        "location": location,
        "location_hint": location_hint[:150],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/enhance")
def api_enhance(file: UploadFile = File(...)):
    img = _read_upload_image(file)
    out = enhance_image(img, strong=True)
    path = _unique_path("enhanced", ".jpg")
    cv2.imwrite(str(path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return {
        "message": "Enhanced image created",
        "download_url": f"/download/{path.name}",
        "path": str(path),
    }


@app.post("/classify")
def api_classify(file: UploadFile = File(...)):
    img = _read_upload_image(file)
    processed = enhance_image(img, strong=False)
    objects = detect_and_classify_objects(processed)
    return {"objects": objects}


@app.post("/edit")
def api_edit(
    file: UploadFile = File(...),
    brightness: float = Form(1.0),
    saturation: float = Form(1.0),
    warmth: float = Form(1.0),
    gamma: float = Form(1.0),
    sharpness: float = Form(1.0),
    denoise: bool = Form(True),
    blur: bool = Form(False),
    detail: bool = Form(True),
    upscale: bool = Form(False),
):
    img = _read_upload_image(file)
    params = EditParams(
        brightness=brightness,
        saturation=saturation,
        warmth=warmth,
        gamma=gamma,
        sharpness=sharpness,
        denoise=denoise,
        blur=blur,
        detail=detail,
        upscale=upscale,
    )
    out = apply_edits(img, params)
    path = _unique_path("edited", ".jpg")
    cv2.imwrite(str(path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    return {
        "message": "Edited image created",
        "download_url": f"/download/{path.name}",
        "path": str(path),
    }


@app.post("/report")
def api_report(file: UploadFile = File(...)):
    img = _read_upload_image(file)
    processed = enhance_image(img, strong=True)
    desc = generate_description(processed)
    objects = detect_and_classify_objects(processed)
    return {
        "description": desc["description"],
        "location": desc["location"],
        "location_hint": desc["location_hint"],
        "objects": objects[:20],
    }


@app.get("/download/{filename}")
def download_file(filename: str):
    path = WORKDIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path), filename=filename)


@app.post("/pipeline")
def api_pipeline(
    file: UploadFile = File(...),
    brightness: float = Form(1.0),
    saturation: float = Form(1.0),
    warmth: float = Form(1.0),
    gamma: float = Form(1.0),
    sharpness: float = Form(1.0),
):
    """One-shot pipeline:
    1) enhance
    2) classify objects
    3) apply edits
    4) create report
    """
    img = _read_upload_image(file)

    enhanced = enhance_image(img, strong=True)
    enhanced_path = _unique_path("pipeline_enhanced", ".jpg")
    cv2.imwrite(str(enhanced_path), enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    objects = detect_and_classify_objects(enhanced)
    report = generate_description(enhanced)

    edited = apply_edits(
        enhanced,
        EditParams(
            brightness=brightness,
            saturation=saturation,
            warmth=warmth,
            gamma=gamma,
            sharpness=sharpness,
            denoise=True,
            blur=False,
            detail=True,
            upscale=False,
        ),
    )
    edited_path = _unique_path("pipeline_edited", ".jpg")
    cv2.imwrite(str(edited_path), edited, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

    return {
        "enhanced_download_url": f"/download/{enhanced_path.name}",
        "edited_download_url": f"/download/{edited_path.name}",
        "report": report,
        "objects": objects,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("clearview_backend_single_file:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
