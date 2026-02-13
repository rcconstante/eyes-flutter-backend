"""
/api/analyze endpoint – the main image processing pipeline.

Flow:
  1. Receive JPEG image from mobile app
  2. Check brightness → run Zero-DCE enhancement if low-light
  3. Run YOLO object detection
  4. Run MiDaS depth estimation
  5. Map detections to distances
  6. Classify scene, detect currency, prioritise critical objects
  7. Return JSON matching Flutter's ResultModel
"""

import io
import logging
import time

import numpy as np
from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image

from app.config import settings
from app.services.scene_classifier import classify_scene
from app.services.priority_engine import pick_priority_object, generate_alerts
from app.services.currency_detector import detect_currency

logger = logging.getLogger("eyes.analyze")

router = APIRouter()


@router.post("/analyze")
async def analyze_image(request: Request, image: UploadFile = File(...)):
    """Process a camera frame and return structured results."""
    t0 = time.time()

    manager = request.app.state.model_manager

    # ── 1. Read image ──────────────────────────────────────────
    raw_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    original_size = pil_image.size  # (W, H)

    # ── 2. Low-light enhancement ───────────────────────────────
    enhanced = False
    zero_dce = manager.get_zero_dce()
    if zero_dce.is_low_light(pil_image, threshold=settings.LOW_LIGHT_THRESHOLD):
        pil_image = zero_dce.enhance(pil_image)
        enhanced = True
        logger.info("Low-light detected → image enhanced")

    # ── 3. Object detection ────────────────────────────────────
    yolo = manager.get_yolo()
    detections = yolo.detect(pil_image)
    logger.info(f"Detected {len(detections)} objects")

    # ── 4. Depth estimation ────────────────────────────────────
    depth_map = None
    if detections:
        midas = manager.get_midas()
        depth_map = midas.estimate_depth_map(pil_image)

    # ── 5. Map each detection to a distance ────────────────────
    detection_results: list[dict] = []
    image_h = original_size[1]

    for det in detections:
        distance = 0.0
        if depth_map is not None:
            midas = manager.get_midas()
            distance = midas.estimate_distance(
                depth_map=depth_map,
                bbox=det.bbox,
                label=det.label,
                bbox_height_px=det.bbox_height_px,
                image_height=image_h,
            )

        detection_results.append({
            "label": det.label,
            "confidence": round(det.confidence, 3),
            "bbox": list(det.bbox),
            "distance": distance,
        })

    # ── 6. Scene classification ────────────────────────────────
    scene_type = classify_scene(detections)

    # ── 7. Currency detection ──────────────────────────────────
    currency = detect_currency(detections)

    # ── 8. Priority & alerts ───────────────────────────────────
    priority = pick_priority_object(detection_results)
    alerts = generate_alerts(detection_results)

    elapsed = round(time.time() - t0, 3)
    logger.info(f"Pipeline done in {elapsed}s | priority={priority['label']} | scene={scene_type}")

    # ── 9. Build response matching Flutter ResultModel ─────────
    return {
        "priority_object": priority["label"],
        "distance": priority["distance"],
        "currency": currency,
        "scene_type": scene_type,
        "alerts": alerts,
        "detections": detection_results,
        "enhanced": enhanced,
        "processing_time": elapsed,
    }
