"""
/api/analyze endpoint – the main image processing pipeline.

Flow:
  1. Receive JPEG image from mobile app
  2. Check brightness → run Zero-DCE enhancement if low-light
  3. Run YOLO object detection
  4. Run MiDaS depth estimation
  5. Map detections to distances
  6. Classify scene, detect currency, prioritise critical objects
    7. Return detections, annotated image, and backend-generated voice alert
"""

import io
import logging
import time
import base64

from fastapi import APIRouter, File, Form, Request, UploadFile
from PIL import Image, ImageDraw, ImageFont

from app.config import settings
from app.services.scene_classifier import classify_scene
from app.services.priority_engine import pick_priority_object, generate_alerts
from app.services.currency_detector import detect_currency

logger = logging.getLogger("eyes.analyze")

router = APIRouter()


def _encode_jpeg_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    pad = 4
    text_w, text_h = _text_size(draw, text, font)
    x, y = xy
    draw.rectangle((x, y, x + text_w + pad * 2, y + text_h + pad * 2), fill=fill)
    draw.text((x + pad, y + pad), text, fill=(255, 255, 255), font=font)


def _draw_result_image(
    image: Image.Image,
    detection_results: list[dict],
    priority_label: str,
    is_critical: bool,
    scene_type: str,
    enhanced: bool,
) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    width, height = annotated.size
    line_width = max(2, int(min(width, height) / 240))

    banner = f"Scene: {scene_type}"
    if enhanced:
        banner += " | Enhanced low-light frame"
    _draw_label(draw, banner, (10, 10), font, (20, 20, 20))

    for det in detection_results:
        bbox = det.get("bbox") or []
        if len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        is_priority = det.get("label") == priority_label
        color = (255, 64, 84) if is_priority and is_critical else (34, 197, 94) if is_priority else (56, 189, 248)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        confidence = int(float(det.get("confidence", 0)) * 100)
        distance = float(det.get("distance", 0))
        distance_text = f" {distance:.1f}m" if distance > 0 else ""
        label = f"{det.get('label', 'object')}{distance_text} {confidence}%"
        label_y = max(0, y1 - 18)
        _draw_label(draw, label, (x1, label_y), font, color)

    if not detection_results:
        message = "No object detected"
        text_w, text_h = _text_size(draw, message, font)
        _draw_label(
            draw,
            message,
            ((width - text_w) // 2, max(10, (height - text_h) // 2)),
            font,
            (20, 20, 20),
        )

    return annotated


def _build_voice_alert(
    priority_label: str,
    distance: float,
    is_critical: bool,
    alerts: list[str],
    scene_type: str,
    currency_result,
    language: str,
) -> str:
    lang = (language or "en").lower()

    if currency_result is not None:
        if lang == "fil":
            return f"Nakakita ako ng pera. Kabuuan: {currency_result.total_amount:,.0f} piso."
        return f"Currency detected. Total {currency_result.total_amount:,.0f} pesos."

    if priority_label in {"No object", "Unknown"}:
        if lang == "fil":
            return f"Wala akong nakitang malinaw na bagay. Eksena: {scene_type}."
        return f"No clear object detected. Scene is {scene_type}."

    if lang == "fil":
        parts = []
        if is_critical and 0 < distance < settings.DISTANCE_CLOSE:
            parts.append("Babala!")
        sentence = f"Nakita ang {priority_label}"
        if distance > 0:
            sentence += f", mga {distance:.1f} metro sa harap."
        else:
            sentence += "."
        parts.append(sentence)
        parts.append(f"Eksena: {scene_type}.")
        if alerts:
            parts.append("Mag-ingat sa malapit na bagay.")
        return " ".join(parts)

    parts = []
    if is_critical and 0 < distance < settings.DISTANCE_CLOSE:
        parts.append("Warning!")
    sentence = f"{priority_label} detected"
    if distance > 0:
        sentence += f", {distance:.1f} meters ahead."
    else:
        sentence += "."
    parts.append(sentence)
    parts.append(f"Scene is {scene_type}.")
    if alerts:
        parts.append("Very close, be careful.")
    return " ".join(parts)


@router.post("/analyze")
async def analyze_image(
    request: Request,
    image: UploadFile = File(...),
    language: str = Form("en"),
):
    """Process a camera frame and return structured results."""
    t0 = time.time()

    manager = request.app.state.model_manager

    # ── 1. Read image ──────────────────────────────────────────
    raw_bytes = await image.read()
    logger.info(f"Received image: {len(raw_bytes)} bytes")
    pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    original_size = pil_image.size  # (W, H)
    logger.info(f"Image size: {original_size[0]}x{original_size[1]}")

    # ── 2. Low-light check (enhance if dim, aggressive if very dark) ─
    enhanced = False
    very_dark = False
    zero_dce = manager.get_zero_dce()
    stats = zero_dce.get_image_stats(pil_image)
    mean_brightness = stats["mean"]
    logger.info(
        f"Image stats: mean={stats['mean']:.3f} max={stats['max']:.3f} "
        f"std={stats['std']:.3f} (low-light<{settings.LOW_LIGHT_THRESHOLD}, "
        f"very-low<{settings.VERY_LOW_LIGHT_THRESHOLD})"
    )
    if mean_brightness < settings.VERY_LOW_LIGHT_THRESHOLD:
        very_dark = True
        logger.info("Very low-light detected → aggressive enhancement")
        pil_image = zero_dce.enhance(pil_image, very_dark=True)
        enhanced = True
        enhanced_brightness = zero_dce.get_brightness(pil_image)
        logger.info(f"Brightness after enhancement: {enhanced_brightness:.3f}")
    elif mean_brightness < settings.LOW_LIGHT_THRESHOLD:
        pil_image = zero_dce.enhance(pil_image, very_dark=False)
        enhanced = True
        enhanced_brightness = zero_dce.get_brightness(pil_image)
        logger.info(f"Low-light detected → image enhanced (brightness now {enhanced_brightness:.3f})")

    # ── 3. Object detection ────────────────────────────────────
    # Use a lower confidence threshold for dark images so faint
    # detections are not discarded prematurely. The enhanced image is
    # *always* what we run YOLO on — falling back to the original dark
    # frame here would just guarantee zero detections, which defeats
    # the whole point of enhancement.
    yolo = manager.get_yolo()
    det_conf = settings.LOW_LIGHT_CONFIDENCE if enhanced else None
    detections = yolo.detect(pil_image, conf=det_conf)
    logger.info(f"Detected {len(detections)} objects (conf={det_conf or settings.CONFIDENCE_THRESHOLD})")

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
    currency_result = detect_currency(detections)

    # ── 8. Priority & alerts ───────────────────────────────────
    # When currency is detected, disable normal object priority and
    # switch to "currency mode": report the total sum instead.
    currency_mode = currency_result is not None

    priority = pick_priority_object(detection_results)
    alerts = generate_alerts(detection_results)

    elapsed = round(time.time() - t0, 3)
    logger.info(f"Pipeline done in {elapsed}s | priority={priority['label']} | scene={scene_type}")

    # ── 9. Build response matching Flutter ResultModel ─────────
    # Include the bounding box of the priority object so the app can
    # draw a highlight rectangle around it.
    priority_bbox = priority.get("bbox")  # may be None for "No object"

    if currency_mode:
        # Currency mode suppresses prioritized-object visuals and distance.
        response_priority = currency_result.summary
        response_distance = 0.0
        is_critical = False
        priority_bbox = None
        response_currency = currency_result.summary
        response_currency_total = currency_result.total_amount
    else:
        response_priority = priority["label"]
        response_distance = priority["distance"]
        is_critical = priority["label"] in settings.CRITICAL_OBJECTS
        response_currency = None
        response_currency_total = None

    result_image = _draw_result_image(
        pil_image,
        detection_results,
        response_priority,
        is_critical,
        scene_type,
        enhanced,
    )
    voice_alert = _build_voice_alert(
        response_priority,
        response_distance,
        is_critical,
        alerts,
        scene_type,
        currency_result,
        language,
    )

    return {
        "priority_object": response_priority,
        "distance": response_distance,
        "is_critical": is_critical,
        "priority_bbox": priority_bbox,
        "currency": response_currency,
        "currency_total": response_currency_total,
        "currency_mode": currency_mode,
        "scene_type": scene_type,
        "alerts": alerts,
        "detections": detection_results,
        "voice_alert": voice_alert,
        "image_data_url": _encode_jpeg_data_url(result_image),
        "image_width": result_image.size[0],
        "image_height": result_image.size[1],
        "enhanced": enhanced,
        "processing_time": elapsed,
    }
