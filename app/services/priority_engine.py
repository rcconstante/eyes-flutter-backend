"""
Priority engine – picks the most important objects (up to 3) and generates
safety alerts for the spoken / haptic feedback layer.
"""

from app.config import settings


def pick_priority_objects(detection_results: list[dict], max_items: int = 3) -> list[dict]:
    """
    Choose the top 1–3 most important detections.

    Scoring:
      - Critical objects get +100 base score
      - Closer distance = higher score
      - Higher confidence = higher score
    
    Returns a list of top objects sorted by importance.
    """
    if not detection_results:
        return [{"label": "No object", "distance": 0.0, "confidence": 0.0}]

    scored = []
    for det in detection_results:
        score = 0.0
        dist = det.get("distance", 0.0)
        conf = det.get("confidence", 0.0)

        # Critical objects get a big boost
        if det["label"] in settings.CRITICAL_OBJECTS:
            score += 100.0

        # Closer = more important (inverse distance score, max 50 pts)
        if dist > 0:
            score += max(0, 50.0 - (dist * 5.0))
        else:
            score += 10.0  # unknown distance, moderate score

        # Higher confidence = slightly more important (max 20 pts)
        score += conf * 20.0

        scored.append((score, det))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by label (keep first/best of each type)
    seen_labels = set()
    top = []
    for _, det in scored:
        if det["label"] not in seen_labels:
            seen_labels.add(det["label"])
            top.append(det)
            if len(top) >= max_items:
                break

    return top


def pick_priority_object(detection_results: list[dict]) -> dict:
    """
    Legacy single-object picker. Returns the #1 priority.
    """
    top = pick_priority_objects(detection_results, max_items=1)
    return {"label": top[0]["label"], "distance": top[0].get("distance", 0.0)}


def generate_alerts(detection_results: list[dict]) -> list[str]:
    """
    Generate safety alert strings for objects that are dangerously close.
    """
    alerts: list[str] = []

    for det in detection_results:
        dist = det["distance"]
        label = det["label"]
        if dist <= 0:
            continue

        if dist <= settings.DISTANCE_VERY_CLOSE:
            alerts.append(f"⚠️ {label} very close – {dist:.1f}m")
        elif dist <= settings.DISTANCE_CLOSE and label in settings.CRITICAL_OBJECTS:
            alerts.append(f"{label} nearby – {dist:.1f}m")

    return alerts
