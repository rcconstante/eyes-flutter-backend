"""
Priority engine – picks the most important object and generates
safety alerts for the spoken / haptic feedback layer.
"""

from app.config import settings


def pick_priority_object(detection_results: list[dict]) -> dict:
    """
    Choose the single most important detection.

    Priority order:
      1. Critical objects (vehicles, hazards) – closest first.
      2. Any object – closest first.
      3. Fallback "No object" if nothing detected.
    """
    if not detection_results:
        return {"label": "No object", "distance": 0.0}

    # Separate critical and non-critical
    critical = [
        d for d in detection_results if d["label"] in settings.CRITICAL_OBJECTS
    ]

    pool = critical if critical else detection_results

    # Sort by distance ascending (closest first); treat 0 as "unknown far"
    pool_sorted = sorted(
        pool,
        key=lambda d: d["distance"] if d["distance"] > 0 else 999,
    )

    best = pool_sorted[0]
    return {"label": best["label"], "distance": best["distance"]}


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
