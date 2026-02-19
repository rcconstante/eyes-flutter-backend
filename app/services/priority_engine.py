"""
Priority engine – picks the most important object and generates
safety alerts for the spoken / haptic feedback layer.
"""

from app.config import settings


def pick_priority_object(detection_results: list[dict]) -> dict:
    """
    Choose the single most important detection.

    Priority order:
      1. Tier-1 CRITICAL objects (vehicles, weapons, dangerous animals) – closest first.
      2. Tier-2 HAZARD objects that are very close (< DISTANCE_CLOSE) – closest first.
      3. Any remaining detected object – closest first.
      4. Fallback "No object" if nothing detected.
    """
    if not detection_results:
        return {"label": "No object", "distance": 0.0}

    def sort_key(d: dict) -> float:
        return d["distance"] if d["distance"] > 0 else 999.0

    # Tier 1: always-critical regardless of distance
    critical = [d for d in detection_results if d["label"] in settings.CRITICAL_OBJECTS]
    if critical:
        return min(critical, key=sort_key)

    # Tier 2: hazard objects elevated only when close
    hazards_close = [
        d for d in detection_results
        if d["label"] in settings.HAZARD_OBJECTS
        and 0 < d["distance"] <= settings.DISTANCE_CLOSE
    ]
    if hazards_close:
        return min(hazards_close, key=sort_key)

    # Tier 3: any detection, closest first
    return min(detection_results, key=sort_key)


def generate_alerts(detection_results: list[dict]) -> list[str]:
    """
    Generate safety alert strings for objects that are dangerously close.

    Alert rules:
      - Any object within DISTANCE_VERY_CLOSE  → urgent warning.
      - Critical object within DISTANCE_CLOSE  → standard proximity warning.
      - Hazard object within DISTANCE_VERY_CLOSE → trip/collision warning.
    """
    alerts: list[str] = []

    for det in detection_results:
        dist = det["distance"]
        label = det["label"]
        if dist <= 0:
            continue

        if dist <= settings.DISTANCE_VERY_CLOSE:
            if label in settings.CRITICAL_OBJECTS:
                alerts.append(f"⚠️ {label} very close – {dist:.1f}m")
            elif label in settings.HAZARD_OBJECTS:
                alerts.append(f"⚠️ {label} in path – {dist:.1f}m")
            else:
                alerts.append(f"{label} very close – {dist:.1f}m")
        elif dist <= settings.DISTANCE_CLOSE and label in settings.CRITICAL_OBJECTS:
            alerts.append(f"{label} nearby – {dist:.1f}m")

    return alerts
