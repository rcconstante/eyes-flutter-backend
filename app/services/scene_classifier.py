"""
Scene classifier – rule-based scene type from detected objects.

Produces labels like "indoor", "outdoor / street", "kitchen", etc.
to provide context-awareness for spoken feedback.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.yolo_detector import Detection


_SCENE_RULES: list[tuple[set[str], str]] = [
    # Most specific first
    ({"oven", "refrigerator", "sink", "microwave"}, "Kitchen"),
    ({"toilet", "sink"}, "Bathroom"),
    ({"bed", "clock"}, "Bedroom"),
    ({"couch", "tv", "remote"}, "Living room"),
    ({"dining table", "cup", "fork", "knife", "spoon", "bowl"}, "Dining area"),
    ({"laptop", "keyboard", "mouse", "monitor"}, "Office / Desk"),
    ({"car", "truck", "bus", "traffic light", "stop sign"}, "Outdoor / Street"),
    ({"bicycle", "motorcycle"}, "Outdoor / Road"),
    ({"bench", "potted plant", "bird"}, "Outdoor / Park"),
    ({"person"}, "General area"),
]


def classify_scene(detections: list[Detection]) -> str:
    """Return the best-matching scene label for the set of detections."""
    if not detections:
        return "Unknown"

    labels = {d.label for d in detections}

    best_match = "General area"
    best_overlap = 0

    for required, scene in _SCENE_RULES:
        overlap = len(labels & required)
        # Require at least 2 matches for multi-item rules,
        # or 1 if the rule itself has only 1 element
        min_needed = min(2, len(required))
        if overlap >= min_needed and overlap > best_overlap:
            best_overlap = overlap
            best_match = scene

    return best_match
