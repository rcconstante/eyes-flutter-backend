"""
Currency detector – extracts currency info from YOLO detections.

If the YOLO model was trained with Philippine peso bill / coin classes
(e.g. "20_peso", "100_peso", "coin_5"), this module aggregates them
into a human-readable string and computes the total.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.yolo_detector import Detection

from app.config import settings

# Map YOLO class names → (display name, numeric value)
_CURRENCY_MAP: dict[str, tuple[str, float]] = {
    "20_peso": ("₱20 bill", 20),
    "50_peso": ("₱50 bill", 50),
    "100_peso": ("₱100 bill", 100),
    "200_peso": ("₱200 bill", 200),
    "500_peso": ("₱500 bill", 500),
    "1000_peso": ("₱1000 bill", 1000),
    "coin_1": ("₱1 coin", 1),
    "coin_5": ("₱5 coin", 5),
    "coin_10": ("₱10 coin", 10),
}


def detect_currency(detections: list[Detection]) -> str | None:
    """
    Return a spoken-friendly string summarising detected currency.

    Groups duplicate denominations and sums the total.
    Example: "2× ₱100 bill, 1× ₱20 bill – total ₱220"
    Returns None if no currency is detected.
    """
    from collections import Counter

    counts: Counter[str] = Counter()

    for det in detections:
        if det.label in _CURRENCY_MAP:
            counts[det.label] += 1

    if not counts:
        return None

    total = 0.0
    parts: list[str] = []
    for label, count in counts.items():
        display_name, value = _CURRENCY_MAP[label]
        total += value * count
        if count > 1:
            parts.append(f"{count}× {display_name}")
        else:
            parts.append(display_name)

    names = ", ".join(parts)
    return f"{names} – total ₱{total:,.0f}"
