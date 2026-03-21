"""
Currency detector – extracts currency info from YOLO detections.

If the YOLO model was trained with Philippine peso bill / coin classes
(e.g. "20_peso", "100_peso", "coin_5"), this module aggregates them
into a human-readable string, computes the total, and returns both.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.yolo_detector import Detection

from app.config import settings

# Map YOLO class names → (display name, numeric value)
_CURRENCY_MAP: dict[str, tuple[str, float]] = {
    "20_Pesos": ("₱20 bill", 20),
    "50_Pesos": ("₱50 bill", 50),
    "100_Pesos": ("₱100 bill", 100),
    "200_Pesos": ("₱200 bill", 200),
    "500_Pesos": ("₱500 bill", 500),
    "1000_Pesos": ("₱1000 bill", 1000),
    "Polymer_1000_Pesos": ("₱1000 bill", 1000),
    "1_Coin_New": ("₱1 coin", 1),
    "1_Coin_Old": ("₱1 coin", 1),
    "5_Coin_New": ("₱5 coin", 5),
    "5_Coin_Old": ("₱5 coin", 5),
    "10_Coin_New": ("₱10 coin", 10) ,
    "10_Coin_Old": ("₱10 coin", 10),
    "20_Coin": ("₱20 coin", 20)
}


@dataclass
class CurrencyResult:
    """Structured currency detection result."""
    summary: str          # e.g. "Total: P220"
    total_amount: float   # e.g. 220.0
    item_count: int       # total number of bills/coins detected


def detect_currency(detections: list[Detection]) -> CurrencyResult | None:
    """
    Return a structured result summarising detected currency.

    Groups duplicate denominations and sums the total.
    Example summary: "Total: P220"
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
    item_count = 0
    for label, count in counts.items():
        _, value = _CURRENCY_MAP[label]
        total += value * count
        item_count += count

    summary = f"Total: P{total:,.0f}"
    return CurrencyResult(summary=summary, total_amount=total, item_count=item_count)
