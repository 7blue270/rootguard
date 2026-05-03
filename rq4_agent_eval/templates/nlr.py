"""NLR (Neutrophil-to-Lymphocyte Ratio) template.

Formula:
    NLR = neu / lym

Risk thresholds (2-class):
    Low  (0):  NLR <  3.0
    High (1):  NLR >= 3.0
"""

from typing import Dict

from templates.base import Template


class NLR(Template):
    name = "NLR"
    roots = ["neu", "lym"]

    bounds = {
        "neu": (0.4, 15.7),
        "lym": (0.5, 65.9),
    }

    target_bounds = (0.09, 19.0)
    num_classes = 2

    def compute(self, values: Dict[str, float]) -> float:
        neu = values["neu"]
        lym = values["lym"]
        if lym <= 0:
            raise ValueError(f"NLR: lym={lym} must be positive")
        return neu / lym

    def risk_class(self, target_value: float) -> int:
        return 1 if target_value >= 3.0 else 0
