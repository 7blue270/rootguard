"""AIP (Atherogenic Index of Plasma) template.

Formula:
    AIP = log10(tg / hdl)

Risk thresholds:
    Low      (0): AIP <= 0.11
    Moderate (1): 0.11 < AIP <= 0.21
    High     (2): AIP > 0.21
"""

import math
from typing import Dict

from templates.base import Template


class AIP(Template):
    name = "AIP"
    roots = ["tg", "hdl"]

    bounds = {
        "tg":  (21.0, 2684.0),
        "hdl": (11.0, 147.0),
    }

    target_bounds = (-0.64, 1.97)
    num_classes = 3

    def compute(self, values: Dict[str, float]) -> float:
        tg = values["tg"]
        hdl = values["hdl"]
        if tg <= 0 or hdl <= 0:
            raise ValueError(f"AIP: tg={tg}, hdl={hdl} must be positive")
        return math.log10(tg / hdl)

    def risk_class(self, target_value: float) -> int:
        if target_value > 0.21:
            return 2
        if target_value > 0.11:
            return 1
        return 0
