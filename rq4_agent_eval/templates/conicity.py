"""Conicity Index template (visceral-adiposity proxy).

Formula:
    CI = waist / (0.109 * sqrt(wt / ht))

where waist is in meters, wt in kg, ht in meters.

Risk thresholds (2-class, common cutoff at 1.25):
    Low  (0):  CI <= 1.25
    High (1):  CI >  1.25
"""

import math
from typing import Dict

from templates.base import Template


class CONICITY(Template):
    name = "CONICITY"
    roots = ["waist", "wt", "ht"]

    bounds = {
        "waist": (0.62, 1.64),    # meters
        "wt":    (36.8, 191.4),   # kg
        "ht":    (1.48, 1.98),    # meters
    }

    target_bounds = (1.01, 1.6)
    num_classes = 2

    def compute(self, values: Dict[str, float]) -> float:
        waist = values["waist"]
        wt = values["wt"]
        ht = values["ht"]
        if wt <= 0 or ht <= 0:
            raise ValueError(f"CONICITY: wt={wt}, ht={ht} must be positive")
        return waist / (0.109 * math.sqrt(wt / ht))

    def risk_class(self, target_value: float) -> int:
        return 1 if target_value > 1.25 else 0
