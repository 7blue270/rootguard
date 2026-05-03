"""TyG (Triglyceride-Glucose Index) template.

Formula:
    TyG = log( (tg * glu) / 2 )    (natural log; tg in mg/dL, glu in mg/dL)

Risk thresholds (2-class):
    Low  (0):  TyG <= 8.5
    High (1):  TyG >  8.5
"""

import math
from typing import Dict

from templates.base import Template


class TYG(Template):
    name = "TYG"
    roots = ["tg", "glu"]

    bounds = {
        "tg":  (21.0, 2684.0),
        "glu": (64.0, 451.0),
    }

    target_bounds = (6.77, 12.31)
    num_classes = 2

    def compute(self, values: Dict[str, float]) -> float:
        tg = values["tg"]
        glu = values["glu"]
        if tg <= 0 or glu <= 0:
            raise ValueError(f"TYG: tg={tg}, glu={glu} must be positive")
        return math.log((tg * glu) / 2.0)

    def risk_class(self, target_value: float) -> int:
        return 1 if target_value > 8.5 else 0
