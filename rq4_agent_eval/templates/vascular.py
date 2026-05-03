"""VASCULAR (Pulse Pressure Index) template.

Formula:
    PPI = pp / sbp = (sbp - dbp) / sbp = 1 - dbp/sbp

where sbp / dbp are systolic / diastolic blood pressure (mmHg).

Risk thresholds (2-class):
    Low  (0):  PPI <= 0.60
    High (1):  PPI >  0.60
"""

from typing import Dict

from templates.base import Template


class VASCULAR(Template):
    name = "VASCULAR"
    roots = ["sbp", "dbp"]

    bounds = {
        "sbp": (72.0, 216.0),
        "dbp": (0.01, 136.0),
    }

    target_bounds = (0.17, 1.0)
    num_classes = 2

    def compute(self, values: Dict[str, float]) -> float:
        sbp = values["sbp"]
        dbp = values["dbp"]
        if sbp <= 0:
            raise ValueError(f"VASCULAR: sbp={sbp} must be positive")
        return (sbp - dbp) / sbp

    def risk_class(self, target_value: float) -> int:
        return 1 if target_value > 0.60 else 0
