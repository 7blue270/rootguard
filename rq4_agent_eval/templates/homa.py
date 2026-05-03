"""HOMA-IR (insulin resistance) template.

Formula:
    HOMA = (Glu * Ins) / 405

where Glu is fasting glucose (mg/dL) and Ins is fasting insulin (uIU/mL).

Risk thresholds (3-class, common clinical cutoffs):
    Low (0):           HOMA < 1.0   (insulin sensitive)
    Borderline (1):    1.0 <= HOMA < 2.5
    Insulin resistant (2): HOMA >= 2.5
"""

from typing import Dict

from templates.base import Template


class HOMA(Template):
    name = "HOMA"
    roots = ["Glu", "Ins"]

    # NHANES 2017-2018 adult-male per-template observed min--max.
    # TODO: replace with exact values from the existing Sec 5 codebase if these differ.
    bounds = {
        "Glu":  (60.0, 350.0),    # mg/dL fasting glucose
        "Ins":  (1.0, 250.0),     # uIU/mL fasting insulin
    }

    target_bounds = (0.15, 155.0)  # from Sec 5: width ~155, typical ~4
    num_classes = 3

    def compute(self, values: Dict[str, float]) -> float:
        return (values["Glu"] * values["Ins"]) / 405.0

    def risk_class(self, target_value: float) -> int:
        if target_value < 1.0:
            return 0
        if target_value < 2.5:
            return 1
        return 2
