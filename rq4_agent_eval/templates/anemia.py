"""Anemia classification (MCHC) template.

Formula:
    MCHC = (Hb / Hct) * 100

where Hb is hemoglobin (g/dL), Hct is hematocrit (%).
RBC is included as a root for the paper's experimental setting (it has zero
sensitivity to MCHC, illustrating the zero-sensitivity-root case in RQ3).

Risk thresholds (3-class):
    Hypochromic (0):  MCHC < 32
    Normochromic (1): 32 <= MCHC <= 36
    Hyperchromic (2): MCHC > 36
"""

from typing import Dict

from templates.base import Template


class ANEMIA(Template):
    name = "ANEMIA"
    roots = ["Hb", "Hct", "RBC"]

    # NHANES 2017-2018 adult-male per-template observed min--max.
    # TODO: replace with exact values from the existing Sec 5 codebase if these differ.
    bounds = {
        "Hb":   (8.0, 19.0),      # g/dL
        "Hct":  (25.0, 55.0),     # %
        "RBC":  (3.0, 7.0),       # 10^6/uL  (zero sensitivity to MCHC)
    }

    target_bounds = (28.0, 36.58)  # MCHC width ~8.58 (matches paper)
    num_classes = 3

    def compute(self, values: Dict[str, float]) -> float:
        hb = values["Hb"]
        hct = values["Hct"]
        if hct <= 0:
            raise ValueError(f"Hct must be positive, got {hct}")
        return (hb / hct) * 100.0

    def risk_class(self, target_value: float) -> int:
        if target_value < 32.0:
            return 0
        if target_value <= 36.0:
            return 1
        return 2
