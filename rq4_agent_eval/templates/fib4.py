"""FIB-4 (liver fibrosis) template.

Formula (Sterling et al. 2006):
    FIB-4 = (age * AST) / (PLT * sqrt(ALT))

Risk thresholds:
    Low (0):           FIB-4 < 1.30
    Indeterminate (1): 1.30 <= FIB-4 <= 2.67
    High (2):          FIB-4 > 2.67

Domain bounds for roots and target are pulled from the NHANES 2017-2018
adult-male subpopulation used for FIB-4 in the paper. The values below are
the bounds used in the paper's Sec 5; if running on a different cohort, regenerate.
"""

import math
from typing import Dict

from templates.base import Template


class FIB4(Template):
    name = "FIB4"
    roots = ["age", "AST", "PLT", "ALT"]

    # NHANES 2017-2018 adult-male per-template observed min--max.
    # TODO: replace with exact values from the existing Sec 5 codebase if these differ.
    bounds = {
        "age":  (18.0, 80.0),     # years
        "AST":  (8.0, 200.0),     # U/L
        "PLT":  (50.0, 500.0),    # 10^9/L
        "ALT":  (5.0, 250.0),     # U/L
    }

    target_bounds = (0.12, 37.95)  # from Sec 5
    num_classes = 3

    def compute(self, values: Dict[str, float]) -> float:
        age = values["age"]
        ast = values["AST"]
        plt = values["PLT"]
        alt = values["ALT"]
        if alt <= 0 or plt <= 0:
            raise ValueError(f"Invalid input: ALT={alt}, PLT={plt} must be positive")
        return (age * ast) / (plt * math.sqrt(alt))

    def risk_class(self, target_value: float) -> int:
        if target_value < 1.30:
            return 0
        if target_value <= 2.67:
            return 1
        return 2
