"""
Template interface: encapsulates a clinical diagnostic formula.

Each template defines:
    - roots: list of private root attribute names (in canonical order)
    - bounds: dict of (lo, hi) per root, from NHANES per-template subpopulation
    - compute(values_dict) -> float: the target function T = g(X)
    - risk_class(target_value) -> int: clinical risk class (0, 1, 2, ...)
    - num_classes: total number of risk classes for RCE normalization

Subclasses live in templates/{fib4,homa,anemia}.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class Template(ABC):
    name: str = ""
    roots: List[str] = []
    bounds: Dict[str, Tuple[float, float]] = {}
    target_bounds: Tuple[float, float] = (0.0, 1.0)
    num_classes: int = 2

    @abstractmethod
    def compute(self, values: Dict[str, float]) -> float:
        """Apply the diagnostic formula to a set of root values."""

    @abstractmethod
    def risk_class(self, target_value: float) -> int:
        """Map a target value to its clinical risk class (0-indexed)."""

    def __repr__(self) -> str:
        return f"<Template {self.name} k={len(self.roots)}>"
