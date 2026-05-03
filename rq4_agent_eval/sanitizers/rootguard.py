"""
RootGuard: batch-sanitize a patient's roots once and cache the noised values.

Unlike M-All (which sanitizes per-release), RootGuard noises every root exactly
once at session initialization. All subsequent tool calls return cached values.
Derived values (the target T = g(X)) are computed deterministically from the
cached noised roots via post-processing.

Usage:
    rg = RootGuard.initialize(patient, template, eps_per_root, rng)
    rg.get("AST")          # cached noised value
    rg.get("AST")          # identical -- same value
    rg.compute_target()    # template's formula on cached roots
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from sanitizers.mechanisms import sanitize, EXPONENTIAL


@dataclass
class RootGuard:
    cached: Dict[str, float] = field(default_factory=dict)
    template: object = None  # set on initialize()
    raw: Dict[str, float] = field(default_factory=dict)  # for analysis only

    @classmethod
    def initialize(
        cls,
        patient: Dict[str, float],
        template,
        eps_per_root: Dict[str, float],
        mechanism: str = EXPONENTIAL,
        m: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> "RootGuard":
        """
        Pre-sanitize each root once.

        patient : Dict[str, float]
            Raw root values, e.g. {"age": 54, "AST": 23, "PLT": 220, "ALT": 30}
        template : Template
            Provides domain bounds per root (lo_i, hi_i).
        eps_per_root : Dict[str, float]
            Per-root privacy parameter. Sum is the total budget B.
        """
        if rng is None:
            rng = np.random.default_rng()
        cached = {}
        for root_name in template.roots:
            if root_name not in patient:
                raise KeyError(f"Patient missing root {root_name!r}")
            if root_name not in eps_per_root:
                raise KeyError(f"No budget for root {root_name!r}")
            lo, hi = template.bounds[root_name]
            cached[root_name] = sanitize(
                value=patient[root_name],
                lo=lo,
                hi=hi,
                eps=eps_per_root[root_name],
                mechanism=mechanism,
                m=m,
                rng=rng,
            )
        return cls(cached=cached, template=template, raw=dict(patient))

    def get(self, root_name: str) -> float:
        """Return cached noised value (same on every call)."""
        if root_name not in self.cached:
            raise KeyError(f"Unknown root {root_name!r}")
        return self.cached[root_name]

    def compute_target(self) -> float:
        """Apply the template's formula to the cached noised roots."""
        return self.template.compute(self.cached)

    def all_cached(self) -> Dict[str, float]:
        """Return a copy of all cached noised values."""
        return dict(self.cached)
