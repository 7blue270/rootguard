"""
Index-space sanitization mechanisms (Discrete Exponential, Bounded Laplace, Staircase).

This file is a clean reimplementation matching Sec 3.2 / App A of the paper.
The sanitization pipeline is:

    1. value x in domain [lo, hi] -> index t = (m-1)*(x - lo)/(hi - lo)
    2. mechanism samples noisy index s in {0, ..., m-1}
    3. clamp s to [0, m-1] (already discrete)
    4. map back: x' = s * delta + lo, where delta = (hi - lo)/(m - 1)

Every mechanism takes (eps) as the privacy parameter under the index-space
metric d(t, t') = |t - t'|. Sensitivity is 1 in this metric for all three
mechanisms.

Public API:

    sanitize(value, lo, hi, eps, mechanism="exp", m=1000, rng=None) -> float
        Returns one noised value in domain space.

    Mechanisms.{EXPONENTIAL, BLAPLACE, STAIRCASE}: enum-style identifiers.

For batch sanitization of a root vector with per-root budgets, see
sanitizers/rootguard.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Iterable

import numpy as np


# ----------------------------------------------------------------------------
# Mechanism names
# ----------------------------------------------------------------------------

EXPONENTIAL = "exp"
BLAPLACE = "blap"
STAIRCASE = "stair"

VALID_MECHANISMS = (EXPONENTIAL, BLAPLACE, STAIRCASE)


# ----------------------------------------------------------------------------
# Index-space mapping
# ----------------------------------------------------------------------------

def value_to_index(x: float, lo: float, hi: float, m: int) -> float:
    """Map domain value x in [lo, hi] to a real-valued index in [0, m-1]."""
    if hi <= lo:
        raise ValueError(f"Invalid domain: lo={lo} >= hi={hi}")
    return (m - 1) * (x - lo) / (hi - lo)


def index_to_value(s: int, lo: float, hi: float, m: int) -> float:
    """Map integer index s in {0, ..., m-1} back to domain space."""
    delta = (hi - lo) / (m - 1)
    return s * delta + lo


def grid_spacing(lo: float, hi: float, m: int) -> float:
    """delta_i = (hi - lo) / (m - 1)."""
    return (hi - lo) / (m - 1)


# ----------------------------------------------------------------------------
# Discrete exponential mechanism
# ----------------------------------------------------------------------------

def sample_exp(t: float, eps: float, m: int, rng: np.random.Generator) -> int:
    """
    Discrete exponential mechanism in index space.

    P(s) ∝ exp(-eps * |round(t) - s| / 2) for s in {0, ..., m-1}.

    Sensitivity = 1 in the index-space metric, so the standard exponential
    mechanism with score u(s) = -|t - s| achieves eps-mDP per release.
    """
    t_round = int(round(t))
    indices = np.arange(m)
    log_weights = -eps * np.abs(t_round - indices) / 2.0
    # Numerically stable softmax-like normalization
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()
    return int(rng.choice(m, p=weights))


# ----------------------------------------------------------------------------
# Bounded Laplace
# ----------------------------------------------------------------------------

def sample_blap(t: float, eps: float, m: int, rng: np.random.Generator) -> int:
    """
    Bounded Laplace: draw Z ~ Lap(0, 1/eps) in index space, add to t,
    clamp to [0, m-1], round to nearest integer.

    Clamping + rounding are deterministic post-processing.
    """
    z = rng.laplace(loc=0.0, scale=1.0 / eps)
    s_continuous = t + z
    s_clamped = max(0.0, min(float(m - 1), s_continuous))
    return int(round(s_clamped))


# ----------------------------------------------------------------------------
# Staircase mechanism (Geng-Viswanath, simplified discrete version)
# ----------------------------------------------------------------------------

def _staircase_optimal_gamma(eps: float) -> float:
    """
    Optimal gamma for the staircase distribution at sensitivity 1.
    Geng & Viswanath give gamma* = 1 / (1 + e^{eps/2}).
    """
    return 1.0 / (1.0 + math.exp(eps / 2.0))


def sample_staircase(t: float, eps: float, m: int, rng: np.random.Generator) -> int:
    """
    Staircase mechanism in index space at sensitivity 1.

    The staircase noise distribution has density:
        f(z) = a(eps) * exp(-eps * floor(|z| + 1/2 - gamma)) for z in R

    where gamma* is the optimal staircase parameter and a(eps) is the
    normalization constant. We sample by:
      1. Choose integer step k with P(k) ∝ exp(-eps * k) for k = 0, 1, ...
      2. Choose offset within the step: uniform on [-gamma, 1-gamma] if k=0
         else either [-(1-gamma), gamma] or [-gamma, 1-gamma] (sign chosen).
      3. Sign uniform.

    Since we ultimately round to an integer index, we use a simplified
    discrete form: P(s) ∝ exp(-eps * floor(|t - s| / 1 + 1 - 2*gamma) / 1)
    truncated to {0, ..., m-1}. For practical purposes at the discretization
    levels we use, this is numerically very close to the discrete Laplace.

    Note: this is a *practical* staircase implementation. The exact
    continuous staircase optimality argument applies to mDP under sensitivity
    1; in our discrete index space the discrete Laplace and discrete
    staircase are nearly identical for eps in the regime we test.
    """
    # Geometric step distribution
    # P(k) ∝ exp(-eps * k), k = 0, 1, 2, ...
    # Equivalent to: k = floor(Geometric(1 - exp(-eps))).
    # We truncate k to ensure the resulting index lies in [0, m-1].
    p_zero = 1.0 - math.exp(-eps)
    # Sample k via inverse CDF
    u = rng.random()
    if p_zero >= 1.0:
        k = 0
    else:
        k = int(math.floor(math.log(1.0 - u) / math.log(1.0 - p_zero)))
    # Sign uniform
    sign = 1 if rng.random() < 0.5 else -1
    # gamma offset (within-step uniform)
    gamma = _staircase_optimal_gamma(eps)
    if k == 0:
        offset = rng.uniform(-gamma, 1.0 - gamma)
    else:
        # Choose half: either [-(1-gamma), -gamma] or [gamma, 1-gamma]
        # but normalized; in the symmetric form, just sample uniformly in a
        # single "step" of width 1 with the gamma offset.
        offset = rng.uniform(-gamma, 1.0 - gamma)
    z = sign * (k + offset)
    s_continuous = t + z
    s_clamped = max(0.0, min(float(m - 1), s_continuous))
    return int(round(s_clamped))


# ----------------------------------------------------------------------------
# Public sanitize() API
# ----------------------------------------------------------------------------

@dataclass
class SanitizationParams:
    lo: float
    hi: float
    eps: float
    m: int = 1000
    mechanism: str = EXPONENTIAL


def sanitize(
    value: float,
    lo: float,
    hi: float,
    eps: float,
    mechanism: str = EXPONENTIAL,
    m: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Sanitize a single value with the given mechanism.

    Parameters
    ----------
    value : float
        Raw input value, must lie in [lo, hi].
    lo, hi : float
        Domain bounds.
    eps : float
        Privacy parameter under the index-space metric.
    mechanism : str
        One of "exp", "blap", "stair".
    m : int
        Discretization grid size; defaults to 1000.
    rng : np.random.Generator, optional
        Reproducible randomness source.

    Returns
    -------
    float
        Noised value in domain space, snapped to the grid.
    """
    if mechanism not in VALID_MECHANISMS:
        raise ValueError(
            f"Unknown mechanism {mechanism!r}; expected one of {VALID_MECHANISMS}"
        )
    if rng is None:
        rng = np.random.default_rng()

    # Clamp value to [lo, hi] before mapping (safety; raw values
    # outside the population range are treated as boundary)
    v_clamped = max(lo, min(hi, float(value)))

    t = value_to_index(v_clamped, lo, hi, m)

    if mechanism == EXPONENTIAL:
        s = sample_exp(t, eps, m, rng)
    elif mechanism == BLAPLACE:
        s = sample_blap(t, eps, m, rng)
    elif mechanism == STAIRCASE:
        s = sample_staircase(t, eps, m, rng)
    else:
        raise AssertionError("unreachable")

    return index_to_value(s, lo, hi, m)


def sanitize_many(
    values: Iterable[float],
    lo: float,
    hi: float,
    eps: float,
    mechanism: str = EXPONENTIAL,
    m: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> list[float]:
    """Sanitize each value independently. Used for M-All ablation comparisons."""
    return [
        sanitize(v, lo, hi, eps, mechanism=mechanism, m=m, rng=rng) for v in values
    ]
