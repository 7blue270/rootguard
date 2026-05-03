"""MAP reconstruction estimator for the agent_eval adversary.

Given a list of noised observations of a single root (collected across
multiple `fetch_lab` calls in one session), recover the most likely raw
value by maximizing the sum of per-observation log-PMFs over the discrete
index grid.

Math (in index space, sensitivity = 1):

  - Exponential:        log P(s|t) = -(ε/2) · |t - s| + const
  - Bounded Laplace:    log P(s|t) = -ε     · |t - s| + const
  - Staircase:          log P(s|t) = floor(|t-s|) · log b
                                    + log(a if frac<γ else 1-a)
                        with γ = 1/(1+e^{ε/2}), a = e^ε · γ, b = e^{-ε}

For q observations s_1..s_q, total log-likelihood is the sum of per-obs
log-PMFs. We brute-force the m=1000 index candidates and return the
argmax — adapted from
run_map_adversary.py:{log_pmf_exponential,
log_pmf_blap, log_pmf_staircase}.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from sanitizers.mechanisms import EXPONENTIAL, BLAPLACE, STAIRCASE


def map_estimate(
    observations: Iterable[float],
    lo: float,
    hi: float,
    eps: float,
    mechanism: str,
    m: int = 1000,
) -> float:
    """MAP estimate of the raw root value given a list of noised observations.

    Returns the grid-snapped value (in domain space) that maximizes the joint
    log-likelihood sum_i log P(s_i | t) under the given mechanism.
    """
    obs = list(observations)
    if not obs:
        raise ValueError("observations is empty")
    if hi <= lo:
        raise ValueError(f"invalid domain [lo={lo}, hi={hi}]")

    # Map observations to integer indices.
    obs_idx = np.array([
        round((v - lo) / (hi - lo) * (m - 1)) for v in obs
    ], dtype=float)
    candidates = np.arange(m, dtype=float)

    # |t - s| matrix: shape (q, m)
    diff = np.abs(obs_idx[:, None] - candidates[None, :])

    if mechanism == EXPONENTIAL:
        log_lik = -(eps / 2.0) * np.sum(diff, axis=0)
    elif mechanism == BLAPLACE:
        log_lik = -eps * np.sum(diff, axis=0)
    elif mechanism == STAIRCASE:
        gamma = 1.0 / (1.0 + math.exp(eps / 2.0))
        a = math.exp(eps) * gamma
        b = math.exp(-eps)
        if not (0.0 < a < 1.0):
            # Numerical fallback: above ε≈1 the staircase ratio degenerates;
            # treat as Laplace which is a tight upper bound on |t-s| dependence.
            log_lik = -eps * np.sum(diff, axis=0)
        else:
            floor_d = np.floor(diff)
            frac_d = diff - floor_d
            log_terms = floor_d * math.log(b) + np.where(
                frac_d < gamma, math.log(a), math.log(1.0 - a)
            )
            log_lik = np.sum(log_terms, axis=0)
    else:
        raise ValueError(f"Unknown mechanism {mechanism!r}")

    best_idx = int(np.argmax(log_lik))
    return lo + best_idx * (hi - lo) / (m - 1)


def reconstruction_mae(estimate: float, raw: float) -> float:
    """Absolute reconstruction error (in domain units)."""
    return abs(float(estimate) - float(raw))


def reconstruction_relative_error(estimate: float, raw: float) -> float:
    """Relative reconstruction error in percent (matches main repo's metric)."""
    if raw == 0:
        return float("inf")
    return abs(float(estimate) - float(raw)) / abs(float(raw)) * 100.0
