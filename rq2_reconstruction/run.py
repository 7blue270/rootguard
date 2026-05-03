#!/usr/bin/env python3
"""
RQ3 Adversarial Queries v3: Reconstruction under repeated queries (root-only).

Difference from v2: there is no separate "target release" feeding into the MAP
likelihood. Instead, the adversary's last turn directly requests root values,
giving exactly +1 extra observation per affected root:

  Strategy A: q adversarial queries on r_star + 1 extra release on r_star
              => q+1 observations on r_star.
  Strategy B: q adversarial queries per root + 1 extra release per root
              => q+1 observations on every root.

For M-All, the extra release is an independent eps_r noise draw on each
affected root (draw_idx = q). For cached variants (M-Roots / M-Opt), the
extra release replays the single cached value (cached methods get no extra
information from the +1 turn regardless of q).

MAP collapses to per-root 1-D argmax in every (strategy, prior, variant)
cell — there is no derived-node coupling without a target release.

Usage:
  python run_rq3_adversarial_v3.py                    # full run
  python run_rq3_adversarial_v3.py --method vanilla   # single method
  python run_rq3_adversarial_v3.py --strategy A       # single strategy
  python run_rq3_adversarial_v3.py --template ANEMIA  # single template
  python run_rq3_adversarial_v3.py --verify           # quick sanity checks
"""
# _REPO_ROOT_BOOTSTRAP: ensure repo root is on sys.path so that
# 'from utils.*' / 'from preempt.*' imports resolve when this script
# is run from its own subfolder.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import get_topological_order
from utils.med_domain.all_templates import (
    template_nodes, template_edges, template_expressions,
)
from preempt.sanitizer import Sanitizer, TEMPLATE_DOMAINS, MEDICAL_DOMAINS
from baselines.bounded_laplace_baseline import bounded_laplace_noise
from baselines.staircase_baseline import staircase_noise

# ── CLI ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="RQ3 adversarial queries v2 experiment")
parser.add_argument("--method", type=str, default=None,
                    help="Run only this method (e.g. vanilla, vanilla_roots, popabs)")
parser.add_argument("--strategy", type=str, default=None, choices=["A", "B"],
                    help="Run only this strategy")
parser.add_argument("--template", type=str, default=None,
                    help="Run only this template (e.g. ANEMIA)")
parser.add_argument("--epsilon", type=float, default=None,
                    help="Run only this epsilon (eps_r)")
parser.add_argument("--q", type=int, default=None,
                    help="Run only this q value")
parser.add_argument("--prior", type=str, default=None,
                    choices=["uniform", "informed"],
                    help="Run only this prior")
parser.add_argument("--verify", action="store_true",
                    help="Run quick sanity checks and exit")
parser.add_argument("--workers", type=int, default=16,
                    help="Number of parallel workers")
cli_args = parser.parse_args()

# ── Configuration ────────────────────────────────────────────────────────

EPSILONS = [0.01, 0.05, 0.1, 0.5, 1.0]
Q_VALUES = [1, 4, 8, 16]
STRATEGIES = ["A", "B"]
PRIORS = ["uniform", "informed"]
NUM_SAMPLES = 200
NUM_CANDIDATES = 1000
NUM_WORKERS = cli_args.workers

ALLOCATIONS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "allocations_v2")
OUTPUT_BASE = "./results_rq3_adversarial_v3"
HOLDOUT_STATS_PATH = "../data/holdout_population_means.json"
BENCHMARK_PATH = "../data/nhanes_benchmark_200.json"

STAIRCASE_MAX_ROOT_EPS = 0.962  # per-root cap for staircase PMF validity

METHODS = {
    # M-All
    "vanilla":                 ("exponential",      "all"),
    "blap_all":                ("bounded_laplace",  "all"),
    "staircase_all":           ("staircase",        "all"),
    # M-Roots (single cached draw per root at uniform eps_r)
    "vanilla_roots":           ("exponential",      "roots_uniform"),
    "blap_roots_uniform":      ("bounded_laplace",  "roots_uniform"),
    "staircase_roots_uniform": ("staircase",        "roots_uniform"),
    # M-Opt (single cached draw per root at sensitivity-weighted eps_i^Opt)
    "popabs":                  ("exponential",      "roots_opt"),
    "blap_roots_opt":          ("bounded_laplace",  "roots_opt"),
    "staircase_roots_opt":     ("staircase",        "roots_opt"),
}

# Variants that use a cached per-root draw
CACHED_VARIANTS = ("roots_opt", "roots_uniform")

# Map method to its M-Opt counterpart (for r_star selection)
METHOD_TO_OPT = {
    "vanilla":                 "popabs",
    "blap_all":                "blap_roots_opt",
    "staircase_all":           "staircase_roots_opt",
    "vanilla_roots":           "popabs",
    "blap_roots_uniform":      "blap_roots_opt",
    "staircase_roots_uniform": "staircase_roots_opt",
    "popabs":                  "popabs",
    "blap_roots_opt":          "blap_roots_opt",
    "staircase_roots_opt":     "staircase_roots_opt",
}

# Legacy alias used below
ALL_TO_OPT = METHOD_TO_OPT

TEMPLATES = {
    "ANEMIA":   {"roots": ["hb", "hct", "rbc"],
                 "derived": ["mcv", "mch", "mchc"]},
    "FIB4":     {"roots": ["age", "ast", "alt", "plt"],
                 "derived": ["fib4_prod", "fib4_denom", "fib4"]},
    "AIP":      {"roots": ["tc", "hdl", "tg"],
                 "derived": ["non_hdl", "ldl", "aip"]},
    "CONICITY": {"roots": ["wt", "ht", "waist"],
                 "derived": ["bmi", "wthr", "conicity"]},
    "VASCULAR": {"roots": ["sbp", "dbp"],
                 "derived": ["pp", "map", "mbp", "ppi"]},
    "TYG":      {"roots": ["tg", "glu"],
                 "derived": ["tyg_prod", "tyg_half", "tyg"]},
    "HOMA":     {"roots": ["glu", "ins"],
                 "derived": ["homa_prod", "homa"]},
    "NLR":      {"roots": ["neu", "lym"],
                 "derived": ["nlr_sum", "nlr_diff", "nlr"]},
}

SKIP_NODES = {"anemia_class", "fib4_risk", "aip_risk", "ci_risk",
              "ppi_risk", "tyg_class", "homa_class", "nlr_class"}


# ── Population statistics ────────────────────────────────────────────────

def load_population_stats():
    """Load holdout population means and stds from pre-computed file."""
    with open(HOLDOUT_STATS_PATH) as f:
        raw = json.load(f)
    stats = {}
    for tmpl_name in TEMPLATES:
        stats[tmpl_name] = {
            "means": raw["per_template_means"][tmpl_name],
            "stds": raw["per_template_stds"][tmpl_name],
        }
        if "per_template_covariances" in raw:
            cov_data = raw["per_template_covariances"].get(tmpl_name)
            if cov_data:
                cov = np.array(cov_data["covariance"])
                cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
                stats[tmpl_name]["cov_root_order"] = cov_data["root_order"]
                stats[tmpl_name]["cov_means"] = np.array(cov_data["means"])
                stats[tmpl_name]["cov_matrix"] = cov
                stats[tmpl_name]["cov_inv"] = np.linalg.inv(cov_reg)
    return stats


# ── Smooth absolute value ────────────────────────────────────────────────

def smooth_abs(x, eta=1e-16):
    """Differentiable approximation to |x| for L-BFGS-B."""
    return np.sqrt(x * x + eta)


# ── Log-PMF functions (one per mechanism family) ─────────────────────────

_INDICES = np.arange(NUM_CANDIDATES, dtype=float)


def log_pmf_exponential(x_true, released_index, epsilon, lo, hi):
    """Discrete exponential mechanism: P(s|t) ∝ exp(-ε/2 · |t - s|)."""
    t = np.clip((x_true - lo) / (hi - lo) * (NUM_CANDIDATES - 1),
                0, NUM_CANDIDATES - 1)
    log_unnorm = -(epsilon / 2.0) * smooth_abs(t - released_index)
    log_terms = -(epsilon / 2.0) * smooth_abs(t - _INDICES)
    return log_unnorm - logsumexp(log_terms)


def log_pmf_blap(x_true, released_index, epsilon, lo, hi):
    """Bounded Laplace: Lap(0, 1/ε) in index space → P(s|t) ∝ exp(-ε·|t-s|)."""
    t = np.clip((x_true - lo) / (hi - lo) * (NUM_CANDIDATES - 1),
                0, NUM_CANDIDATES - 1)
    log_unnorm = -epsilon * smooth_abs(t - released_index)
    log_terms = -epsilon * smooth_abs(t - _INDICES)
    return log_unnorm - logsumexp(log_terms)


def log_pmf_staircase(x_true, released_index, epsilon, lo, hi):
    """Staircase mechanism (Geng & Viswanath 2015) in index space."""
    t = np.clip((x_true - lo) / (hi - lo) * (NUM_CANDIDATES - 1),
                0, NUM_CANDIDATES - 1)
    gamma = 1.0 / (1.0 + np.exp(epsilon / 2.0))
    a = np.exp(epsilon) * gamma
    b = np.exp(-epsilon)
    dist = smooth_abs(t - _INDICES)
    floor_dist = np.floor(dist)
    frac_dist = dist - floor_dist
    log_base = floor_dist * np.log(b)
    log_weight = np.where(frac_dist < gamma, np.log(a), np.log(1.0 - a))
    log_terms = log_base + log_weight
    s_dist = smooth_abs(t - released_index)
    s_floor = np.floor(s_dist)
    s_frac = s_dist - s_floor
    log_unnorm = s_floor * np.log(b) + (
        np.log(a) if s_frac < gamma else np.log(1.0 - a))
    return log_unnorm - logsumexp(log_terms)


LOG_PMF_FNS = {
    "exponential": log_pmf_exponential,
    "bounded_laplace": log_pmf_blap,
    "staircase": log_pmf_staircase,
}


# ── Batch log-PMF: compute normalization once for k observations ─────────
# For k observations of the same root at the same epsilon, the normalization
# constant log_Z(x, eps) = logsumexp(log_terms) is identical. Computing it
# once instead of k times gives ~k× speedup in PMF evaluation.

def log_pmf_batch_exponential(x_true, obs_indices, epsilon, lo, hi):
    """Sum of log-PMFs for k observations, single normalization."""
    t = np.clip((x_true - lo) / (hi - lo) * (NUM_CANDIDATES - 1),
                0, NUM_CANDIDATES - 1)
    log_terms = -(epsilon / 2.0) * smooth_abs(t - _INDICES)
    log_Z = logsumexp(log_terms)
    total = 0.0
    for obs_i in obs_indices:
        log_unnorm = -(epsilon / 2.0) * smooth_abs(t - obs_i)
        total += log_unnorm - log_Z
    return total


def log_pmf_batch_blap(x_true, obs_indices, epsilon, lo, hi):
    """Sum of log-PMFs for k observations, single normalization."""
    t = np.clip((x_true - lo) / (hi - lo) * (NUM_CANDIDATES - 1),
                0, NUM_CANDIDATES - 1)
    log_terms = -epsilon * smooth_abs(t - _INDICES)
    log_Z = logsumexp(log_terms)
    total = 0.0
    for obs_i in obs_indices:
        log_unnorm = -epsilon * smooth_abs(t - obs_i)
        total += log_unnorm - log_Z
    return total


def log_pmf_batch_staircase(x_true, obs_indices, epsilon, lo, hi):
    """Sum of log-PMFs for k observations, single normalization."""
    t = np.clip((x_true - lo) / (hi - lo) * (NUM_CANDIDATES - 1),
                0, NUM_CANDIDATES - 1)
    gamma = 1.0 / (1.0 + np.exp(epsilon / 2.0))
    a = np.exp(epsilon) * gamma
    b = np.exp(-epsilon)
    log_a = np.log(a)
    log_1ma = np.log(1.0 - a)
    log_b = np.log(b)
    dist = smooth_abs(t - _INDICES)
    floor_dist = np.floor(dist)
    frac_dist = dist - floor_dist
    log_base = floor_dist * log_b
    log_weight = np.where(frac_dist < gamma, log_a, log_1ma)
    log_terms = log_base + log_weight
    log_Z = logsumexp(log_terms)
    total = 0.0
    for obs_i in obs_indices:
        s_dist = smooth_abs(t - obs_i)
        s_floor = np.floor(s_dist)
        s_frac = s_dist - s_floor
        log_unnorm = s_floor * log_b + (log_a if s_frac < gamma else log_1ma)
        total += log_unnorm - log_Z
    return total


LOG_PMF_BATCH_FNS = {
    "exponential": log_pmf_batch_exponential,
    "bounded_laplace": log_pmf_batch_blap,
    "staircase": log_pmf_batch_staircase,
}


# ── Vectorized PMF matrix (config-level precompute) ──────────────────────
# For a given (mech, eps, M), log_terms[i, j] is unnormalized log P(j | i)
# in index space. log_Z[i] is the per-x normalization (logsumexp over j).
# Both are independent of the observed value and the ground truth, so we
# build them once per (mech, eps) per config and reuse across all 200 samples.
# Per-sample MAP becomes:
#   ll[i] = sum_k log_terms[i, obs_k] - K * log_Z[i]   (+ optional Gaussian prior)
#   x_map = x_grid[argmax(ll)]
# vs the previous per-sample 1000×1000 inner loop.

def build_pmf_matrix(mech_family, epsilon, M=NUM_CANDIDATES):
    """Precompute (log_terms[M,M], log_Z[M]) for the index-space PMF."""
    indices = np.arange(M, dtype=float)
    diff = np.abs(indices[:, None] - indices[None, :])
    diff_s = np.sqrt(diff * diff + 1e-16)
    if mech_family == "exponential":
        log_terms = -(epsilon / 2.0) * diff_s
    elif mech_family == "bounded_laplace":
        log_terms = -epsilon * diff_s
    elif mech_family == "staircase":
        gamma = 1.0 / (1.0 + np.exp(epsilon / 2.0))
        a = np.exp(epsilon) * gamma
        b = np.exp(-epsilon)
        log_a = np.log(a)
        log_1ma = np.log(1.0 - a)
        log_b = np.log(b)
        floor_diff = np.floor(diff_s)
        frac_diff = diff_s - floor_diff
        log_terms = floor_diff * log_b + np.where(
            frac_diff < gamma, log_a, log_1ma)
    else:
        raise ValueError(f"Unknown mech_family: {mech_family}")
    log_Z = logsumexp(log_terms, axis=1)
    return log_terms, log_Z


def _argmax_1d_vec(obs_indices, log_terms, log_Z, x_grid,
                   prior_mu=None, prior_sigma=None):
    """Vectorized 1-D argmax MAP using a precomputed PMF matrix."""
    if not obs_indices:
        return float(x_grid[len(x_grid) // 2])
    obs_arr = np.asarray(obs_indices, dtype=int)
    K = len(obs_arr)
    ll = log_terms[:, obs_arr].sum(axis=1) - K * log_Z
    if prior_mu is not None and prior_sigma is not None and prior_sigma > 0:
        ll = ll - 0.5 * ((x_grid - prior_mu) / prior_sigma) ** 2
    return float(x_grid[int(np.argmax(ll))])


# ── Derived value computation ────────────────────────────────────────────

def get_derived_order(template_name):
    """Get topologically sorted derived node names for a template."""
    derived = TEMPLATES[template_name]["derived"]
    roots = set(TEMPLATES[template_name]["roots"])
    order = []
    resolved = set(roots)
    remaining = list(derived)
    max_iter = len(remaining) * len(remaining)
    i = 0
    while remaining and i < max_iter:
        i += 1
        for node in list(remaining):
            deps = template_nodes.get(node, [])
            if deps is None:
                deps = []
            if all(d in resolved for d in deps):
                order.append(node)
                resolved.add(node)
                remaining.remove(node)
    return order


def compute_derived(template_name, root_values):
    """Compute all derived values from root values using template formulas."""
    values = dict(root_values)
    for node in get_derived_order(template_name):
        if node in template_expressions and node not in SKIP_NODES:
            try:
                val = eval(template_expressions[node],
                           {"np": np, "abs": abs, "math": math}, values)
                values[node] = val
            except Exception:
                values[node] = float('nan')
    return values


# ── Index recovery ───────────────────────────────────────────────────────

def value_to_index(val, lo, hi, m=NUM_CANDIDATES):
    """Recover the integer grid index from a released value."""
    return int(round((val - lo) / (hi - lo) * (m - 1)))


# ── Prior functions ──────────────────────────────────────────────────────

def log_prior_independent_gaussian(x_array, roots, pop_stats, template_name):
    """Independent Gaussian prior per root from NHANES holdout stats."""
    stats = pop_stats[template_name]
    mu = np.array([stats["means"][r] for r in roots])
    sigma = np.array([stats["stds"][r] for r in roots])
    diff = x_array - mu
    return -0.5 * np.sum((diff / sigma) ** 2)


def log_prior_multivariate_gaussian(x_array, roots, pop_stats, template_name):
    """Multivariate Gaussian prior with full covariance from holdout."""
    stats = pop_stats[template_name]
    root_order = stats["cov_root_order"]
    idx_map = {r: i for i, r in enumerate(roots)}
    x_ordered = np.array([x_array[idx_map[r]] for r in root_order])
    diff = x_ordered - stats["cov_means"]
    return -0.5 * diff @ stats["cov_inv"] @ diff


# ── Noise function wrappers ─────────────────────────────────────────────

_san = Sanitizer()

NOISE_FNS = {
    "exponential":     lambda v, e, lo, hi: _san.M_exponential_discrete(
                           v, e, lo, hi, NUM_CANDIDATES),
    "bounded_laplace": lambda v, e, lo, hi: bounded_laplace_noise(
                           v, e, lo, hi, NUM_CANDIDATES),
    "staircase":       lambda v, e, lo, hi: staircase_noise(
                           v, e, lo, hi, NUM_CANDIDATES),
}


# ── Allocation loading (v2) ─────────────────────────────────────────────

_OPT_MECH_MAP = {
    "popabs":              "exponential",
    "blap_roots_opt":      "bounded_laplace",
    "staircase_roots_opt": "staircase",
}


def get_uniform_allocation(template_name, epsilon):
    """M-Roots allocation: uniform eps_r per root. Total B = k*eps_r."""
    return {r: float(epsilon) for r in TEMPLATES[template_name]["roots"]}


def load_mopt_allocation_v2(mechanism_family, template_name, epsilon):
    """Load M-Opt allocation from allocations_v2/ (computed at B = k*eps_r)."""
    path = (f"{ALLOCATIONS_BASE}/{mechanism_family}/"
            f"{template_name}_eps{epsilon}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return {r: float(v) for r, v in data["allocation"].items()}


def load_opt_allocation(tmpl_name, opt_method, epsilon):
    """Load M-Opt allocation for r_star selection (via allocations_v2/)."""
    mech = _OPT_MECH_MAP.get(opt_method)
    if mech is None:
        return None
    return load_mopt_allocation_v2(mech, tmpl_name, epsilon)


def load_allocation_for_method(method_name, template_name, epsilon):
    """Dispatcher: per-method allocation used in sanitization and MAP."""
    _, variant = METHODS[method_name]
    if variant == "all":
        return None
    if variant == "roots_uniform":
        return get_uniform_allocation(template_name, epsilon)
    if variant == "roots_opt":
        return load_mopt_allocation_v2(
            METHODS[method_name][0], template_name, epsilon)
    raise ValueError(f"Unknown variant: {variant}")


# ── Method-agnostic RNG seeding ─────────────────────────────────────────

def _seed_for_draw(template, root, sample_idx, draw_idx):
    """Seed for a single noise draw. Method-agnostic: M-All draw 0 on
    (template, root, sample) matches M-Roots cached draw (same seed)."""
    return (hash((template, root, sample_idx, int(draw_idx))) % (2**31))


# ── Observation generation ──────────────────────────────────────────────

def _draw_with_seed(noise_fn, val, eps, lo, hi, seed):
    """Deterministic noise draw. Seeds numpy and python random before calling.
    Works for any mechanism that uses np.random internally (all three here do)."""
    import random as _random
    np.random.seed(seed)
    _random.seed(seed)
    return noise_fn(val, eps, lo, hi)


def generate_adversarial_obs(gt_roots, template_name, method_name, epsilon,
                             allocation, noise_fn, strategy, q,
                             domains, r_star, sample_idx):
    """Generate q adversarial observations + 1 extra root release per spec v3.

    Strategy A: q queries on r_star + 1 extra release on r_star = q+1 obs.
    Strategy B: q queries per root + 1 extra release per root = q+1 obs each.

    For M-All: q+1 independent draws (draw_idx 0..q-1 + draw_idx=q for extra).
    For cached variants: q+1 copies of the single cached draw (no info gain
    from the extra release).

    Seeding is method-agnostic: (template, root, sample_idx, draw_idx) →
    M-All draw 0 and M-Roots cached draw produce identical per-root values.
    """
    roots = TEMPLATES[template_name]["roots"]
    _, variant = METHODS[method_name]

    observations = {r: [] for r in roots}

    if strategy == "A":
        affected = [r_star]
    else:
        affected = list(roots)

    for r in affected:
        lo, hi = domains[r]
        if variant == "all":
            # q adversarial draws (draw_idx 0..q-1)
            for di in range(q):
                seed = _seed_for_draw(template_name, r, sample_idx, di)
                observations[r].append(
                    _draw_with_seed(noise_fn, gt_roots[r], epsilon, lo, hi, seed))
            # +1 extra release (draw_idx = q, independent of adv draws)
            seed = _seed_for_draw(template_name, r, sample_idx, q)
            observations[r].append(
                _draw_with_seed(noise_fn, gt_roots[r], epsilon, lo, hi, seed))
        else:  # cached variant
            eps_r_alloc = allocation[r] if allocation else epsilon
            seed = _seed_for_draw(template_name, r, sample_idx, 0)
            cached = _draw_with_seed(
                noise_fn, gt_roots[r], eps_r_alloc, lo, hi, seed)
            # q+1 identical cached copies (extra release replays cached)
            observations[r] = [cached] * (q + 1)

    return observations


def generate_target_release(gt_values, template_name, method_name, epsilon,
                            allocation, noise_fn, domains,
                            adv_obs=None, sample_idx=0):
    """Generate the legitimate utility release.

    For cached variants (roots_opt / roots_uniform): the root values released
    here MUST match the single cached draw used in generate_adversarial_obs
    (so the target release T(x_hat) is deterministic given cached roots, as
    required by spec). We reuse adv_obs[r][0] as the cached root release.

    For M-All: independently noise every node at eps_r using target-release
    draw seed (draw_idx = "target" bucket to avoid collision with adv draws).
    """
    roots = TEMPLATES[template_name]["roots"]
    derived = TEMPLATES[template_name]["derived"]
    _, variant = METHODS[method_name]

    release = {}

    if variant == "all":
        # M-All: independently sanitize every node, seeded distinctly from adv draws
        for node in roots + derived:
            if node in SKIP_NODES or node not in domains:
                continue
            lo, hi = domains[node]
            # Use a target-specific draw index (large integer namespace) to
            # avoid colliding with adversarial draws 0..q-1
            seed = _seed_for_draw(
                template_name, node, sample_idx, 10_000_000)
            release[node] = _draw_with_seed(
                noise_fn, float(gt_values[node]), epsilon, lo, hi, seed)
    else:
        # Cached variants: reuse the single cached root draw from adv_obs if
        # provided (ensures T(x_hat) is exactly what was released to the
        # adversary). If adv_obs is missing that root, fall back to a fresh
        # seeded draw at the allocated eps_r.
        noised_roots = {}
        for r in roots:
            lo, hi = domains[r]
            eps_r = allocation[r] if allocation else epsilon
            if adv_obs and r in adv_obs and adv_obs[r]:
                noised_roots[r] = adv_obs[r][0]
            else:
                seed = _seed_for_draw(template_name, r, sample_idx, 0)
                noised_roots[r] = _draw_with_seed(
                    noise_fn, float(gt_values[r]), eps_r, lo, hi, seed)
            release[r] = noised_roots[r]
        derived_vals = compute_derived(template_name, noised_roots)
        for d in derived:
            if d in derived_vals and d not in SKIP_NODES:
                release[d] = derived_vals[d]

    return release


# ── MAP estimation with multiple observations ────────────────────────────

def _argmax_1d_root(root, obs_values, eps_r, domain_lo, domain_hi,
                    log_pmf_batch_fn, prior_mu=None, prior_sigma=None):
    """1-D grid-search argmax for a single root given obs_values.

    log posterior = log_pmf_batch(x, obs_indices, eps_r, lo, hi) +
                    log Gaussian(x | mu, sigma)  [if prior_mu given]
    """
    obs_idx = [value_to_index(v, domain_lo, domain_hi) for v in obs_values]
    x_grid = np.linspace(domain_lo, domain_hi, NUM_CANDIDATES)
    best_val = -np.inf
    best_x = domain_lo
    for xi in range(NUM_CANDIDATES):
        x = x_grid[xi]
        ll = log_pmf_batch_fn(x, obs_idx, eps_r, domain_lo, domain_hi)
        if prior_mu is not None and prior_sigma is not None and prior_sigma > 0:
            ll += -0.5 * ((x - prior_mu) / prior_sigma) ** 2
        if ll > best_val:
            best_val = ll
            best_x = float(x)
    return best_x


def map_estimate_multi_obs(root_observations, target_release,
                           template_name, method_name, epsilon,
                           allocation, log_pmf_fn, prior_type, pop_stats,
                           mech_family=None, strategy="B", r_star=None,
                           pmf_cache=None):
    """v3 MAP: per-root 1-D argmax for every (strategy, prior, variant) cell.

    Adversary's observation set per affected root has q+1 entries:
      M-All:  q independent adv draws + 1 independent extra release.
      Cached: q+1 copies of the single cached value.

    There is no target release in the likelihood — root observations only.

    Returns dict {root: MAP_estimate}:
      Strategy A: r_star gets MAP from q+1 obs (+ optional Gaussian prior).
                  Other roots have no obs → return prior mean (informed) or
                  domain midpoint (uniform).
      Strategy B: every root gets MAP from its q+1 obs (+ optional prior).

    `target_release` arg is kept for signature compatibility but unused.
    """
    roots = TEMPLATES[template_name]["roots"]
    _, variant = METHODS[method_name]
    domains = TEMPLATE_DOMAINS[template_name]
    log_pmf_batch_fn = (LOG_PMF_BATCH_FNS.get(mech_family)
                        if mech_family else None)

    def _alloc_eps(r):
        return allocation[r] if allocation else epsilon

    def _prior_for(r):
        if prior_type == "informed" and pop_stats:
            return (pop_stats[template_name]["means"][r],
                    pop_stats[template_name]["stds"][r])
        return (None, None)

    if strategy == "A":
        target_roots = [r_star] if r_star else [roots[0]]
    else:
        target_roots = list(roots)

    out = {}
    for r in roots:
        lo, hi = domains[r]
        obs = root_observations.get(r, [])
        if r in target_roots and obs:
            mu, sigma = _prior_for(r)
            eps_r = _alloc_eps(r)
            # Vectorized fast path: per-config PMF matrix already built.
            if pmf_cache is not None and (r, eps_r) in pmf_cache:
                log_terms, log_Z, x_grid = pmf_cache[(r, eps_r)]
                obs_idx = [value_to_index(v, lo, hi) for v in obs]
                out[r] = _argmax_1d_vec(
                    obs_idx, log_terms, log_Z, x_grid, mu, sigma)
            else:
                # Fallback: build on demand (used by --verify and ad-hoc calls).
                obs_idx = [value_to_index(v, lo, hi) for v in obs]
                log_terms, log_Z = build_pmf_matrix(mech_family, eps_r)
                x_grid = np.linspace(lo, hi, NUM_CANDIDATES)
                out[r] = _argmax_1d_vec(
                    obs_idx, log_terms, log_Z, x_grid, mu, sigma)
        else:
            mu, _ = _prior_for(r)
            out[r] = mu if mu is not None else (lo + hi) / 2.0
    return out


# ── Parallel worker ──────────────────────────────────────────────────────

def _adversarial_map_worker(args):
    """Top-level function for ProcessPoolExecutor."""
    (idx, root_observations, tmpl_name, method_name,
     epsilon, allocation, mech_family, prior_type, pop_stats,
     strategy, r_star) = args
    log_pmf_fn = LOG_PMF_FNS[mech_family]
    est = map_estimate_multi_obs(
        root_observations, {}, tmpl_name, method_name,
        epsilon, allocation, log_pmf_fn, prior_type, pop_stats,
        mech_family=mech_family, strategy=strategy, r_star=r_star)
    return idx, est


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Loading population statistics...", flush=True)
    pop_stats = load_population_stats()

    print("Loading ground truth...", flush=True)
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    # Pre-extract ground truth
    gt_by_template = {}
    for tmpl_name in TEMPLATES:
        samples = benchmark[tmpl_name][:NUM_SAMPLES]
        gt_values = []
        for sample in samples:
            _, vals = get_topological_order(sample, template_edges)
            gt_values.append(vals)
        gt_by_template[tmpl_name] = gt_values

    # Filter parameters by CLI args
    epsilons = [cli_args.epsilon] if cli_args.epsilon else EPSILONS
    templates = [cli_args.template] if cli_args.template else list(TEMPLATES.keys())
    methods = {cli_args.method: METHODS[cli_args.method]} if cli_args.method else METHODS
    strategies = [cli_args.strategy] if cli_args.strategy else STRATEGIES
    q_values = [cli_args.q] if cli_args.q else Q_VALUES
    priors = [cli_args.prior] if cli_args.prior else PRIORS

    total_configs = (len(epsilons) * len(templates) * len(methods)
                     * len(priors) * len(strategies) * len(q_values))
    print(f"Epsilons: {epsilons}")
    print(f"Templates: {templates}")
    print(f"Methods: {list(methods.keys())}")
    print(f"Priors: {priors}")
    print(f"Strategies: {strategies}")
    print(f"Q values: {q_values}")
    print(f"Total configs: {total_configs}")
    print(f"Samples per template: {NUM_SAMPLES}", flush=True)

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    configs_done = 0

    for epsilon in epsilons:
        eps_dir = f"{OUTPUT_BASE}/epsilon_{epsilon}"
        os.makedirs(eps_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Epsilon = {epsilon}")
        print(f"{'='*70}", flush=True)

        for tmpl_name in templates:
            if tmpl_name not in TEMPLATES:
                print(f"  WARNING: Unknown template {tmpl_name}, skipping")
                continue

            tmpl = TEMPLATES[tmpl_name]
            roots = tmpl["roots"]
            derived = tmpl["derived"]
            gt_values = gt_by_template[tmpl_name]
            domains = TEMPLATE_DOMAINS[tmpl_name]

            tmpl_dir = f"{eps_dir}/{tmpl_name}"
            os.makedirs(tmpl_dir, exist_ok=True)

            # Pre-load M-Opt allocations for r_star selection
            opt_allocations = {}
            for opt_method in ["popabs", "blap_roots_opt", "staircase_roots_opt"]:
                alloc = load_opt_allocation(tmpl_name, opt_method, epsilon)
                if alloc:
                    opt_allocations[opt_method] = alloc

            for method_name, (mech_family, variant) in methods.items():
                if method_name not in METHODS:
                    continue

                # Load per-method allocation (None for M-All; uniform for
                # M-Roots; optimized for M-Opt)
                allocation = load_allocation_for_method(
                    method_name, tmpl_name, epsilon)
                if variant != "all" and allocation is None:
                    print(f"  {tmpl_name}/{method_name} — no allocation, skipping",
                          flush=True)
                    continue

                # Determine r_star from the M-Opt counterpart's allocation
                # (so r_star is consistent across M-All/M-Roots/M-Opt)
                opt_method = ALL_TO_OPT.get(method_name, method_name)
                opt_alloc = opt_allocations.get(opt_method)
                if opt_alloc:
                    r_star = max(opt_alloc, key=opt_alloc.get)
                elif allocation:
                    r_star = max(allocation, key=allocation.get)
                else:
                    r_star = roots[0]

                # Staircase validity check (applies to M-All, M-Roots, M-Opt)
                if mech_family == "staircase":
                    if allocation:
                        max_root_eps = max(allocation.values())
                    else:
                        # M-All: each node gets full epsilon
                        max_root_eps = epsilon
                    if max_root_eps > STAIRCASE_MAX_ROOT_EPS:
                        print(f"  {tmpl_name}/{method_name} — staircase per-root "
                              f"eps {max_root_eps:.3f} > {STAIRCASE_MAX_ROOT_EPS}, "
                              f"skipping", flush=True)
                        continue

                noise_fn = NOISE_FNS[mech_family]

                for prior_type in priors:
                    for strategy in strategies:
                        for q in q_values:
                            label = (f"{method_name}_{prior_type}_"
                                     f"{strategy}_q{q}")
                            out_path = f"{tmpl_dir}/{label}_results.json"

                            # Skip if result exists
                            if os.path.exists(out_path):
                                configs_done += 1
                                continue

                            # Cached variants: results identical across q
                            # (MAP depends only on cached single obs + prior).
                            # Copy from q=1 if available.
                            is_cached = variant in CACHED_VARIANTS
                            if is_cached and q > 1:
                                q1_path = (f"{tmpl_dir}/{method_name}_"
                                           f"{prior_type}_{strategy}_"
                                           f"q1_results.json")
                                if os.path.exists(q1_path):
                                    with open(q1_path) as f:
                                        q1_data = json.load(f)
                                    q1_data["q"] = q
                                    with open(out_path, "w") as f:
                                        json.dump(q1_data, f, indent=2)
                                    configs_done += 1
                                    continue

                            t0 = time.time()
                            print(f"  {tmpl_name} {label} "
                                  f"({NUM_SAMPLES} samples)...",
                                  end="", flush=True)

                            # Generate observations and run MAP for all samples
                            per_sample_results = []
                            map_errors = {r: [] for r in roots}
                            naive_errors = {r: [] for r in roots}

                            # Step 1: Generate all observations upfront
                            # adv_obs[r] now has q+1 entries on affected roots
                            # (q adversarial + 1 extra release; cached methods
                            # repeat the cached value q+1 times).
                            all_obs_data = []
                            for i in range(NUM_SAMPLES):
                                gt = gt_values[i]
                                gt_roots = {r: float(gt[r]) for r in roots}

                                adv_obs = generate_adversarial_obs(
                                    gt_roots, tmpl_name, method_name,
                                    epsilon, allocation, noise_fn,
                                    strategy, q, domains, r_star, i)

                                all_obs_data.append((i, adv_obs, gt_roots, gt))

                            # Step 2: Build per-config PMF cache for the
                            # vectorized 1-D argmax (one matrix per unique
                            # (root, eps_r), shared across all 200 samples).
                            pmf_cache = {}
                            target_roots_iter = (
                                [r_star] if strategy == "A" else roots)
                            for r in target_roots_iter:
                                eps_r = (allocation[r] if allocation
                                         else epsilon)
                                key = (r, eps_r)
                                if key in pmf_cache:
                                    continue
                                lo_r, hi_r = domains[r]
                                lt, lz = build_pmf_matrix(mech_family, eps_r)
                                xg = np.linspace(lo_r, hi_r, NUM_CANDIDATES)
                                pmf_cache[key] = (lt, lz, xg)

                            # Step 3: Run MAP estimation single-threaded
                            # (vectorized, microseconds per sample).
                            log_pmf_fn = LOG_PMF_FNS[mech_family]
                            estimates = {}
                            for i, adv_obs, gt_roots, gt in all_obs_data:
                                estimates[i] = map_estimate_multi_obs(
                                    adv_obs, {}, tmpl_name, method_name,
                                    epsilon, allocation, log_pmf_fn,
                                    prior_type, pop_stats,
                                    mech_family=mech_family, strategy=strategy,
                                    r_star=r_star, pmf_cache=pmf_cache)

                            # Step 4: Compute metrics and collect per-sample data
                            target_roots_for_metrics = (
                                [r_star] if strategy == "A" else roots)
                            for i, adv_obs, gt_roots, gt in all_obs_data:
                                est = estimates[i]

                                # Skip samples with near-zero ground truth
                                if any(r in gt and abs(gt[r]) < 1e-6
                                       for r in roots):
                                    continue

                                # Naive: mean of root observations (q+1 obs on
                                # affected roots; no obs on unaffected roots
                                # under Strategy A).
                                naive_est = {}
                                for r in roots:
                                    obs_r = adv_obs.get(r, [])
                                    if obs_r:
                                        naive_est[r] = float(np.mean(obs_r))
                                    else:
                                        naive_est[r] = None

                                sample_result = {
                                    "sample_idx": i,
                                    "ground_truth": {r: float(gt[r])
                                                     for r in roots},
                                    "map_estimates": {r: float(est[r])
                                                      for r in roots
                                                      if r in est},
                                    "naive_estimates": {
                                        r: (float(v) if v is not None else None)
                                        for r, v in naive_est.items()},
                                    "adversarial_obs": {
                                        r: [float(v) for v in adv_obs.get(r, [])]
                                        for r in roots},
                                }
                                per_sample_results.append(sample_result)

                                for r in target_roots_for_metrics:
                                    if r in gt and gt[r] != 0 and r in est:
                                        m_err = (abs(est[r] - gt[r])
                                                 / abs(gt[r]) * 100)
                                        map_errors[r].append(m_err)
                                        if naive_est.get(r) is not None:
                                            n_err = (abs(naive_est[r] - gt[r])
                                                     / abs(gt[r]) * 100)
                                            naive_errors[r].append(n_err)

                            # Aggregate per-root stats
                            per_root = {}
                            for r in roots:
                                me = map_errors[r]
                                ne = naive_errors[r]
                                deltas = [n - m for n, m in zip(ne, me)]
                                per_root[r] = {
                                    "map_mae": float(np.mean(me)) if me else None,
                                    "naive_mae": float(np.mean(ne)) if ne else None,
                                    "map_std": float(np.std(me)) if me else None,
                                    "naive_std": float(np.std(ne)) if ne else None,
                                    "delta_mean": (float(np.mean(deltas))
                                                   if deltas else None),
                                    "delta_std": (float(np.std(deltas))
                                                  if deltas else None),
                                    "n": len(me),
                                    "map_errors": [float(x) for x in me],
                                    "naive_errors": [float(x) for x in ne],
                                }

                            # Compute wMAPE summary
                            # For Strategy A: only r_star. For B: all roots.
                            target_roots = ([r_star] if strategy == "A"
                                            else roots)
                            wmape_map_vals = []
                            wmape_naive_vals = []
                            for r in target_roots:
                                me = map_errors[r]
                                ne = naive_errors[r]
                                if not me:
                                    continue
                                # Get ground truth abs values for weighting
                                gt_abs = []
                                for sr in per_sample_results:
                                    if r in sr["ground_truth"]:
                                        gt_abs.append(
                                            abs(sr["ground_truth"][r]))
                                gt_abs = np.array(gt_abs[:len(me)])
                                denom = np.sum(gt_abs)
                                if denom > 0:
                                    wmape_map_vals.append(
                                        np.sum(np.array(me) * gt_abs) / denom)
                                    wmape_naive_vals.append(
                                        np.sum(np.array(ne) * gt_abs) / denom)

                            wmape_map = (float(np.mean(wmape_map_vals))
                                         if wmape_map_vals else None)
                            wmape_naive = (float(np.mean(wmape_naive_vals))
                                           if wmape_naive_vals else None)
                            wmape_delta = (float(wmape_naive - wmape_map)
                                           if wmape_map is not None
                                           and wmape_naive is not None
                                           else None)

                            elapsed = time.time() - t0
                            print(f" wMAPE_map={wmape_map:.2f}% "
                                  f"wMAPE_naive={wmape_naive:.2f}% "
                                  f"Δ={wmape_delta:+.2f}% "
                                  f"({elapsed:.1f}s)"
                                  if wmape_map is not None
                                  else f" no data ({elapsed:.1f}s)",
                                  flush=True)

                            # Save
                            out = {
                                "per_root": per_root,
                                "per_sample": per_sample_results,
                                "summary": {
                                    "wmape_map": wmape_map,
                                    "wmape_naive": wmape_naive,
                                    "wmape_delta": wmape_delta,
                                },
                                "r_star": r_star if strategy == "A" else None,
                                "method": method_name,
                                "mechanism": mech_family,
                                "variant": variant,
                                "epsilon": epsilon,
                                "prior": prior_type,
                                "strategy": strategy,
                                "q": q,
                                "allocation": (
                                    {rr: float(v) for rr, v in allocation.items()}
                                    if allocation else None),
                                "n_samples": NUM_SAMPLES,
                            }
                            with open(out_path, "w") as f:
                                json.dump(out, f, indent=2)

                            configs_done += 1

    print(f"\n{'='*70}")
    print(f"Done. {configs_done} configurations processed.")
    print(f"Results in {OUTPUT_BASE}/")


# ── Verification (v2) ────────────────────────────────────────────────────

def verify():
    """v3 sanity checks.

    1. PMF normalization.
    2. M-All draw 0 == M-Roots cached draw at same (template, sample, eps_r).
    3. obs[r] has length q+1 for affected roots; cached methods have q+1 copies.
    4. M-Roots / M-Opt MAP invariance across q.
    5. Uniform + cached + Strategy B: per-root MAP == cached value.
    6. M-All q-scaling: MAE decreases as q grows (soft).
    """
    print("Running v3 verification checks...\n", flush=True)
    pop_stats = load_population_stats()

    # 1. PMF normalization
    print("1. PMF normalization check:")
    for name, pmf_fn in LOG_PMF_FNS.items():
        eps_test = 0.5 if name == "staircase" else 0.1
        lo, hi = 0.0, 100.0
        x_true = 50.0
        total = 0.0
        for s in range(NUM_CANDIDATES):
            total += np.exp(pmf_fn(x_true, s, eps_test, lo, hi))
        print(f"   {name} (eps={eps_test}): sum(PMF) = {total:.6f} "
              f"{'OK' if abs(total - 1.0) < 1e-4 else 'FAIL'}")

    tmpl = "ANEMIA"
    eps = 0.1
    domains = TEMPLATE_DOMAINS[tmpl]
    roots = TEMPLATES[tmpl]["roots"]
    noise_fn = NOISE_FNS["exponential"]
    gt_roots = {"hb": 14.5, "hct": 42.0, "rbc": 5.2}
    alloc_uniform = get_uniform_allocation(tmpl, eps)

    # 2. M-All draw 0 == M-Roots cached draw (method-agnostic seed)
    print("\n2. M-All draw 0 == M-Roots cached draw (seeded):")
    obs_all = generate_adversarial_obs(
        gt_roots, tmpl, "vanilla", eps, None, noise_fn,
        "B", 1, domains, None, 0)
    obs_roots = generate_adversarial_obs(
        gt_roots, tmpl, "vanilla_roots", eps, alloc_uniform, noise_fn,
        "B", 1, domains, None, 0)
    all_ok = True
    for r in roots:
        v_all = obs_all[r][0]
        v_roots = obs_roots[r][0]
        ok = abs(v_all - v_roots) < 1e-9
        all_ok = all_ok and ok
        print(f"   {r}: M-All draw0={v_all:.6f}  M-Roots cached={v_roots:.6f}"
              f"  {'OK' if ok else 'FAIL'}")
    print(f"   Overall: {'PASS' if all_ok else 'FAIL'}")

    # 3. obs length and cached structure
    print("\n3. obs length q+1 for affected roots:")
    for q_test in [1, 4, 8, 16]:
        # Strategy B: every root has q+1 obs
        obs_b = generate_adversarial_obs(
            gt_roots, tmpl, "vanilla_roots", eps, alloc_uniform, noise_fn,
            "B", q_test, domains, None, 0)
        ok_len = all(len(obs_b[r]) == q_test + 1 for r in roots)
        # Cached: all q+1 copies identical
        ok_cached = all(
            all(abs(v - obs_b[r][0]) < 1e-12 for v in obs_b[r])
            for r in roots)
        # Strategy A: only r_star has obs, length q+1
        obs_a = generate_adversarial_obs(
            gt_roots, tmpl, "vanilla", eps, None, noise_fn,
            "A", q_test, domains, "hb", 0)
        ok_a = (len(obs_a["hb"]) == q_test + 1
                and all(len(obs_a[r]) == 0 for r in roots if r != "hb"))
        print(f"   q={q_test}: B+1 length={ok_len}, cached identical={ok_cached},"
              f" A only r_star={ok_a}  "
              f"{'OK' if (ok_len and ok_cached and ok_a) else 'FAIL'}")

    # 4. M-Roots MAP invariance across q
    print("\n4. M-Roots MAP invariance across q (Strategy B, informed):")
    map_q = {}
    for q_test in [1, 4, 8, 16]:
        obs_q = generate_adversarial_obs(
            gt_roots, tmpl, "vanilla_roots", eps, alloc_uniform, noise_fn,
            "B", q_test, domains, None, 0)
        est = map_estimate_multi_obs(
            obs_q, {}, tmpl, "vanilla_roots", eps, alloc_uniform,
            log_pmf_exponential, "informed", pop_stats,
            mech_family="exponential", strategy="B", r_star=None)
        map_q[q_test] = {r: est[r] for r in roots}
    base = map_q[1]
    inv_ok = all(
        all(abs(map_q[q][r] - base[r]) < 1e-6 for r in roots)
        for q in [4, 8, 16])
    print(f"   q=1 vs {{4,8,16}} match: {'OK' if inv_ok else 'FAIL'}")

    # 5. Uniform + cached + Strategy B: per-root MAP == cached value
    print("\n5. Uniform + cached + B: MAP == cached:")
    obs1 = generate_adversarial_obs(
        gt_roots, tmpl, "vanilla_roots", eps, alloc_uniform, noise_fn,
        "B", 1, domains, None, 0)
    est = map_estimate_multi_obs(
        obs1, {}, tmpl, "vanilla_roots", eps, alloc_uniform,
        log_pmf_exponential, "uniform", pop_stats,
        mech_family="exponential", strategy="B", r_star=None)
    max_err = max(abs(est[r] - obs1[r][0]) for r in roots)
    print(f"   max |MAP - cached| = {max_err:.6f}  "
          f"{'OK' if max_err < 1e-2 else 'FAIL'}")  # 1-D grid resolution

    # 6. M-All q-scaling (soft check)
    print("\n6. M-All q-scaling (MAE vs q, Strategy A, r_star=hb):")
    r_star_t = "hb"
    errors = {}
    for q_test in [1, 4, 8, 16]:
        obs = generate_adversarial_obs(
            gt_roots, tmpl, "vanilla", eps, None, noise_fn,
            "A", q_test, domains, r_star_t, 0)
        est = map_estimate_multi_obs(
            obs, {}, tmpl, "vanilla", eps, None,
            log_pmf_exponential, "uniform", pop_stats,
            mech_family="exponential", strategy="A", r_star=r_star_t)
        err = abs(est[r_star_t] - gt_roots[r_star_t])
        errors[q_test] = err
        print(f"   q={q_test}: |MAP - gt|[r_star] = {err:.4f}")
    print("   MAE decreases with q:" + (
        " OK" if errors[16] < errors[1] else " WARN (single-sample, noisy)"))

    print("\nVerification complete.", flush=True)


if __name__ == "__main__":
    if cli_args.verify:
        verify()
    else:
        main()
