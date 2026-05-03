"""Shared loaders and formatters for RQ3 v3 generator scripts."""
# _REPO_ROOT_BOOTSTRAP: ensure repo root is on sys.path so that
# 'from utils.*' / 'from preempt.*' imports resolve when this script
# is run from its own subfolder.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.utils import get_topological_order
from utils.med_domain.all_templates import template_edges

RECON_DIR = "results_rq3_adversarial_v3"
ALLOC_DIR = "allocations_v2"
# Benchmark data lives at the repo root (../data/), one level up from this RQ folder.
# Fall back to ./data/ for legacy callers that supply a local copy.
_REPO_ROOT_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "nhanes_benchmark_200.json",
)
BENCHMARK_PATH = _REPO_ROOT_DATA if os.path.exists(_REPO_ROOT_DATA) else "./data/nhanes_benchmark_200.json"

EPSILONS = [0.01, 0.05, 0.1, 0.5, 1.0]
Q_VALUES = [1, 4, 8, 16]
PRIORS = ["uniform", "informed"]
STRATEGIES = ["A", "B"]
TEMPLATES = ["ANEMIA", "FIB4", "AIP", "CONICITY", "VASCULAR", "TYG", "HOMA", "NLR"]
NUM_SAMPLES = 200

MECHANISMS = [
    ("Exponential",      "Exp",   "vanilla",          "vanilla_roots",           "popabs"),
    ("Bounded Laplace",  "BLap",  "blap_all",         "blap_roots_uniform",      "blap_roots_opt"),
    ("Staircase",        "Stair", "staircase_all",    "staircase_roots_uniform", "staircase_roots_opt"),
]

METHOD_LABELS = {
    "vanilla":                 "M-All",
    "vanilla_roots":           "M-Roots",
    "popabs":                  "M-Opt",
    "blap_all":                "M-All",
    "blap_roots_uniform":      "M-Roots",
    "blap_roots_opt":          "M-Opt",
    "staircase_all":           "M-All",
    "staircase_roots_uniform": "M-Roots",
    "staircase_roots_opt":     "M-Opt",
}

TEMPLATE_ROOTS = {
    "ANEMIA": ["hb", "hct", "rbc"], "FIB4": ["age", "ast", "alt", "plt"],
    "AIP": ["tc", "hdl", "tg"], "CONICITY": ["wt", "ht", "waist"],
    "VASCULAR": ["sbp", "dbp"], "TYG": ["tg", "glu"],
    "HOMA": ["glu", "ins"], "NLR": ["neu", "lym"],
}

TEMPLATE_PAIRS = [
    ("ANEMIA", "FIB4"),
    ("AIP", "CONICITY"),
    ("VASCULAR", "TYG"),
    ("HOMA", "NLR"),
]

_GT_CACHE = None


def _load_gt():
    global _GT_CACHE
    if _GT_CACHE is not None:
        return _GT_CACHE
    with open(BENCHMARK_PATH) as f:
        bench = json.load(f)
    gt = {}
    for t in TEMPLATES:
        gt[t] = [get_topological_order(s, template_edges)[1]
                 for s in bench[t][:NUM_SAMPLES]]
    _GT_CACHE = gt
    return gt


def load_result(tmpl, method, prior, strategy, q, eps):
    """Load a v2 reconstruction result JSON, or None if missing."""
    label = f"{method}_{prior}_{strategy}_q{q}"
    fp = os.path.join(RECON_DIR, f"epsilon_{eps}", tmpl, f"{label}_results.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)


def load_allocation(mechanism_family, tmpl, eps):
    fp = os.path.join(ALLOC_DIR, mechanism_family, f"{tmpl}_eps{eps}.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)


def compute_wmape(result, tmpl, strategy):
    """wMAPE across target roots. Strategy A: r_star only. B: all roots."""
    if result is None:
        return None, None
    per_root = result.get("per_root", {})
    r_star = result.get("r_star")
    gt = _load_gt()[tmpl]
    roots = list(per_root.keys())

    if strategy == "A" and r_star:
        target_roots = [r_star]
    else:
        target_roots = roots

    vals = []
    for r in target_roots:
        if r not in per_root:
            continue
        me = per_root[r].get("map_errors", [])
        ne = per_root[r].get("naive_errors", [])
        n = len(me)
        if n == 0:
            continue
        included = []
        tmpl_roots = TEMPLATE_ROOTS[tmpl]
        for i in range(min(NUM_SAMPLES, len(gt))):
            s = gt[i]
            if any(rr in s and abs(s[rr]) < 1e-6 for rr in tmpl_roots):
                continue
            included.append(i)
        gt_abs = np.array([
            abs(gt[i][r]) if r in gt[i] and gt[i][r] != 0 else 0.0
            for i in included[:n]
        ])
        denom = np.sum(gt_abs)
        if denom > 0:
            vals.append((
                np.sum(np.array(me[:n]) * gt_abs) / denom,
                np.sum(np.array(ne[:n]) * gt_abs) / denom,
            ))
    if not vals:
        return None, None
    return float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals]))


def fmt(v):
    """Format a MAE/wMAPE value."""
    if v is None:
        return "--"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


def fmt_delta(v):
    if v is None:
        return "--"
    if abs(v) >= 100:
        return f"{v:+.0f}"
    return f"{v:+.1f}"
