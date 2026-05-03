#!/usr/bin/env python3
"""Precompute per-root M-Opt allocations for the agent_eval pipeline.

Adapts precompute_mopt_allocations_v2.py to the agent_eval setting:
- Total budget B = mult * eps with mult in {k+1, 2k+1, 3k+1} (RQ1 turn budgets)
  rather than v2's B = k * eps_r (RQ3 v2).
- Output format is the agent_eval/sanitizers/allocations.py CSV schema:
      template,mechanism,B_setting,config,root,eps
  with config in {"all","roots","opt"}, mechanism in {"exp","blap","stair"},
  and one root="*" row per (template, mechanism, B_setting) for "all".

Usage:
    python precompute_agent_eval_allocations.py
    python precompute_agent_eval_allocations.py --eps 0.1
    python precompute_agent_eval_allocations.py --out path/to/allocations.csv

Default output: agent_eval/agent_eval/data/allocations.csv
"""
# _REPO_ROOT_BOOTSTRAP: ensure repo root is on sys.path so that
# 'from utils.*' / 'from preempt.*' imports resolve when this script
# is run from its own subfolder.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import (
    get_topological_order,
    compute_population_mean_sensitivities,
    optimal_budget_allocation_exact,
    optimal_budget_allocation_blap,
    optimal_budget_allocation_staircase,
)
from utils.med_domain.all_templates import (
    template_nodes, template_edges, template_expressions,
    template_target_keys,
)
from preempt.sanitizer import MEDICAL_DOMAINS


EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
EPSILON_MIN = 0.001
NUM_CANDIDATES = 1000

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "allocations.csv",
)
BENCHMARK_PATH = "../data/nhanes_benchmark_200.json"
HOLDOUT_STATS_PATH = "../data/holdout_population_means.json"

SKIP_NODES = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
              "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

# (mechanism short name in CSV, optimizer fn, optimizer kwargs)
MECHANISMS = [
    ("exp",   "exponential",     optimal_budget_allocation_exact,       {"mode": "abs"}),
    ("blap",  "bounded_laplace", optimal_budget_allocation_blap,        {}),
    ("stair", "staircase",       optimal_budget_allocation_staircase,   {}),
]

# (B_setting label, multiplier on eps)
B_SETTINGS = [("k+1", lambda k: k + 1),
              ("2k+1", lambda k: 2 * k + 1),
              ("3k+1", lambda k: 3 * k + 1)]

# Templates exercised by the agent_eval harness (full set of 8 from the paper).
TEMPLATES = ["HOMA", "ANEMIA", "FIB4", "AIP", "CONICITY", "VASCULAR", "TYG", "NLR"]


def get_template_structure(sample):
    topo, gtv = get_topological_order(sample, template_edges)
    roots = []
    for node in topo:
        if node not in template_nodes:
            continue
        prevs = template_nodes[node] or []
        prevs = [p for p in prevs if p in gtv]
        if not prevs:
            roots.append(node)
    target = None
    for tname, tnode in template_target_keys.items():
        if tnode in gtv and tnode not in SKIP_NODES:
            target = tnode
            break
    return topo, roots, target


def root_display_name(template, root):
    """Map internal root names (lowercase) to the casing used in the
    agent_eval/templates/<>.py files (e.g. 'glu' -> 'Glu', 'age' -> 'age')."""
    aliases = {
        "HOMA":     {"glu": "Glu", "ins": "Ins"},
        "ANEMIA":   {"hb": "Hb", "hct": "Hct", "rbc": "RBC"},
        "FIB4":     {"age": "age", "ast": "AST", "alt": "ALT", "plt": "PLT"},
        "AIP":      {"tg": "tg", "hdl": "hdl"},
        "CONICITY": {"waist": "waist", "wt": "wt", "ht": "ht"},
        "VASCULAR": {"sbp": "sbp", "dbp": "dbp"},
        "TYG":      {"tg": "tg", "glu": "glu"},
        "NLR":      {"neu": "neu", "lym": "lym"},
    }
    return aliases.get(template, {}).get(root, root)


def compute_one(template, mech_short, mech_full, allocator, kwargs,
                eps, mult_fn, samples, holdout_means):
    """Return ((roots_alloc, opt_alloc), B, sensitivities)."""
    sample = samples[template][0]
    topo, roots, target = get_template_structure(sample)
    if target is None:
        raise RuntimeError(f"No target found for {template}")
    k = len(roots)
    B = mult_fn(k) * eps

    sens = compute_population_mean_sensitivities(
        target, roots, template_expressions, SKIP_NODES,
        topo, template_nodes, holdout_means[template])
    dw = {r: (MEDICAL_DOMAINS.get(r, MEDICAL_DOMAINS["_default"])[1] -
              MEDICAL_DOMAINS.get(r, MEDICAL_DOMAINS["_default"])[0]) for r in roots}
    opt_alloc = allocator(
        sens, B, EPSILON_MIN,
        domain_widths=dw, num_candidates=NUM_CANDIDATES, **kwargs,
    )
    s = sum(opt_alloc.values())
    if not np.isclose(s, B, rtol=1e-3):
        print(f"  WARN sum(opt)={s:.6f} != B={B:.6f} for "
              f"{template}/{mech_full}/eps={eps}/B_mult={mult_fn(k)}", flush=True)
    roots_alloc = {r: B / k for r in roots}
    return roots, roots_alloc, opt_alloc, sens, B, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=None,
                    help="Run a single epsilon (default: full sweep)")
    ap.add_argument("--template", type=str, default=None)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(BENCHMARK_PATH) as f:
        samples = json.load(f)
    with open(HOLDOUT_STATS_PATH) as f:
        holdout = json.load(f)
    holdout_means = holdout["per_template_means"]

    epsilons = [args.eps] if args.eps else EPSILONS
    templates = [args.template] if args.template else TEMPLATES

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for tmpl in templates:
        for eps in epsilons:
            for mech_short, mech_full, allocator, kwargs in MECHANISMS:
                for B_label, mult_fn in B_SETTINGS:
                    try:
                        roots, roots_alloc, opt_alloc, sens, B, k = compute_one(
                            tmpl, mech_short, mech_full, allocator, kwargs,
                            eps, mult_fn, samples, holdout_means,
                        )
                    except Exception as e:
                        print(f"  FAILED {tmpl}/{mech_full}/eps={eps}/{B_label}: {e}",
                              flush=True)
                        continue
                    # M-All: per-call eps stays as the input eps (independent of B).
                    rows.append((tmpl, mech_short, B_label, "all",
                                 float(eps), "*", float(eps)))
                    for r in roots:
                        name = root_display_name(tmpl, r)
                        rows.append((tmpl, mech_short, B_label, "roots",
                                     float(eps), name, float(roots_alloc[r])))
                        rows.append((tmpl, mech_short, B_label, "opt",
                                     float(eps), name, float(opt_alloc[r])))
                    print(f"  {tmpl}/{mech_short}/eps={eps}/{B_label} "
                          f"(B={B:.4f}, k={k}): "
                          f"opt=[{', '.join(f'{root_display_name(tmpl, r)}={opt_alloc[r]:.4f}' for r in roots)}]",
                          flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["template", "mechanism", "B_setting", "config",
                    "eps_in", "root", "eps"])
        for row in rows:
            w.writerow(row)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
