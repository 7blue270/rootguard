#!/usr/bin/env python3
"""
Precompute M-Opt allocations for RQ3 v2 with total budget B = k * eps_r.

RQ3 v1 used B = n * eps (n = total non-skip nodes including derived), cached
in results_v10_holdout/. The v2 spec requires B = k * eps_r (k = roots only),
so allocations must be recomputed.

Output: allocations_v2/{mechanism}/{template}_eps{eps_r}.json
        {"allocation": {root: eps_i}, "total_budget": k * eps_r,
         "k_roots": k, "epsilon_r": eps_r, "sensitivities": {...}}

Usage:
    python precompute_mopt_allocations_v2.py
    python precompute_mopt_allocations_v2.py --eps 0.1  # single epsilon
"""
# _REPO_ROOT_BOOTSTRAP: ensure repo root is on sys.path so that
# 'from utils.*' / 'from preempt.*' imports resolve when this script
# is run from its own subfolder.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
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

EPSILONS = [0.01, 0.05, 0.1, 0.5, 1.0]
EPSILON_MIN = 0.001
NUM_CANDIDATES = 1000
OUTPUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "allocations_v2")
BENCHMARK_PATH = "../data/nhanes_benchmark_200.json"
HOLDOUT_STATS_PATH = "../data/holdout_population_means.json"

SKIP_NODES = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
              "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

MECHANISM_ALLOCATORS = {
    "exponential":     optimal_budget_allocation_exact,
    "bounded_laplace": optimal_budget_allocation_blap,
    "staircase":       optimal_budget_allocation_staircase,
}

# Templates in canonical order
TEMPLATES = ["ANEMIA", "FIB4", "AIP", "CONICITY", "VASCULAR", "TYG", "HOMA", "NLR"]


def get_template_structure(sample):
    """Identify root_nodes, derived_nodes, target_node, topo_order."""
    topo_order, gt_values = get_topological_order(sample, template_edges)
    root_nodes = []
    derived_nodes = []
    for node in topo_order:
        if node not in template_nodes:
            continue
        prevs = template_nodes[node]
        if prevs:
            prevs = [p for p in prevs if p in gt_values]
        if not prevs:
            root_nodes.append(node)
        else:
            derived_nodes.append(node)

    target_node = None
    for tname, tnode in template_target_keys.items():
        if tnode in gt_values and tnode not in SKIP_NODES:
            target_node = tnode
            break
    return topo_order, root_nodes, derived_nodes, target_node


def compute_allocation_for(mechanism, template, eps_r, samples, holdout_means):
    """Compute M-Opt allocation at total_budget = k * eps_r for (mechanism, template, eps_r)."""
    sample = samples[template][0]
    topo_order, root_nodes, _, target_node = get_template_structure(sample)

    k = len(root_nodes)
    total_budget = k * eps_r

    if target_node is None or k == 0:
        raise RuntimeError(f"No target/roots found for {template}")

    sensitivities = compute_population_mean_sensitivities(
        target_node, root_nodes, template_expressions, SKIP_NODES,
        topo_order, template_nodes, holdout_means[template])

    dw = {}
    for r in root_nodes:
        lo, hi = MEDICAL_DOMAINS.get(r, MEDICAL_DOMAINS["_default"])
        dw[r] = hi - lo

    allocator = MECHANISM_ALLOCATORS[mechanism]
    if mechanism == "exponential":
        allocation = allocator(
            sensitivities, total_budget, EPSILON_MIN,
            domain_widths=dw, num_candidates=NUM_CANDIDATES, mode="abs")
    else:
        allocation = allocator(
            sensitivities, total_budget, EPSILON_MIN,
            domain_widths=dw, num_candidates=NUM_CANDIDATES)

    alloc_sum = sum(allocation.values())
    if not np.isclose(alloc_sum, total_budget, rtol=1e-4):
        print(f"    WARNING: {template}/{mechanism}/eps={eps_r}: "
              f"sum(alloc)={alloc_sum:.6f} != k*eps_r={total_budget:.6f}",
              flush=True)

    return {
        "allocation": {r: float(v) for r, v in allocation.items()},
        "sensitivities": {r: float(v) for r, v in sensitivities.items()},
        "total_budget": float(total_budget),
        "k_roots": k,
        "epsilon_r": float(eps_r),
        "domain_widths": {r: float(v) for r, v in dw.items()},
        "target_node": target_node,
        "alloc_sum": float(alloc_sum),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=None,
                    help="Run a single epsilon only")
    ap.add_argument("--template", type=str, default=None)
    ap.add_argument("--mechanism", type=str, default=None,
                    choices=list(MECHANISM_ALLOCATORS.keys()))
    args = ap.parse_args()

    print("Loading benchmark and holdout stats...", flush=True)
    with open(BENCHMARK_PATH) as f:
        samples = json.load(f)
    with open(HOLDOUT_STATS_PATH) as f:
        holdout = json.load(f)
    holdout_means = holdout["per_template_means"]

    epsilons = [args.eps] if args.eps else EPSILONS
    templates = [args.template] if args.template else TEMPLATES
    mechanisms = ([args.mechanism] if args.mechanism
                  else list(MECHANISM_ALLOCATORS.keys()))

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    total = len(mechanisms) * len(templates) * len(epsilons)
    done = 0

    for mech in mechanisms:
        mech_dir = f"{OUTPUT_BASE}/{mech}"
        os.makedirs(mech_dir, exist_ok=True)
        for tmpl in templates:
            for eps_r in epsilons:
                out_path = f"{mech_dir}/{tmpl}_eps{eps_r}.json"
                try:
                    rec = compute_allocation_for(
                        mech, tmpl, eps_r, samples, holdout_means)
                    with open(out_path, "w") as f:
                        json.dump(rec, f, indent=2)
                    alloc_str = ", ".join(
                        f"{r}={v:.4f}" for r, v in rec["allocation"].items())
                    print(f"  [{done+1}/{total}] {mech}/{tmpl}/eps={eps_r}: "
                          f"B={rec['total_budget']:.4f} ({alloc_str})",
                          flush=True)
                except Exception as e:
                    print(f"  [{done+1}/{total}] {mech}/{tmpl}/eps={eps_r}: "
                          f"FAILED: {e}", flush=True)
                done += 1

    print(f"\nDone. Allocations written to {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
