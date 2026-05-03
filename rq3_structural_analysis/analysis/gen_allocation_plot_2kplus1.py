#!/usr/bin/env python3
"""
Power-law allocation plots using budget B = (2k + 1) * eps,
where k = number of root nodes.

Outputs (at eps = 0.1):
  - plots_rq1/rq2_allocation_powerlaw_2kplus1_eps0p1.{pdf,png}  (1x3, all mechs)
  - plots_rq1/rq2_allocation_powerlaw_2kplus1_Exp_eps0p1.{pdf,png}  (single Exp panel)
"""
# _REPO_ROOT_BOOTSTRAP: ensure repo root is on sys.path so that
# 'from utils.*' / 'from preempt.*' imports resolve when this script
# is run from its own subfolder.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import os

# `gen_allocation_plot.py` is the canonical sibling module that exposes the
# shared TEMPLATES / MECHANISMS / plotting helpers used here. The earlier
# module name `generate_rq2_allocation_plot` was retired during cleanup.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_allocation_plot import (
    TEMPLATES,
    MECHANISMS,
    EPS_MIN,
    M,
    OUTPUT_DIR,
    compute_sensitivities_and_dw,
    plot_power_law,
    plot_power_law_single,
)
import gen_allocation_plot as base
from preempt.sanitizer import set_template_domains


def compute_allocations_2kplus1(eps, tmpl_info):
    """Allocations with B = (2k + 1) * eps, k = #roots."""
    allocs = {m[0]: {} for m in MECHANISMS}
    for tmpl in TEMPLATES:
        ti = tmpl_info[tmpl]
        k = len(ti["roots"])
        total_budget = (2 * k + 1) * eps
        set_template_domains(tmpl)
        for mech_key, _, alloc_fn in MECHANISMS:
            alloc = alloc_fn(ti["sens"], total_budget, EPS_MIN,
                             domain_widths=ti["dw"], num_candidates=M)
            allocs[mech_key][tmpl] = alloc
    return allocs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmpl_info = compute_sensitivities_and_dw()

    # Monkey-patch compute_allocations so the reused plot functions
    # use the (2k+1)*eps budget.
    original = base.compute_allocations
    base.compute_allocations = compute_allocations_2kplus1
    try:
        for eps in [0.1, 0.05, 0.5]:
            print(f"\n--- Power law (B=(2k+1)*eps, eps={eps}, all mechs) ---")
            plot_power_law(tmpl_info, eps=eps)
            _rename_outputs(
                src=f"rq2_allocation_powerlaw_eps{_suffix(eps)}",
                dst=f"rq2_allocation_powerlaw_2kplus1_eps{_suffix(eps)}",
            )

            print(f"\n--- Power law (B=(2k+1)*eps, eps={eps}, Exp only) ---")
            plot_power_law_single(tmpl_info, eps=eps, mech_idx=0)
            _rename_outputs(
                src=f"rq2_allocation_powerlaw_Exp_eps{_suffix(eps)}",
                dst=f"rq2_allocation_powerlaw_2kplus1_Exp_eps{_suffix(eps)}",
            )
    finally:
        base.compute_allocations = original


def _suffix(eps):
    return str(eps).replace(".", "p")


def _rename_outputs(src, dst):
    for ext in ["pdf", "png"]:
        s = os.path.join(OUTPUT_DIR, f"{src}.{ext}")
        d = os.path.join(OUTPUT_DIR, f"{dst}.{ext}")
        if os.path.exists(s):
            os.replace(s, d)
            print(f"  -> {d}")


if __name__ == "__main__":
    main()
