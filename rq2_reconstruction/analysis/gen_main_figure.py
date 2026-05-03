#!/usr/bin/env python3
"""
RQ3 v3 main figure: reconstruction wMAPE vs q.

Two panels (Strategy A, Strategy B).
Lines: M-All, M-Roots, M-Opt.
Fixed: Exponential mechanism, eps_r=0.1, informed prior.
Y: mean wMAPE across 8 templates. X: q in {1, 4, 8}.

Output: plots_rq3_v2/fig_mae_vs_q_eps0.1_informed.pdf
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import (
    TEMPLATES, Q_VALUES, METHOD_LABELS,
    load_result, compute_wmape,
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "plots", "main",
)
EPS = 0.1
PRIORS_TO_PLOT = ["uniform", "informed"]
METHODS_TO_PLOT = ["vanilla", "vanilla_roots", "popabs"]
COLORS = {"vanilla": "#d62728", "vanilla_roots": "#2ca02c", "popabs": "#1f77b4"}
MARKERS = {"vanilla": "o", "vanilla_roots": "s", "popabs": "^"}
STRATEGIES = ["A", "B"]
STRAT_TITLES = {"A": "Strategy A (single-root, r*)",
                "B": "Strategy B (all roots)"}


def aggregate_mae(method, strategy, q, prior):
    """Mean wMAPE across templates. Skip templates with missing data."""
    vals = []
    for t in TEMPLATES:
        r = load_result(t, method, prior, strategy, q, EPS)
        m, _ = compute_wmape(r, t, strategy)
        if m is not None:
            vals.append(m)
    return float(np.mean(vals)) if vals else None


def make_figure(prior):
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, strat in zip(axes, STRATEGIES):
        for method in METHODS_TO_PLOT:
            ys = [aggregate_mae(method, strat, q, prior) for q in Q_VALUES]
            ax.plot(Q_VALUES, ys, marker=MARKERS[method], markersize=9,
                    linewidth=2.4, color=COLORS[method],
                    label=METHOD_LABELS[method])
        ax.set_xlabel("q (adversarial queries)")
        ax.set_xticks(Q_VALUES)
        ax.set_title(STRAT_TITLES[strat])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Reconstruction wMAPE (%)")
    axes[0].legend(loc="best")
    fig.suptitle(
        f"Reconstruction wMAPE vs q — Exponential, $\\varepsilon_r={EPS}$, {prior} prior "
        f"(mean across templates)", fontsize=16)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"fig_mae_vs_q_eps{EPS}_{prior}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} (+png)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for prior in PRIORS_TO_PLOT:
        make_figure(prior)


if __name__ == "__main__":
    main()
