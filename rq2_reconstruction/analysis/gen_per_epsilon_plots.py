#!/usr/bin/env python3
"""
RQ3 v2 per-epsilon reconstruction error plots.

For each epsilon (5 values), generates:
  - A 2-panel figure (Strategy A / Strategy B) of wMAPE vs q, three lines
    (M-All / M-Roots / M-Opt), aggregated (mean) across the 8 templates,
    Exponential mechanism, informed prior.

Also generates one grid figure per mechanism family showing all 5 epsilons
on one canvas (2 rows = strategies, 5 cols = epsilons).

Output: ../plots/per_epsilon/*.pdf/png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import (
    TEMPLATES, Q_VALUES, EPSILONS,
    load_result, compute_wmape,
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "plots", "per_epsilon",
)
PRIORS = ["uniform", "informed"]

MECH_FAMILIES = {
    "exp":   ("Exponential",     "vanilla",       "vanilla_roots",           "popabs"),
    "blap":  ("Bounded Laplace", "blap_all",      "blap_roots_uniform",      "blap_roots_opt"),
    "stair": ("Staircase",       "staircase_all", "staircase_roots_uniform", "staircase_roots_opt"),
}

COLORS = {"M-All": "#d62728", "M-Roots": "#2ca02c", "M-Opt": "#1f77b4"}
MARKERS = {"M-All": "o", "M-Roots": "s", "M-Opt": "^"}
STRAT_TITLES = {"A": "Strategy A (target r*)", "B": "Strategy B (all roots)"}


def aggregate_line(method, strat, eps, q_list, prior):
    """Mean wMAPE across templates for each q. Skips None values per-template."""
    ys = []
    for q in q_list:
        vals = []
        for t in TEMPLATES:
            r = load_result(t, method, prior, strat, q, eps)
            m, _ = compute_wmape(r, t, strat)
            if m is not None:
                vals.append(m)
        ys.append(float(np.mean(vals)) if vals else None)
    return ys


def draw_panel(ax, mech_tuple, strat, eps, prior, xlabel=True, ylabel=True):
    _mech_label, m_all, m_roots, m_opt = mech_tuple
    methods = [("M-All", m_all), ("M-Roots", m_roots), ("M-Opt", m_opt)]
    for label, method in methods:
        ys = aggregate_line(method, strat, eps, Q_VALUES, prior)
        if all(v is None for v in ys):
            continue
        ys_clean = [v if v is not None else float("nan") for v in ys]
        ax.plot(Q_VALUES, ys_clean, marker=MARKERS[label], markersize=6,
                linewidth=1.8, color=COLORS[label], label=label)
    ax.set_xticks(Q_VALUES)
    ax.grid(True, alpha=0.3)
    if xlabel:
        ax.set_xlabel("q")
    if ylabel:
        ax.set_ylabel("wMAPE (%)")


def make_per_epsilon(mech_key="exp", prior="informed"):
    """One figure per epsilon: 2 panels (Strategy A / B)."""
    mech_label, m_all, m_roots, m_opt = MECH_FAMILIES[mech_key]
    mech_tuple = (mech_label, m_all, m_roots, m_opt)
    os.makedirs(OUT_DIR, exist_ok=True)
    for eps in EPSILONS:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, strat in zip(axes, ["A", "B"]):
            draw_panel(ax, mech_tuple, strat, eps, prior)
            ax.set_title(STRAT_TITLES[strat])
        axes[0].legend(loc="best", fontsize=9)
        fig.suptitle(
            f"wMAPE vs q — {mech_label}, $\\varepsilon_r={eps}$, {prior} prior "
            f"(mean across 8 templates)", fontsize=11)
        fig.tight_layout()
        base = f"{OUT_DIR}/pe_{mech_key}_eps{eps}_{prior}"
        fig.savefig(f"{base}.pdf", bbox_inches="tight")
        fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  eps={eps}: {base}.pdf/.png")


def make_grid(mech_key="exp", prior="informed"):
    """2x5 grid: rows=strategies, cols=epsilons; aggregated across templates."""
    mech_label, m_all, m_roots, m_opt = MECH_FAMILIES[mech_key]
    mech_tuple = (mech_label, m_all, m_roots, m_opt)
    fig, axes = plt.subplots(2, len(EPSILONS), figsize=(16, 6.2), sharex=True)
    for col, eps in enumerate(EPSILONS):
        for row, strat in enumerate(["A", "B"]):
            ax = axes[row, col]
            draw_panel(ax, mech_tuple, strat, eps, prior,
                       xlabel=(row == 1), ylabel=(col == 0))
            if row == 0:
                ax.set_title(f"$\\varepsilon_r={eps}$", fontsize=11)
            if col == 0:
                ax.text(-0.30, 0.5, f"Strategy {strat}",
                        transform=ax.transAxes, fontsize=11,
                        va="center", rotation=90)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"Per-epsilon wMAPE vs q — {mech_label}, {prior} prior "
        f"(mean across 8 templates)", fontsize=12)
    fig.tight_layout()
    base = f"{OUT_DIR}/grid_{mech_key}_all_eps_{prior}"
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  grid: {base}.pdf/.png")


def make_eps_vs_wmape(mech_key="exp", prior="informed"):
    """Alternative view: wMAPE vs eps (x-axis), one line per q, per strategy.
    2 panels, 3 method subgroups each. Aggregated across templates."""
    mech_label, m_all, m_roots, m_opt = MECH_FAMILIES[mech_key]
    methods = [("M-All", m_all, "solid"),
               ("M-Roots", m_roots, "dashed"),
               ("M-Opt", m_opt, "dotted")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    cmap = plt.cm.plasma
    q_colors = {q: cmap(i / max(1, len(Q_VALUES) - 1))
                for i, q in enumerate(Q_VALUES)}

    for ax, strat in zip(axes, ["A", "B"]):
        for label, method, ls in methods:
            for q in Q_VALUES:
                ys = []
                for eps in EPSILONS:
                    vals = []
                    for t in TEMPLATES:
                        r = load_result(t, method, prior, strat, q, eps)
                        m, _ = compute_wmape(r, t, strat)
                        if m is not None:
                            vals.append(m)
                    ys.append(float(np.mean(vals)) if vals else float("nan"))
                ax.plot(EPSILONS, ys, linestyle=ls, color=q_colors[q],
                        linewidth=1.6, alpha=0.9, marker="o", markersize=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$\\varepsilon_r$")
        ax.set_title(STRAT_TITLES[strat])
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("wMAPE (%, log)")

    from matplotlib.lines import Line2D
    q_handles = [Line2D([0], [0], color=q_colors[q], lw=2, label=f"q={q}")
                 for q in Q_VALUES]
    style_handles = [Line2D([0], [0], color="black", linestyle=ls, lw=2,
                            label=label) for label, _, ls in methods]
    fig.legend(handles=q_handles, loc="upper right",
               bbox_to_anchor=(0.995, 0.965), fontsize=9, title="q")
    fig.legend(handles=style_handles, loc="upper right",
               bbox_to_anchor=(0.905, 0.965), fontsize=9, title="Method")
    fig.suptitle(
        f"wMAPE vs $\\varepsilon_r$ — {mech_label}, {prior} prior "
        f"(mean across 8 templates)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    base = f"{OUT_DIR}/wmape_vs_eps_{mech_key}_{prior}"
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wmape-vs-eps: {base}.pdf/.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for prior in PRIORS:
        print(f"\n=== Prior = {prior} ===")

        print("Per-epsilon plots (Exponential, Strategy A+B):")
        make_per_epsilon("exp", prior)

        print("Per-epsilon grids (all 5 eps at a glance, per mechanism):")
        for key in ["exp", "blap", "stair"]:
            make_grid(key, prior)

        print("wMAPE-vs-epsilon plots (log-log):")
        for key in ["exp", "blap", "stair"]:
            make_eps_vs_wmape(key, prior)

    print(f"\nAll plots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
