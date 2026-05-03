#!/usr/bin/env python3
"""
RQ3 v2 per-template reconstruction error plots.

Generates:
  - 8 per-template plots (one per template): 2 panels (Strategy A, B),
    3 lines (M-All, M-Roots, M-Opt), Exponential mechanism, informed prior,
    eps_r = 0.1.
  - 1 grid figure: 4x2 layout showing all 8 templates (Strategy B only).
  - 3 multi-epsilon per-mechanism grids: for each mechanism family, 8-panel
    figure showing per-template lines across epsilons at Strategy B + informed.

Output: ../plots/per_template/*.pdf/png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import (
    TEMPLATES, Q_VALUES, EPSILONS, MECHANISMS, METHOD_LABELS,
    load_result, compute_wmape,
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "plots", "per_template",
)

EPS_FIXED = 0.1
PRIORS = ["uniform", "informed"]
MECH_FAMILIES = {
    "exp":   ("Exponential",     "vanilla",       "vanilla_roots",           "popabs"),
    "blap":  ("Bounded Laplace", "blap_all",      "blap_roots_uniform",      "blap_roots_opt"),
    "stair": ("Staircase",       "staircase_all", "staircase_roots_uniform", "staircase_roots_opt"),
}

COLORS = {"M-All": "#d62728", "M-Roots": "#2ca02c", "M-Opt": "#1f77b4"}
MARKERS = {"M-All": "o", "M-Roots": "s", "M-Opt": "^"}
STRAT_TITLES = {"A": "Strategy A (target r*)", "B": "Strategy B (all roots)"}


def line_for(tmpl, method, label, strat, q_list, eps, prior):
    ys = []
    for q in q_list:
        r = load_result(tmpl, method, prior, strat, q, eps)
        m, _ = compute_wmape(r, tmpl, strat)
        ys.append(m)
    return ys


def draw_panel(ax, tmpl, mech_tuple, strat, eps, prior, xlabel=True, ylabel=True):
    _mech_label, m_all, m_roots, m_opt = mech_tuple
    methods = [("M-All", m_all), ("M-Roots", m_roots), ("M-Opt", m_opt)]
    any_data = False
    for label, method in methods:
        ys = line_for(tmpl, method, label, strat, Q_VALUES, eps, prior)
        if all(v is None for v in ys):
            continue
        any_data = True
        ys_clean = [v if v is not None else float("nan") for v in ys]
        ax.plot(Q_VALUES, ys_clean, marker=MARKERS[label], markersize=6,
                linewidth=1.8, color=COLORS[label], label=label)
    ax.set_xticks(Q_VALUES)
    ax.grid(True, alpha=0.3)
    if xlabel:
        ax.set_xlabel("q")
    if ylabel:
        ax.set_ylabel("wMAPE (%)")
    return any_data


def make_per_template(mech_key="exp", prior="informed"):
    """One figure per template: 2 panels (A, B)."""
    mech_label, m_all, m_roots, m_opt = MECH_FAMILIES[mech_key]
    mech_tuple = (mech_label, m_all, m_roots, m_opt)
    os.makedirs(OUT_DIR, exist_ok=True)
    for tmpl in TEMPLATES:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, strat in zip(axes, ["A", "B"]):
            draw_panel(ax, tmpl, mech_tuple, strat, EPS_FIXED, prior)
            ax.set_title(f"{STRAT_TITLES[strat]}")
        axes[0].legend(loc="best", fontsize=9)
        fig.suptitle(
            f"{tmpl} — reconstruction wMAPE vs q ({mech_label}, "
            f"$\\varepsilon_r={EPS_FIXED}$, {prior} prior)", fontsize=11)
        fig.tight_layout()
        base = f"{OUT_DIR}/pt_{mech_key}_{tmpl}_eps{EPS_FIXED}_{prior}"
        fig.savefig(f"{base}.pdf", bbox_inches="tight")
        fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {tmpl}: {base}.pdf/.png")


def make_grid(mech_key="exp", strategy="B", prior="informed"):
    """4x2 grid: all 8 templates at fixed (mechanism, strategy, eps, prior)."""
    mech_label, m_all, m_roots, m_opt = MECH_FAMILIES[mech_key]
    mech_tuple = (mech_label, m_all, m_roots, m_opt)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for ax, tmpl in zip(axes.flatten(), TEMPLATES):
        draw_panel(ax, tmpl, mech_tuple, strategy, EPS_FIXED, prior,
                   xlabel=False, ylabel=False)
        ax.set_title(tmpl, fontsize=10)
    axes[0, 0].legend(loc="best", fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("q")
    for ax in axes[:, 0]:
        ax.set_ylabel("wMAPE (%)")
    fig.suptitle(
        f"Per-template reconstruction wMAPE vs q — {mech_label}, "
        f"Strategy {strategy}, $\\varepsilon_r={EPS_FIXED}$, {prior} prior",
        fontsize=12)
    fig.tight_layout()
    base = f"{OUT_DIR}/grid_{mech_key}_{strategy}_eps{EPS_FIXED}_{prior}"
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  grid: {base}.pdf/.png")


def make_eps_sweep(mech_key="exp", strategy="B", prior="informed"):
    """Per-template 2x4 grid: rows=strategies? no — here we sweep eps.
    4 subplots per row, 2 rows. For one mechanism, one strategy:
    each subplot = one template, showing 5 lines (one per eps) of wMAPE vs q
    for M-All (solid), M-Roots (dashed), M-Opt (dotted)."""
    mech_label, m_all, m_roots, m_opt = MECH_FAMILIES[mech_key]
    methods = [("M-All", m_all, "solid"),
               ("M-Roots", m_roots, "dashed"),
               ("M-Opt", m_opt, "dotted")]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    cmap = plt.cm.viridis
    eps_colors = {e: cmap(i / (len(EPSILONS) - 1)) for i, e in enumerate(EPSILONS)}

    for ax, tmpl in zip(axes.flatten(), TEMPLATES):
        for eps in EPSILONS:
            for label, method, ls in methods:
                ys = line_for(tmpl, method, label, strategy, Q_VALUES, eps, prior)
                if all(v is None for v in ys):
                    continue
                ys_c = [v if v is not None else float("nan") for v in ys]
                ax.plot(Q_VALUES, ys_c, linestyle=ls,
                        color=eps_colors[eps], linewidth=1.3, alpha=0.85)
        ax.set_title(tmpl, fontsize=10)
        ax.set_xticks(Q_VALUES)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    for ax in axes[1, :]:
        ax.set_xlabel("q")
    for ax in axes[:, 0]:
        ax.set_ylabel("wMAPE (%, log)")

    # Two legends: eps (color) and method (line style)
    from matplotlib.lines import Line2D
    eps_handles = [Line2D([0], [0], color=eps_colors[e], lw=2,
                          label=f"$\\varepsilon_r={e}$") for e in EPSILONS]
    style_handles = [Line2D([0], [0], color="black", linestyle=ls, lw=2,
                            label=label)
                     for label, _, ls in methods]
    fig.legend(handles=eps_handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.98), fontsize=9, title="$\\varepsilon_r$")
    fig.legend(handles=style_handles, loc="upper right",
               bbox_to_anchor=(0.88, 0.98), fontsize=9, title="Method")
    fig.suptitle(
        f"Per-template wMAPE vs q across all $\\varepsilon_r$ — {mech_label}, "
        f"Strategy {strategy}, {prior} prior",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    base = f"{OUT_DIR}/grid_{mech_key}_{strategy}_alleps_{prior}"
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  all-eps grid: {base}.pdf/.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for prior in PRIORS:
        print(f"\n=== Prior = {prior} ===")

        print("Per-template plots (one per template, Exponential, Strategy A+B):")
        make_per_template("exp", prior)
        print("  done.")

        for strat in ["B", "A"]:
            print(f"Grids ({strat}) per mechanism:")
            for key in ["exp", "blap", "stair"]:
                make_grid(key, strat, prior)

        for strat in ["B", "A"]:
            print(f"Epsilon sweep grids ({strat}, log-y, all eps overlaid):")
            for key in ["exp", "blap", "stair"]:
                make_eps_sweep(key, strat, prior)

    print(f"\nAll plots written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
