"""Emit PDF plots from aggregates.json."""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parent.parent
DEFAULT_AGG = REPO / "aggregates.json"
DEFAULT_PLOTS = REPO / "plots"

CONFIGS = ("all", "roots", "opt")
CONFIG_LABEL = {"all": "M-All", "roots": "M-Roots", "opt": "M-Opt"}
B_SETTINGS = ("k+1", "2k+1", "3k+1")
TEMPLATES = ("HOMA", "ANEMIA", "FIB4", "AIP", "CONICITY", "VASCULAR", "TYG", "NLR")
EPS_LIST = (0.01, 0.05, 0.1)

# Style adapted from gen_rq1_new_plots.py
CONFIG_COLORS = {"all": "#0072B2", "roots": "#E69F00", "opt": "#009E73"}
CONFIG_MARKERS = {"all": "o", "roots": "s", "opt": "D"}
CONFIG_LW = {"all": 2.0, "roots": 2.5, "opt": 3.0}
CONFIG_MS = {"all": 8, "roots": 9, "opt": 10}
CONFIG_LS = {"all": "-", "roots": "--", "opt": "-"}


def cell(cells, **kw):
    for c in cells:
        if all(c.get(k) == v for k, v in kw.items()):
            return c
    return None


def plot_double_asymmetry(cells, eps: float, out_path: Path):
    """8-panel grid (one per template), x=B_setting, y=wMAPE, three lines per panel."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=False)
    x = np.arange(len(B_SETTINGS))
    for i, tmpl in enumerate(TEMPLATES):
        ax = axes[i // 4, i % 4]
        for cfg in CONFIGS:
            ys, errs = [], []
            for B in B_SETTINGS:
                c = cell(cells, template=tmpl, config=cfg, B_setting=B, eps=eps)
                ys.append(c["wmape_mean"] if c else float("nan"))
                errs.append(c["wmape_se"] if c else float("nan"))
            ax.errorbar(
                x, ys, yerr=errs,
                marker=CONFIG_MARKERS[cfg], color=CONFIG_COLORS[cfg],
                linestyle=CONFIG_LS[cfg], linewidth=CONFIG_LW[cfg] * 0.7,
                markersize=CONFIG_MS[cfg] * 0.8,
                label=CONFIG_LABEL[cfg], capsize=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(B_SETTINGS, fontsize=10)
        ax.set_title(tmpl, fontsize=14, fontweight="bold")
        ax.set_xlabel("Adversarial budget", fontsize=11)
        if i % 4 == 0:
            ax.set_ylabel("wMAPE (%)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=11, loc="upper right")
    fig.suptitle(f"Agent-eval wMAPE vs adversarial budget at $\\varepsilon={eps}$",
                 fontsize=15, fontweight="bold", y=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_privacy_utility(cells, B_setting: str, out_path: Path):
    """8-panel log-log: x=ε, y=wMAPE, three lines per panel, fixed B_setting."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=False)
    for i, tmpl in enumerate(TEMPLATES):
        ax = axes[i // 4, i % 4]
        for cfg in CONFIGS:
            xs, ys, errs = [], [], []
            for eps in EPS_LIST:
                c = cell(cells, template=tmpl, config=cfg, B_setting=B_setting, eps=eps)
                if c:
                    xs.append(eps)
                    ys.append(c["wmape_mean"])
                    errs.append(c["wmape_se"])
            ax.errorbar(
                xs, ys, yerr=errs,
                marker=CONFIG_MARKERS[cfg], color=CONFIG_COLORS[cfg],
                linestyle=CONFIG_LS[cfg], linewidth=CONFIG_LW[cfg] * 0.7,
                markersize=CONFIG_MS[cfg] * 0.8,
                label=CONFIG_LABEL[cfg], capsize=3,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(EPS_LIST)
        ax.set_xticklabels([str(e) for e in EPS_LIST], fontsize=9)
        ax.set_title(tmpl, fontsize=14, fontweight="bold")
        ax.set_xlabel(r"$\varepsilon$", fontsize=12)
        if i % 4 == 0:
            ax.set_ylabel("wMAPE (%)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=11, loc="upper right")
    fig.suptitle(f"Privacy-utility tradeoff (B={B_setting})",
                 fontsize=15, fontweight="bold", y=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_rce_grid(cells, eps: float, out_path: Path):
    """8-panel grid, x=B_setting, y=RCE."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=False)
    x = np.arange(len(B_SETTINGS))
    for i, tmpl in enumerate(TEMPLATES):
        ax = axes[i // 4, i % 4]
        for cfg in CONFIGS:
            ys, errs = [], []
            for B in B_SETTINGS:
                c = cell(cells, template=tmpl, config=cfg, B_setting=B, eps=eps)
                ys.append(c["rce_mean"] if c else float("nan"))
                errs.append(c["rce_se"] if c else float("nan"))
            ax.errorbar(x, ys, yerr=errs,
                        marker=CONFIG_MARKERS[cfg], color=CONFIG_COLORS[cfg],
                        linestyle=CONFIG_LS[cfg], linewidth=CONFIG_LW[cfg] * 0.7,
                        markersize=CONFIG_MS[cfg] * 0.8,
                        label=CONFIG_LABEL[cfg], capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(B_SETTINGS, fontsize=10)
        ax.set_title(tmpl, fontsize=14, fontweight="bold")
        ax.set_xlabel("Adversarial budget", fontsize=11)
        if i % 4 == 0:
            ax.set_ylabel("RCE (%)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=11, loc="upper right")
    fig.suptitle(f"Agent-eval Risk Class Error vs B at $\\varepsilon={eps}$",
                 fontsize=15, fontweight="bold", y=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_config_comparison(cells, out_path: Path):
    """Single figure summary: aggregate wMAPE across templates per (config, B) at each ε."""
    fig, axes = plt.subplots(1, len(EPS_LIST), figsize=(18, 5), sharey=True)
    x = np.arange(len(B_SETTINGS))
    width = 0.27
    for i, eps in enumerate(EPS_LIST):
        ax = axes[i]
        for j, cfg in enumerate(CONFIGS):
            means = []
            for B in B_SETTINGS:
                vals = []
                for tmpl in TEMPLATES:
                    c = cell(cells, template=tmpl, config=cfg, B_setting=B, eps=eps)
                    if c:
                        vals.append(c["wmape_mean"])
                means.append(float(np.mean(vals)) if vals else float("nan"))
            ax.bar(x + (j - 1) * width, means, width, color=CONFIG_COLORS[cfg],
                   label=CONFIG_LABEL[cfg])
        ax.set_xticks(x)
        ax.set_xticklabels(B_SETTINGS, fontsize=11)
        ax.set_title(f"$\\varepsilon = {eps}$", fontsize=13, fontweight="bold")
        ax.set_xlabel("Adversarial budget", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.25, axis="y")
        if i == 0:
            ax.set_ylabel("wMAPE (%) averaged over 8 templates", fontsize=12)
            ax.legend(fontsize=11, loc="upper right")
    fig.suptitle("Cross-template aggregate wMAPE", fontsize=14, fontweight="bold")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default=str(DEFAULT_AGG))
    ap.add_argument("--out-dir", default=str(DEFAULT_PLOTS))
    args = ap.parse_args()
    with open(args.agg) as f:
        agg = json.load(f)
    cells = agg["cells"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_double_asymmetry(cells, 0.1, out_dir / "double_asymmetry_eps0.1.pdf")
    plot_double_asymmetry(cells, 0.05, out_dir / "double_asymmetry_eps0.05.pdf")
    plot_double_asymmetry(cells, 0.01, out_dir / "double_asymmetry_eps0.01.pdf")
    plot_privacy_utility(cells, "3k+1", out_dir / "privacy_utility.pdf")
    plot_rce_grid(cells, 0.1, out_dir / "rce_grid.pdf")
    plot_config_comparison(cells, out_dir / "config_comparison.pdf")


if __name__ == "__main__":
    main()
