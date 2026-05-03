#!/usr/bin/env python3
"""
Budget allocation plots:
  1. 1×6 bar chart grid (Exp/BLap/Stair at ε=0.1 and ε=1.0)
  2. 1×3 power law plot (log-log scatter of ε_i vs |h_i|·D_i per mechanism)
"""
# _REPO_ROOT_BOOTSTRAP: ensure repo root is on sys.path so that
# 'from utils.*' / 'from preempt.*' imports resolve when this script
# is run from its own subfolder.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gmean

from utils.utils import (
    compute_population_mean_sensitivities,
    optimal_budget_allocation_exact,
    optimal_budget_allocation_blap,
    optimal_budget_allocation_staircase,
)
from utils.med_domain.all_templates import (
    template_expressions, template_target_keys, template_nodes,
)
from preempt.sanitizer import set_template_domains, MEDICAL_DOMAINS

# ── Configuration ────────────────────────────────────────────────────────────

EPSILONS = [0.1, 1.0]
EPS_MIN = 0.001
M = 1000

TEMPLATES = ["ANEMIA", "FIB4", "AIP", "CONICITY",
             "VASCULAR", "TYG", "HOMA", "NLR"]
SKIP_NODES = {"anemia_class", "aip_class", "aip_risk", "ci_risk",
              "ppi_risk", "fib4_risk", "tyg_class", "homa_class", "nlr_class",
              "const"}
TEMPLATE_INFO = {
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

FORMULA_DEPS = {
    "mcv": ["hct", "rbc"], "mch": ["hb", "rbc"], "mchc": ["hb", "hct"],
    "non_hdl": ["tc", "hdl"], "ldl": ["tc", "hdl", "tg"], "aip": ["hdl", "tg"],
    "bmi": ["wt", "ht"], "wthr": ["waist", "ht"],
    "conicity": ["waist", "wt", "ht"],
    "fib4_prod": ["age", "ast"], "fib4_denom": ["plt", "alt"],
    "fib4": ["fib4_prod", "fib4_denom"],
    "homa_prod": ["glu", "ins"], "homa": ["homa_prod"],
    "nlr_sum": ["neu", "lym"], "nlr_diff": ["neu", "lym"], "nlr": ["neu", "lym"],
    "pp": ["sbp", "dbp"], "map": ["dbp", "pp"],
    "mbp": ["sbp", "dbp"], "ppi": ["pp", "sbp"],
    "tyg_prod": ["tg", "glu"], "tyg_half": ["tyg_prod"], "tyg": ["tyg_half"],
}

MECHANISMS = [
    ("Exp",   "Exp-Opt",   optimal_budget_allocation_exact),
    ("BLap",  "BLap-Opt",  optimal_budget_allocation_blap),
    ("Stair", "Stair-Opt", optimal_budget_allocation_staircase),
]

ROOT_LABELS = {
    "hb": "Hb", "hct": "Hct", "rbc": "RBC",
    "tc": "TC", "hdl": "HDL", "tg": "TG",
    "wt": "Wt", "ht": "Ht", "waist": "Waist",
    "sbp": "SBP", "dbp": "DBP",
    "age": "Age", "ast": "AST", "alt": "ALT", "plt": "PLT",
    "glu": "Glu", "ins": "Ins",
    "neu": "Neu", "lym": "Lym",
}

TEMPLATE_STYLE = {
    "ANEMIA":   {"color": "#0072B2", "marker": "o", "hatch": ""},
    "AIP":      {"color": "#E69F00", "marker": "s", "hatch": "//"},
    "CONICITY": {"color": "#009E73", "marker": "D", "hatch": "\\\\"},
    "FIB4":     {"color": "#CC79A7", "marker": "^", "hatch": "xx"},
    "HOMA":     {"color": "#D55E00", "marker": "v", "hatch": ".."},
    "NLR":      {"color": "#56B4E9", "marker": "<", "hatch": "++"},
    "TYG":      {"color": "#F0E442", "marker": ">", "hatch": "||"},
    "VASCULAR": {"color": "#000000", "marker": "P", "hatch": "--"},
}

LABEL_OFFSETS = {
    ("ANEMIA", "hct"): (6, -10), ("ANEMIA", "hb"): (6, 5),
    ("AIP", "hdl"): (-28, -10), ("AIP", "tg"): (6, 5),
    ("CONICITY", "waist"): (6, 5), ("CONICITY", "ht"): (6, 5),
    ("CONICITY", "wt"): (6, -10),
    ("FIB4", "alt"): (6, -10), ("FIB4", "plt"): (-22, -10),
    ("FIB4", "age"): (-28, 5), ("FIB4", "ast"): (6, 5),
    ("HOMA", "glu"): (6, -10), ("HOMA", "ins"): (6, -10),
    ("NLR", "neu"): (-26, -10), ("NLR", "lym"): (6, -10),
    ("TYG", "tg"): (-16, 6), ("TYG", "glu"): (-26, -10),
    ("VASCULAR", "dbp"): (6, -10), ("VASCULAR", "sbp"): (-26, 6),
}

OUTPUT_DIR = "plots_rq1"


def topo_sort(roots, derived):
    order = []
    visited = set(roots)
    def visit(node):
        if node in visited:
            return
        for dep in FORMULA_DEPS.get(node, []):
            visit(dep)
        visited.add(node)
        order.append(node)
    for d in derived:
        visit(d)
    return order


def load_holdout_means():
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "holdout_population_means.json",
    )
    with open(data_path) as f:
        return json.load(f)["per_template_means"]


def compute_sensitivities_and_dw():
    """Compute sensitivities and domain widths for all templates (mechanism-independent)."""
    HOLDOUT_MEANS = load_holdout_means()
    result = {}
    for tmpl in TEMPLATES:
        info = TEMPLATE_INFO[tmpl]
        root_nodes = info["roots"]
        derived_order = topo_sort(root_nodes, info["derived"])
        full_topo = root_nodes + derived_order
        target_node = template_target_keys[tmpl]

        set_template_domains(tmpl)
        sens = compute_population_mean_sensitivities(
            target_node, root_nodes, template_expressions, SKIP_NODES,
            full_topo, template_nodes, HOLDOUT_MEANS[tmpl])

        dw = {}
        for root in root_nodes:
            lo, hi = MEDICAL_DOMAINS.get(root, MEDICAL_DOMAINS["_default"])
            dw[root] = hi - lo

        n_nodes = len(root_nodes + derived_order)
        result[tmpl] = {"sens": sens, "dw": dw, "roots": root_nodes,
                        "n_nodes": n_nodes}
    return result


def compute_allocations(eps, tmpl_info):
    """Compute allocations for all (template, mechanism) at given epsilon."""
    allocs = {m[0]: {} for m in MECHANISMS}
    for tmpl in TEMPLATES:
        ti = tmpl_info[tmpl]
        total_budget = ti["n_nodes"] * eps
        set_template_domains(tmpl)
        for mech_key, _, alloc_fn in MECHANISMS:
            alloc = alloc_fn(ti["sens"], total_budget, EPS_MIN,
                             domain_widths=ti["dw"], num_candidates=M)
            allocs[mech_key][tmpl] = alloc
    return allocs


# ── 1×6 bar chart ────────────────────────────────────────────────────────────

def draw_bar_panel(ax, allocs, eps, mech_key, mech_label):
    labels = []
    fractions = []
    colors = []
    hatches = []
    tmpl_dividers = []
    pos = 0

    for tmpl in TEMPLATES:
        alloc = allocs[mech_key][tmpl]
        total = sum(alloc.values())
        roots = list(alloc.keys())
        if pos > 0:
            tmpl_dividers.append(pos - 0.5)
        for root in roots:
            frac = alloc[root] / total * 100
            labels.append(ROOT_LABELS.get(root, root))
            fractions.append(frac)
            colors.append(TEMPLATE_STYLE[tmpl]["color"])
            hatches.append(TEMPLATE_STYLE[tmpl]["hatch"])
            pos += 1

    y = np.arange(len(labels))
    bars = ax.barh(y, fractions, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        bar.set_edgecolor("#333333")
        bar.set_linewidth(0.3)
    for i, (bar, frac) in enumerate(zip(bars, fractions)):
        if frac > 3:
            ax.text(frac + 0.5, i, f"{frac:.0f}%",
                    va="center", fontsize=6.5, color="#333333")
    for div_y in tmpl_dividers:
        ax.axhline(div_y, color="#cccccc", linewidth=0.5, linestyle="-")
    pos = 0
    for tmpl in TEMPLATES:
        n_roots = len(allocs[mech_key][tmpl])
        mid = pos + (n_roots - 1) / 2
        ax.text(102, mid, tmpl, va="center", ha="left", fontsize=6.5,
                fontweight="bold", color=TEMPLATE_STYLE[tmpl]["color"],
                clip_on=False)
        pos += n_roots
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Share of budget (%)", fontsize=8)
    ax.set_title(f"{mech_label}\n$\\varepsilon = {eps}$",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='x', labelsize=7)


def plot_bar_charts(all_allocs):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, eps in enumerate(EPSILONS):
        for col, (mech_key, mech_label, _) in enumerate(MECHANISMS):
            draw_bar_panel(axes[row, col], all_allocs[eps], eps,
                           mech_key, mech_label)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        outpath = os.path.join(OUTPUT_DIR, f"rq2_allocation_by_mechanism.{ext}")
        fig.savefig(outpath, bbox_inches="tight", dpi=300 if ext == "pdf" else 150)
    print("Saved bar chart to plots_rq1/rq2_allocation_by_mechanism.{pdf,png}")
    plt.close(fig)


# ── 1×3 power law plot ───────────────────────────────────────────────────────

def plot_power_law(tmpl_info, eps=0.1):
    """1×3 log-log scatter: normalized ε_i vs |h_i|·D_i, one panel per mechanism."""

    allocs = compute_allocations(eps, tmpl_info)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for col, (mech_key, mech_label, _) in enumerate(MECHANISMS):
        ax = axes[col]

        # Build per-root data for this mechanism
        tmpl_roots = {}  # {tmpl: [{root, eps_i, sw, clamped}, ...]}
        for tmpl in TEMPLATES:
            alloc = allocs[mech_key][tmpl]
            ti = tmpl_info[tmpl]
            roots_data = []
            for root in ti["roots"]:
                s = abs(ti["sens"][root])
                w = ti["dw"][root]
                sw = s * w
                eps_i = alloc[root]
                clamped = eps_i < EPS_MIN * 1.5
                roots_data.append({
                    "root": root, "eps_i": eps_i, "sw": sw, "clamped": clamped,
                })
            tmpl_roots[tmpl] = roots_data

        # Normalize per template by geometric mean of unclamped roots
        points = []
        for tmpl, roots_data in tmpl_roots.items():
            unclamped = [r for r in roots_data
                         if not r["clamped"] and r["sw"] > 0]
            if len(unclamped) < 1:
                continue
            eps_gm = gmean([r["eps_i"] for r in unclamped])
            sw_gm = gmean([r["sw"] for r in unclamped])
            for r in roots_data:
                if r["sw"] <= 0:
                    continue
                points.append({
                    "tmpl": tmpl, "root": r["root"],
                    "x": r["sw"] / sw_gm,
                    "y": r["eps_i"] / eps_gm,
                    "clamped": r["clamped"],
                })

        # Plot points
        legend_handles = {}
        for pt in points:
            style = TEMPLATE_STYLE[pt["tmpl"]]
            fc = style["color"] if not pt["clamped"] else "none"
            handle = ax.plot(
                pt["x"], pt["y"],
                marker=style["marker"], color=style["color"],
                markerfacecolor=fc, markeredgecolor=style["color"],
                markeredgewidth=1.5, markersize=9, linestyle="none",
                zorder=5,
            )[0]
            if pt["tmpl"] not in legend_handles:
                legend_handles[pt["tmpl"]] = handle

            key = (pt["tmpl"], pt["root"])
            dx, dy = LABEL_OFFSETS.get(key, (6, 4))
            ax.annotate(pt["root"].upper(), (pt["x"], pt["y"]),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=7.5, color="#333333")

        # Fit power law on unclamped points
        unc = [p for p in points if not p["clamped"]]
        log_x = np.log10([p["x"] for p in unc])
        log_y = np.log10([p["y"] for p in unc])
        coeffs = np.polyfit(log_x, log_y, 1)

        x_range = np.logspace(
            np.log10(min(p["x"] for p in points)) - 0.15,
            np.log10(max(p["x"] for p in points)) + 0.15, 100)

        # Reference slope = 0.5 line
        ax.plot(x_range, x_range ** 0.5, "k--", linewidth=1.2, alpha=0.6,
                label=r"slope $= 0.5$", zorder=2)
        # Fitted line
        y_fit = 10 ** (coeffs[0] * np.log10(x_range) + coeffs[1])
        ax.plot(x_range, y_fit, color="#888888", linewidth=1.0, linestyle=":",
                alpha=0.7, label=f"fit: slope $= {coeffs[0]:.2f}$", zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Normalized $|h_i| \cdot D_i$", fontsize=12)
        if col == 0:
            ax.set_ylabel(r"Normalized $\epsilon_i$", fontsize=12)
        ax.set_title(f"{mech_label}", fontsize=13, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.25, zorder=0)

        # Legend: templates + fit lines
        tmpl_handles = [legend_handles[t] for t in TEMPLATES if t in legend_handles]
        tmpl_labels = [t for t in TEMPLATES if t in legend_handles]
        leg1 = ax.legend(tmpl_handles, tmpl_labels, fontsize=7,
                         loc="upper left", ncol=2, columnspacing=1.0,
                         handletextpad=0.4, borderpad=0.4)
        ax.add_artist(leg1)
        ax.legend(fontsize=8, loc="lower right")

        print(f"{mech_label}: fitted slope = {coeffs[0]:.4f}")

    fig.suptitle(
        f"Budget Allocation Power Law ($\\varepsilon = {eps}$)",
        fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    suffix = str(eps).replace(".", "p")
    for ext in ["pdf", "png"]:
        outpath = os.path.join(OUTPUT_DIR, f"rq2_allocation_powerlaw_eps{suffix}.{ext}")
        fig.savefig(outpath, bbox_inches="tight", dpi=300 if ext == "pdf" else 150)
    print(f"Saved power law plot to plots_rq1/rq2_allocation_powerlaw_eps{suffix}.{{pdf,png}}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tmpl_info = compute_sensitivities_and_dw()

    # 1×6 bar chart
    all_allocs = {}
    for eps in EPSILONS:
        print(f"\n--- epsilon = {eps} ---")
        all_allocs[eps] = compute_allocations(eps, tmpl_info)
    plot_bar_charts(all_allocs)

    # 1×3 power law at three epsilons (appendix)
    for pl_eps in [0.01, 0.1, 1.0]:
        print(f"\n--- Power law plot (eps={pl_eps}) ---")
        plot_power_law(tmpl_info, eps=pl_eps)

    # Main-text: single Staircase panel at eps=1.0
    print("\n--- Main-text: Staircase power law (eps=1.0) ---")
    plot_power_law_single(tmpl_info, eps=1.0, mech_idx=2)


def plot_power_law_single(tmpl_info, eps=1.0, mech_idx=2):
    """Single-panel power law plot for one mechanism (main text)."""

    allocs = compute_allocations(eps, tmpl_info)
    mech_key, mech_label, _ = MECHANISMS[mech_idx]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))

    tmpl_roots = {}
    for tmpl in TEMPLATES:
        alloc = allocs[mech_key][tmpl]
        ti = tmpl_info[tmpl]
        roots_data = []
        for root in ti["roots"]:
            s = abs(ti["sens"][root])
            w = ti["dw"][root]
            sw = s * w
            eps_i = alloc[root]
            clamped = eps_i < EPS_MIN * 1.5
            roots_data.append({
                "root": root, "eps_i": eps_i, "sw": sw, "clamped": clamped,
            })
        tmpl_roots[tmpl] = roots_data

    points = []
    for tmpl, roots_data in tmpl_roots.items():
        unclamped = [r for r in roots_data
                     if not r["clamped"] and r["sw"] > 0]
        if len(unclamped) < 1:
            continue
        eps_gm = gmean([r["eps_i"] for r in unclamped])
        sw_gm = gmean([r["sw"] for r in unclamped])
        for r in roots_data:
            if r["sw"] <= 0:
                continue
            points.append({
                "tmpl": tmpl, "root": r["root"],
                "x": r["sw"] / sw_gm,
                "y": r["eps_i"] / eps_gm,
                "clamped": r["clamped"],
            })

    legend_handles = {}
    for pt in points:
        style = TEMPLATE_STYLE[pt["tmpl"]]
        fc = style["color"] if not pt["clamped"] else "none"
        handle = ax.plot(
            pt["x"], pt["y"],
            marker=style["marker"], color=style["color"],
            markerfacecolor=fc, markeredgecolor=style["color"],
            markeredgewidth=1.5, markersize=10, linestyle="none",
            zorder=5,
        )[0]
        if pt["tmpl"] not in legend_handles:
            legend_handles[pt["tmpl"]] = handle

        key = (pt["tmpl"], pt["root"])
        dx, dy = LABEL_OFFSETS.get(key, (6, 4))
        ax.annotate(pt["root"].upper(), (pt["x"], pt["y"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=8.5, color="#333333")

    unc = [p for p in points if not p["clamped"]]
    log_x = np.log10([p["x"] for p in unc])
    log_y = np.log10([p["y"] for p in unc])
    coeffs = np.polyfit(log_x, log_y, 1)

    x_range = np.logspace(
        np.log10(min(p["x"] for p in points)) - 0.15,
        np.log10(max(p["x"] for p in points)) + 0.15, 100)

    ax.plot(x_range, x_range ** 0.5, "k--", linewidth=1.2, alpha=0.6,
            label=r"slope $= 0.5$", zorder=2)
    y_fit = 10 ** (coeffs[0] * np.log10(x_range) + coeffs[1])
    ax.plot(x_range, y_fit, color="#888888", linewidth=1.0, linestyle=":",
            alpha=0.7, label=f"fit: slope $= {coeffs[0]:.2f}$", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Normalized $|h_i| \cdot D_i$", fontsize=13)
    ax.set_ylabel(r"Normalized $\varepsilon_i$", fontsize=13)
    ax.set_title(f"{mech_label} ($\\varepsilon = {eps}$)",
                 fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.25, zorder=0)

    tmpl_handles = [legend_handles[t] for t in TEMPLATES if t in legend_handles]
    tmpl_labels = [t for t in TEMPLATES if t in legend_handles]
    leg1 = ax.legend(tmpl_handles, tmpl_labels, fontsize=8,
                     loc="upper left", ncol=2, columnspacing=1.0,
                     handletextpad=0.4, borderpad=0.4)
    ax.add_artist(leg1)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    suffix = str(eps).replace(".", "p")
    for ext in ["pdf", "png"]:
        outpath = os.path.join(
            OUTPUT_DIR, f"rq2_allocation_powerlaw_{mech_key}_eps{suffix}.{ext}")
        fig.savefig(outpath, bbox_inches="tight", dpi=300 if ext == "pdf" else 150)
    print(f"Saved single-panel plot to plots_rq1/"
          f"rq2_allocation_powerlaw_{mech_key}_eps{suffix}.{{pdf,png}}")
    print(f"  fitted slope = {coeffs[0]:.4f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
