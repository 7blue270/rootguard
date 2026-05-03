#!/usr/bin/env python3
"""Generate RQ2 LaTeX tables with the labels referenced in the paper.

This is the canonical RQ2 table generator. It emits every label the
paper text cites (tab:recon_main, tab:rq2_per_template, the four
tab:rq2_exp_{A,B}_{uniform,informed} appendix tables, plus
tab:rq2_blap_agg and tab:rq2_stair_agg) into rq2_reconstruction/tables/.

Outputs (under rq2_reconstruction/tables/):

* recon_main.tex                — tab:recon_main          (§5 RQ2 main text)
* rq2_per_template.tex          — tab:rq2_per_template    (appendix per-template)
* rq2_exp_A_uniform.tex         — tab:rq2_exp_A_uniform   (appendix Exp Strategy A uniform)
* rq2_exp_A_informed.tex        — tab:rq2_exp_A_informed  (Exp A informed)
* rq2_exp_B_uniform.tex         — tab:rq2_exp_B_uniform   (Exp B uniform)
* rq2_exp_B_informed.tex        — tab:rq2_exp_B_informed  (Exp B informed; same content as recon_main)
* rq2_blap_agg.tex              — tab:rq2_blap_agg        (BoundedLaplace B informed)
* rq2_stair_agg.tex             — tab:rq2_stair_agg       (Staircase B informed)

Run from rq2_reconstruction/:

    python analysis/gen_paper_tables.py

Requires ``results_rq3_adversarial_v3/`` and ``allocations_v2/`` to be present
or symlinked at the cwd.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import (
    TEMPLATES, EPSILONS, Q_VALUES, METHOD_LABELS,
    load_result, compute_wmape, fmt,
)

# ε_r grid that the paper actually shows in its aggregate tables
EPS_PAPER = [0.01, 0.05, 0.1, 0.5, 1.0]

# Method names in the result-JSON namespace, per mechanism family
METHODS_BY_MECH = {
    "exponential":  ("vanilla",       "vanilla_roots",        "popabs"),
    "boundedlaplace": ("blap_all",      "blap_roots_uniform",   "blap_roots_opt"),
    "staircase":    ("staircase_all", "staircase_roots_uniform", "staircase_roots_opt"),
}


def aggregate(method, prior, strategy, eps, q):
    vals = []
    for t in TEMPLATES:
        r = load_result(t, method, prior, strategy, q, eps)
        m, _ = compute_wmape(r, t, strategy)
        if m is not None:
            vals.append(m)
    return float(np.mean(vals)) if vals else None


# ───────── eps × q × (All, Rts, Opt) layout (used by recon_main + 6 of the 8) ─────────
def render_eps_q_table(mech, prior, strategy, label, caption):
    m_all, m_rts, m_opt = METHODS_BY_MECH[mech]
    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append(r"\setlength{\tabcolsep}{2.5pt}")
    out.append(r"\small")
    out.append(r"\caption{" + caption + r"}")
    out.append(r"\label{" + label + r"}")
    out.append(r"\begin{tabular}{c|ccc|ccc|ccc|ccc}")
    out.append(r"\toprule")
    top = [""] + [r"\multicolumn{3}{c|}{$q=" + str(q) + r"$}" if i < len(Q_VALUES) - 1
                  else r"\multicolumn{3}{c}{$q=" + str(q) + r"$}"
                  for i, q in enumerate(Q_VALUES)]
    out.append(" & ".join(top) + r" \\")
    sub = [r"$\varepsilon_r$"]
    for _ in Q_VALUES:
        sub += ["All", "Rts", "Opt"]
    out.append(" & ".join(sub) + r" \\")
    out.append(r"\midrule")
    for eps in EPS_PAPER:
        cells = [str(eps)]
        for q in Q_VALUES:
            for method in (m_all, m_rts, m_opt):
                v = aggregate(method, prior, strategy, eps, q)
                cells.append(fmt(v))
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


# ───────── per-template (template-rows × q-All cols + Roots/Opt) ─────────
def render_per_template():
    """tab:rq2_per_template — Exp Strategy B informed, ε_r=0.1.

    Layout: rows = templates, cols = M-All q={1,4,8,16} | M-Roots (any q) | M-Opt (any q).
    M-Roots and M-Opt are q-invariant so a single column each suffices.
    """
    m_all, m_rts, m_opt = METHODS_BY_MECH["exponential"]
    eps = 0.1
    prior = "informed"
    strategy = "B"
    PAPER_TMPL_ORDER = ["ANEMIA", "FIB4", "AIP", "CONICITY", "VASCULAR", "TYG", "HOMA", "NLR"]

    out = []
    out.append(r"\begin{table}[h]")
    out.append(r"\centering")
    out.append(r"\setlength{\tabcolsep}{4pt}")
    out.append(r"\small")
    out.append(r"\caption{Per-template reconstruction wMAPE (\%) at $\varepsilon_r = 0.1$ "
               r"(Exponential, Strategy B, informed prior). $\mathcal{M}$-Roots and "
               r"$\mathcal{M}$-Opt are invariant in $q$, so a single column each suffices.}")
    out.append(r"\label{tab:rq2_per_template}")
    out.append(r"\begin{tabular}{l|cccc|cc}")
    out.append(r"\toprule")
    out.append(r"& \multicolumn{4}{c|}{$\mathcal{M}$-All} & $\mathcal{M}$-Roots & $\mathcal{M}$-Opt \\")
    out.append(r"\textbf{Template} & $q{=}1$ & $q{=}4$ & $q{=}8$ & $q{=}16$ & (any $q$) & (any $q$) \\")
    out.append(r"\midrule")
    for t in PAPER_TMPL_ORDER:
        cells = [t]
        # M-All across q
        for q in Q_VALUES:
            r = load_result(t, m_all, prior, strategy, q, eps)
            v, _ = compute_wmape(r, t, strategy)
            cells.append(fmt(v))
        # M-Roots, M-Opt at q=1 (invariant in q for cached methods)
        for method in (m_rts, m_opt):
            r = load_result(t, method, prior, strategy, 1, eps)
            v, _ = compute_wmape(r, t, strategy)
            cells.append(fmt(v))
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables")
    os.makedirs(out_dir, exist_ok=True)

    files = {}

    # tab:recon_main — main text §5 RQ2
    files["recon_main.tex"] = render_eps_q_table(
        "exponential", "informed", "B", "tab:recon_main",
        r"Aggregate reconstruction wMAPE (\%) across privacy levels (Exponential, "
        r"Strategy B, informed prior, mean across 8 templates). $\mathcal{M}$-All "
        r"decreases with $q$ at every $\varepsilon_r$; $\mathcal{M}$-Roots and "
        r"$\mathcal{M}$-Opt are invariant. Lower values mean the adversary "
        r"reconstructs more accurately (worse for privacy)."
    )

    # tab:rq2_per_template — appendix per-template
    files["rq2_per_template.tex"] = render_per_template()

    # tab:rq2_exp_{A,B}_{uniform,informed} — appendix per-strategy/prior aggregates
    for strategy in ("A", "B"):
        for prior in ("uniform", "informed"):
            label = f"tab:rq2_exp_{strategy}_{prior}"
            cap = (
                r"Reconstruction wMAPE (\%) under the Exponential mechanism, "
                rf"Strategy {strategy}, {prior} prior. Mean across 8 templates."
            )
            files[f"rq2_exp_{strategy}_{prior}.tex"] = render_eps_q_table(
                "exponential", prior, strategy, label, cap
            )

    # tab:rq2_blap_agg, tab:rq2_stair_agg — Strategy B, informed prior
    files["rq2_blap_agg.tex"] = render_eps_q_table(
        "boundedlaplace", "informed", "B", "tab:rq2_blap_agg",
        r"Reconstruction wMAPE (\%) under the Bounded Laplace mechanism, "
        r"Strategy B, informed prior. Mean across 8 templates."
    )
    files["rq2_stair_agg.tex"] = render_eps_q_table(
        "staircase", "informed", "B", "tab:rq2_stair_agg",
        r"Reconstruction wMAPE (\%) under the Staircase mechanism, "
        r"Strategy B, informed prior. Mean across 8 templates."
    )

    for name, content in files.items():
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
