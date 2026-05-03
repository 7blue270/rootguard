"""Emit LaTeX tables from aggregates.json."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_AGG = REPO / "aggregates.json"
DEFAULT_TABLES = REPO / "tables"

CONFIGS = ("all", "roots", "opt")
CONFIG_LABEL = {"all": "M-All", "roots": "M-Roots", "opt": "M-Opt"}
B_SETTINGS = ("k+1", "2k+1", "3k+1")
TEMPLATES = ("HOMA", "ANEMIA", "FIB4", "AIP", "CONICITY", "VASCULAR", "TYG", "NLR")
PAPER_TEMPLATE_ORDER = ("HOMA", "AIP", "FIB4", "NLR", "TYG", "ANEMIA", "CONICITY", "VASCULAR")
EPS_LIST = (0.01, 0.05, 0.1)
PMSTD = "\\providecommand{\\pmstd}[1]{{\\scriptsize $\\pm$#1}}"


def fmt(v: float) -> str:
    if v != v:
        return "--"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


def fmt_se(v: float) -> str:
    if v != v:
        return "--"
    if v >= 10:
        return f"{v:.0f}"
    return f"{v:.1f}"


def fmt_cell(mean: float, se: float, bold: bool = False) -> str:
    s = f"{fmt(mean)}\\pmstd{{{fmt_se(se)}}}"
    return f"\\textbf{{{s}}}" if bold else s


def cell(cells, **kw):
    """Find one cell matching all kw key-value pairs (None means any)."""
    for c in cells:
        if all(c.get(k) == v for k, v in kw.items()):
            return c
    return None


def gen_double_asymmetry(cells, eps: float) -> str:
    out = []
    out.append(PMSTD)
    out.append("")
    out.append("\\begin{table*}[t]")
    out.append("\\centering")
    out.append("\\setlength{\\tabcolsep}{2.8pt}")
    out.append("\\small")
    out.append(f"\\caption{{Aggregate wMAPE (\\%) at $\\varepsilon={eps}$ across the three "
               "agent-eval configurations and three turn budgets, with bootstrap "
               "standard errors. Per-template breakdown.}}")
    out.append(f"\\label{{tab:agent_double_asymmetry_eps{eps}}}")
    out.append("\\begin{tabular}{l|ccc|ccc|ccc}")
    out.append("\\toprule")
    out.append("& \\multicolumn{3}{c|}{\\textbf{M-All}} & "
               "\\multicolumn{3}{c|}{\\textbf{M-Roots}} & "
               "\\multicolumn{3}{c}{\\textbf{M-Opt}} \\\\")
    out.append("Template & " + " & ".join([f"$B={B}$" for B in B_SETTINGS] * 3) + " \\\\")
    out.append("\\midrule")
    for tmpl in TEMPLATES:
        row = [tmpl]
        # Find best across this row
        means = []
        for cfg in CONFIGS:
            for B in B_SETTINGS:
                c = cell(cells, template=tmpl, config=cfg, B_setting=B, eps=eps)
                means.append(c["wmape_mean"] if c else float("nan"))
        best_idx = min(range(len(means)),
                       key=lambda i: means[i] if means[i] == means[i] else float("inf"))
        idx = 0
        for cfg in CONFIGS:
            for B in B_SETTINGS:
                c = cell(cells, template=tmpl, config=cfg, B_setting=B, eps=eps)
                if c is None:
                    row.append("--")
                else:
                    row.append(fmt_cell(c["wmape_mean"], c["wmape_se"], idx == best_idx))
                idx += 1
        out.append(" & ".join(row) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table*}")
    return "\n".join(out) + "\n"


def gen_aggregate_wmape(cells) -> str:
    """Cross-template aggregate (mean of per-template wMAPE) per (config, B, ε)."""
    import numpy as np
    out = []
    out.append(PMSTD)
    out.append("")
    out.append("\\begin{table*}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\caption{Aggregate wMAPE (\\%) averaged across 8 templates, per "
               "(config, $B$, $\\varepsilon$). Bootstrap SEs (composed across "
               "templates) shown.}")
    out.append("\\label{tab:agent_aggregate_wmape}")
    out.append("\\begin{tabular}{c|ccc|ccc|ccc}")
    out.append("\\toprule")
    out.append("& \\multicolumn{3}{c|}{\\textbf{M-All}} & "
               "\\multicolumn{3}{c|}{\\textbf{M-Roots}} & "
               "\\multicolumn{3}{c}{\\textbf{M-Opt}} \\\\")
    out.append("$\\varepsilon$ & " + " & ".join([f"$B={B}$" for B in B_SETTINGS] * 3) + " \\\\")
    out.append("\\midrule")
    for eps in EPS_LIST:
        row = [str(eps)]
        for cfg in CONFIGS:
            for B in B_SETTINGS:
                tmpl_means = [cell(cells, template=t, config=cfg, B_setting=B, eps=eps)
                              for t in TEMPLATES]
                means = [c["wmape_mean"] for c in tmpl_means if c]
                ses = [c["wmape_se"] for c in tmpl_means if c]
                if not means:
                    row.append("--")
                    continue
                m = float(np.mean(means))
                s = float(np.sqrt(np.sum(np.array(ses) ** 2)) / len(ses))
                row.append(f"{fmt(m)}\\pmstd{{{fmt_se(s)}}}")
        out.append(" & ".join(row) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table*}")
    return "\n".join(out) + "\n"


def gen_rce(cells) -> str:
    """Per-template RCE (\\%) at ε=0.1 across (config, B). Legacy label retained for backward compat."""
    out = []
    out.append(PMSTD)
    out.append("")
    out.append("\\begin{table*}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\caption{Risk Class Error (\\%) at $\\varepsilon=0.1$ per template "
               "and (config, $B$). Bootstrap SEs.}")
    out.append("\\label{tab:agent_rce_eps0.1}")
    out.append("\\begin{tabular}{l|ccc|ccc|ccc}")
    out.append("\\toprule")
    out.append("& \\multicolumn{3}{c|}{\\textbf{M-All}} & "
               "\\multicolumn{3}{c|}{\\textbf{M-Roots}} & "
               "\\multicolumn{3}{c}{\\textbf{M-Opt}} \\\\")
    out.append("Template & " + " & ".join([f"$B={B}$" for B in B_SETTINGS] * 3) + " \\\\")
    out.append("\\midrule")
    for tmpl in TEMPLATES:
        row = [tmpl]
        means = []
        for cfg in CONFIGS:
            for B in B_SETTINGS:
                c = cell(cells, template=tmpl, config=cfg, B_setting=B, eps=0.1)
                means.append(c["rce_mean"] if c else float("nan"))
        best_idx = min(range(len(means)),
                       key=lambda i: means[i] if means[i] == means[i] else float("inf"))
        idx = 0
        for cfg in CONFIGS:
            for B in B_SETTINGS:
                c = cell(cells, template=tmpl, config=cfg, B_setting=B, eps=0.1)
                if c is None:
                    row.append("--")
                else:
                    row.append(fmt_cell(c["rce_mean"], c["rce_se"], idx == best_idx))
                idx += 1
        out.append(" & ".join(row) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table*}")
    return "\n".join(out) + "\n"


def gen_rce_appendix(cells) -> str:
    """Per-template agent-eval RCE at ε=0.1 across (config × B). Appendix table.

    Same shape as gen_rce() but uses paper row order and the new label
    ``tab:agent_rce_eps0.1_app`` referenced from the paper appendix.
    """
    out = []
    out.append(PMSTD)
    out.append("")
    out.append("\\begin{table*}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\setlength{\\tabcolsep}{3pt}")
    out.append("\\caption{Per-template agent-eval Risk Class Error (\\%) at "
               "$\\varepsilon = 0.1$, with bootstrap SE over patients ($n = 100$ "
               "per template-cell). Best per row in \\textbf{bold}.}")
    out.append("\\label{tab:agent_rce_eps0.1_app}")
    out.append("\\begin{tabular}{l|ccc|ccc|ccc}")
    out.append("\\toprule")
    out.append("& \\multicolumn{3}{c|}{\\textbf{$\\mathcal{M}$-All}} & "
               "\\multicolumn{3}{c|}{\\textbf{$\\mathcal{M}$-Roots}} & "
               "\\multicolumn{3}{c}{\\textbf{$\\mathcal{M}$-Opt}} \\\\")
    out.append("Template & " + " & ".join([f"$B{{=}}{B}$" for B in B_SETTINGS] * 3) + " \\\\")
    out.append("\\midrule")
    for tmpl in PAPER_TEMPLATE_ORDER:
        row = [tmpl]
        means = []
        for cfg in CONFIGS:
            for B in B_SETTINGS:
                c = cell(cells, template=tmpl, config=cfg, B_setting=B, eps=0.1)
                means.append(c["rce_mean"] if c else float("nan"))
        best_idx = min(range(len(means)),
                       key=lambda i: means[i] if means[i] == means[i] else float("inf"))
        idx = 0
        for cfg in CONFIGS:
            for B in B_SETTINGS:
                c = cell(cells, template=tmpl, config=cfg, B_setting=B, eps=0.1)
                if c is None:
                    row.append("--")
                else:
                    row.append(fmt_cell(c["rce_mean"], c["rce_se"], idx == best_idx))
                idx += 1
        out.append(" & ".join(row) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table*}")
    return "\n".join(out) + "\n"


def gen_rce_body(cells) -> str:
    """Body RQ4 RCE: per-template at ε=0.1 over {2k+1, 3k+1} × {M-All, M-Opt}."""
    body_B = ("2k+1", "3k+1")
    body_cfg = ("all", "opt")
    out = []
    out.append(PMSTD)
    out.append("")
    out.append("\\begin{table}[h]")
    out.append("\\centering")
    out.append("\\setlength{\\tabcolsep}{2pt}")
    out.append("\\caption{Agent-eval Risk Class Error (\\%) per template at "
               "$\\varepsilon = 0.1$ under deployment (RQ4), with bootstrap SE over "
               "patients ($n = 100$). Showing the two larger adversarial budgets; "
               "the $B = (k{+}1)\\varepsilon$ column and the $\\mathcal{M}$-Roots "
               "configuration are in App.~\\ref{app:agent-eval-rce} "
               "(Table~\\ref{tab:agent_rce_eps0.1_app}). Lower is better.}")
    out.append("\\label{tab:rce_per_template}")
    out.append("\\begin{tabular}{l|cc|cc}")
    out.append("\\toprule")
    out.append("& \\multicolumn{2}{c|}{\\textbf{$\\mathcal{M}$-All}} & "
               "\\multicolumn{2}{c}{\\textbf{$\\mathcal{M}$-Opt}} \\\\")
    out.append("Template & $2k{+}1$ & $3k{+}1$ & $2k{+}1$ & $3k{+}1$ \\\\")
    out.append("\\midrule")
    for tmpl in PAPER_TEMPLATE_ORDER:
        row = [tmpl]
        means = []
        for cfg in body_cfg:
            for B in body_B:
                c = cell(cells, template=tmpl, config=cfg, B_setting=B, eps=0.1)
                means.append(c["rce_mean"] if c else float("nan"))
        best_idx = min(range(len(means)),
                       key=lambda i: means[i] if means[i] == means[i] else float("inf"))
        idx = 0
        for cfg in body_cfg:
            for B in body_B:
                c = cell(cells, template=tmpl, config=cfg, B_setting=B, eps=0.1)
                if c is None:
                    row.append("--")
                else:
                    row.append(fmt_cell(c["rce_mean"], c["rce_se"], idx == best_idx))
                idx += 1
        out.append(" & ".join(row) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def gen_aggregate_rce(cells) -> str:
    """Cross-template aggregate (mean of per-template RCE) per (config, B, ε).

    Mirrors gen_aggregate_wmape's structure; reports mean and aggregate SE
    composed across templates as sqrt(sum SE_i^2)/k.
    """
    import numpy as np
    out = []
    out.append(PMSTD)
    out.append("")
    out.append("\\begin{table*}[t]")
    out.append("\\centering")
    out.append("\\small")
    out.append("\\caption{Aggregate Risk Class Error (\\%) averaged across 8 "
               "templates, per (config, $B$, $\\varepsilon$). Bootstrap SEs "
               "(composed across templates) shown.}")
    out.append("\\label{tab:agent_aggregate_rce}")
    out.append("\\begin{tabular}{c|ccc|ccc|ccc}")
    out.append("\\toprule")
    out.append("& \\multicolumn{3}{c|}{\\textbf{$\\mathcal{M}$-All}} & "
               "\\multicolumn{3}{c|}{\\textbf{$\\mathcal{M}$-Roots}} & "
               "\\multicolumn{3}{c}{\\textbf{$\\mathcal{M}$-Opt}} \\\\")
    out.append("$\\varepsilon$ & " + " & ".join([f"$B{{=}}{B}$" for B in B_SETTINGS] * 3) + " \\\\")
    out.append("\\midrule")
    for eps in EPS_LIST:
        row = [str(eps)]
        for cfg in CONFIGS:
            for B in B_SETTINGS:
                tmpl_cells = [cell(cells, template=t, config=cfg, B_setting=B, eps=eps)
                              for t in TEMPLATES]
                means = [c["rce_mean"] for c in tmpl_cells if c]
                ses = [c["rce_se"] for c in tmpl_cells if c]
                if not means:
                    row.append("--")
                    continue
                m = float(np.mean(means))
                s = float(np.sqrt(np.sum(np.array(ses) ** 2)) / len(ses))
                row.append(f"{fmt(m)}\\pmstd{{{fmt_se(s)}}}")
        out.append(" & ".join(row) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table*}")
    return "\n".join(out) + "\n"


def gen_per_template_wmape(cells, eps: float) -> str:
    return gen_double_asymmetry(cells, eps)  # same shape, alias


def gen_per_root_mae(cells) -> str:
    """Per-root MAP MAE per template at ε=0.1, across configs and B_settings."""
    out = []
    out.append(PMSTD)
    out.append("")
    out.append("\\begin{table*}[t]")
    out.append("\\centering\\footnotesize")
    out.append("\\caption{Per-root MAP reconstruction MAE at $\\varepsilon=0.1$, $B=3k{+}1$. "
               "Bootstrap SEs.}")
    out.append("\\label{tab:agent_per_root_mae}")
    out.append("\\begin{tabular}{l|l|ccc}")
    out.append("\\toprule")
    out.append("Template & Root & M-All & M-Roots & M-Opt \\\\")
    out.append("\\midrule")
    for tmpl in TEMPLATES:
        c_all = cell(cells, template=tmpl, config="all", B_setting="3k+1", eps=0.1)
        if c_all is None:
            continue
        roots = list(c_all["per_root_map_mae"].keys())
        for i, r in enumerate(roots):
            row = [tmpl if i == 0 else "", r]
            for cfg in CONFIGS:
                c = cell(cells, template=tmpl, config=cfg, B_setting="3k+1", eps=0.1)
                if c is None:
                    row.append("--")
                else:
                    e = c["per_root_map_mae"][r]
                    row.append(f"{fmt(e['mean'])}\\pmstd{{{fmt_se(e['se'])}}}")
            out.append(" & ".join(row) + " \\\\")
        if tmpl != TEMPLATES[-1]:
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table*}")
    return "\n".join(out) + "\n"


def gen_win_rate(cells) -> str:
    """M-Opt vs M-All win rate per template (across all (B, ε) cells)."""
    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering\\small")
    out.append("\\caption{$\\mathcal{M}$-Opt vs $\\mathcal{M}$-All wMAPE win rate per "
               "template across all 9 $(B, \\varepsilon)$ cells.}")
    out.append("\\label{tab:agent_win_rate}")
    out.append("\\begin{tabular}{lcc}")
    out.append("\\toprule")
    out.append("Template & $\\mathcal{M}$-Opt vs $\\mathcal{M}$-All & "
               "$\\mathcal{M}$-Opt vs $\\mathcal{M}$-Roots \\\\")
    out.append("\\midrule")
    for tmpl in TEMPLATES:
        wins_all, wins_roots, total = 0, 0, 0
        for B in B_SETTINGS:
            for eps in EPS_LIST:
                c_all = cell(cells, template=tmpl, config="all", B_setting=B, eps=eps)
                c_roots = cell(cells, template=tmpl, config="roots", B_setting=B, eps=eps)
                c_opt = cell(cells, template=tmpl, config="opt", B_setting=B, eps=eps)
                if not all((c_all, c_roots, c_opt)):
                    continue
                if c_opt["wmape_mean"] < c_all["wmape_mean"]:
                    wins_all += 1
                if c_opt["wmape_mean"] < c_roots["wmape_mean"]:
                    wins_roots += 1
                total += 1
        out.append(f"{tmpl} & {wins_all}/{total} & {wins_roots}/{total} \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default=str(DEFAULT_AGG))
    ap.add_argument("--out-dir", default=str(DEFAULT_TABLES))
    args = ap.parse_args()

    with open(args.agg) as f:
        agg = json.load(f)
    cells = agg["cells"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "double_asymmetry_eps0.1.tex": gen_double_asymmetry(cells, 0.1),
        "double_asymmetry_eps0.05.tex": gen_double_asymmetry(cells, 0.05),
        "double_asymmetry_eps0.01.tex": gen_double_asymmetry(cells, 0.01),
        "aggregate_wmape.tex": gen_aggregate_wmape(cells),
        "per_template_wmape_eps0.1.tex": gen_per_template_wmape(cells, 0.1),
        "rce_eps0.1.tex": gen_rce(cells),
        "rce_eps0.1_appendix.tex": gen_rce_appendix(cells),
        "rce_eps0.1_body.tex": gen_rce_body(cells),
        "aggregate_rce.tex": gen_aggregate_rce(cells),
        "per_root_mae.tex": gen_per_root_mae(cells),
        "win_rate.tex": gen_win_rate(cells),
    }

    for name, content in files.items():
        path = out_dir / name
        path.write_text(content)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
