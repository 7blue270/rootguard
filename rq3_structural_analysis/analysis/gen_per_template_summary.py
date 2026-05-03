#!/usr/bin/env python3
"""Generate the §5 RQ3 body table tab:per_template_summary.

Per-template wMAPE at \\varepsilon = 0.1, B = (2k+1)\\varepsilon, Exponential
mechanism. Eight templates split into two regimes (paper §5.7):

* Compressing — large absolute gaps (logs / divisions / large constants):
  AIP, HOMA, NLR, FIB4
* Amplifying — small absolute gaps:
  VASCULAR, TYG, CONICITY, ANEMIA

Each row shows the template name, its target formula (paper-canonical
LaTeX), the M-All wMAPE, the M-Opt wMAPE, and their ratio.

Reads from ``results_rq1_adversarial_2kplus1_wmape_root_based/`` (root-space
sweep) — the same data the paper uses for §5 RQ1/RQ3.

Output: ``tables/per_template_summary.tex``  (label tab:per_template_summary)
"""
import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EPS = 0.1

# Paper's row order within each regime
COMPRESSING_ORDER = ['AIP', 'HOMA', 'NLR', 'FIB4']
AMPLIFYING_ORDER  = ['VASCULAR', 'TYG', 'CONICITY', 'ANEMIA']

# Formula labels copied from PAPER/sections/5_experiments_worst_case_adversary.tex
FORMULAS = {
    'AIP':      r'$\log_{10}(\text{TG}/\text{HDL})$',
    'HOMA':     r'$\text{Glu}\cdot\text{Ins}/405$',
    'NLR':      r'$\text{NEU}/\text{LYM}$',
    'FIB4':     r'$\text{age}\cdot\text{AST}/(\text{PLT}\sqrt{\text{ALT}})$',
    'VASCULAR': r'$100(\text{SBP}{-}\text{DBP})/\text{SBP}$',
    'TYG':      r'$\ln(\text{TG}\cdot\text{Glu}/2)$',
    'CONICITY': r'$\text{waist}/(0.109\sqrt{\text{wt}/\text{ht}})$',
    'ANEMIA':   r'$100\cdot\text{Hb}/\text{Hct}$',
}


def load(results_dir, tmpl, method):
    p = Path(results_dir) / f'epsilon_{EPS}' / tmpl / f'{method}_results.json'
    with open(p) as f:
        return json.load(f)['summary']


def fmt(v):
    return f'{v:.1f}'


def make_table(results_dir):
    rows = {}
    for tmpl in COMPRESSING_ORDER + AMPLIFYING_ORDER:
        a = load(results_dir, tmpl, 'exp_all')['wmape']
        o = load(results_dir, tmpl, 'exp_opt')['wmape']
        rows[tmpl] = (a, o, a / o if o > 0 else float('inf'))

    out = []
    out.append(r'\begin{table}[h]')
    out.append(r'\centering')
    out.append(r'\setlength{\tabcolsep}{4pt}')
    out.append(r'\small')
    out.append(r'\caption{Per-template wMAPE (\%) at $\varepsilon = 0.1$, '
               r'$B = (2k{+}1)\varepsilon$ (Exponential). \emph{Compressing} '
               r'formulas show large absolute gaps; \emph{amplifying} formulas '
               r'show small absolute gaps but comparable multiplicative ratios. '
               r'Full per-template tables in App.~\ref{app:rq1-tables}.}')
    out.append(r'\label{tab:per_template_summary}')
    out.append(r'\begin{tabular}{l l c c c}')
    out.append(r'\toprule')
    out.append(r'\textbf{Template} & \textbf{Formula} & \textbf{All} & \textbf{Opt} & \textbf{Ratio} \\')
    out.append(r'\midrule')
    out.append(r'\multicolumn{5}{l}{\emph{Compressing — large absolute gaps}} \\')
    for tmpl in COMPRESSING_ORDER:
        a, o, r = rows[tmpl]
        out.append(f'{tmpl} & {FORMULAS[tmpl]} & {fmt(a)} & {fmt(o)} & ${r:.1f}\\times$ \\\\')
    out.append(r'\midrule')
    out.append(r'\multicolumn{5}{l}{\emph{Amplifying — small absolute gaps}} \\')
    for tmpl in AMPLIFYING_ORDER:
        a, o, r = rows[tmpl]
        out.append(f'{tmpl} & {FORMULAS[tmpl]} & {fmt(a)} & {fmt(o)} & ${r:.1f}\\times$ \\\\')
    out.append(r'\bottomrule')
    out.append(r'\end{tabular}')
    out.append(r'\end{table}')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n', 1)[0])
    # The RQ1 sweep emits its results under rq1_target_utility/. Look there
    # by default (sibling of this RQ3 folder); fall back to the repo root for
    # users who symlink results outside the folder structure.
    _default_rq1 = (REPO.parent / 'rq1_target_utility'
                    / 'results_rq1_adversarial_2kplus1_wmape_root_based')
    if not _default_rq1.exists():
        _default_rq1 = REPO.parent / 'results_rq1_adversarial_2kplus1_wmape_root_based'
    ap.add_argument('--results-dir',
                    default=str(_default_rq1),
                    help='Path to the root-space (2k+1) RQ1 sweep results '
                         '(default: ../rq1_target_utility/results_rq1_adversarial_2kplus1_wmape_root_based)')
    ap.add_argument('--out-dir', default=str(REPO / 'tables'),
                    help='Where to write the .tex (default: rq3_structural_analysis/tables)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'per_template_summary.tex'
    path.write_text(make_table(args.results_dir))
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
