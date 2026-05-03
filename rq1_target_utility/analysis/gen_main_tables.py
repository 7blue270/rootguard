"""Generate RQ1 adversarial tables using wMAPE metric.

Reads from results_rq1_adversarial_tN_wmape/ and generates latex_rq1_adversarial_tN.tex with:
  - Table 1: Aggregate wMAPE (mean across templates)
  - Table 2: Aggregate Risk Class Error (mean across templates)
  - Table 3: Per-template wMAPE at epsilon=0.1
  - Table 4: Per-template RCE at epsilon=0.1

Usage:
    python gen_rq1_adversarial_tables.py [--turns N]
"""
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--turns", type=int, default=6)
parser.add_argument("--budget_mode", choices=["fixed", "kplus1"], default="fixed")
_args = parser.parse_args()
T = _args.turns
BUDGET_MODE = _args.budget_mode

if BUDGET_MODE == "kplus1":
    RESULTS_BASE = "./results_rq1_adversarial_kplus1_wmape"
else:
    RESULTS_BASE = "./results_rq1_adversarial_t{}_wmape".format(T)
TEMPLATES = ['ANEMIA', 'FIB4', 'AIP', 'CONICITY', 'VASCULAR', 'TYG', 'HOMA', 'NLR']
METHODS = [
    'exp_all', 'exp_roots', 'exp_opt',
    'blap_all', 'blap_roots', 'blap_opt',
    'stair_all', 'stair_roots', 'stair_opt',
]
EPS_VALS = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
# B_MAP is only used for fixed-turns mode display; kplus1 varies per template
B_MAP = {e: e * T for e in EPS_VALS}

K_ROOTS = {'ANEMIA': 3, 'FIB4': 4, 'AIP': 3, 'CONICITY': 3,
           'VASCULAR': 2, 'TYG': 2, 'HOMA': 2, 'NLR': 2}


def fmt(v):
    if v >= 100:
        return '{:.0f}'.format(v)
    else:
        return '{:.1f}'.format(v)


def load_summary(eps, tmpl, method):
    fname = '{}/epsilon_{}/{}/{}_results.json'.format(RESULTS_BASE, eps, tmpl, method)
    with open(fname) as f:
        return json.load(f)['summary']


# Preload all data
data = {}
for eps in EPS_VALS:
    data[eps] = {}
    for tmpl in TEMPLATES:
        data[eps][tmpl] = {}
        for m in METHODS:
            data[eps][tmpl][m] = load_summary(eps, tmpl, m)

out = []

out.append(r'%% ======================================================================')
out.append(r'%% RQ1 Adversarial Results — B = 6*epsilon threat model')
out.append(r'%% Data: results_rq1_adversarial_wmape/')
out.append(r'%% wMAPE = sum(|y_san - y_gt|) / sum(|y_gt|) * 100')
out.append(r'%% ======================================================================')
out.append('')

# ── Table 1: Aggregate wMAPE ─────────────────────────────────────────────
out.append(r'\begin{table*}[t]')
out.append(r'\centering')
out.append(r'\setlength{\tabcolsep}{3.0pt}')
out.append(r'\small')
out.append(r'\caption{{Aggregate wMAPE (\%) averaged across 8 clinical templates (200 adult male patients each, NHANES 2017--2018). Adversarial threat model: $B = {0}\varepsilon$, $t = {0}$ turns. $\mathcal{{M}}$-All sanitizes the target directly with $\varepsilon$; $\mathcal{{M}}$-Roots and $\mathcal{{M}}$-Opt allocate $B = {0}\varepsilon$ across root nodes. Best mean per row in \textbf{{bold}}.}}'.format(T))
out.append(r'\label{tab:rq1_adv_wmape}')
out.append(r'\begin{tabular}{cc|ccc|ccc|ccc}')
out.append(r'\toprule')
out.append(r'& & \multicolumn{3}{c|}{\textbf{Exponential}} & \multicolumn{3}{c|}{\textbf{Bounded Laplace}} & \multicolumn{3}{c}{\textbf{Staircase}} \\')
out.append(r'$\varepsilon$ & $B$ & All & Roots & Opt & All & Roots & Opt & All & Roots & Opt \\')
out.append(r'\midrule')

for eps in EPS_VALS:
    row_means = []
    for m in METHODS:
        per_tmpl = [data[eps][t][m]['wmape'] for t in TEMPLATES]
        row_means.append(np.mean(per_tmpl))

    best_idx = min(range(9), key=lambda i: row_means[i])

    cells = []
    for i, v in enumerate(row_means):
        s = fmt(v)
        if i == best_idx:
            cells.append('\\textbf{{{}}}'.format(s))
        else:
            cells.append(s)

    B = B_MAP[eps]
    out.append('{} & {:.2f} & {} \\\\'.format(eps, B, ' & '.join(cells)))

out.append(r'\bottomrule')
out.append(r'\end{tabular}')
out.append(r'\end{table*}')
out.append('')

# ── Table 2: Aggregate Risk Class Error ──────────────────────────────────
out.append(r'\begin{table*}[t]')
out.append(r'\centering')
out.append(r'\setlength{\tabcolsep}{3.0pt}')
out.append(r'\small')
out.append(r'\caption{{Aggregate Risk Class Error (\%) averaged across 8 clinical templates. Same adversarial threat model ($B = {}\varepsilon$). Best mean per row in \textbf{{bold}}.}}'.format(T))
out.append(r'\label{tab:rq1_adv_rce}')
out.append(r'\begin{tabular}{cc|ccc|ccc|ccc}')
out.append(r'\toprule')
out.append(r'& & \multicolumn{3}{c|}{\textbf{Exponential}} & \multicolumn{3}{c|}{\textbf{Bounded Laplace}} & \multicolumn{3}{c}{\textbf{Staircase}} \\')
out.append(r'$\varepsilon$ & $B$ & All & Roots & Opt & All & Roots & Opt & All & Roots & Opt \\')
out.append(r'\midrule')

for eps in EPS_VALS:
    row_means = []
    for m in METHODS:
        per_tmpl = [data[eps][t][m]['risk_mae'] for t in TEMPLATES]
        row_means.append(np.mean(per_tmpl))

    best_idx = min(range(9), key=lambda i: row_means[i])

    cells = []
    for i, v in enumerate(row_means):
        s = fmt(v)
        if i == best_idx:
            cells.append('\\textbf{{{}}}'.format(s))
        else:
            cells.append(s)

    B = B_MAP[eps]
    out.append('{} & {:.2f} & {} \\\\'.format(eps, B, ' & '.join(cells)))

out.append(r'\bottomrule')
out.append(r'\end{tabular}')
out.append(r'\end{table*}')
out.append('')

# ── Table 3: Per-template wMAPE at eps=0.1 ───────────────────────────────
eps_detail = 0.1
out.append(r'\begin{table*}[t]')
out.append(r'\centering')
out.append(r'\setlength{\tabcolsep}{3.0pt}')
out.append(r'\small')
out.append(r'\caption{{Per-template wMAPE (\%) at $\varepsilon = 0.1$ ($B = {}$). Best per row in \textbf{{bold}}.}}'.format(T * 0.1))
out.append(r'\label{tab:rq1_adv_per_template_wmape}')
out.append(r'\begin{tabular}{l|ccc|ccc|ccc}')
out.append(r'\toprule')
out.append(r'& \multicolumn{3}{c|}{\textbf{Exponential}} & \multicolumn{3}{c|}{\textbf{Bounded Laplace}} & \multicolumn{3}{c}{\textbf{Staircase}} \\')
out.append(r'Template & All & Roots & Opt & All & Roots & Opt & All & Roots & Opt \\')
out.append(r'\midrule')

for tmpl in TEMPLATES:
    vals = [data[eps_detail][tmpl][m]['wmape'] for m in METHODS]
    best_idx = min(range(9), key=lambda i: vals[i])
    cells = []
    for i, v in enumerate(vals):
        s = fmt(v)
        if i == best_idx:
            cells.append('\\textbf{{{}}}'.format(s))
        else:
            cells.append(s)
    out.append('{} & {} \\\\'.format(tmpl, ' & '.join(cells)))

out.append(r'\bottomrule')
out.append(r'\end{tabular}')
out.append(r'\end{table*}')
out.append('')

# ── Table 4: Per-template RCE at eps=0.1 ─────────────────────────────────
out.append(r'\begin{table*}[t]')
out.append(r'\centering')
out.append(r'\setlength{\tabcolsep}{3.0pt}')
out.append(r'\small')
out.append(r'\caption{{Per-template Risk Class Error (\%) at $\varepsilon = 0.1$ ($B = {}$). Best per row in \textbf{{bold}}.}}'.format(T * 0.1))
out.append(r'\label{tab:rq1_adv_per_template_rce}')
out.append(r'\begin{tabular}{l|ccc|ccc|ccc}')
out.append(r'\toprule')
out.append(r'& \multicolumn{3}{c|}{\textbf{Exponential}} & \multicolumn{3}{c|}{\textbf{Bounded Laplace}} & \multicolumn{3}{c}{\textbf{Staircase}} \\')
out.append(r'Template & All & Roots & Opt & All & Roots & Opt & All & Roots & Opt \\')
out.append(r'\midrule')

for tmpl in TEMPLATES:
    vals = [data[eps_detail][tmpl][m]['risk_mae'] for m in METHODS]
    best_idx = min(range(9), key=lambda i: vals[i])
    cells = []
    for i, v in enumerate(vals):
        s = fmt(v)
        if i == best_idx:
            cells.append('\\textbf{{{}}}'.format(s))
        else:
            cells.append(s)
    out.append('{} & {} \\\\'.format(tmpl, ' & '.join(cells)))

out.append(r'\bottomrule')
out.append(r'\end{tabular}')
out.append(r'\end{table*}')
out.append('')

if BUDGET_MODE == "kplus1":
    outfile = 'latex_rq1_adversarial_kplus1.tex'
else:
    outfile = 'latex_rq1_adversarial_t{}.tex'.format(T)
with open(outfile, 'w') as f:
    f.write('\n'.join(out) + '\n')
print('Written {} lines to {}'.format(len(out), outfile))
