"""Generate the double asymmetry table for RQ1.

Rows: eps in {0.5, 0.2, 0.1, 0.05, 0.02, 0.01}
Column groups: k+1, 2k+1, 3k+1 (each with All/Roots/Opt)
Each group has a B_bar column showing average budget
SE values shown beneath each mean as \\pmstd{...}
Mechanism: Staircase only
Output: writes to latex_rq1_double_asymmetry.tex
"""
import json
import numpy as np

TEMPLATES = ['ANEMIA', 'FIB4', 'AIP', 'CONICITY', 'VASCULAR', 'TYG', 'HOMA', 'NLR']
K = {'ANEMIA': 3, 'FIB4': 4, 'AIP': 3, 'CONICITY': 3,
     'VASCULAR': 2, 'TYG': 2, 'HOMA': 2, 'NLR': 2}
EPS_VALS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

RUNS = [
    ('kplus1',  'k+1',  'results_rq1_adversarial_kplus1_wmape',  lambda k: k + 1),
    ('2kplus1', '2k+1', 'results_rq1_adversarial_2kplus1_wmape', lambda k: 2 * k + 1),
    ('3kplus1', '3k+1', 'results_rq1_adversarial_3kplus1_wmape', lambda k: 3 * k + 1),
]


def load_summary(base, eps, tmpl, method):
    fname = '{}/epsilon_{}/{}/{}_results.json'.format(base, eps, tmpl, method)
    with open(fname) as f:
        return json.load(f)['summary']


def fmt(v):
    if v >= 100:
        return '{:.0f}'.format(v)
    elif v >= 10:
        return '{:.1f}'.format(v)
    else:
        return '{:.1f}'.format(v)


def fmt_se(v):
    if v >= 10:
        return '{:.0f}'.format(v)
    else:
        return '{:.1f}'.format(v)


out = []
out.append('\\providecommand{\\pmstd}[1]{{\\scriptsize $\\pm$#1}}')
out.append('')
out.append('\\begin{table*}[t]')
out.append('\\centering')
out.append('\\setlength{\\tabcolsep}{2.8pt}')
out.append('\\small')
out.append('\\caption{Aggregate wMAPE (\\%) at mid-to-high privacy levels across three adversary budget levels (Exponential mechanism). $\\bar{B}$: average total budget across templates. $\\mathcal{M}$-All is invariant to the budget level; $\\mathcal{M}$-Opt improves with each additional adversarial turn. Bootstrap standard errors shown beneath each mean. At $\\varepsilon \\geq 1.0$, all methods achieve $\\leq 1\\%$ wMAPE and differences are negligible (see App.~\\ref{app:rq1-tables}).}')
out.append('\\label{tab:double_asymmetry}')
out.append('\\begin{tabular}{c|cccc|cccc|cccc}')
out.append('\\toprule')
out.append('& \\multicolumn{4}{c|}{$B = (k{+}1)\\varepsilon$} & \\multicolumn{4}{c|}{$B = (2k{+}1)\\varepsilon$} & \\multicolumn{4}{c}{$B = (3k{+}1)\\varepsilon$} \\\\')
out.append('$\\varepsilon$ & $\\bar{B}$ & All & Roots & Opt & $\\bar{B}$ & All & Roots & Opt & $\\bar{B}$ & All & Roots & Opt \\\\')
out.append('\\midrule')

for ei, eps in enumerate(EPS_VALS):
    # Collect means and SEs
    means = []
    ses = []
    bbars = []
    for rk, rl, base, bfn in RUNS:
        bbar = np.mean([bfn(K[t]) * eps for t in TEMPLATES])
        bbars.append(bbar)
        for var in ['all', 'roots', 'opt']:
            summaries = [load_summary(base, eps, t, 'exp_{}'.format(var)) for t in TEMPLATES]
            per_tmpl_wmape = [s['wmape'] for s in summaries]
            per_tmpl_se = [s['wmape_se'] for s in summaries]
            means.append(np.mean(per_tmpl_wmape))
            # SE of the aggregate: propagate bootstrap SEs across templates
            ses.append(np.sqrt(np.sum(np.array(per_tmpl_se) ** 2)) / len(TEMPLATES))

    # Find best (lowest mean) across all 9 data cells
    best_idx = min(range(9), key=lambda i: means[i])

    # Build single row with mean ± SE
    cells = []
    for run_idx, (rk, rl, base, bfn) in enumerate(RUNS):
        # B_bar column
        cells.append('{:.2f}'.format(bbars[run_idx]))
        for var_idx in range(3):
            i = run_idx * 3 + var_idx
            m_str = fmt(means[i])
            se_str = fmt_se(ses[i])
            val = '{}\\pmstd{{{}}}'.format(m_str, se_str)
            if i == best_idx:
                cells.append('\\textbf{{{}}}'.format(val))
            else:
                cells.append(val)

    out.append('{} & {} \\\\'.format(eps, ' & '.join(cells)))

out.append('\\bottomrule')
out.append('\\end{tabular}')
out.append('\\end{table*}')

tex = '\n'.join(out) + '\n'

with open('latex_rq1_double_asymmetry.tex', 'w') as f:
    f.write(tex)

print(tex)
print('Written to latex_rq1_double_asymmetry.tex')
