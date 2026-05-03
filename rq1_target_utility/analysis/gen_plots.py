"""Generate all RQ1 figures for the new adversarial threat model.

Outputs to plots_rq1_new/:
  - rq1_double_asymmetry.pdf  (Figure 1: headline — budget scaling)
  - rq1_privacy_utility.pdf   (Figure 2: eps sweep at 2k+1)
  - rq1_per_template_grid.pdf (Appendix: 8-panel per-template breakdown)
"""
import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PLOTS_BASE = "./plots_rq1_new"
EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
TEMPLATES = ['ANEMIA', 'FIB4', 'AIP', 'CONICITY',
             'VASCULAR', 'TYG', 'HOMA', 'NLR']
K = {'ANEMIA': 3, 'FIB4': 4, 'AIP': 3, 'CONICITY': 3,
     'VASCULAR': 2, 'TYG': 2, 'HOMA': 2, 'NLR': 2}

RUNS = [
    ('k+1',  'results_rq1_adversarial_kplus1_wmape',  lambda k: k + 1),
    ('2k+1', 'results_rq1_adversarial_2kplus1_wmape', lambda k: 2 * k + 1),
    ('3k+1', 'results_rq1_adversarial_3kplus1_wmape', lambda k: 3 * k + 1),
]

FAMILY_COLORS = {'Exp': '#0072B2', 'BLap': '#E69F00', 'Stair': '#009E73'}
VARIANT_COLORS = {'All': '#999999', 'Roots': '#555555', 'Opt': '#000000'}
VARIANT_LW = {'All': 2.0, 'Roots': 2.5, 'Opt': 3.0}
VARIANT_LS = {'All': '-', 'Roots': '--', 'Opt': '-'}
VARIANT_MARKERS = {'All': 'o', 'Roots': 's', 'Opt': 'D'}
VARIANT_MS = {'All': 8, 'Roots': 9, 'Opt': 10}
VARIANT_ALPHA = {'All': 0.55, 'Roots': 0.75, 'Opt': 1.0}

FAMILIES = [
    ('Exp',   'Exponential',     'exp'),
    ('BLap',  'Bounded Laplace', 'blap'),
    ('Stair', 'Staircase',       'stair'),
]

os.makedirs(PLOTS_BASE, exist_ok=True)


def load(base, eps, tmpl, method):
    f = '{}/epsilon_{}/{}/{}_results.json'.format(base, eps, tmpl, method)
    with open(f) as fh:
        return json.load(fh)['summary']['wmape']


# ═════════════════════════════════════════════════════════════════════════
# Figure 1: Double Asymmetry (headline)
# ═════════════════════════════════════════════════════════════════════════

def plot_double_asymmetry(output_path, eps=0.1):
    budget_labels = [r[0] for r in RUNS]
    x = np.arange(len(budget_labels))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for col, (fam_key, fam_title, mech_key) in enumerate(FAMILIES):
        ax = axes[col]
        fam_color = FAMILY_COLORS[fam_key]

        for variant, var_key in [('All', 'all'), ('Roots', 'roots'), ('Opt', 'opt')]:
            vals = []
            for run_label, base, bfn in RUNS:
                per_tmpl = [load(base, eps, t, '{}_{}'.format(mech_key, var_key))
                            for t in TEMPLATES]
                vals.append(np.mean(per_tmpl))

            ax.plot(x, vals,
                    marker=VARIANT_MARKERS[variant],
                    linestyle=VARIANT_LS[variant],
                    color=fam_color,
                    linewidth=VARIANT_LW[variant],
                    markersize=VARIANT_MS[variant],
                    alpha=VARIANT_ALPHA[variant],
                    label='{}-{}'.format(fam_key, variant))

            # Annotate values
            for i, v in enumerate(vals):
                offset = 0.4 if variant == 'All' else (-0.6 if variant == 'Opt' else 0.2)
                ax.annotate('{:.1f}'.format(v),
                            xy=(i, v), fontsize=8,
                            textcoords='offset points',
                            xytext=(12, offset * 10),
                            ha='left', va='center')

        ax.set_xticks(x)
        ax.set_xticklabels(['$B = (k{+}1)\\varepsilon$',
                            '$B = (2k{+}1)\\varepsilon$',
                            '$B = (3k{+}1)\\varepsilon$'],
                           fontsize=10)
        ax.set_title(fam_title, fontsize=14, fontweight='bold', color=fam_color)
        ax.grid(True, linestyle='--', alpha=0.3)

        if col == 0:
            ax.set_ylabel('Aggregate wMAPE (%) at $\\varepsilon = {}$'.format(eps),
                           fontsize=12)
        ax.legend(fontsize=12, loc='upper right')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Saved: {}".format(output_path))


# ═════════════════════════════════════════════════════════════════════════
# Figure 2: Privacy-Utility Tradeoff (2k+1)
# ═════════════════════════════════════════════════════════════════════════

def plot_privacy_utility(output_path):
    base = 'results_rq1_adversarial_2kplus1_wmape'

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for col, (fam_key, fam_title, mech_key) in enumerate(FAMILIES):
        ax = axes[col]
        fam_color = FAMILY_COLORS[fam_key]

        for variant, var_key in [('All', 'all'), ('Roots', 'roots'), ('Opt', 'opt')]:
            means, eps_used = [], []
            for eps in EPSILONS:
                per_tmpl = [load(base, eps, t, '{}_{}'.format(mech_key, var_key))
                            for t in TEMPLATES]
                means.append(np.mean(per_tmpl))
                eps_used.append(eps)

            means = np.array(means)
            ax.plot(eps_used, means,
                    marker=VARIANT_MARKERS[variant],
                    linestyle=VARIANT_LS[variant],
                    color=fam_color,
                    linewidth=VARIANT_LW[variant],
                    markersize=VARIANT_MS[variant],
                    alpha=VARIANT_ALPHA[variant],
                    label='{}-{}'.format(fam_key, variant),
                    zorder=3)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(EPSILONS)
        ax.set_xticklabels(['{:g}'.format(e) for e in EPSILONS],
                           fontsize=8, rotation=45, ha='right')
        ax.set_xlabel('$\\varepsilon$', fontsize=13)
        if col == 0:
            ax.set_ylabel('Aggregate wMAPE (%)', fontsize=13)
        ax.set_title(fam_title, fontsize=14, fontweight='bold', color=fam_color)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=12, loc='upper right')

    fig.suptitle('Privacy--Utility Tradeoff ($B = (2k{+}1)\\varepsilon$)',
                 fontsize=15, fontweight='bold', y=1.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Saved: {}".format(output_path))


# ═════════════════════════════════════════════════════════════════════════
# Appendix Figure: Per-Template Breakdown (2k+1)
# ═════════════════════════════════════════════════════════════════════════

def plot_per_template_grid(output_path):
    base = 'results_rq1_adversarial_2kplus1_wmape'
    TMPL_ORDER = ['FIB4', 'HOMA', 'VASCULAR', 'CONICITY',
                  'ANEMIA', 'AIP', 'NLR', 'TYG']
    ROW_LABELS = [r'$\mathcal{M}$-Opt wins or ties',
                  r'$\mathcal{M}$-All wins']
    TARGET_NAMES = {
        'ANEMIA': 'mchc', 'FIB4': 'fib4', 'AIP': 'aip',
        'CONICITY': 'conicity', 'VASCULAR': 'ppi',
        'TYG': 'tyg', 'HOMA': 'homa', 'NLR': 'nlr',
    }

    METHODS = [
        ('exp_all',    'Exp-All',    'Exp',   'All'),
        ('exp_roots',  'Exp-Roots',  'Exp',   'Roots'),
        ('exp_opt',    'Exp-Opt',    'Exp',   'Opt'),
        ('blap_all',   'BLap-All',   'BLap',  'All'),
        ('blap_roots', 'BLap-Roots', 'BLap',  'Roots'),
        ('blap_opt',   'BLap-Opt',   'BLap',  'Opt'),
        ('stair_all',  'Stair-All',  'Stair', 'All'),
        ('stair_roots','Stair-Roots','Stair', 'Roots'),
        ('stair_opt',  'Stair-Opt',  'Stair', 'Opt'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharey=False)

    for idx, tmpl in enumerate(TMPL_ORDER):
        row, col = idx // 4, idx % 4
        ax = axes[row, col]
        target = TARGET_NAMES[tmpl]

        for json_key, label, family, variant in METHODS:
            color = FAMILY_COLORS[family]
            marker = VARIANT_MARKERS[variant]
            ms = VARIANT_MS[variant]

            vals_list, eps_used = [], []
            for eps in EPSILONS:
                v = load(base, eps, tmpl, json_key)
                vals_list.append(v)
                eps_used.append(eps)

            ax.plot(eps_used, vals_list, marker=marker,
                    linestyle=VARIANT_LS[variant],
                    color=color,
                    linewidth=VARIANT_LW[variant] * 0.7,
                    markersize=ms * 0.8, label=label,
                    alpha=VARIANT_ALPHA[variant])

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('{} (target: {})'.format(tmpl, target),
                     fontsize=16, fontweight='bold')
        ax.set_xticks(EPSILONS)
        ax.set_xticklabels(['{:g}'.format(e) for e in EPSILONS],
                           fontsize=10, rotation=45, ha='right')
        ax.set_xlabel(r'$\varepsilon$', fontsize=14)
        if col == 0:
            ax.set_ylabel('wMAPE (%)', fontsize=14)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.25)

        if idx == 0:
            ax.legend(fontsize=10, loc='upper right', ncol=2)

    for row_idx, label in enumerate(ROW_LABELS):
        fig.text(0.005, 0.75 - row_idx * 0.48, label,
                 fontsize=15, fontweight='bold', rotation=90,
                 va='center', ha='left', color='#444444')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=[0.03, 0, 1, 1])
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("Saved: {}".format(output_path))


# ═════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    plot_double_asymmetry(os.path.join(PLOTS_BASE, 'rq1_double_asymmetry.pdf'))
    plot_privacy_utility(os.path.join(PLOTS_BASE, 'rq1_privacy_utility.pdf'))
    plot_per_template_grid(os.path.join(PLOTS_BASE, 'rq1_per_template_grid.pdf'))
