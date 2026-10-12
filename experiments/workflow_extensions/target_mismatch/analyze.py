"""Patient-paired bootstrap; repeats are not independent patients."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    cfg = json.loads((args.run / 'config.json').read_text())
    cells = json.loads((args.run / 'cells.json').read_text())
    draws = np.load(args.run / 'draws.npz')
    groups = {}
    with (args.run / 'patients.csv').open() as f:
        for row in csv.DictReader(f):
            groups.setdefault(row['cell'], []).append(row)
    references = {(c['template'], c['B'], c['method']): c for c in cells if c['method'] != 'opt_proxy'}
    boot_indices = {}
    results = []
    for c in cells:
        rows = groups[c['cell']]
        truth = np.array([abs(float(r['truth'])) for r in rows])
        e = np.array([float(r['mean_abs_error']) for r in rows])
        w = np.array([float(r['wrong_fraction']) for r in rows])
        if c['template'] not in boot_indices:
            rng = np.random.default_rng(cfg['seed'] + 100)
            boot_indices[c['template']] = rng.integers(0, len(rows), size=(cfg['bootstrap'], len(rows)))
        idx = boot_indices[c['template']]
        denom = truth[idx].sum(axis=1)
        out = dict(c)
        out['wmape_ci95'] = np.quantile(100 * e[idx].sum(axis=1) / denom, [.025, .975]).tolist()
        out['rce_ci95'] = np.quantile(100 * w[idx].mean(axis=1), [.025, .975]).tolist()
        for method, label in [('roots', 'roots'), ('opt_true', 'true')]:
            ref = references[c['template'], c['B'], method]
            rr = groups[ref['cell']]
            assert [r['patient'] for r in rows] == [r['patient'] for r in rr]
            de = e - np.array([float(r['mean_abs_error']) for r in rr])
            dw = w - np.array([float(r['wrong_fraction']) for r in rr])
            out[f'delta_wmape_vs_{label}'] = float(100 * de.sum() / truth.sum())
            out[f'delta_wmape_vs_{label}_ci95'] = np.quantile(100 * de[idx].sum(axis=1) / denom, [.025, .975]).tolist()
            out[f'delta_rce_vs_{label}'] = float(100 * dw.mean())
            out[f'delta_rce_vs_{label}_ci95'] = np.quantile(100 * dw[idx].mean(axis=1), [.025, .975]).tolist()
            # Noise-only uncertainty for the fixed cohort. Shared draws preserve
            # pairing; each repeat column is an independent full-cohort run.
            dr = draws[c['cell'] + ':error'] - draws[ref['cell'] + ':error']
            per_repeat = 100. * dr.sum(axis=0) / truth.sum()
            se = float(per_repeat.std(ddof=1) / np.sqrt(len(per_repeat)))
            out[f'delta_wmape_vs_{label}_mc_se'] = se
            margin = float(student_t.ppf(.975, len(per_repeat)-1)) * se
            center = out[f'delta_wmape_vs_{label}']
            out[f'delta_wmape_vs_{label}_mc_ci95'] = [center-margin, center+margin]
        results.append(out)
    (args.run / 'analysis.json').write_text(json.dumps(results, indent=2))
    zero = [r for r in results if r['method'] == 'opt_proxy' and r['level'] == 0]
    if any(abs(r['delta_wmape_vs_true']) > 1e-9 or abs(r['delta_rce_vs_true']) > 1e-9 for r in zero):
        raise RuntimeError('Zero mismatch did not recover true M-Opt')
    (args.run / 'checks.json').write_text(json.dumps({'zero_mismatch_cells': len(zero),
        'zero_mismatch_exact_recovery': True, 'patient_pairing_checked': True,
        'uncertainty': 'patient bootstrap and fixed-cohort noise-only t intervals reported separately'}, indent=2))
    fields = ['template', 'B', 'method', 'family', 'root', 'level', 'proxy_wmape', 'importance_l1', 'wmape', 'rce',
              'delta_wmape_vs_roots', 'delta_wmape_vs_roots_ci95', 'delta_wmape_vs_true',
              'delta_rce_vs_roots', 'delta_rce_vs_roots_ci95']
    with (args.run / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    lines = ['# Target-function mismatch: run summary', '',
             'Errors are percentages; paired gaps are percentage points. CIs resample patients after averaging noise repeats.', '',
             '| Template | B | M-Roots wMAPE | True M-Opt wMAPE | M-Roots RCE | True M-Opt RCE |',
             '|---|---:|---:|---:|---:|---:|']
    for name in cfg['templates']:
        for B in cfg['budgets']:
            a = references[name, B, 'roots']
            b = references[name, B, 'opt_true']
            lines.append(f"| {name} | {B} | {a['wmape']:.3f} | {b['wmape']:.3f} | {a['rce']:.3f} | {b['rce']:.3f} |")
    lines += ['', 'Full cells and paired intervals: `analysis.json` and `summary.csv`.',
              'Exploratory 95% pointwise intervals; no multiplicity correction or population-generalization claim.']
    (args.run / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
