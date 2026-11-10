"""Patient-level/hierarchical summaries and finite-grid iso-utility frontiers."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summaries(rows, bootstrap=500, seed=20261110):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'], row['template'], row['method'], row['parameter'], row['B'])].append(row)
    result = []
    for key, rr in sorted(groups.items()):
        by_subject = defaultdict(list)
        for row in rr:
            by_subject[row['subject_id']].append(row)
        ids = sorted(by_subject)
        # Multiple admissions/windows are not counted as independent people.
        values = []
        for sid in ids:
            reps = defaultdict(list)
            for r in by_subject[sid]:
                reps[r['repeat']].append(r['mae'])
            values.append(np.array([np.mean(reps[k]) for k in sorted(reps)]))
        patient_means = np.array([v.mean() for v in values])
        mean = float(patient_means.mean())
        rng = np.random.default_rng(seed)  # Same resamples across paired method/B cells.
        hierarchical, patient_only, mechanism_only = [], [], []
        for _ in range(bootstrap):
            indices = rng.integers(len(ids), size=len(ids))
            patient_only.append(float(patient_means[indices].mean()))
            hierarchical.append(float(np.mean([rng.choice(values[i], len(values[i]), replace=True).mean() for i in indices])))
            mechanism_only.append(float(np.mean([rng.choice(v, len(v), replace=True).mean() for v in values])))
        interval = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
        base = dict(zip(['split', 'template', 'method', 'parameter', 'B'], key))
        within = [float(np.var(v, ddof=1)) / len(v) if len(v) > 1 else 0. for v in values]
        between = float(np.var(patient_means, ddof=1)) if len(values) > 1 else 0.
        base.update(n_patients=len(ids), n_runs=len(rr), mean=mean,
                    bootstrap_means=hierarchical,
                    ci_low=interval(hierarchical)[0], ci_high=interval(hierarchical)[1],
                    # One-sided 95% bound, not the 97.5% upper two-sided endpoint.
                    ucb95=max(mean, float(np.quantile(hierarchical, .95))),
                    patient_ci=interval(patient_only), mechanism_ci=interval(mechanism_only),
                    variance_mechanism_mean=float(np.sum(within) / len(values) ** 2),
                    variance_between_mean=max(0., between - float(np.mean(within))) / len(values))
        for name in ['mae_protected', 'p95_abs_error', 'input_clipping_mae', 'query_loss', 'workload_loss', 'class_error',
                     'spent', 'remaining_fraction', 'refresh_count', 'post_stop_mae', 'J', 'aoi_mean',
                     'cache_projection_fraction', 'publication_projection_fraction', 'input_clipping_fraction',
                     'early_budget_freeze', 'root_release_count', 'reason_gate', 'reason_age_force', 'reason_both',
                     'reason_periodic', 'reason_none', 'reason_empty', 'reason_frozen']:
            per_patient = []
            for sid in ids:
                vv = [r[name] for r in by_subject[sid] if r.get(name) is not None]
                if vv:
                    per_patient.append(float(np.mean(vv)))
            base[name] = float(np.mean(per_patient)) if per_patient else None
        den = sum(r['significant_events'] for r in rr)
        refs = sum(r['refresh_count'] for r in rr)
        base['miss_rate'] = sum(r['missed_events'] for r in rr) / den if den else None
        base['frac_age'] = sum(r['age_refreshes'] for r in rr) / refs if refs else None
        base['spent_max'] = max(r['spent'] for r in rr)
        assert base['spent_max'] <= base['B'] + 1e-8
        result.append(base)
    return result


def first_reached(curve, tau, metric='mean'):
    reached = [r['B'] for r in curve if r[metric] <= tau]
    return min(reached) if reached else None


def select(summaries_, template, method, tau):
    curves = defaultdict(list)
    for row in summaries_:
        if row['split'] == 'validation' and row['template'] == template and row['method'] == method:
            curves[row['parameter']].append(row)
    if not curves:
        raise ValueError(f'No validation candidates for {template}/{method}')
    # Unreachable candidates tie-break by average validation error, then parameter.
    def rank(item):
        parameter, curve = item
        b = first_reached(curve, tau)
        return (float('inf') if b is None else b, float(np.mean([r['mean'] for r in curve])), parameter)
    parameter, curve = min(curves.items(), key=rank)
    return {'parameter': parameter, 'validation_B_min': first_reached(curve, tau),
            'validation_reached': first_reached(curve, tau) is not None}


def selection_stability(summary, targets):
    """Paired patient/mechanism resamples, using validation rows exclusively."""
    result = []
    for template, taus in targets.items():
        for method in ['B2', 'B2p', 'M1-a']:
            curves = defaultdict(list)
            for row in summary:
                if row['split'] == 'validation' and row['template'] == template and row['method'] == method:
                    curves[row['parameter']].append(row)
            if not curves:
                continue
            n = len(next(iter(curves.values()))[0]['bootstrap_means'])
            for tau in taus:
                frequencies = defaultdict(int)
                reached = 0
                for draw in range(n):
                    ranks = []
                    for param, curve in curves.items():
                        bs = [r['B'] for r in curve if r['bootstrap_means'][draw] <= tau]
                        ranks.append((min(bs) if bs else float('inf'),
                                      float(np.mean([r['bootstrap_means'][draw] for r in curve])), param))
                    winner = min(ranks)
                    frequencies[winner[2]] += 1
                    reached += int(np.isfinite(winner[0]))
                result.append({'template': template, 'method': method, 'tau': tau,
                               'selection_frequency': {k: v / n for k, v in sorted(frequencies.items())},
                               'bootstrap_reached_fraction': reached / n})
    return result


def frontiers(summary, selections, targets):
    result = []
    for template, taus in targets.items():
        for tau in taus:
            chosen = selections[template][str(tau)]
            for metric in ['mean', 'ucb95']:
                cells = {}
                for method, selection in chosen.items():
                    curve = [r for r in summary if r['split'] == 'test' and r['template'] == template
                             and r['method'] == method and r['parameter'] == selection['parameter']]
                    if not curve:
                        continue
                    value = first_reached(curve, tau, metric)
                    at = next((r for r in curve if r['B'] == value), None)
                    cells[method] = {'B_min': value, 'B_max_scanned': max(r['B'] for r in curve),
                                     'status': 'reached' if value is not None else 'not_reached_within_grid',
                                     'miss_rate': at['miss_rate'] if at else None,
                                     'frac_age': at['frac_age'] if at else None}
                for method, cell in cells.items():
                    reference = cells.get('B2p', {}).get('B_min')
                    saving = 1 - cell['B_min'] / reference if cell['B_min'] is not None and reference is not None else None
                    uncertainty = {}
                    if metric == 'mean':
                        def draws(m):
                            cc = [r for r in summary if r['split'] == 'test' and r['template'] == template
                                  and r['method'] == m and r['parameter'] == chosen[m]['parameter']]
                            n = len(cc[0]['bootstrap_means'])
                            return [min((r['B'] for r in cc if r['bootstrap_means'][i] <= tau), default=None) for i in range(n)]
                        bb, ref = draws(method), draws('B2p')
                        finite_b = [b for b in bb if b is not None]
                        savings = [1 - b / r for b, r in zip(bb, ref) if b is not None and r is not None]
                        ci = lambda vv: [float(np.quantile(vv, .025)), float(np.quantile(vv, .975))] if vv else None
                        uncertainty = {'B_min_ci_conditional_on_reached': ci(finite_b),
                                       'bootstrap_reached_fraction': len(finite_b) / len(bb),
                                       'saving_ci_conditional_on_both_reached': ci(savings),
                                       'bootstrap_both_reached_fraction': len(savings) / len(bb)}
                    result.append({'template': template, 'tau': tau, 'criterion': metric, 'method': method,
                                   'parameter': chosen[method]['parameter'], **cell, 'saving_vs_B2p': saving,
                                   **uncertainty})
    return result


def plots(directory, summary, frontiers_):
    # Plain matplotlib; no dependency on local drawing skills or remote services.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    for template in sorted({r['template'] for r in frontiers_}):
        for criterion in ['mean', 'ucb95']:
            fig, ax = plt.subplots(figsize=(6.4, 4.2), layout='constrained')
            for method in sorted({r['method'] for r in frontiers_}):
                rr = sorted([r for r in frontiers_ if r['template'] == template and r['criterion'] == criterion
                             and r['method'] == method and r['B_min'] is not None], key=lambda r: r['tau'])
                if rr:
                    ax.plot([r['tau'] for r in rr], [r['B_min'] for r in rr], marker='o', label=method)
            ax.set(xlabel='Target final-node MAE', ylabel='Minimum scanned budget B',
                   title=f'{template} | {criterion} | pilot, finite grid')
            ax.legend()
            for ext in ['png', 'pdf']:
                fig.savefig(directory / f'{template}_frontier_{criterion}.{ext}', dpi=180)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.4, 4.2), layout='constrained')
        grouped = defaultdict(list)
        for row in summary:
            if row['template'] == template and row['split'] == 'test':
                grouped[(row['method'], row['parameter'])].append(row)
        for (method, param), rr in sorted(grouped.items()):
            rr.sort(key=lambda r: r['B'])
            ax.plot([r['B'] for r in rr], [r['mean'] for r in rr], marker='.', label=f'{method}/{param}')
        ax.set(xlabel='Session budget B', ylabel='Final-node MAE over all T rounds', title=f'{template} | pilot')
        ax.legend(fontsize=6)
        fig.savefig(directory / f'{template}_mae.png', dpi=180)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--plots', action='store_true')
    a = p.parse_args()
    metadata = json.loads((a.directory / 'metadata.json').read_text())
    rows = [json.loads(s) for s in (a.directory / 'metrics.jsonl').read_text().splitlines()]
    ss = summaries(rows, a.bootstrap, metadata['config']['seed'])
    ff = frontiers(ss, metadata['selections'], metadata['targets'])
    write_json(a.directory / 'summary.json', ss)
    csv_rows(a.directory / 'frontiers.csv', ff)
    if a.plots:
        plots(a.directory, ss, ff)


if __name__ == '__main__':
    main()
