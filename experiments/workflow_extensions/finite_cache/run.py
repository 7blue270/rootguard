"""Train -> validation selection -> test, with an explicitly causal mechanism."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from . import data, public_stats
from .workload import Workload, FORMULA_VERSION
from .mechanisms import Cache, paired_uniforms
from .analyze import write_json, csv_rows, summaries, select, frontiers, plots, first_reached, selection_stability

ROOT = Path(__file__).resolve().parents[3]


def grid(low, high, ratio):
    if low <= 0 or high < low or ratio <= 1:
        raise ValueError('Invalid geometric budget grid')
    values, v = [float(low)], float(low)
    while v * ratio < high * (1 - 1e-12):
        v *= ratio
        values.append(round(v, 8))
    return sorted(set(values + [float(high)]))


def refinement_points(budgets, crossing_budgets):
    extra = set()
    for b in crossing_budgets:
        if b in budgets and budgets.index(b) > 0:
            a = budgets[budgets.index(b) - 1]
            if b / a > 1.05 + 1e-9:
                extra.update(grid(a, b, 1.05)[1:-1])
    return sorted(extra - set(budgets))


def simulate(rec, workload, stats, method, parameter, B, repeat, config, *, detailed=False):
    h = stats['h']
    q = float(parameter.split('=')[1]) if parameter.startswith('q=') else config['quantiles'][0]
    period = int(parameter.split('=')[1]) if parameter.startswith('P=') else config.get('fixed_gate_period', 4)
    cache = Cache(workload.roots, h, stats['C'][str(q)], B, method,
                  rec['T'] - 1, period=period, **config['mechanism'])
    raw = data.replay(rec)
    z = workload.normalize(raw)
    truth, protected = workload.evaluate(raw), workload.evaluate(workload.denormalize(z))
    identity = [config['seed'], rec['subject_id'], rec['trajectory_id'], B, repeat]
    uniforms = lambda turn: paired_uniforms(*identity, turn, len(workload.roots))
    cache.initialize(z[0], uniforms(1))
    events = {e['turn']: e for e in rec['events']}
    queries = {i: [] for i in range(1, rec['T'] + 1)}
    for q_ in rec['queries']:
        queries[q_['turn']].append(q_['node'])
    estimates, snapshots, logs = [], [], []
    errors, protected_errors, query_errors, workload_errors, Js, aois, project = [], [], [], [], [], [], []
    last_release_turn = np.ones(len(h), dtype=int)
    missed = significant = age_refreshes = 0
    reasons = Counter()
    M = np.array(stats['M'])
    weights, scales = np.array(stats['weights']), np.array(stats['scales'])
    previous = z[0]
    for turn in range(1, rec['T'] + 1):
        refreshed = set()
        if turn in events:
            event = events[turn]
            decision = cache.decide(data.public_batch(event))
            reasons[decision.trigger] += 1
            before = cache.snapshot()
            if decision.trigger not in ('none', 'empty', 'frozen'):
                cache.publish(decision, z[turn - 1], uniforms(turn))
                refreshed = {workload.roots[i] for i in decision.indices}
                last_release_turn[list(decision.indices)] = turn
                age_refreshes += int(decision.trigger in ('age_force', 'both'))
            changed = data.significant_roots(previous, z[turn - 1], h, workload.roots,
                                              config['significant_delta'], config.get('significant_tolerance', 1e-12))
            if changed:
                significant += 1
                missed += int(not changed <= refreshed)
            previous = z[turn - 1]
            if detailed:
                logs.append({'turn': turn, 'decision': asdict(decision), 'before': before, 'after': cache.snapshot()})
        est = workload.from_cache(cache.cache)
        target = float(est[workload.target])
        estimates.append(target)
        errors.append(abs(target - truth[workload.target][turn - 1]))
        protected_errors.append(abs(target - protected[workload.target][turn - 1]))
        per_node = np.array([abs(float(est[node]) - truth[node][turn - 1]) for node in workload.nodes]) / scales
        workload_errors.append(float(weights @ per_node))
        for node in queries[turn]:
            if node in workload.nodes:
                query_errors.append(float(per_node[workload.nodes.index(node)]))
        e = np.clip(cache.cache, 0, 1) - z[turn - 1]
        Js.append(float(e @ M @ e))
        aois.append(float(np.mean((turn - last_release_turn)[cache.active])))
        project.append(float(np.mean(((cache.cache < 0) | (cache.cache > 1))[cache.active])))
        if detailed:
            snapshots.append({'turn': turn, **cache.snapshot(), 'target_estimate': target})
    classes = workload.classes(np.array(estimates))
    true_classes = workload.classes(truth[workload.target])
    stop = cache.stop_turn
    post = errors[stop:] if stop is not None else []
    published = [value for release in cache.transcript for value in release['values']]
    metrics = {'subject_id': rec['subject_id'], 'trajectory_id': rec['trajectory_id'], 'template': rec['template'],
               'method': method, 'parameter': parameter, 'B': float(B), 'repeat': repeat,
               'mae': float(np.mean(errors)), 'mae_protected': float(np.mean(protected_errors)),
               'p95_abs_error': float(np.quantile(errors, .95)),
               'input_clipping_mae': float(np.mean(np.abs(truth[workload.target] - protected[workload.target]))),
               'input_outside_root_rounds': int(((raw < workload.lo) | (raw > workload.hi)).sum()),
               'query_loss': float(np.mean(query_errors)) if query_errors else None,
               'workload_loss': float(np.mean(workload_errors)), 'J': float(np.mean(Js)),
               'class_error': float(np.mean(classes != true_classes)) if classes is not None else None,
               'spent': cache.spent, 'remaining_fraction': cache.remaining / B,
               'refresh_count': cache.refresh_count, 'age_refreshes': age_refreshes,
               'significant_events': significant, 'missed_events': missed, 'stop_turn': stop,
               'post_stop_mae': float(np.mean(post)) if post else None,
               'aoi_mean': float(np.mean(aois)), 'cache_projection_fraction': float(np.mean(project))}
    metrics.update(publication_projection_fraction=float(np.mean([(v < 0 or v > 1) for v in published])),
                   input_clipping_fraction=float(((raw < workload.lo) | (raw > workload.hi)).mean()),
                   early_budget_freeze=int(method != 'B0' and stop is not None and stop < rec['T']),
                   root_release_count=len(published))
    for reason in ['gate', 'age_force', 'both', 'periodic', 'none', 'empty', 'frozen']:
        metrics['reason_' + reason] = reasons[reason]
    assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, float))
    return metrics, {'decisions': logs, 'rounds': snapshots, 'transcript': cache.transcript}


def run(config, output, data_directory, method_document=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = data.load(data_directory, config['templates'])
    assignment = data.split(records, config['seed'], *config['split_fractions'])
    write_json(output / 'splits.json', assignment)
    budgets = config.get('budgets') or grid(**config['budget_grid'])
    config = dict(config, budgets=budgets)
    if not config['repeats'] >= 1 or config['bootstrap'] < 20:
        raise ValueError('Need positive repeats and >=20 bootstrap draws')
    workloads, statistics = {}, {}
    for template in config['templates']:
        for part in ['train', 'validation', 'test']:
            if not data.partition(records, assignment, part, template):
                raise ValueError(f'{template}: no {part} patients; use a larger cohort or omit this auxiliary template')
        w = Workload(template, config.get('domains', {}).get(template))
        workloads[template] = w
        statistics[template] = public_stats.fit(data.partition(records, assignment, 'train', template), w,
                                                 config['quantiles'], config.get('workload_scales', {}).get(template))
        statistics[template]['domains'] = dict(zip(w.roots, zip(w.lo.tolist(), w.hi.tolist())))
    write_json(output / 'public_stats.json', statistics)
    write_json(output / 'config.json', config)
    rows, finished = [], set()
    metrics_path, log_path = output / 'metrics.jsonl', output / 'ledger.jsonl'
    with metrics_path.open('w') as mf, log_path.open('w') as lf:
        def execute(part, template, method, parameter, values):
            for B in values:
                key = (part, template, method, parameter, B)
                if key in finished:
                    continue
                for rec in data.partition(records, assignment, part, template):
                    assert not rec['subject_id'] in statistics[template]['train_subject_ids']
                    if config.get('batch_repeats', False):
                        from .batch import simulate_repeats
                        results = simulate_repeats(rec, workloads[template], statistics[template], method,
                                                   parameter, B, config['repeats'], config)
                    else:
                        results = [simulate(rec, workloads[template], statistics[template], method, parameter,
                                            B, repeat, config, detailed=config.get('save_logs', True))
                                   for repeat in range(config['repeats'])]
                    for repeat, (metric, log) in enumerate(results):
                        metric['split'] = part
                        rows.append(metric)
                        mf.write(json.dumps(metric, allow_nan=False) + '\n')
                        if config.get('save_logs', True) and repeat < config.get('log_repeats', config['repeats']):
                            lf.write(json.dumps({'split': part, 'trajectory_id': rec['trajectory_id'], 'method': method,
                                                 'parameter': parameter, 'B': B, 'repeat': repeat, **log}, allow_nan=False) + '\n')
                finished.add(key)
            mf.flush()
            lf.flush()
            print(f'{part}/{template}/{method}/{parameter}: {len(values)} budgets complete', flush=True)

        def candidates():
            yield 'B0', 'static'
            for method in ['B2', 'B2p']:
                for period in config['periods']:
                    yield method, f'P={period}'
            for q in config['quantiles']:
                yield 'M1-a', f'q={q}'
                yield 'M1-g', f'q={q}'

        for template in config['templates']:
            for method, parameter in candidates():
                execute('validation', template, method, parameter, budgets)
            print(f'{template}: validation scan complete', flush=True)
        val = summaries(rows, config['bootstrap'], config['seed'])
        targets = config['targets']
        # Optional Step-0-derived targets use validation ONLY, frozen before test.
        if not targets:
            targets = {}
            for template in config['templates']:
                rr = [r for r in val if r['template'] == template and r['method'] == 'B2']
                targets[template] = []
                for b in config['target_reference_budgets']:
                    nearest = min(budgets, key=lambda v: abs(v - b))
                    targets[template].append(float(min(r['mean'] for r in rr if r['B'] == nearest)))
                targets[template] = sorted(set(targets[template]))
        selections = {}

        def choose():
            selected = {}
            for template in config['templates']:
                selected[template] = {}
                for tau in targets[template]:
                    chosen = {method: select(val, template, method, tau) for method in ['B0', 'B2', 'B2p', 'M1-a']}
                    # Age ablation uses the SAME q, not a separately tuned variant.
                    param = chosen['M1-a']['parameter']
                    age_curve = [r for r in val if r['template'] == template and r['method'] == 'M1-g' and r['parameter'] == param]
                    b = first_reached(age_curve, tau)
                    chosen['M1-g'] = {'parameter': param, 'selection_source': 'M1-a',
                                      'validation_B_min': b, 'validation_reached': b is not None}
                    selected[template][str(tau)] = chosen
            return selected

        selections = choose()
        # Predeclared one-step refinement. Split each coarse interval geometrically;
        # candidates are selected only from validation; refined grid is common to test.
        if config.get('refine', False):
            crossings = []
            for template in config['templates']:
                for tau in targets[template]:
                    for method in ['B2p', 'M1-a']:
                        chosen = selections[template][str(tau)][method]
                        b = chosen['validation_B_min']
                        crossings.append(b)
            extra = refinement_points(budgets, crossings)
            for template in config['templates']:
                for method, parameter in candidates():
                    execute('validation', template, method, parameter, extra)
                print(f'{template}: refinement complete ({len(extra)} additional budgets)', flush=True)
            budgets = sorted(set(budgets) | set(extra))
            config['budgets'] = budgets
            val = summaries(rows, config['bootstrap'], config['seed'])
            selections = choose()
        write_json(output / 'selections.json', selections)
        write_json(output / 'selection_stability.json', selection_stability(val, targets))
        write_json(output / 'targets.json', targets)
        for template in config['templates']:
            selected = {(method, sel['parameter']) for cc in selections[template].values() for method, sel in cc.items()}
            for method, parameter in sorted(selected):
                execute('test', template, method, parameter, budgets)
            print(f'{template}: selected test runs complete', flush=True)
    source_files = sorted(Path(__file__).parent.rglob('*.py'))
    source_files += [ROOT / p for p in ['data_mine/finite_trajectories.py', 'data_mine/build_datasets.py',
                                      'data_mine/build_meld_na.py', 'utils/risk_class.py']]
    method_path = Path(method_document or '/home/方法整体描述.md')
    metadata = {'config': config, 'targets': targets, 'selections': selections,
                'status': 'pilot_internal_validation_not_paper_claim', 'formula_version': FORMULA_VERSION,
                'privacy': {'unit': 'single user trajectory conditional on public metadata', 'metric': 'max_t max_i |dz_i|',
                            'epsilon_unit': 'standard_normalized_Laplace', 'delta': 0,
                            'root_alloc_B2': 'sqrt_h', 'root_alloc_B2p_M1': 'sqrt_h_times_soft_quota'},
                'input_sha256': data.input_hashes(data_directory, config['templates']),
                'dataset_manifest_sha256': hashlib.sha256((Path(data_directory) / 'manifest.json').read_bytes()).hexdigest(),
                'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                'method_document_sha256': hashlib.sha256(method_path.read_bytes()).hexdigest() if method_path.exists() else None,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'patient_counts': {t: {part: len(data.partition(records, assignment, part, t)) for part in
                                           ['train', 'validation', 'test']} for t in config['templates']},
                'logs_contain_local_evaluation_diagnostics': True}
    write_json(output / 'metadata.json', metadata)
    ss = summaries(rows, config['bootstrap'], config['seed'])
    ff = frontiers(ss, selections, targets)
    write_json(output / 'summary.json', ss)
    csv_rows(output / 'frontiers.csv', ff)
    write_json(output / 'config.json', config)
    if config.get('plots', False):
        plots(output, ss, ff)
    print(json.dumps({'output': str(output), 'runs': len(rows), 'patient_counts': metadata['patient_counts']}, indent=2), flush=True)
    return metadata


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/pilot.json')
    p.add_argument('--data', type=Path, default=ROOT / 'data_mine/finite_cache')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method-document', type=Path, default=Path('/home/方法整体描述.md'))
    a = p.parse_args()
    run(json.loads(a.config.read_text()), a.output, a.data, a.method_document)


if __name__ == '__main__':
    main()
