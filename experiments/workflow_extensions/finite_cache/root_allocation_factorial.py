"""Frozen T x root-package x allocation factorial replays, no gate changes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from . import data
from .analyze import write_json
from .matched_timing import action_uniforms, digest, make_cache
from .mechanisms import laplace_standard
from .pure_timing import Action, M1_TIME, UNIFORM_TIME, replay_actions
from .run import ROOT
from .workload import Workload

TIMES = ('M1_time', 'uniform_time')
ROOTS = ('M1_roots', 'uniform_roots')
ALLOCS = ('quota', 'sqrt_h')
CELLS = tuple(product(TIMES, ROOTS, ALLOCS))


def label(cell):
    return '|'.join(cell)


def policy_config(config, allocation):
    cfg = dict(config, mechanism=dict(config['mechanism']))
    if allocation == 'sqrt_h':
        cfg['mechanism']['gamma'] = 0.
    elif allocation != 'quota':
        raise ValueError('Unknown allocation')
    return cfg


def allocate_package(rec, w, stats, parameter, B, config, root_packages, totals, allocation):
    """One independent soft-quota ledger per S/A, independent of truth/time/noise.

    Dummy publication values are never used for evaluation. Only the original
    Cache allocator/ledger run here. Both time levels replay these same charges.
    """
    cfg = policy_config(config, allocation)
    cache = make_cache(rec, w, stats, parameter, B, cfg)
    k = len(w.roots)
    cache.initialize(np.zeros(k), np.full(k, .5))
    assert tuple(cache.transcript[0]['roots']) == tuple(root_packages[0])
    assert abs(sum(cache.transcript[0]['eps']) - totals[0]) < 1e-10
    epsilons = [tuple(cache.transcript[0]['eps'])]
    ledger = [{'action_index': 0, 'roots': list(root_packages[0]), 'eps': list(epsilons[0]),
               'total_epsilon': float(sum(epsilons[0])), 'root_remaining_after': cache.root_remaining.tolist()}]
    for j, (roots, total) in enumerate(zip(root_packages[1:], totals[1:]), 1):
        idx = [w.roots.index(r) for r in roots]
        assert idx and len(set(idx)) == len(idx) and all(cache.active[i] for i in idx)
        remaining = cache.root_remaining.copy()
        ratio = np.maximum(remaining[idx] / cache.quota[idx], cache.q_min)
        weights = np.sqrt(cache.h[idx]) * ratio ** cache.gamma
        eps = cache.allocate(idx, total)
        cache._publish(j + 1, idx, eps, np.zeros(k), np.full(k, .5))
        cache.refresh_count += 1
        assert abs(float(eps.sum()) - total) < 1e-10
        epsilons.append(tuple(float(e) for e in eps))
        ledger.append({'action_index': j, 'roots': list(roots), 'eps': eps.tolist(),
                       'total_epsilon': float(eps.sum()), 'gamma': cache.gamma,
                       'root_remaining_before': remaining.tolist(), 'soft_quota_ratios': ratio.tolist(),
                       'allocation_weights': weights.tolist(), 'root_remaining_after': cache.root_remaining.tolist()})
    assert abs(math.fsum(e for row in epsilons for e in row) - math.fsum(totals)) < 1e-10
    return epsilons, ledger, cfg


def make_actions(rec, w, B, repeat, config, root_packages, epsilons):
    result = []
    for j, (roots, eps) in enumerate(zip(root_packages, epsilons)):
        units = laplace_standard(action_uniforms(config['seed'], rec['subject_id'], rec['trajectory_id'],
                                                 B, repeat, j, w.roots))
        result.append(Action(j, tuple(roots), tuple(eps), tuple(float(units[w.roots.index(r)]) for r in roots)))
    return tuple(result)


def replay_cube(rec, w, stats, parameter, B, repeat, config, pure, matched):
    original = pure[M1_TIME]['transcript']
    old_uniform = matched['matched_periodic']['transcript']
    calendars = {'M1_time': tuple(r['turn'] for r in original),
                 'uniform_time': tuple(r['turn'] for r in old_uniform)}
    assert calendars['uniform_time'] == tuple(r['turn'] for r in pure[UNIFORM_TIME]['transcript'])
    roots = {'M1_roots': tuple(tuple(r['roots']) for r in original),
             'uniform_roots': tuple(tuple(r['roots']) for r in old_uniform)}
    totals = tuple(float(sum(r['eps'])) for r in original)
    assert all(abs(total - sum(r['eps'])) < 1e-10 for total, r in zip(totals, old_uniform))
    assert len(original) == len(old_uniform) and roots['M1_roots'][0] == roots['uniform_roots'][0]
    common = [set(a) & set(b) for a, b in zip(roots['M1_roots'], roots['uniform_roots'])]
    unions = [set(a) | set(b) for a, b in zip(roots['M1_roots'], roots['uniform_roots'])]
    key = {k: pure[k] for k in ['subject_id', 'trajectory_id', 'template', 'B', 'repeat', 'parameter']}
    rows, logs = [], []
    allocations = {}
    for s, a in product(ROOTS, ALLOCS):
        allocations[(s, a)] = allocate_package(rec, w, stats, parameter, B, config, roots[s], totals, a)
    for t, s, a in CELLS:
        epsilon, ledger, cfg = allocations[(s, a)]
        actions = make_actions(rec, w, B, repeat, config, roots[s], epsilon)
        metric, log = replay_actions(rec, w, stats, parameter, B, repeat, cfg, actions, calendars[t], label((t, s, a)))
        assert abs(metric['spent'] - math.fsum(totals)) < 1e-10 and metric['spent'] <= B + 1e-10
        metric.update(T=t, S=s, A=a,
                      root_set_size_mean=float(np.mean([len(r) for r in roots[s][1:]])) if len(actions) > 1 else None,
                      root_jaccard_mean=float(np.mean([len(c) / len(u) for c, u in zip(common[1:], unions[1:])])) if len(actions) > 1 else None,
                      common_root_fraction=float(np.mean([len(c) / len(r) for c, r in zip(common[1:], roots[s][1:])])) if len(actions) > 1 else None)
        for root in w.roots:
            metric['root_releases__' + root] = sum(root in r for r in roots[s][1:])
            metric['root_total_epsilon__' + root] = math.fsum(e for action in actions
                                                             for r, e in zip(action.roots, action.eps) if r == root)
        for j, (release, entry) in enumerate(zip(log['transcript'], ledger)):
            assert release['eps'] == entry['eps'] and release['roots'] == entry['roots']
            assert abs(sum(release['eps']) - totals[j]) < 1e-10
            release['allocation_ledger'] = entry
            release['common_roots_between_S_levels'] = sorted(common[j])
            release['root_jaccard_between_S_levels'] = len(common[j]) / len(unions[j])
        rows.append(metric)
        logs.append(dict(key, T=t, S=s, A=a, method=label((t, s, a)),
                         fixed_action_totals=totals, transcript=log['transcript'], changes=log['changes']))
    # Common root noises are exactly shared across T/S/A; A changes noise/epsilon.
    units = {}
    initial = logs[0]['transcript'][0]
    for row in logs:
        init = row['transcript'][0]
        for field in ['turn', 'roots', 'eps', 'values', 'action']:
            assert init[field] == initial[field]
        for j, release in enumerate(row['transcript']):
            for root, unit in zip(release['roots'], release['action']['unit_noise']):
                key_ = j, root
                if key_ in units:
                    assert units[key_] == unit
                else:
                    units[key_] = unit
    references = {('M1_time', 'M1_roots', 'quota'): original,
                  ('uniform_time', 'M1_roots', 'quota'): pure[UNIFORM_TIME]['transcript'],
                  ('uniform_time', 'uniform_roots', 'quota'): old_uniform}
    max_eps = max_value = 0.
    for row in logs:
        cell = row['T'], row['S'], row['A']
        if cell in references:
            for actual, old in zip(row['transcript'], references[cell]):
                assert actual['turn'] == old['turn'] and actual['roots'] == old['roots']
                difference = float(np.max(np.abs(np.array(actual['eps']) - old['eps'])))
                value_difference = float(np.max(np.abs(np.array(actual['values']) - old['values'])))
                max_eps, max_value = max(max_eps, difference), max(max_value, value_difference)
                assert difference < 1e-10 and value_difference < 1e-10
    return rows, logs, {'max_reference_epsilon_difference': max_eps, 'max_reference_value_difference': max_value}


def code_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}


def run(pure_source, matched_source, output, data_directory, plan, repeats, precheck=None):
    start = time.monotonic()
    pure_source, matched_source, output = map(lambda p: Path(p).resolve(), [pure_source, matched_source, output])
    source_files = {'pure': (pure_source, ['protocol.json', 'audit.json', 'pure_timing_metrics.jsonl', 'pure_timing_schedules.jsonl',
                                         'frozen_config.json', 'frozen_public_stats.json', 'frozen_splits.json', 'frozen_selections.json', 'frozen_metadata.json']),
                    'matched': (matched_source, ['protocol.json', 'audit.json', 'matched_metrics.jsonl', 'matched_schedules.jsonl'])}
    frozen = {name: {n: digest(p / n) for n in files} for name, (p, files) in source_files.items()}
    prior = json.loads((pure_source / 'protocol.json').read_text())
    old_audit = json.loads((pure_source / 'audit.json').read_text())
    assert old_audit['passed'] and prior['repeats'] == 50
    assert plan['primary_budgets'] == prior['primary_budgets'] and plan['availability'] == prior['availability']
    assert repeats in (2, plan['repeats']) and plan['bootstrap'] == 2000
    code = code_hashes()
    for name in ['mechanisms.py', 'pure_timing.py', 'matched_timing.py', 'workload.py', 'data.py']:
        key = str(Path(__file__).with_name(name).relative_to(ROOT))
        assert code[key] == old_audit['code_sha256'][key]
    config = json.loads((pure_source / 'frozen_config.json').read_text())
    stats = json.loads((pure_source / 'frozen_public_stats.json').read_text())
    metadata = json.loads((pure_source / 'frozen_metadata.json').read_text())
    assert data.input_hashes(data_directory, ['MELD_NA']) == metadata['input_sha256']
    if repeats == 2:
        tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s',
                                'experiments/workflow_extensions/finite_cache/tests', '-t', '.', '-v'],
                               cwd=ROOT, capture_output=True, text=True)
        if tests.returncode:
            raise RuntimeError(tests.stdout + tests.stderr)
        test_log = tests.stdout + tests.stderr
        budgets = plan['precheck_budgets']
    else:
        if precheck is None:
            raise ValueError('Formal run requires successful R=2 precheck')
        old = json.loads((Path(precheck) / 'audit.json').read_text())
        assert old['passed'] and old['repeats'] == 2 and old['source_sha256'] == frozen
        assert old['code_sha256'] == code and old['plan'] == plan and old['unit_tests_passed']
        test_log = (Path(precheck) / 'unit_tests.txt').read_text()
        budgets = plan['primary_budgets']
    def case_key(r):
        return r['trajectory_id'], r['B'], r['repeat']
    pure_logs = {case_key(r): r for r in map(json.loads, (pure_source / 'pure_timing_schedules.jsonl').open())
                 if r['B'] in budgets and r['repeat'] < repeats}
    matched_logs = {case_key(r): r for r in map(json.loads, (matched_source / 'matched_schedules.jsonl').open())
                    if case_key(r) in pure_logs}
    pure_metrics = {case_key(r) + (r['method'],): r for r in map(json.loads, (pure_source / 'pure_timing_metrics.jsonl').open())}
    matched_metrics = {case_key(r) + (r['method'],): r for r in map(json.loads, (matched_source / 'matched_metrics.jsonl').open())}
    records = {r['trajectory_id']: r for r in data.load(data_directory, ['MELD_NA'])}
    w = Workload('MELD_NA', stats['MELD_NA']['domains'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'unit_tests.txt').write_text(test_log)
    protocol = dict(plan, repeats=repeats, executed_budgets=budgets, seed=config['seed'],
                    pure_source=str(pure_source), matched_source=str(matched_source), precheck=str(precheck) if precheck else None,
                    noise_key='seed,subject,trajectory,B,repeat,action_index,root',
                    primary_scope='patient_equal_mean_of_prespecified_budget_mean; noise_R_not_independent_patients',
                    conditional_intervals='pointwise nominal95; no test-selected effect or budget',
                    direct_original_gap='Y(M1_time,M1_roots,quota)-Y(uniform_time,uniform_roots,quota)',
                    direct_attribution='two-factor T/S Shapley at fixed quota; direct A contribution=0',
                    three_factor_reference='(uniform_time,uniform_roots,sqrt_h)',
                    shapley_difference='phi(full_111)-phi(baseline_001); sums ORIGINAL gap; A is context interaction, not an endpoint switch',
                    long_term='omit one release, hold all later packages and realized inputs fixed; effects not additive')
    write_json(output / 'protocol.json', protocol)
    audit = {'passed': False, 'repeats': repeats, 'plan': plan, 'code_sha256': code, 'source_sha256': frozen,
             'unit_tests_passed': True, 'cases': 0, 'combinations': 0, 'max_total_epsilon_difference': 0.,
             'max_reference_epsilon_difference': 0., 'max_reference_value_difference': 0., 'max_reference_metric_difference': 0.,
             'max_budget_ratio': 0., 'all_initializations_exact': True, 'common_root_unit_noise_exact': True,
             'independent_soft_quota_ledgers': True, 'strict_increasing_legal_latest_observed_times': True}
    rows = []
    with (output / 'factorial_metrics.jsonl').open('w') as mf, (output / 'factorial_actions.jsonl').open('w') as af:
        for key, pure in sorted(pure_logs.items()):
            tid, B, repeat = key
            rec = records[tid]
            rr, logs, check = replay_cube(rec, w, stats['MELD_NA'], pure['parameter'], B, repeat, config,
                                           pure, matched_logs[key])
            for field, value in check.items():
                audit[field] = max(audit[field], value)
            reference_keys = {('M1_time', 'M1_roots', 'quota'): pure_metrics[key + (M1_TIME,)],
                              ('uniform_time', 'M1_roots', 'quota'): pure_metrics[key + (UNIFORM_TIME,)],
                              ('uniform_time', 'uniform_roots', 'quota'): matched_metrics[key + ('matched_periodic',)]}
            for r, log in zip(rr, logs):
                if (r['T'], r['S'], r['A']) in reference_keys:
                    old = reference_keys[(r['T'], r['S'], r['A'])]
                    for field in ['mae', 'p95_abs_error', 'workload_loss', 'query_loss', 'spent', 'aoi_mean',
                                  'publication_projection_fraction', 'cache_projection_fraction']:
                        d = abs(r[field] - old[field])
                        assert d < 1e-10, field
                        audit['max_reference_metric_difference'] = max(audit['max_reference_metric_difference'], d)
                audit['max_total_epsilon_difference'] = max(audit['max_total_epsilon_difference'], abs(r['spent'] - rr[0]['spent']))
                audit['max_budget_ratio'] = max(audit['max_budget_ratio'], r['spent'] / B)
                rows.append(r)
                mf.write(json.dumps(r, allow_nan=False) + '\n')
                af.write(json.dumps(log, allow_nan=False) + '\n')
            audit['cases'] += 1
            audit['combinations'] += len(rr)
            if repeat == repeats - 1:
                mf.flush(); af.flush()
                print(f'{tid}/B={B:g}: all 8 combinations, R={repeats} complete', flush=True)
    from .root_allocation_effects import analyze
    analyze(rows, output, protocol)
    assert code == code_hashes()
    assert frozen == {name: {n: digest(p / n) for n in files} for name, (p, files) in source_files.items()}
    audit.update(passed=True, source_outputs_unchanged=True, code_unchanged_during_run=True,
                 elapsed_seconds=time.monotonic() - start)
    write_json(output / 'audit.json', audit)
    print(json.dumps({'output': str(output), 'audit_passed': True, 'cases': audit['cases'],
                      'combinations': audit['combinations'], 'elapsed_seconds': audit['elapsed_seconds']}), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--pure-source', type=Path, required=True)
    p.add_argument('--matched-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data_mine/finite_cache')
    p.add_argument('--plan', type=Path, default=Path(__file__).with_name('configs') / 'root_allocation_factorial.json')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--precheck', type=Path)
    a = p.parse_args()
    run(a.pure_source, a.matched_source, a.output, a.data, json.loads(a.plan.read_text()), a.repeats, a.precheck)


if __name__ == '__main__':
    main()
