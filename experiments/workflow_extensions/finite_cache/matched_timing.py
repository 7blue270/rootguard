"""Frozen, action-paired M1-a versus equal-count/equal-cost uniform timing.

The comparator is an OFFLINE diagnostic: its K and ordered costs come from M1,
and its legal-opportunity calendar is visible in full. M1 itself stays causal.
No fitting, parameter selection or frontier search is performed here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

from . import data
from .analyze import write_json
from .mechanisms import Cache, laplace_standard
from .run import ROOT
from .workload import FORMULA_VERSION, Workload


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action_uniforms(seed, subject, trajectory, B, repeat, action_index, roots):
    """One stream per action AND named root, independent of turn/root subset."""
    result = []
    for root in roots:
        key = json.dumps([seed, subject, trajectory, float(B), repeat, action_index, root],
                         separators=(',', ':'))
        value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], 'little')
        result.append(np.random.default_rng(value).random())
    return np.clip(result, np.finfo(float).eps, 1 - np.finfo(float).eps)


def uniform_turns(legal_turns, K, T):
    """Globally closest K distinct legal turns to evenly spaced dialogue targets.

    Dynamic programming uses squared distances and lexicographically earlier
    calendars to break ties. Initialization at turn 1 is separate. Targets divide
    [1,T] into K+1 gaps; all future public opportunities are allowed to comparator.
    """
    turns = tuple(sorted(legal_turns))
    if len(set(turns)) != len(turns) or any(t <= 1 or t > T for t in turns):
        raise ValueError('Legal turns must be distinct and after initialization')
    if not 0 <= K <= len(turns):
        raise ValueError('Cannot place the requested number of releases')
    states = {0: (0, ())}
    for turn in turns:
        for j in range(min(K, len(turns)), 0, -1):
            if j - 1 in states:
                old_cost, old_path = states[j - 1]
                # Integer objective avoids floating tie-breaking differences.
                cost = old_cost + ((turn - 1) * (K + 1) - j * (T - 1)) ** 2
                candidate = cost, old_path + (turn,)
                if j not in states or candidate < states[j]:
                    states[j] = candidate
    result = states[K][1]
    assert len(result) == K and len(set(result)) == K
    return result


def make_cache(rec, w, stats, parameter, B, config):
    q = parameter.split('=')[1]
    return Cache(w.roots, stats['h'], stats['C'][q], B, 'M1-a', rec['T'] - 1,
                 period=config.get('fixed_gate_period', 4), **config['mechanism'])


def run_method(rec, w, stats, parameter, B, repeat, config, reference=None):
    """Actual numerical releases through Cache.publish, followed by evaluation."""
    cache = make_cache(rec, w, stats, parameter, B, config)
    raw = data.replay(rec)  # Trusted offline evaluator/publication only.
    z, T = w.normalize(raw), rec['T']
    truth = w.evaluate(raw)
    k = len(w.roots)
    identity = [config['seed'], rec['subject_id'], rec['trajectory_id'], B, repeat]
    uniforms = lambda action: action_uniforms(*identity, action, w.roots)
    cache.initialize(z[0], uniforms(0))
    active = set(np.array(w.roots)[cache.active])
    # Future public calendar is used ONLY by the offline comparator.
    selected = None
    if reference is not None:
        legal = [e['turn'] for e in rec['events'] if active.intersection(e['measured_roots'])]
        turns = uniform_turns(legal, len(reference) - 1, T)
        assert turns == uniform_turns(legal, len(reference) - 1, T)
        selected = {t: float(sum(r['eps'])) for t, r in zip(turns, reference[1:])}
    events = {e['turn']: e for e in rec['events']}
    histories = np.zeros((T, k))
    before_releases, actions, reasons = [], [], []
    last_turn = np.ones(k, int)
    aoi, per_root_aoi = [], []
    significant, missed, changes = 0, 0, []
    previous = z[0]
    action = 0
    decisions = []
    for turn in range(1, T + 1):
        refreshed = set()
        if turn in events:
            batch = data.public_batch(events[turn])
            assert isinstance(batch, data.PublicBatch) and not hasattr(batch, 'values')
            # No true state, event list, future calendar or reference enters M1.decide.
            decision = (cache.decide(batch) if selected is None else
                        cache.decide_scheduled(batch, selected.get(turn)))
            reasons.append(decision.trigger)
            decisions.append(asdict(decision))
            if decision.trigger not in ('none', 'empty', 'frozen'):
                before_releases.append(cache.cache.copy())
                action += 1
                cache.publish(decision, z[turn - 1], uniforms(action))
                refreshed = {w.roots[i] for i in decision.indices}
                assert refreshed <= set(batch.measured_roots)
                last_turn[list(decision.indices)] = turn
                actions.append(turn)
            changed = data.significant_roots(previous, z[turn - 1], stats['h'], w.roots,
                                              config['significant_delta'],
                                              config.get('significant_tolerance', 1e-12))
            if changed:
                significant += 1
                missed += int(not changed <= refreshed)
                changes.append({'turn': turn, 'roots': sorted(changed)})
            previous = z[turn - 1]
        histories[turn - 1] = cache.cache
        root_age = turn - last_turn
        per_root_aoi.append(root_age)
        aoi.append(float(root_age[cache.active].mean()))
    estimates = w.from_cache(histories)
    errors = np.abs(estimates[w.target] - truth[w.target])
    scales, weights = np.asarray(stats['scales']), np.asarray(stats['weights'])
    node_errors = np.stack([np.abs(estimates[n] - truth[n]) for n in w.nodes], axis=-1)
    scaled = node_errors / scales
    query_errors = [scaled[q['turn'] - 1, w.nodes.index(q['node'])]
                    for q in rec['queries'] if q['node'] in w.nodes]
    transcript = cache.transcript
    total = math.fsum(e for r in transcript for e in r['eps'])
    assert abs(total - cache.spent) < 1e-10 and total <= B + 1e-10
    projected = [v < 0 or v > 1 for r in transcript for v in r['values']]
    # Immediate error decrease holds truth fixed to the CURRENT release turn.
    gains, scaled_gains = [], []
    for old_cache, release in zip(before_releases, transcript[1:]):
        turn = release['turn']
        before = w.from_cache(old_cache)
        gains.append(float(abs(before[w.target] - truth[w.target][turn - 1]) - errors[turn - 1]))
        err_before = np.array([abs(before[n] - truth[n][turn - 1]) for n in w.nodes]) / scales
        scaled_gains.append(float(weights @ (err_before - scaled[turn - 1])))
    changes_turns = [e['turn'] for e in changes]
    wait_any, wait_roots = [], []
    for change in changes:
        t = change['turn']
        following = [r['turn'] for r in transcript[1:] if r['turn'] >= t]
        nxt = min(following) if following else None
        root_next = {root: min((r['turn'] for r in transcript[1:]
                              if r['turn'] >= t and root in r['roots']), default=None)
                     for root in change['roots']}
        complete = max(root_next.values()) if all(v is not None for v in root_next.values()) else None
        def waiting(n):
            return {'observed_wait': n - t if n is not None else None,
                    'restricted_wait': (n if n is not None else T + 1) - t,
                    'censored': n is None}
        wait_any.append(waiting(nxt))
        wait_roots.append(waiting(complete))
        change.update(next_refresh_turn=nxt, next_root_refresh_turns=root_next,
                      next_complete_refresh_turn=complete)
    frac = lambda vs: float(np.mean(vs)) if len(vs) else None
    refresh_eps = math.fsum(e for r in transcript[1:] for e in r['eps'])
    for i, release in enumerate(transcript):
        release.update(action_index=i, epsilon_total=float(sum(release['eps'])),
                       unit_laplace={root: float(v) for root, v in
                                     zip(w.roots, laplace_standard(uniforms(i))) if root in release['roots']},
                       immediate_error_drop=gains[i - 1] if i else None,
                       immediate_scaled_drop=scaled_gains[i - 1] if i else None)
    metric = {
        'subject_id': rec['subject_id'], 'trajectory_id': rec['trajectory_id'], 'template': rec['template'],
        'method': 'M1-a' if reference is None else 'matched_periodic', 'parameter': parameter,
        'B': float(B), 'repeat': repeat, 'mae': float(errors.mean()),
        'p95_abs_error': float(np.quantile(errors, .95)),
        'query_loss': frac(query_errors), 'workload_loss': float((scaled @ weights).mean()),
        'spent': total, 'initial_spent': math.fsum(transcript[0]['eps']),
        'refresh_spent': refresh_eps, 'refresh_count': action,
        'miss_rate': missed / significant if significant else None,
        'missed_events': missed, 'significant_events': significant,
        'aoi_mean': frac(aoi), 'publication_projection_fraction': frac(projected),
        'cache_projection_fraction': float(np.mean(((histories < 0) | (histories > 1))[:, cache.active])),
        'refresh_near_change_fraction': frac([any(abs(t - c) <= 1 for c in changes_turns) for t in actions]),
        'refresh_after_change_fraction': frac([any(0 <= t - c <= 1 for c in changes_turns) for t in actions]),
        'refresh_before_change_fraction': frac([any(0 <= c - t <= 1 for c in changes_turns) for t in actions]),
        'immediate_error_drop_per_refresh': frac(gains),
        'immediate_error_drop_per_epsilon': sum(gains) / refresh_eps if refresh_eps else None,
        'immediate_scaled_drop_per_refresh': frac(scaled_gains),
        'immediate_scaled_drop_per_epsilon': sum(scaled_gains) / refresh_eps if refresh_eps else None,
        'budget_exhaustion_turn': next((r['turn'] for r in transcript if
                                       B - math.fsum(e for s in transcript[:r['action_index'] + 1]
                                                     for e in s['eps']) < config['mechanism'].get('eps_min', .01)), None),
        'stop_turn': cache.stop_turn, 'last_refresh_turn': actions[-1] if actions else None,
        'mean_refresh_turn': frac(actions), 'root_release_count': sum(len(r['roots']) for r in transcript),
    }
    for label, waiting in [('wait_any', wait_any), ('wait_roots', wait_roots)]:
        metric[label + '_observed_mean'] = frac([v['observed_wait'] for v in waiting if not v['censored']])
        metric[label + '_restricted_mean'] = frac([v['restricted_wait'] for v in waiting])
        metric[label + '_censored_fraction'] = frac([v['censored'] for v in waiting])
    for i, node in enumerate(w.nodes):
        metric['node_mae__' + node] = float(node_errors[:, i].mean())
        metric['node_scaled__' + node] = float(scaled[:, i].mean())
    for i, root in enumerate(w.roots):
        metric['aoi__' + root] = float(np.mean(per_root_aoi, axis=0)[i])
        metric['root_eps__' + root] = math.fsum(e for r in transcript for n, e in zip(r['roots'], r['eps']) if n == root)
        metric['root_releases__' + root] = sum(root in r['roots'] for r in transcript[1:])
    schedule = {'transcript': transcript, 'changes': changes, 'decisions': decisions,
                'legal_turns': [e['turn'] for e in rec['events'] if active.intersection(e['measured_roots'])],
                'stop_turn': cache.stop_turn, 'spent': total}
    return metric, schedule


def run_pair(rec, w, stats, parameter, B, repeat, config):
    m, a = run_method(rec, w, stats, parameter, B, repeat, config)
    p, b = run_method(rec, w, stats, parameter, B, repeat, config, a['transcript'])
    assert m['refresh_count'] == p['refresh_count']
    assert abs(m['spent'] - p['spent']) < 1e-10
    assert abs(m['initial_spent'] - p['initial_spent']) < 1e-10
    assert m['spent'] <= B + 1e-10 and p['spent'] <= B + 1e-10
    comparisons = []
    for i, (left, right) in enumerate(zip(a['transcript'], b['transcript'])):
        assert abs(sum(left['eps']) - sum(right['eps'])) < 1e-10
        common = set(left['roots']) & set(right['roots'])
        assert all(left['unit_laplace'][r] == right['unit_laplace'][r] for r in common)
        comparisons.append({'action_index': i, 'same_roots': set(left['roots']) == set(right['roots']),
                            'common_roots': sorted(common),
                            'root_jaccard': len(common) / len(set(left['roots']) | set(right['roots'])),
                            'm1_turn': left['turn'], 'periodic_turn': right['turn'],
                            'epsilon_difference': sum(left['eps']) - sum(right['eps'])})
    for metric in [m, p]:
        metric['paired_root_mismatch_fraction'] = (float(np.mean([not c['same_roots'] for c in comparisons[1:]]))
                                                    if len(comparisons) > 1 else None)
    return [m, p], {'subject_id': rec['subject_id'], 'trajectory_id': rec['trajectory_id'],
                   'template': rec['template'], 'B': B, 'repeat': repeat, 'parameter': parameter,
                   'M1-a': a, 'matched_periodic': b, 'comparisons': comparisons}


def legacy_schedule_index(source):
    result = {}
    with (source / 'ledger.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if r['split'] == 'test' and r['method'] == 'M1-a' and r['repeat'] == 0:
                result[(r['trajectory_id'], r['B'], r['parameter'])] = r['transcript']
    return result


def run(source, output, data_directory, plan, repeats, precheck=None):
    started = time.monotonic()
    source, output, data_directory = Path(source).resolve(), Path(output).resolve(), Path(data_directory)
    # Check frozen input and formula BEFORE creating results. Never write to source.
    required = ['config.json', 'metadata.json', 'splits.json', 'selections.json', 'public_stats.json', 'targets.json']
    frozen = {name: json.loads((source / name).read_text()) for name in required}
    config, metadata = frozen['config.json'], frozen['metadata.json']
    assert data.input_hashes(data_directory, config['templates']) == metadata['input_sha256']
    assert FORMULA_VERSION == metadata['formula_version']
    old_workload = metadata['code_sha256']['experiments/workflow_extensions/finite_cache/workload.py']
    assert digest(Path(__file__).with_name('workload.py')) == old_workload, 'Task implementation changed'
    assert set(plan['primary_budgets']) <= set(config['budgets'])
    if repeats < 2:
        raise ValueError('Need at least two noise repeats')
    if repeats > 2:
        if precheck is None:
            raise ValueError('Formal run requires a completed R=2 precheck directory')
        pre = json.loads((Path(precheck) / 'audit.json').read_text())
        assert pre['passed'] and pre['repeats'] == 2
        assert pre['source_sha256'] == {n: digest(source / n) for n in required}
        assert pre['plan'] == plan
        assert pre['code_sha256'] == {str(p.relative_to(ROOT)): digest(p) for p in
                                      sorted(Path(__file__).parent.rglob('*.py'))}
    records = data.load(data_directory, config['templates'])
    assignment = frozen['splits.json']  # No split() and no public_stats.fit().
    selections, statistics = frozen['selections.json'], frozen['public_stats.json']
    legacy = legacy_schedule_index(source)
    test_result = None
    if repeats == 2:
        test_result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
                                     'experiments/workflow_extensions/finite_cache/tests', '-t', '.', '-v'],
                                    cwd=ROOT, capture_output=True, text=True)
        if test_result.returncode:
            raise RuntimeError('Unit tests failed before precheck:\n' + test_result.stderr)
    output.mkdir(parents=True, exist_ok=False)
    if test_result is not None:
        (output / 'unit_tests.txt').write_text(test_result.stdout + test_result.stderr)
    elif precheck is not None:
        shutil.copyfile(Path(precheck) / 'unit_tests.txt', output / 'unit_tests.txt')
    for name in required:
        shutil.copyfile(source / name, output / ('frozen_' + name))
    protocol = dict(plan, source=str(source), repeats=repeats, budgets=config['budgets'],
                    bootstrap=plan['bootstrap'], precheck=str(precheck) if precheck else None,
                    paired_noise_key='seed,subject,trajectory,B,repeat,action_index,root',
                    initialization_action_index=0,
                    inference='two_sided_percentile_patient_paired_and_hierarchical_bootstrap',
                    unit='patients_equal; trajectories_equal_within_patient; noise_repeats_equal',
                    waiting_censoring='T+1 endpoint; report observed-only AND restricted waiting plus censor fraction',
                    per_budget_CI='pointwise_nominal; primary aggregate prespecified, no best-budget selection',
                    single_task_warning='MELD_NA only; intermediate nodes are dependent diagnostics, not multiple tasks')
    write_json(output / 'protocol.json', protocol)  # Written before any new evaluation.
    audit = {'passed': False, 'repeats': repeats, 'plan': plan,
             'source_sha256': {n: digest(source / n) for n in required},
             'input_sha256': data.input_hashes(data_directory, config['templates']),
             'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))},
             'pairs': 0, 'legacy_schedule_matches': 0, 'max_epsilon_difference': 0.,
             'max_action_epsilon_difference': 0., 'max_budget_ratio': 0., 'n_test_patients': {},
             'm1_decision_input': ['current_PublicBatch', 'public_statistics', 'own_past_ledger_and_ages'],
             'uniform_schedule_deterministic_and_unique': True, 'common_root_unit_noise_equal': True}
    audit['unit_tests_passed'] = True
    audit['unit_tests_log_sha256'] = digest(output / 'unit_tests.txt')
    rows = []
    with (output / 'matched_metrics.jsonl').open('w') as mf, (output / 'matched_schedules.jsonl').open('w') as sf:
        for template in config['templates']:
            parameters = sorted({s['M1-a']['parameter'] for s in selections[template].values()})
            w = Workload(template, statistics[template].get('domains'))
            test = data.partition(records, assignment, 'test', template)
            audit['n_test_patients'][template] = len({r['subject_id'] for r in test})
            for parameter in parameters:
                for B in config['budgets']:
                    for rec in test:
                        assert rec['subject_id'] not in statistics[template]['train_subject_ids']
                        for repeat in range(repeats):
                            pair, schedule = run_pair(rec, w, statistics[template], parameter, B, repeat, config)
                            if repeat == 0:
                                old = legacy[(rec['trajectory_id'], B, parameter)]
                                new = schedule['M1-a']['transcript']
                                assert len(old) == len(new)
                                for a, b in zip(old, new):
                                    assert a['turn'] == b['turn'] and a['roots'] == b['roots']
                                    np.testing.assert_allclose(a['eps'], b['eps'], rtol=0, atol=1e-12)
                                audit['legacy_schedule_matches'] += 1
                            audit['pairs'] += 1
                            audit['max_epsilon_difference'] = max(audit['max_epsilon_difference'], abs(pair[0]['spent'] - pair[1]['spent']))
                            audit['max_action_epsilon_difference'] = max(audit['max_action_epsilon_difference'],
                                                                       max(abs(c['epsilon_difference']) for c in schedule['comparisons']))
                            audit['max_budget_ratio'] = max(audit['max_budget_ratio'], *(r['spent'] / B for r in pair))
                            for row in pair:
                                rows.append(row)
                                mf.write(json.dumps(row, allow_nan=False) + '\n')
                            sf.write(json.dumps(schedule, allow_nan=False) + '\n')
                    mf.flush()
                    sf.flush()
                    print(f'{template}/{parameter}/B={B:g}: {repeats} paired repeats complete', flush=True)
    from .matched_analysis import analyze
    analyze(rows, output, protocol, config['seed'])
    assert audit['source_sha256'] == {n: digest(source / n) for n in required}
    audit.update(passed=True, elapsed_seconds=time.monotonic() - started,
                 code_unchanged_during_run=audit['code_sha256'] == {
                     str(p.relative_to(ROOT)): digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))})
    assert audit['code_unchanged_during_run']
    write_json(output / 'audit.json', audit)
    print(json.dumps({'output': str(output), 'pairs': audit['pairs'], 'audit_passed': True,
                      'elapsed_seconds': audit['elapsed_seconds']}), flush=True)
    return audit


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data_mine/finite_cache')
    p.add_argument('--plan', type=Path, default=Path(__file__).parent / 'configs/matched_timing.json')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--precheck', type=Path)
    a = p.parse_args()
    run(a.source, a.output, a.data, json.loads(a.plan.read_text()), a.repeats, a.precheck)


if __name__ == '__main__':
    main()
