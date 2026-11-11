"""Pure timing diagnostic: fixed ordered roots, epsilon and action/root noise.

M1 generation uses the unmodified causal policy. Both replays execute immutable
action packages through Cache._publish; neither reselects roots or reallocates
epsilon. Only the public release calendar and the latest observed inputs differ.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, asdict
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
from .matched_timing import action_uniforms, digest, make_cache, run_method, legacy_schedule_index
from .mechanisms import laplace_standard
from .run import ROOT
from .workload import FORMULA_VERSION, Workload

M1_TIME = 'm1_time_fixed_action'
UNIFORM_TIME = 'uniform_time_fixed_action'


@dataclass(frozen=True)
class Action:
    action_index: int
    roots: tuple[str, ...]
    eps: tuple[float, ...]
    unit_noise: tuple[float, ...]


def packages(transcript):
    result = []
    for i, r in enumerate(transcript):
        roots = tuple(r['roots'])
        assert len(set(roots)) == len(roots) and roots
        result.append(Action(i, roots, tuple(r['eps']), tuple(r['unit_laplace'][root] for root in roots)))
    return tuple(result)


def public_opportunities(rec):
    """Availability metadata ONLY: values are not read while constructing calendar."""
    observed = set(rec['initial_state'])
    result = []
    for event in rec['events']:
        observed.update(event['measured_roots'])
        result.append((event['turn'], frozenset(observed)))
    return tuple(result)


def match_actions(actions, opportunities, T):
    """Ordered injection minimizing sum_k (t_k-[1+k(T-1)/(K+1)])**2.

    Only root names, public observation availability and turns enter matching.
    Integer costs give deterministic ties. A failing whole pair is excluded;
    action packages are never truncated, reordered or altered to make it fit.
    """
    K = len(actions)
    turns = [t for t, _ in opportunities]
    if len(set(turns)) != len(turns) or turns != sorted(turns) or any(t <= 1 or t > T for t in turns):
        raise ValueError('Invalid public opportunities')
    candidates = [[t for t, roots in opportunities if set(a.roots) <= roots] for a in actions]
    states = {0: (0, ())}
    for turn, observed in opportunities:
        for j in range(K, 0, -1):
            if j - 1 in states and set(actions[j - 1].roots) <= observed:
                old, path = states[j - 1]
                proposed = old + ((turn - 1) * (K + 1) - j * (T - 1)) ** 2, path + (turn,)
                if j not in states or proposed < states[j]:
                    states[j] = proposed
    if K not in states:
        reason = ('insufficient_distinct_opportunities' if len(opportunities) < K else
                  'unobserved_action_roots' if any(not c for c in candidates) else
                  'no_order_preserving_complete_assignment')
        return None, {'reason': reason, 'K': K, 'candidate_counts': [len(c) for c in candidates],
                      'candidate_turns': candidates, 'longest_feasible_prefix': max(states)}
    cost, calendar = states[K]
    assert len(calendar) == K and len(set(calendar)) == K
    assert all(calendar[i] in candidates[i] for i in range(K))
    return calendar, {'integer_cost': cost, 'squared_cost': cost / (K + 1) ** 2,
                      'ideal_turns': [1 + i * (T - 1) / (K + 1) for i in range(1, K + 1)],
                      'candidate_counts': [len(c) for c in candidates]}


def replay_actions(rec, w, stats, parameter, B, repeat, config, actions, calendar, method):
    """One shared causal replayer for both calendars; no scheduling/allocation."""
    assert len(actions) == len(calendar) and calendar[0] == 1
    assert len(set(calendar)) == len(calendar) and list(calendar) == sorted(calendar)
    assert all(1 <= t <= rec['T'] for t in calendar)
    opportunities = dict(public_opportunities(rec))
    assert all(t in opportunities and set(a.roots) <= opportunities[t] for a, t in zip(actions[1:], calendar[1:]))
    cache = make_cache(rec, w, stats, parameter, B, config)
    events = {e['turn']: e for e in rec['events']}
    # Latest values are populated by streaming past/current observations only.
    state = dict(rec['initial_state'])
    source_turn = {r: 1 for r in state}
    identity = [config['seed'], rec['subject_id'], rec['trajectory_id'], B, repeat]
    schedule = dict(zip(calendar, actions))
    histories, before, latest_turns = [], [], []
    last_release = np.ones(len(w.roots), int)
    aois = []
    sources, spent_path = [], []
    for turn in range(1, rec['T'] + 1):
        if turn in events:
            state.update(events[turn]['values'])
            source_turn.update({r: turn for r in events[turn]['measured_roots']})
        if turn in schedule:
            a = schedule[turn]
            assert all(r in state and source_turn[r] <= turn for r in a.roots)
            z = w.normalize([state.get(r, w.lo[i]) for i, r in enumerate(w.roots)])
            uniforms = action_uniforms(*identity, a.action_index, w.roots)
            idx = [w.roots.index(r) for r in a.roots]
            np.testing.assert_array_equal(laplace_standard(uniforms)[idx], a.unit_noise)
            before.append(cache.cache.copy())
            if a.action_index == 0:
                cache.initialize(z, uniforms)
                assert tuple(cache.transcript[-1]['eps']) == a.eps
                assert tuple(cache.transcript[-1]['roots']) == a.roots
            else:
                # EXACT original per-root eps; no allocator / gate in either replay.
                cache._publish(turn, idx, a.eps, z, uniforms)
                cache.refresh_count += 1
                cache._check_stop(turn)
            release = cache.transcript[-1]
            release.update(action=asdict(a), source_turns={r: source_turn[r] for r in a.roots},
                           normalized_inputs={r: float(z[i]) for r, i in zip(a.roots, idx)})
            for i, root, noise, eps, value in zip(idx, a.roots, a.unit_noise, a.eps, release['values']):
                assert value == float(z[i] + noise / eps)
            last_release[idx] = turn
            sources.extend(release['source_turns'].values())
            spent_path.append(cache.spent)
        histories.append(cache.cache.copy())
        latest_turns.append([source_turn.get(r, 0) for r in w.roots])
        aois.append(float((turn - last_release)[cache.active].mean()))
    assert len(cache.transcript) == len(actions) and cache.refresh_count == len(actions) - 1
    total = math.fsum(e for a in actions for e in a.eps)
    assert abs(total - cache.spent) < 1e-10 and total <= B + 1e-10
    raw = data.replay(rec)  # Full truth is used ONLY below for offline evaluation.
    histories = np.asarray(histories)
    truth, estimates = w.evaluate(raw), w.from_cache(histories)
    errors = np.abs(estimates[w.target] - truth[w.target])
    scales, weights = np.asarray(stats['scales']), np.asarray(stats['weights'])
    node_errors = np.stack([np.abs(estimates[n] - truth[n]) for n in w.nodes], axis=-1)
    scaled = node_errors / scales
    gains, scaled_gains, long_gains, long_sums = [], [], [], []
    for j, (a, turn) in enumerate(zip(actions[1:], calendar[1:]), 1):
        old = w.from_cache(before[j])
        immediate = float(abs(old[w.target] - truth[w.target][turn - 1]) - errors[turn - 1])
        old_scaled = np.array([abs(old[n] - truth[n][turn - 1]) for n in w.nodes]) / scales
        scaled_gain = float(weights @ (old_scaled - scaled[turn - 1]))
        # Omit exactly action j, keep all other releases/inputs/noise unchanged.
        omitted = histories.copy()
        for root in a.roots:
            i = w.roots.index(root)
            nxt = next((t for later, t in zip(actions[j + 1:], calendar[j + 1:]) if root in later.roots), rec['T'] + 1)
            omitted[turn - 1:nxt - 1, i] = before[j][i]
        omitted_err = np.abs(w.from_cache(omitted)[w.target] - truth[w.target])
        effect = omitted_err[turn - 1:] - errors[turn - 1:]
        assert abs(float(effect[0]) - immediate) < 1e-12
        gains.append(immediate)
        scaled_gains.append(scaled_gain)
        long_gains.append(float(effect.mean()))
        long_sums.append(float(effect.sum()))
        cache.transcript[j].update(immediate_error_drop=immediate, immediate_scaled_drop=scaled_gain,
                                   long_term_mean_error_drop=long_gains[-1],
                                   long_term_cumulative_error_drop=long_sums[-1],
                                   long_term_horizon=[turn, rec['T']])
    norm = w.normalize(raw)
    changes = []
    previous = norm[0]
    for e in rec['events']:
        roots = data.significant_roots(previous, norm[e['turn'] - 1], stats['h'], w.roots,
                                       config['significant_delta'], config.get('significant_tolerance', 1e-12))
        if roots:
            changes.append({'turn': e['turn'], 'roots': sorted(roots)})
        previous = norm[e['turn'] - 1]
    at_turn = {t: set(a.roots) for a, t in zip(actions[1:], calendar[1:])}
    missed = sum(not set(c['roots']) <= at_turn.get(c['turn'], set()) for c in changes)
    waiting = []
    root_waiting = []
    for c in changes:
        t = c['turn']
        nxt = next((v for v in calendar[1:] if v >= t), None)
        per_root = {root: next((v for a, v in zip(actions[1:], calendar[1:]) if v >= t and root in a.roots), None)
                    for root in c['roots']}
        complete = max(per_root.values()) if all(v is not None for v in per_root.values()) else None
        def item(v):
            return {'observed_wait': v - t if v is not None else None,
                    'restricted_wait': (v if v is not None else rec['T'] + 1) - t, 'censored': v is None}
        waiting.append(item(nxt))
        root_waiting.append(item(complete))
        c.update(next_refresh_turn=nxt, next_root_turns=per_root, next_complete_refresh_turn=complete)
    mean = lambda values: float(np.mean(values)) if len(values) else None
    refresh_eps = math.fsum(e for a in actions[1:] for e in a.eps)
    pub = [v < 0 or v > 1 for r in cache.transcript for v in r['values']]
    m = {'subject_id': rec['subject_id'], 'trajectory_id': rec['trajectory_id'], 'template': rec['template'],
         'B': B, 'repeat': repeat, 'parameter': parameter, 'method': method,
         'mae': float(errors.mean()), 'p95_abs_error': float(np.quantile(errors, .95)),
         'workload_loss': float((scaled @ weights).mean()),
         'query_loss': mean([scaled[q['turn'] - 1, w.nodes.index(q['node'])] for q in rec['queries'] if q['node'] in w.nodes]),
         'refresh_count': len(actions) - 1, 'spent': total, 'initial_spent': math.fsum(actions[0].eps),
         'refresh_spent': refresh_eps, 'missed_events': missed, 'significant_events': len(changes),
         'miss_rate': missed / len(changes) if changes else None, 'aoi_mean': mean(aois),
         'publication_projection_fraction': mean(pub),
         'cache_projection_fraction': float(((histories < 0) | (histories > 1))[:, cache.active].mean()),
         'immediate_error_drop_per_refresh': mean(gains),
         'immediate_error_drop_per_epsilon': sum(gains) / refresh_eps if refresh_eps else None,
         'immediate_scaled_drop_per_refresh': mean(scaled_gains),
         'long_term_error_drop_per_refresh': mean(long_gains),
         'long_term_cumulative_drop_per_refresh': mean(long_sums),
         'long_term_cumulative_drop_per_epsilon': sum(long_sums) / refresh_eps if refresh_eps else None,
         'mean_refresh_turn': mean(calendar[1:])}
    for label, wait in [('wait_any', waiting), ('wait_roots', root_waiting)]:
        m[label + '_observed_mean'] = mean([v['observed_wait'] for v in wait if not v['censored']])
        m[label + '_restricted_mean'] = mean([v['restricted_wait'] for v in wait])
        m[label + '_censored_fraction'] = mean([v['censored'] for v in wait])
    for i, n in enumerate(w.nodes):
        m['node_mae__' + n] = float(node_errors[:, i].mean())
        m['node_scaled__' + n] = float(scaled[:, i].mean())
    log = {'transcript': cache.transcript, 'changes': changes, 'stop_turn': cache.stop_turn,
           'spent_path': spent_path, 'latest_observation_turns': latest_turns}
    return m, log


def run_pair(rec, w, stats, parameter, B, repeat, config):
    generated_metric, generated = run_method(rec, w, stats, parameter, B, repeat, config)
    actions = packages(generated['transcript'])
    m1_calendar = tuple(r['turn'] for r in generated['transcript'])
    opportunities = public_opportunities(rec)
    assert all(set(a.roots) <= dict(opportunities)[t] for a, t in zip(actions[1:], m1_calendar[1:]))
    uniform, matching = match_actions(actions[1:], opportunities, rec['T'])
    key = {k: generated_metric[k] for k in ['subject_id', 'trajectory_id', 'template', 'B', 'repeat', 'parameter']}
    base = dict(key, K=len(actions) - 1, actions=[asdict(a) for a in actions],
                generated_m1_mae=generated_metric['mae'],
                m1_original_calendar=m1_calendar, matching=matching,
                public_opportunities=[{'turn': t, 'observed_roots': sorted(roots)} for t, roots in opportunities])
    if uniform is None:
        return [], dict(base, status='excluded_matching_failure'), dict(base, **matching)
    assert (uniform, matching) == match_actions(actions[1:], opportunities, rec['T'])
    pair, logs = [], {}
    for method, calendar in [(M1_TIME, m1_calendar), (UNIFORM_TIME, (1,) + uniform)]:
        metric, log = replay_actions(rec, w, stats, parameter, B, repeat, config, actions, calendar, method)
        pair.append(metric)
        logs[method] = log
    assert abs(pair[0]['spent'] - pair[1]['spent']) < 1e-10
    assert pair[0]['spent'] == pair[1]['spent']
    assert pair[0]['refresh_count'] == pair[1]['refresh_count']
    assert logs[M1_TIME]['transcript'][0] == logs[UNIFORM_TIME]['transcript'][0]
    for i, (a, b, original) in enumerate(zip(logs[M1_TIME]['transcript'], logs[UNIFORM_TIME]['transcript'], generated['transcript'])):
        assert a['action'] == b['action'] == asdict(actions[i])
        assert a['roots'] == b['roots'] == original['roots']
        assert a['eps'] == b['eps'] == original['eps']
        assert a['values'] == original['values'] and a['turn'] == original['turn']
    for field in ['mae', 'p95_abs_error', 'query_loss', 'workload_loss', 'aoi_mean',
                  'cache_projection_fraction', 'publication_projection_fraction', 'spent', 'miss_rate']:
        if generated_metric[field] is None:
            assert pair[0][field] is None
        else:
            assert abs(pair[0][field] - generated_metric[field]) < 1e-10, field
    shifts = [t - s for t, s in zip(uniform, m1_calendar[1:])]
    for metric in pair:
        metric['paired_time_changed_fraction'] = float(np.mean(np.array(shifts) != 0)) if shifts else None
        metric['paired_absolute_time_shift'] = float(np.mean(np.abs(shifts))) if shifts else None
    base['matching']['time_shifts_uniform_minus_m1'] = shifts
    return pair, dict(base, status='strictly_matched', **logs), None


def current_code_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}


def summarize_coverage(attempts, repeats):
    """Whole pairs only: no partial action contributes to the metric sample."""
    def cell(rr):
        ok = [r for r in rr if r['success']]
        patients = {r['subject_id'] for r in rr}
        successful = {r['subject_id'] for r in ok}
        trajectories = {r['trajectory_id'] for r in rr}
        successful_t = {r['trajectory_id'] for r in ok}
        actions = sum(r['K'] for r in rr)
        matched = sum(r['K'] for r in ok)
        frac = lambda a, b: a / b if b else None
        return {'attempted_pairs': len(rr), 'successful_pairs': len(ok), 'failed_pairs': len(rr) - len(ok),
                'pair_coverage': frac(len(ok), len(rr)), 'attempted_patients': len(patients),
                'successful_patients_any': len(successful), 'patient_coverage_any': frac(len(successful), len(patients)),
                'successful_patients_complete': sum(all(r['success'] for r in rr if r['subject_id'] == sid) for sid in patients),
                'attempted_trajectories': len(trajectories), 'successful_trajectories_any': len(successful_t),
                'trajectory_coverage_any': frac(len(successful_t), len(trajectories)),
                'successful_trajectories_complete': sum(all(r['success'] for r in rr if r['trajectory_id'] == tid) for tid in trajectories),
                'attempted_actions_including_repeats': actions, 'matched_actions_including_repeats': matched,
                'action_coverage': frac(matched, actions),
                'unique_case_actions': actions // repeats, 'unique_case_matched_actions': matched // repeats,
                'failed_patient_ids': sorted(patients - successful),
                'excluded_cases': [{k: r[k] for k in ['subject_id', 'trajectory_id', 'B', 'parameter', 'K', 'reason']}
                                   for r in rr if not r['success']]}
    result = {'overall': cell(attempts), 'by_budget': {}}
    for B in sorted({r['B'] for r in attempts}):
        result['by_budget'][str(B)] = cell([r for r in attempts if r['B'] == B])
    return result


def run(source, prior, output, data_directory, plan, repeats, precheck=None):
    start = time.monotonic()
    source, prior, output = Path(source).resolve(), Path(prior).resolve(), Path(output).resolve()
    required = ['config.json', 'metadata.json', 'splits.json', 'public_stats.json', 'selections.json', 'targets.json']
    frozen = {name: json.loads((source / name).read_text()) for name in required}
    config, meta, stats = frozen['config.json'], frozen['metadata.json'], frozen['public_stats.json']
    assert config['templates'] == ['MELD_NA'] and FORMULA_VERSION == meta['formula_version']
    assert data.input_hashes(data_directory, config['templates']) == meta['input_sha256']
    prior_audit = json.loads((prior / 'audit.json').read_text())
    prior_protocol = json.loads((prior / 'protocol.json').read_text())
    assert prior_audit['passed']
    assert plan['primary_budgets'] == prior_protocol['primary_budgets']
    assert all(b in config['budgets'] for b in plan['primary_budgets'])
    assert plan['availability'] == 'latest_observed_at_original_measurement_opportunities'
    assert plan['bootstrap'] == 2000
    code = current_code_hashes()
    for filename in ['mechanisms.py', 'workload.py', 'data.py', 'matched_timing.py']:
        key = str(Path(__file__).with_name(filename).relative_to(ROOT))
        assert code[key] == prior_audit['code_sha256'][key], f'Frozen mechanism changed: {filename}'
    source_hash = {n: digest(source / n) for n in required}
    assert source_hash == prior_audit['source_sha256']
    prior_hash = {n: digest(prior / n) for n in ['audit.json', 'protocol.json']}
    if repeats < 2:
        raise ValueError('At least two mechanism repeats required')
    if repeats > 2:
        if precheck is None:
            raise ValueError('Formal run requires successful R=2 precheck')
        a = json.loads((Path(precheck) / 'audit.json').read_text())
        assert a['passed'] and a['repeats'] == 2 and a['unit_tests_passed']
        assert a['source_sha256'] == source_hash and a['prior_sha256'] == prior_hash
        assert a['code_sha256'] == code and a['plan'] == plan
        test_log = (Path(precheck) / 'unit_tests.txt').read_text()
    else:
        proc = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s',
                               'experiments/workflow_extensions/finite_cache/tests', '-t', '.', '-v'],
                              cwd=ROOT, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(proc.stdout + proc.stderr)
        test_log = proc.stdout + proc.stderr
    records = data.load(data_directory, config['templates'])
    test = data.partition(records, frozen['splits.json'], 'test', 'MELD_NA')
    parameters = sorted({v['M1-a']['parameter'] for v in frozen['selections.json']['MELD_NA'].values()})
    w = Workload('MELD_NA', stats['MELD_NA']['domains'])
    legacy = legacy_schedule_index(source)
    output.mkdir(parents=True, exist_ok=False)
    for name in required:
        shutil.copyfile(source / name, output / ('frozen_' + name))
    (output / 'unit_tests.txt').write_text(test_log)
    protocol = dict(plan, repeats=repeats, seed=config['seed'], source=str(source), prior=str(prior),
                    original_budget_grid=config['budgets'], executed_budgets=plan['primary_budgets'],
                    selected_parameters=parameters, precheck=str(precheck) if precheck else None,
                    matching_targets='1+k*(T-1)/(K+1); ordered distinct public batch opportunities',
                    same_turn_observation='batch observations become available before publication and queries',
                    initialization_action_index=0, paired_noise_key='seed,subject,trajectory,B,repeat,action_index,root',
                    complete_case_primary='only trajectories matched at ALL six budgets; no budget-specific patient mix',
                    waiting_censoring='T+1; observed, restricted means and censored fraction all reported',
                    intervals='pointwise two-sided95; primary six-budget patient mean; no test-based budget selection',
                    long_term_scope='offline omit-one fixed-action counterfactual; effects overlap and are not additive',
                    method_names=[M1_TIME, UNIFORM_TIME])
    write_json(output / 'protocol.json', protocol)
    audit = {'passed': False, 'repeats': repeats, 'plan': plan, 'code_sha256': code,
             'source_sha256': source_hash, 'prior_sha256': prior_hash,
             'input_sha256': data.input_hashes(data_directory, config['templates']), 'unit_tests_passed': True,
             'generated_pairs': 0, 'matched_pairs': 0, 'legacy_schedule_matches': 0,
             'max_epsilon_difference': 0., 'max_budget_ratio': 0., 'max_m1_replay_mae_difference': 0.,
             'fixed_ordered_roots_eps_unit_noise': True, 'initialization_exact': True,
             'latest_observed_values_only': True, 'shared_replay_implementation': True,
             'ordered_unique_deterministic_matching': True, 'unchanged_m1_gate_and_parameters': True}
    rows, attempts = [], []
    with (output / 'pure_timing_metrics.jsonl').open('w') as mf, (output / 'pure_timing_schedules.jsonl').open('w') as sf, \
            (output / 'matching_failures.jsonl').open('w') as ff:
        for parameter in parameters:
            for B in plan['primary_budgets']:
                for rec in test:
                    for repeat in range(repeats):
                        pair, log, failure = run_pair(rec, w, stats['MELD_NA'], parameter, B, repeat, config)
                        attempt = {k: log[k] for k in ['subject_id', 'trajectory_id', 'B', 'parameter', 'K']}
                        attempt.update(success=failure is None, reason=failure['reason'] if failure else None)
                        attempts.append(attempt)
                        sf.write(json.dumps(log, allow_nan=False) + '\n')
                        audit['generated_pairs'] += 1
                        if failure:
                            ff.write(json.dumps(failure, allow_nan=False) + '\n')
                            continue
                        if repeat == 0:
                            old = legacy[(rec['trajectory_id'], B, parameter)]
                            new = log[M1_TIME]['transcript']
                            assert len(old) == len(new)
                            for x, y in zip(old, new):
                                assert x['turn'] == y['turn'] and x['roots'] == y['roots'] and x['eps'] == y['eps']
                            audit['legacy_schedule_matches'] += 1
                        audit['matched_pairs'] += 1
                        audit['max_epsilon_difference'] = max(audit['max_epsilon_difference'], abs(pair[0]['spent'] - pair[1]['spent']))
                        audit['max_budget_ratio'] = max(audit['max_budget_ratio'], *(r['spent'] / B for r in pair))
                        audit['max_m1_replay_mae_difference'] = max(audit['max_m1_replay_mae_difference'],
                                                                   abs(pair[0]['mae'] - log['generated_m1_mae']))
                        for r in pair:
                            rows.append(r)
                            mf.write(json.dumps(r, allow_nan=False) + '\n')
                mf.flush()
                sf.flush()
                ff.flush()
                print(f'pure_timing/{parameter}/B={B:g}: R={repeats} complete', flush=True)
    coverage = summarize_coverage(attempts, repeats)
    write_json(output / 'coverage.json', coverage)
    from .pure_timing_analysis import analyze
    analyze(rows, output, protocol, coverage)
    assert code == current_code_hashes()
    assert source_hash == {n: digest(source / n) for n in required}
    assert prior_hash == {n: digest(prior / n) for n in prior_hash}
    audit.update(passed=True, coverage=coverage['overall'], elapsed_seconds=time.monotonic() - start,
                 code_unchanged_during_run=True, source_and_prior_outputs_unchanged=True)
    write_json(output / 'audit.json', audit)
    print(json.dumps({'output': str(output), 'audit_passed': True, 'pairs': audit['matched_pairs'],
                      'elapsed_seconds': audit['elapsed_seconds']}), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data_mine/finite_cache')
    p.add_argument('--plan', type=Path, default=Path(__file__).with_name('configs') / 'pure_timing.json')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--precheck', type=Path)
    a = p.parse_args()
    run(a.source, a.prior, a.output, a.data, json.loads(a.plan.read_text()), a.repeats, a.precheck)


if __name__ == '__main__':
    main()
