"""Uniform random calendars and exact exhaustive oracle for frozen action packs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
from itertools import combinations
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
from .matched_timing import digest
from .pure_timing import Action, M1_TIME, UNIFORM_TIME, public_opportunities, replay_actions
from .run import ROOT
from .workload import Workload

RANDOM = 'random_time_fixed_action'
ORACLE = 'oracle_time_fixed_action'


class ScheduleSpace:
    """Exact suffix counts, lexicographic unranking and exhaustive enumeration."""
    def __init__(self, actions, opportunities):
        self.turns = tuple(t for t, _ in opportunities)
        assert list(self.turns) == sorted(set(self.turns))
        self.K, self.n = len(actions), len(opportunities)
        self.legal = np.array([[set(a.roots) <= observed for _, observed in opportunities]
                               for a in actions], bool).reshape(self.K, self.n)
        self.counts = [[0] * (self.n + 1) for _ in range(self.K + 1)]
        self.counts[self.K] = [1] * (self.n + 1)
        for j in range(self.K - 1, -1, -1):
            for p in range(self.n - 1, -1, -1):
                self.counts[j][p] = self.counts[j][p + 1] + (
                    self.counts[j + 1][p + 1] if self.legal[j, p] else 0)
        self.total = self.counts[0][0]

    def unrank(self, rank):
        if not 0 <= rank < self.total:
            raise ValueError('Calendar rank outside feasible space')
        result, start = [1], 0
        for j in range(self.K):
            for p in range(start, self.n):
                if not self.legal[j, p]:
                    continue
                block = self.counts[j + 1][p + 1]
                if rank < block:
                    result.append(self.turns[p])
                    start = p + 1
                    break
                rank -= block
            else:
                raise AssertionError('Suffix counts/unranking disagree')
        return tuple(result)

    def enumerate(self):
        paths = [(1,) + tuple(self.turns[i] for i in indices)
                 for indices in combinations(range(self.n), self.K)
                 if all(self.legal[j, i] for j, i in enumerate(indices))]
        assert len(paths) == self.total
        return np.array(paths, int).reshape(self.total, self.K + 1)

    def sample(self, schedule_seed, subject, trajectory, B, repeat, size):
        # Separate from publication noise; no values or evaluation enter this key.
        key = json.dumps([schedule_seed, subject, trajectory, float(B), repeat], separators=(',', ':'))
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], 'little')
        if not 0 < self.total <= np.iinfo(np.int64).max:
            raise ValueError('Use arbitrary-integer rejection sampling for larger spaces')
        rng = np.random.default_rng(seed)
        ranks = rng.integers(self.total, size=size).tolist()
        return np.array([self.unrank(r) for r in ranks], int), ranks, f'{seed:032x}'


class Evaluator:
    """Numerical cache replay; verified against original Cache replay for baselines.

    Candidate oracle scoring sums piecewise-constant cache estimates using exact
    empirical CDF prefixes. It never assumes time/root independence of MELD.
    Full replay remains available for all selected random/oracle diagnostics.
    """
    def __init__(self, rec, w, stats, config):
        self.rec, self.w, self.stats, self.config = rec, w, stats, config
        self.T, self.k = rec['T'], len(w.roots)
        self.raw = data.replay(rec)
        self.z = w.normalize(self.raw)
        self.truth = w.evaluate(self.raw)
        self.target = np.asarray(self.truth[w.target])
        self.opportunities = dict(public_opportunities(rec))
        self.provenance = []
        source = {r: 1 for r in rec['initial_state']}
        events = {e['turn']: e for e in rec['events']}
        for t in range(1, self.T + 1):
            if t in events:
                source.update({r: t for r in events[t]['measured_roots']})
            self.provenance.append(dict(source))
        unique = np.unique(self.target)
        index = np.searchsorted(unique, self.target)
        less = index[:, None] < np.arange(len(unique) + 1)[None, :]
        self.unique = unique
        self.cdf_counts = np.vstack([np.zeros(len(unique) + 1), np.cumsum(less, axis=0)])
        self.cdf_sums = np.vstack([np.zeros(len(unique) + 1), np.cumsum(less * self.target[:, None], axis=0)])
        self.truth_sum = np.r_[0., np.cumsum(self.target)]
        self.changes = []
        previous = self.z[0]
        for event in rec['events']:
            t = event['turn']
            roots = data.significant_roots(previous, self.z[t - 1], stats['h'], w.roots,
                                           config['significant_delta'], config.get('significant_tolerance', 1e-12))
            if roots:
                self.changes.append((t, roots))
            previous = self.z[t - 1]

    def validate(self, actions, calendars):
        calendars = np.asarray(calendars, int)
        assert calendars.ndim == 2 and calendars.shape[1] == len(actions)
        assert np.all(calendars[:, 0] == 1) and np.all(np.diff(calendars, axis=1) > 0)
        assert np.all(calendars <= self.T)
        for j, action in enumerate(actions[1:], 1):
            assert all(t in self.opportunities and set(action.roots) <= self.opportunities[t]
                       for t in np.unique(calendars[:, j]))
        return calendars

    def released(self, action, turns):
        idx = [self.w.roots.index(r) for r in action.roots]
        # Every row is the latest observed state at that turn, not a future state.
        return idx, self.z[np.asarray(turns) - 1][:, idx] + np.asarray(action.unit_noise) / action.eps

    def interval_cost(self, estimate, start, end):
        """Sum |estimate-truth[t]| for t in [start,end), broadcast over repeats."""
        p = np.searchsorted(self.unique, estimate)
        counts = self.cdf_counts[end, p] - self.cdf_counts[start, p]
        sums = self.cdf_sums[end, p] - self.cdf_sums[start, p]
        result = estimate * (2 * counts - (end - start)) + self.truth_sum[end] - self.truth_sum[start] - 2 * sums
        assert np.min(result) > -1e-9
        return np.maximum(result, 0.)

    def score_many_repeats(self, actions_by_repeat, calendars):
        """Score ALL candidate calendars for each fixed-noise repeat in a batch."""
        calendars = self.validate(actions_by_repeat[0], calendars)
        R, N = len(actions_by_repeat), len(calendars)
        cache = np.zeros((R, N, self.k))
        for r, actions in enumerate(actions_by_repeat):
            idx, v = self.released(actions[0], [1])
            for i, value in zip(idx, v[0]):
                cache[r, :, i] = value
        cost = np.zeros((R, N))
        start = np.zeros((1, N), int)
        for j in range(1, len(actions_by_repeat[0])):
            end = calendars[:, j][None, :] - 1
            value = self.w.from_cache(cache)[self.w.target]
            cost += self.interval_cost(value, start, end)
            idx = [self.w.roots.index(root) for root in actions_by_repeat[0][j].roots]
            for r, actions in enumerate(actions_by_repeat):
                assert actions[j].roots == actions_by_repeat[0][j].roots
                assert actions[j].eps == actions_by_repeat[0][j].eps
                for i, noise, eps in zip(idx, actions[j].unit_noise, actions[j].eps):
                    cache[r, :, i] = self.z[calendars[:, j] - 1, i] + noise / eps
            start = end
        cost += self.interval_cost(self.w.from_cache(cache)[self.w.target], start, self.T)
        return cost / self.T

    def metrics(self, actions, calendars):
        calendars = self.validate(actions, calendars)
        N = len(calendars)
        cache = np.zeros((N, self.k))
        idx, v = self.released(actions[0], [1])
        cache[:, idx] = v[0]
        histories = np.zeros((N, self.T, self.k))
        last = np.ones((N, self.k), int)
        active = np.asarray(self.stats['h']) > 0
        aoi = np.zeros(N)
        published_projection = np.full(N, np.sum((v[0] < 0) | (v[0] > 1)), float)
        publications = len(idx)
        for action in actions[1:]:
            idx, values = self.released(action, calendars[:, action.action_index])
            published_projection += np.sum((values < 0) | (values > 1), axis=1)
            publications += len(idx)
        for turn in range(1, self.T + 1):
            for j, action in enumerate(actions[1:], 1):
                selected = np.flatnonzero(calendars[:, j] == turn)
                if len(selected):
                    idx, values = self.released(action, np.full(len(selected), turn))
                    cache[np.ix_(selected, idx)] = values
                    last[np.ix_(selected, idx)] = turn
            histories[:, turn - 1] = cache
            aoi += np.mean((turn - last)[:, active], axis=1)
        estimated = self.w.from_cache(histories)
        error = np.abs(estimated[self.w.target] - self.target)
        scaled = np.stack([np.abs(estimated[n] - self.truth[n]) for n in self.w.nodes], axis=-1) / self.stats['scales']
        query = [scaled[:, q['turn'] - 1, self.w.nodes.index(q['node'])]
                 for q in self.rec['queries'] if q['node'] in self.w.nodes]
        missed = np.zeros(N)
        waits_any, waits_roots, censor_any, censor_roots = [], [], [], []
        for turn, roots in self.changes:
            covered = np.zeros(N, bool)
            for j, action in enumerate(actions[1:], 1):
                if roots <= set(action.roots):
                    covered |= calendars[:, j] == turn
            missed += ~covered
            future = np.where(calendars[:, 1:] >= turn, calendars[:, 1:], self.T + 1)
            nxt = future.min(axis=1) if len(actions) > 1 else np.full(N, self.T + 1)
            per_root = []
            for root in roots:
                columns = [j for j, a in enumerate(actions[1:], 1) if root in a.roots]
                future_root = np.where(calendars[:, columns] >= turn, calendars[:, columns], self.T + 1)
                per_root.append(future_root.min(axis=1) if columns else np.full(N, self.T + 1))
            all_roots = np.max(per_root, axis=0)
            waits_any.append(nxt - turn)
            waits_roots.append(all_roots - turn)
            censor_any.append(nxt == self.T + 1)
            censor_roots.append(all_roots == self.T + 1)
        total = math.fsum(e for a in actions for e in a.eps)
        result = []
        for s in range(N):
            m = {'mae': float(error[s].mean()), 'p95_abs_error': float(np.quantile(error[s], .95)),
                 'workload_loss': float((scaled[s] @ np.asarray(self.stats['weights'])).mean()),
                 'query_loss': float(np.mean(query, axis=0)[s]) if query else None,
                 'miss_rate': float(missed[s] / len(self.changes)) if self.changes else None,
                 'aoi_mean': float(aoi[s] / self.T), 'spent': total, 'refresh_count': len(actions) - 1,
                 'publication_projection_fraction': float(published_projection[s] / publications),
                 'cache_projection_fraction': float(((histories[s] < 0) | (histories[s] > 1))[:, active].mean())}
            for label, values in [('wait_any_restricted_mean', waits_any), ('wait_roots_restricted_mean', waits_roots),
                                   ('wait_any_censored_fraction', censor_any), ('wait_roots_censored_fraction', censor_roots)]:
                m[label] = float(np.mean(values, axis=0)[s]) if values else None
            for i, node in enumerate(self.w.nodes):
                m['node_mae__' + node] = float(np.abs(estimated[node][s] - self.truth[node]).mean())
            result.append(m)
        return result

    def transcript(self, actions, calendar):
        result = []
        for a, turn in zip(actions, calendar):
            idx, values = self.released(a, [turn])
            assert all(self.provenance[turn - 1][root] <= turn for root in a.roots)
            result.append({'turn': int(turn), 'roots': list(a.roots), 'eps': list(a.eps),
                           'unit_noise': list(a.unit_noise), 'values': values[0].tolist(),
                           'normalized_inputs': {root: float(self.z[turn - 1, i]) for root, i in zip(a.roots, idx)},
                           'source_turns': {root: self.provenance[turn - 1][root] for root in a.roots}})
        return result


def exact_oracle(evaluator, actions_by_repeat, calendars, chunk_size=2048):
    best = np.full(len(actions_by_repeat), np.inf)
    best_index = np.zeros(len(actions_by_repeat), int)
    evaluated = 0
    for start in range(0, len(calendars), chunk_size):
        scores = evaluator.score_many_repeats(actions_by_repeat, calendars[start:start + chunk_size])
        indices = np.argmin(scores, axis=1)
        candidate = scores[np.arange(len(scores)), indices]
        improve = candidate < best
        best[improve] = candidate[improve]
        best_index[improve] = start + indices[improve]
        evaluated += scores.shape[1]
    assert evaluated == len(calendars)
    return calendars[best_index], best, {'solver': 'exact_exhaustive_enumeration',
                                        'evaluated_per_repeat': evaluated, 'optimality_gap': 0.,
                                        'floating_score_tolerance': 1e-10}


def source_hashes(source):
    files = ['protocol.json', 'audit.json', 'coverage.json', 'pure_timing_metrics.jsonl',
             'pure_timing_schedules.jsonl', 'frozen_config.json', 'frozen_public_stats.json',
             'frozen_splits.json', 'frozen_selections.json', 'frozen_metadata.json']
    return {f: digest(source / f) for f in files}


def code_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}


def read_actions(log):
    return tuple(Action(a['action_index'], tuple(a['roots']), tuple(a['eps']), tuple(a['unit_noise'])) for a in log['actions'])


def run(source, output, data_directory, plan, repeats, precheck=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    start = time.monotonic()
    prior_protocol = json.loads((source / 'protocol.json').read_text())
    prior_audit = json.loads((source / 'audit.json').read_text())
    assert prior_audit['passed'] and prior_protocol['repeats'] == 50
    assert plan['primary_budgets'] == prior_protocol['primary_budgets']
    assert plan['bootstrap'] == 2000 and plan['random_schedules_per_pair'] == 50
    assert plan['availability'] == prior_protocol['availability']
    assert repeats in (2, plan['repeats'])
    code, frozen = code_hashes(), source_hashes(source)
    for name in ['pure_timing.py', 'data.py', 'workload.py', 'mechanisms.py', 'matched_timing.py']:
        key = str(Path(__file__).with_name(name).relative_to(ROOT))
        assert code[key] == prior_audit['code_sha256'][key]
    config = json.loads((source / 'frozen_config.json').read_text())
    stats = json.loads((source / 'frozen_public_stats.json').read_text())
    metadata = json.loads((source / 'frozen_metadata.json').read_text())
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
            raise ValueError('Formal supplement requires completed small precheck')
        audit = json.loads((Path(precheck) / 'audit.json').read_text())
        assert audit['passed'] and audit['repeats'] == 2 and audit['unit_tests_passed']
        assert audit['code_sha256'] == code and audit['source_sha256'] == frozen and audit['plan'] == plan
        test_log = (Path(precheck) / 'unit_tests.txt').read_text()
        budgets = plan['primary_budgets']
    source_metrics = {(r['trajectory_id'], r['B'], r['repeat'], r['method']): r
                      for r in map(json.loads, (source / 'pure_timing_metrics.jsonl').open())}
    groups = defaultdict(list)
    for r in map(json.loads, (source / 'pure_timing_schedules.jsonl').open()):
        if r['B'] in budgets and r['repeat'] < repeats:
            assert r['status'] == 'strictly_matched'
            groups[(r['trajectory_id'], r['B'], r['parameter'])].append(r)
    records = {r['trajectory_id']: r for r in data.load(data_directory, ['MELD_NA'])}
    w = Workload('MELD_NA', stats['MELD_NA']['domains'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'unit_tests.txt').write_text(test_log)
    protocol = dict(plan, repeats=repeats, executed_budgets=budgets, source=str(source),
                    precheck=str(precheck) if precheck else None, seed=prior_protocol['seed'],
                    conditional_distribution='50 calendars per fixed patient/trajectory/B/noise-repeat; empirical central95 not CI',
                    m1_percentile='100*(fraction random_MAE<M1_MAE + half ties); lower is better',
                    m1_win_rate='fraction M1_MAE<random_MAE; absolute tie tolerance 1e-12',
                    oracle_information='complete truth AND realized fixed package noise; offline ceiling only',
                    solver_numeric_tolerance=1e-10,
                    primary_inference='patient-equal six-budget paired mean; patient and patient+mechanism bootstrap',
                    nested_random_uncertainty='additional patient+mechanism+schedule bootstrap reported',
                    no_near_oracle_threshold='no equivalence margin predeclared; do not equate nonsignificance to equivalence')
    write_json(output / 'protocol.json', protocol)
    audit = {'passed': False, 'plan': plan, 'repeats': repeats, 'source_sha256': frozen, 'code_sha256': code,
             'unit_tests_passed': True, 'pairs': 0, 'random_schedules': 0, 'oracle_candidates_evaluated': 0,
             'oracle_max_feasible_space': 0, 'oracle_exact_all_pairs': True, 'oracle_optimality_gap': 0.,
             'baseline_metric_max_difference': 0., 'oracle_score_max_difference': 0.,
             'fixed_ordered_roots_eps_noise_all_methods': True, 'latest_observed_values_only': True,
             'max_total_epsilon_difference': 0., 'max_budget_ratio': 0.,
             'iid_uniform_ranks_no_result_filtering': True, 'no_reselection_or_tuning': True}
    rows, distributions = [], []
    with (output / 'supplement_metrics.jsonl').open('w') as mf, \
            (output / 'random_schedules.jsonl').open('w') as rf, \
            (output / 'oracle_schedules.jsonl').open('w') as of, \
            (output / 'random_distribution.jsonl').open('w') as df:
        for (trajectory, B, parameter), logs in sorted(groups.items()):
            logs.sort(key=lambda r: r['repeat'])
            assert [r['repeat'] for r in logs] == list(range(repeats))
            rec = records[trajectory]
            ev = Evaluator(rec, w, stats['MELD_NA'], config)
            actions_r = [read_actions(r) for r in logs]
            for actions in actions_r[1:]:
                assert [(a.roots, a.eps) for a in actions] == [(a.roots, a.eps) for a in actions_r[0]]
            space = ScheduleSpace(actions_r[0][1:], public_opportunities(rec))
            calendars = space.enumerate()
            assert space.total > 0
            best_calendars, scores, solver = exact_oracle(ev, actions_r, calendars, plan['oracle_chunk_size'])
            audit['oracle_candidates_evaluated'] += space.total * repeats
            audit['oracle_max_feasible_space'] = max(audit['oracle_max_feasible_space'], space.total)
            for repeat, (log, actions, optimal, score) in enumerate(zip(logs, actions_r, best_calendars, scores)):
                key = {k: log[k] for k in ['subject_id', 'trajectory_id', 'template', 'B', 'repeat', 'parameter']}
                original_calendars = np.array([[r['turn'] for r in log[m]['transcript']]
                                              for m in [M1_TIME, UNIFORM_TIME]])
                baseline = ev.metrics(actions, original_calendars)
                for method, metric in zip([M1_TIME, UNIFORM_TIME], baseline):
                    old = source_metrics[(trajectory, B, repeat, method)]
                    for field, value in metric.items():
                        if value is None:
                            assert old[field] is None
                        else:
                            difference = abs(value - old[field])
                            audit['baseline_metric_max_difference'] = max(audit['baseline_metric_max_difference'], difference)
                            assert difference < 1e-10, field
                    # Preserve EXACT source values, rather than replace by rounded reruns.
                    copied = {field: old[field] for field in metric}
                    row = dict(key, method=method, **copied)
                    rows.append(row)
                    mf.write(json.dumps(row, allow_nan=False) + '\n')
                random_calendars, ranks, seed_hex = space.sample(plan['schedule_seed'], rec['subject_id'], trajectory, B,
                                                               repeat, plan['random_schedules_per_pair'])
                random_metrics = ev.metrics(actions, random_calendars)
                m1_mae = source_metrics[(trajectory, B, repeat, M1_TIME)]['mae']
                maes = np.array([m['mae'] for m in random_metrics])
                tie = np.abs(maes - m1_mae) <= 1e-12
                dist = dict(key, random_mae_mean=float(maes.mean()), random_mae_median=float(np.median(maes)),
                            random_mae_q025=float(np.quantile(maes, .025)), random_mae_q975=float(np.quantile(maes, .975)),
                            m1_mae=m1_mae, m1_win_rate=float(np.mean((m1_mae < maes) & ~tie)),
                            tie_rate=float(tie.mean()), m1_percentile=float(100 * (np.mean((maes < m1_mae) & ~tie) + .5 * tie.mean())),
                            feasible_calendars=space.total, random_schedules=len(ranks), unique_random_ranks=len(set(ranks)),
                            schedule_rng_seed_hex=seed_hex)
                distributions.append(dist)
                df.write(json.dumps(dist, allow_nan=False) + '\n')
                for s, (calendar, rank, metric) in enumerate(zip(random_calendars, ranks, random_metrics)):
                    transcript = ev.transcript(actions, calendar)
                    assert [r['roots'] for r in transcript] == [r['roots'] for r in log[M1_TIME]['transcript']]
                    assert [r['eps'] for r in transcript] == [r['eps'] for r in log[M1_TIME]['transcript']]
                    assert [r['unit_noise'] for r in transcript] == [list(a.unit_noise) for a in actions]
                    total = math.fsum(e for r in transcript for e in r['eps'])
                    assert total == metric['spent'] == baseline[0]['spent'] <= B + 1e-10
                    schedule_row = dict(key, sample_index=s, schedule_seed_hex=seed_hex, feasible_rank=rank,
                                        feasible_calendars=space.total, transcript=transcript)
                    rf.write(json.dumps(schedule_row, allow_nan=False) + '\n')
                    row = dict(key, method=RANDOM, sample_index=s, **metric)
                    rows.append(row)
                    mf.write(json.dumps(row, allow_nan=False) + '\n')
                oracle_metric = ev.metrics(actions, np.array([optimal]))[0]
                # Selected oracle also runs through original Cache publication path.
                reference, reference_log = replay_actions(rec, w, stats['MELD_NA'], parameter, B, repeat,
                                                          config, actions, tuple(optimal), ORACLE)
                for field, value in oracle_metric.items():
                    if value is not None:
                        assert abs(value - reference[field]) < 1e-10, field
                diff = abs(score - reference['mae'])
                audit['oracle_score_max_difference'] = max(audit['oracle_score_max_difference'], diff)
                assert diff < 1e-10
                assert reference['mae'] <= min(m1_mae, baseline[1]['mae'], float(maes.min())) + 1e-10
                oracle_transcript = ev.transcript(actions, optimal)
                for published, source_release in zip(oracle_transcript, reference_log['transcript']):
                    assert published['values'] == source_release['values']
                    assert published['eps'] == source_release['eps']
                of.write(json.dumps(dict(key, **solver, feasible_calendars=space.total,
                                         objective_mae=reference['mae'], transcript=oracle_transcript), allow_nan=False) + '\n')
                row = dict(key, method=ORACLE, **oracle_metric)
                rows.append(row)
                mf.write(json.dumps(row, allow_nan=False) + '\n')
                audit['pairs'] += 1
                audit['random_schedules'] += len(random_metrics)
                audit['max_budget_ratio'] = max(audit['max_budget_ratio'], baseline[0]['spent'] / B)
            for f in [mf, rf, of, df]:
                f.flush()
            print(f'{trajectory}/B={B:g}: {repeats} repeats, {space.total} feasible calendars each, exact oracle complete', flush=True)
    from .timing_supplement_analysis import analyze
    analyze(rows, distributions, output, protocol)
    assert code == code_hashes() and frozen == source_hashes(source)
    audit.update(passed=True, elapsed_seconds=time.monotonic() - start, code_unchanged_during_run=True,
                 source_outputs_unchanged=True)
    write_json(output / 'audit.json', audit)
    print(json.dumps({'output': str(output), 'audit_passed': True, 'pairs': audit['pairs'],
                      'oracle_candidates': audit['oracle_candidates_evaluated'], 'elapsed_seconds': audit['elapsed_seconds']}), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=ROOT / 'data_mine/finite_cache')
    p.add_argument('--plan', type=Path, default=Path(__file__).with_name('configs') / 'timing_supplement.json')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--precheck', type=Path)
    a = p.parse_args()
    run(a.source, a.output, a.data, json.loads(a.plan.read_text()), a.repeats, a.precheck)


if __name__ == '__main__':
    main()
