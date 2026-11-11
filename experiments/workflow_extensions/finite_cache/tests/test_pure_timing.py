import copy
from itertools import combinations
import unittest
import numpy as np

from experiments.workflow_extensions.finite_cache.public_stats import fit
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.finite_cache.matched_timing import action_uniforms, run_method
from experiments.workflow_extensions.finite_cache.mechanisms import laplace_standard
from experiments.workflow_extensions.finite_cache.pure_timing import (
    Action, M1_TIME, UNIFORM_TIME, packages, match_actions, public_opportunities,
    replay_actions, run_pair, summarize_coverage)
from .test_protocol import fixture


class PureTiming(unittest.TestCase):
    def case(self):
        rec = fixture('test')
        w = Workload('ANEMIA')
        stats = fit([fixture('train')], w, [.5])
        stats['h'] = [1., 1., 1.]
        stats['C']['0.5'] = [.15, .15, .15]
        cfg = {'seed': 17, 'quantiles': [.5], 'significant_delta': .05, 'mechanism': {}}
        return rec, w, stats, cfg

    def test_ordered_match_equals_bruteforce_and_is_deterministic(self):
        opp = [(2, frozenset('a')), (3, frozenset('ab')), (7, frozenset('ab')),
               (11, frozenset('abc')), (16, frozenset('abc')), (19, frozenset('abc'))]
        for roots in [('a',), ('b', 'a'), ('c', 'b', 'a'), ('a', 'b', 'c', 'a')]:
            acts = tuple(Action(i + 1, (r,), (1.,), (.1,)) for i, r in enumerate(roots))
            result, detail = match_actions(acts, opp, 20)
            candidates = [path for path in combinations([t for t, _ in opp], len(acts))
                          if all(set(a.roots) <= dict(opp)[t] for a, t in zip(acts, path))]
            score = lambda path: (sum(((t - 1) * (len(acts) + 1) - j * 19) ** 2
                                     for j, t in enumerate(path, 1)), path)
            self.assertEqual(result, min(candidates, key=score))
            self.assertEqual((result, detail), match_actions(acts, opp, 20))
        self.assertEqual(match_actions((), opp, 20)[0], ())

    def test_matching_failures_do_not_drop_or_reorder_actions(self):
        a = Action(1, ('a',), (2.,), (.3,))
        b = Action(2, ('b',), (3.,), (.4,))
        self.assertEqual(match_actions((a, b), [(2, frozenset('ab'))], 6)[1]['reason'],
                         'insufficient_distinct_opportunities')
        self.assertEqual(match_actions((b,), [(2, frozenset('a'))], 6)[1]['reason'], 'unobserved_action_roots')
        failure = match_actions((a, b), [(2, frozenset('b')), (3, frozenset('ab'))], 6)
        self.assertIsNone(failure[0])
        self.assertEqual(failure[1]['reason'], 'no_order_preserving_complete_assignment')
        self.assertEqual(a.eps, (2.,))

    def test_public_calendar_does_not_read_values(self):
        rec, w, stats, cfg = self.case()
        copied = copy.deepcopy(rec)
        copied['initial_state'] = dict.fromkeys(rec['initial_state'], object())
        for event in copied['events']:
            event['values'] = object()
        self.assertEqual(public_opportunities(rec), public_opportunities(copied))

    def test_fixed_packages_and_original_time_replay_equal_actual_m1(self):
        rec, w, stats, cfg = self.case()
        for repeat in range(3):
            actual, log = run_method(rec, w, stats, 'q=0.5', 320., repeat, cfg)
            acts = packages(log['transcript'])
            calendar = tuple(r['turn'] for r in log['transcript'])
            metric, replayed = replay_actions(rec, w, stats, 'q=0.5', 320., repeat, cfg, acts, calendar, M1_TIME)
            self.assertEqual(metric['mae'], actual['mae'])
            for a, b in zip(log['transcript'], replayed['transcript']):
                self.assertEqual(a['values'], b['values'])
                self.assertEqual(a['eps'], b['eps'])

    def test_replay_uses_historical_latest_value_when_root_not_remeasured(self):
        rec, w, stats, cfg = self.case()
        # hb is measured at3 but not5; replaying an hb-only fixed packet at5 is legal.
        _, log = run_method(rec, w, stats, 'q=0.5', 320., 0, cfg)
        init = packages(log['transcript'])[0]
        noise = float(laplace_standard(action_uniforms(cfg['seed'], rec['subject_id'], rec['trajectory_id'],
                                                    320., 0, 1, ['hb']))[0])
        acts = (init, Action(1, ('hb',), (10.,), (noise,)))
        _, replayed = replay_actions(rec, w, stats, 'q=0.5', 320., 0, cfg, acts, (1, 5), UNIFORM_TIME)
        release = replayed['transcript'][1]
        self.assertEqual(release['source_turns'], {'hb': 3})
        self.assertEqual(release['normalized_inputs']['hb'], w.normalize([11., 30., 4.])[0])
        # Future hb at5 must not enter an earlier release at3.
        future = copy.deepcopy(rec)
        future['events'][1]['measured_roots'].append('hb')
        future['events'][1]['values']['hb'] = 24.
        _, earlier = replay_actions(future, w, stats, 'q=0.5', 320., 0, cfg, acts, (1, 3), UNIFORM_TIME)
        self.assertEqual(earlier['transcript'][1]['source_turns'], {'hb': 3})
        self.assertEqual(earlier['transcript'][1]['values'], release['values'])

    def test_replay_rejects_future_roots_duplicate_times_and_mismatched_noise(self):
        rec, w, stats, cfg = self.case()
        _, log = run_method(rec, w, stats, 'q=0.5', 320., 0, cfg)
        init = packages(log['transcript'])[0]
        with self.assertRaises(AssertionError):
            replay_actions(rec, w, stats, 'q=0.5', 320., 0, cfg,
                           (init, Action(1, ('future',), (1.,), (.1,))), (1, 3), UNIFORM_TIME)
        with self.assertRaises(AssertionError):
            replay_actions(rec, w, stats, 'q=0.5', 320., 0, cfg,
                           (init, init), (1, 1), UNIFORM_TIME)
        with self.assertRaises(AssertionError):
            replay_actions(rec, w, stats, 'q=0.5', 320., 0, cfg,
                           (init, Action(1, ('hb',), (1.,), (999.,))), (1, 3), UNIFORM_TIME)

    def test_pair_all_constraints_and_zero_refresh_boundary(self):
        rec, w, stats, cfg = self.case()
        rec['T'] = 18
        rec['events'] = []
        for j, t in enumerate([2, 4, 7, 9, 12, 15, 17]):
            root = ['hb', 'hct', 'rbc'][j % 3]
            rec['events'].append({'turn': t, 'batch_id': str(j), 'available_time': str(j),
                                  'measured_roots': [root], 'values': {root: 3. + j}})
        pair, schedules, failure = run_pair(rec, w, stats, 'q=0.5', 320., 0, cfg)
        self.assertIsNone(failure)
        self.assertEqual(pair[0]['spent'], pair[1]['spent'])
        self.assertEqual(pair[0]['refresh_count'], pair[1]['refresh_count'])
        for a, b in zip(schedules[M1_TIME]['transcript'], schedules[UNIFORM_TIME]['transcript']):
            self.assertEqual(a['action'], b['action'])
            self.assertEqual(a['eps'], b['eps'])
        stats['C']['0.5'] = [0., 0., 0.]
        pair, schedules, failure = run_pair(rec, w, stats, 'q=0.5', 10., 0, cfg)
        self.assertEqual(pair[0]['refresh_count'], 0)
        self.assertEqual(pair[0]['mae'], pair[1]['mae'])
        self.assertIsNone(pair[0]['long_term_error_drop_per_refresh'])

    def test_coverage_excludes_entire_failed_pair_actions(self):
        attempts = []
        for repeat in range(2):
            attempts += [dict(subject_id='a', trajectory_id='a', B=100., parameter='q=.5', K=2, success=True, reason=None),
                         dict(subject_id='b', trajectory_id='b', B=100., parameter='q=.5', K=3, success=False, reason='no_fit')]
        c = summarize_coverage(attempts, 2)['overall']
        self.assertEqual(c['action_coverage'], .4)
        self.assertEqual(c['pair_coverage'], .5)
        self.assertEqual(c['unique_case_matched_actions'], 2)
        self.assertEqual(c['failed_patient_ids'], ['b'])

    def test_long_term_drop_matches_brute_force_omitting_one_action(self):
        rec, w, stats, cfg = self.case()
        stats['C']['0.5'] = [10., 10., 10.]
        _, generated = run_method(rec, w, stats, 'q=0.5', 320., 0, cfg)
        actions = packages(generated['transcript'])
        calendar = tuple(r['turn'] for r in generated['transcript'])
        _, log = replay_actions(rec, w, stats, 'q=0.5', 320., 0, cfg, actions, calendar, M1_TIME)
        from experiments.workflow_extensions.finite_cache.data import replay
        truth = w.evaluate(replay(rec))[w.target]
        def evaluate(skip):
            cache = np.zeros(len(w.roots))
            estimated = []
            byturn = {r['turn']: (j, r) for j, r in enumerate(log['transcript'])}
            for turn in range(1, rec['T'] + 1):
                if turn in byturn:
                    j, r = byturn[turn]
                    if j != skip:
                        for root, value in zip(r['roots'], r['values']):
                            cache[w.roots.index(root)] = value
                estimated.append(float(w.from_cache(cache)[w.target]))
            return np.abs(np.array(estimated) - truth)
        actual = evaluate(None)
        for j, r in enumerate(log['transcript'][1:], 1):
            delta = evaluate(j)[r['turn'] - 1:] - actual[r['turn'] - 1:]
            self.assertAlmostEqual(r['long_term_mean_error_drop'], float(delta.mean()))
            self.assertAlmostEqual(r['long_term_cumulative_error_drop'], float(delta.sum()))


if __name__ == '__main__':
    unittest.main()
