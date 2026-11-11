import copy
from itertools import combinations
import unittest
import numpy as np

from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.mechanisms import Cache
from experiments.workflow_extensions.finite_cache.public_stats import fit
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.finite_cache.matched_timing import (
    action_uniforms, uniform_turns, run_pair, run_method)
from experiments.workflow_extensions.finite_cache.matched_analysis import paired_interval, summarize_cell
from .test_protocol import fixture


class MatchedTiming(unittest.TestCase):
    def setup_case(self, C=None):
        rec = fixture('test')
        w = Workload('ANEMIA')
        stats = fit([fixture('train')], w, [.5])
        if C is not None:
            stats['C']['0.5'] = C
        cfg = {'seed': 20261110, 'quantiles': [.5], 'significant_delta': .05,
               'mechanism': {'eps_min': .01}}
        return rec, w, stats, cfg

    def test_uniform_calendar_is_exact_deterministic_distinct_optimum(self):
        turns, T = [2, 3, 8, 13, 15, 19], 20
        for K in range(len(turns) + 1):
            actual = uniform_turns(turns, K, T)
            def objective(path):
                return (sum(((t - 1) * (K + 1) - j * (T - 1)) ** 2 for j, t in enumerate(path, 1)), path)
            expected = min(combinations(turns, K), key=objective)
            self.assertEqual(actual, expected)
            self.assertEqual(actual, uniform_turns(turns, K, T))
            self.assertEqual(len(set(actual)), K)
        with self.assertRaises(ValueError):
            uniform_turns([2, 2], 1, T)
        with self.assertRaises(ValueError):
            uniform_turns(turns, 7, T)

    def test_noise_key_is_action_and_named_root_not_turn_or_position(self):
        key = (1, 's', 't', 320., 2, 3)
        a = action_uniforms(*key, ['hb', 'rbc'])
        b = action_uniforms(*key, ['rbc', 'hct', 'hb'])
        np.testing.assert_array_equal(a, b[[2, 0]])
        self.assertNotEqual(a[0], action_uniforms(1, 's', 't', 320., 2, 4, ['hb'])[0])
        self.assertNotEqual(a[0], action_uniforms(1, 's', 't', 320., 3, 3, ['hb'])[0])

    def test_forced_api_uses_same_allocator_and_rejects_unavailable_or_over_budget(self):
        a, b = [Cache(['a', 'b'], [1., 4.], [.1, .2], 100., 'M1-a', 10) for _ in range(2)]
        for c in [a, b]:
            c.initialize([.3, .5], [.5, .5])
        batch = data.PublicBatch(2, ('a', 'b'))
        candidate = a.decide(batch)
        forced = b.decide_scheduled(batch, sum(candidate.eps))
        self.assertEqual(candidate.eps, forced.eps)
        with self.assertRaises(TypeError):
            a.decide({'turn': 3, 'values': [1, 2]})
        c = Cache(['a'], [1.], [.1], 10., 'M1-a', 5)
        c.initialize([.4], [.5])
        with self.assertRaises(RuntimeError):
            c.decide_scheduled(data.PublicBatch(2, ('unavailable',)), 1.)
        with self.assertRaises(ValueError):
            c.decide_scheduled(data.PublicBatch(3, ('a',)), 6.)

    def test_zero_refresh_pair_matches_initialization_and_entire_trajectory(self):
        rec, w, stats, cfg = self.setup_case([0., 0., 0.])
        pair, schedule = run_pair(rec, w, stats, 'q=0.5', 10., 0, cfg)
        self.assertEqual(pair[0]['refresh_count'], 0)
        self.assertEqual(pair[1]['refresh_count'], 0)
        self.assertEqual(pair[0]['mae'], pair[1]['mae'])
        self.assertEqual(pair[0]['spent'], pair[1]['spent'])
        self.assertEqual(schedule['M1-a']['transcript'], schedule['matched_periodic']['transcript'])
        self.assertIsNone(pair[0]['immediate_error_drop_per_refresh'])

    def test_all_opportunities_pair_identical_values_and_cost(self):
        rec, w, stats, cfg = self.setup_case([10., 10., 10.])
        stats['h'] = [1., 1., 1.]  # Include the fixture's otherwise unused rbc.
        pair, schedule = run_pair(rec, w, stats, 'q=0.5', 320., 0, cfg)
        self.assertEqual(pair[0]['refresh_count'], len(rec['events']))
        self.assertEqual(pair[0]['mae'], pair[1]['mae'])
        self.assertEqual(schedule['M1-a']['transcript'], schedule['matched_periodic']['transcript'])

    def test_sparse_root_selection_and_total_cost_matching(self):
        rec, w, stats, cfg = self.setup_case([.15, .15, .15])
        rec['T'] = 18
        rec['events'] = []
        for i, t in enumerate([2, 3, 6, 9, 12, 15, 17]):
            roots = ['hb', 'hct'] if i % 2 else ['rbc']
            rec['events'].append({'turn': t, 'batch_id': str(i), 'available_time': str(i),
                                  'measured_roots': roots,
                                  'values': {r: 3. + i if r == 'rbc' else 15. + i for r in roots}})
        for repeat in range(3):
            pair, schedule = run_pair(rec, w, stats, 'q=0.5', 320., repeat, cfg)
            self.assertGreater(pair[0]['refresh_count'], 0)
            self.assertEqual(pair[0]['refresh_count'], pair[1]['refresh_count'])
            self.assertLess(abs(pair[0]['spent'] - pair[1]['spent']), 1e-10)
            self.assertLessEqual(pair[0]['spent'], 320. + 1e-10)
            legal = {e['turn']: set(e['measured_roots']) for e in rec['events']}
            for method in ['M1-a', 'matched_periodic']:
                for release in schedule[method]['transcript'][1:]:
                    self.assertLessEqual(set(release['roots']), legal[release['turn']])

    def test_m1_decisions_ignore_all_truth_and_future_calendar_prefix(self):
        rec, w, stats, cfg = self.setup_case([.15, .15, .15])
        _, original = run_method(rec, w, stats, 'q=0.5', 320., 0, cfg)
        altered = copy.deepcopy(rec)
        altered['initial_state'] = {'hb': 24., 'hct': 70., 'rbc': 9.}
        for event in altered['events']:
            event['values'] = {root: 9. for root in event['measured_roots']}
        _, mutated = run_method(altered, w, stats, 'q=0.5', 320., 1, cfg)
        self.assertEqual(original['decisions'], mutated['decisions'])
        # Even future public metadata cannot change the prior M1 decision.
        altered['events'][1]['measured_roots'] = ['hb']
        altered['events'][1]['values'] = {'hb': 18.}
        _, changed_future = run_method(altered, w, stats, 'q=0.5', 320., 1, cfg)
        self.assertEqual(original['decisions'][0], changed_future['decisions'][0])

    def test_bootstrap_pairs_methods_and_preserves_patient_equal_weight(self):
        ci = paired_interval([[0., 0.], [0., 0.]], draws=100)
        self.assertEqual(ci['patient_ci'], [0., 0.])
        self.assertEqual(ci['hierarchical_ci'], [0., 0.])
        rows = []
        for sid, trajectories, delta in [('a', 1, -3.), ('b', 3, 1.)]:
            for tr in range(trajectories):
                for repeat in range(2):
                    for method, mae in [('M1-a', 10. + delta), ('matched_periodic', 10.)]:
                        rows.append(dict(subject_id=sid, trajectory_id=f'{sid}:{tr}', B=100., repeat=repeat,
                                         method=method, mae=mae, template='MELD_NA', parameter='q=0.5'))
        summary = summarize_cell(rows, ['mae'], 'test', 100., 100, 1)[0]
        self.assertEqual(summary['delta'], -1.)  # Equal patients, not equal trajectories.
        self.assertEqual(summary['n_patients'], 2)


if __name__ == '__main__':
    unittest.main()
