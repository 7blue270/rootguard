from itertools import combinations
import unittest
import numpy as np

from experiments.workflow_extensions.finite_cache.public_stats import fit
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.finite_cache.matched_timing import run_method
from experiments.workflow_extensions.finite_cache.pure_timing import Action, packages, public_opportunities, replay_actions
from experiments.workflow_extensions.finite_cache.timing_supplement import ScheduleSpace, Evaluator, exact_oracle
from .test_protocol import fixture


class TimingSupplement(unittest.TestCase):
    def case(self):
        rec = fixture('test')
        rec['T'] = 14
        rec['events'] = []
        for j, t in enumerate([2, 4, 6, 9, 11, 13]):
            root = ['hb', 'hct', 'rbc'][j % 3]
            rec['events'].append({'turn': t, 'batch_id': str(j), 'available_time': str(j),
                                  'measured_roots': [root], 'values': {root: 5. + j}})
        w = Workload('ANEMIA')
        stats = fit([rec], w, [.5])
        stats['h'] = [1., 1., 1.]
        stats['C']['0.5'] = [.007, .007, .007]
        cfg = {'seed': 17, 'quantiles': [.5], 'significant_delta': .05, 'mechanism': {}}
        generated = [run_method(rec, w, stats, 'q=0.5', 320., repeat, cfg)[1] for repeat in range(2)]
        actions = [packages(r['transcript']) for r in generated]
        return rec, w, stats, cfg, actions

    def test_exact_counts_and_unranking_are_bijection_on_conditional_legal_space(self):
        actions = (Action(1, ('b',), (1.,), (.1,)), Action(2, ('a',), (1.,), (.2,)))
        opp = [(2, frozenset('a')), (3, frozenset('ab')), (5, frozenset('ab')), (7, frozenset('ab'))]
        space = ScheduleSpace(actions, opp)
        expected = [(1,) + t for t in combinations([2, 3, 5, 7], 2) if t[0] >= 3]
        self.assertEqual(space.total, len(expected))
        self.assertEqual([space.unrank(i) for i in range(space.total)], expected)
        np.testing.assert_array_equal(space.enumerate(), expected)
        with self.assertRaises(ValueError):
            space.unrank(space.total)

    def test_uniform_rank_sampling_seed_is_reproducible_and_keeps_duplicates(self):
        space = ScheduleSpace((Action(1, ('a',), (1.,), (.1,)),), [(2, frozenset('a')), (3, frozenset('a'))])
        a, ranks, seed = space.sample(123, 's', 'trajectory', 100., 0, 50)
        b, rr, ss = space.sample(123, 's', 'trajectory', 100., 0, 50)
        np.testing.assert_array_equal(a, b)
        self.assertEqual((ranks, seed), (rr, ss))
        self.assertEqual(len(ranks), 50)
        self.assertLess(len(set(ranks)), 50)  # No filtering of repeated calendars.
        self.assertTrue(all(tuple(path) == space.unrank(rank) for path, rank in zip(a, ranks)))
        self.assertNotEqual(ranks, space.sample(123, 's', 'trajectory', 100., 1, 50)[1])

    def test_cdf_interval_cost_equals_direct_absolute_sum(self):
        rec, w, stats, cfg, actions = self.case()
        ev = Evaluator(rec, w, stats, cfg)
        rng = np.random.default_rng(7)
        estimates = rng.uniform(0, 100, size=(2, 30))
        starts = rng.integers(rec['T'], size=(1, 30))
        ends = starts + rng.integers(0, rec['T'] - starts + 1)
        result = ev.interval_cost(estimates, starts, ends)
        for r in range(2):
            for j in range(30):
                expected = np.abs(estimates[r, j] - ev.target[starts[0, j]:ends[0, j]]).sum()
                self.assertAlmostEqual(result[r, j], expected, places=10)

    def test_all_calendar_fast_metrics_and_oracle_equal_shared_reference_replay(self):
        rec, w, stats, cfg, actions_r = self.case()
        space = ScheduleSpace(actions_r[0][1:], public_opportunities(rec))
        calendars = space.enumerate()
        self.assertGreater(space.total, 1)
        ev = Evaluator(rec, w, stats, cfg)
        scores = ev.score_many_repeats(actions_r, calendars)
        brute = []
        for repeat, actions in enumerate(actions_r):
            metrics = ev.metrics(actions, calendars)
            errors = []
            for j, calendar in enumerate(calendars):
                reference, _ = replay_actions(rec, w, stats, 'q=0.5', 320., repeat, cfg, actions, tuple(calendar), 'test')
                for key, value in metrics[j].items():
                    if value is None:
                        self.assertIsNone(reference[key])
                    else:
                        self.assertAlmostEqual(value, reference[key], places=10, msg=key)
                self.assertAlmostEqual(scores[repeat, j], reference['mae'], places=10)
                errors.append(reference['mae'])
            brute.append(errors)
        best, objective, info = exact_oracle(ev, actions_r, calendars, chunk_size=3)
        np.testing.assert_allclose(objective, np.min(brute, axis=1), atol=1e-10, rtol=0)
        for repeat, cal in enumerate(best):
            actual = ev.metrics(actions_r[repeat], np.array([cal]))[0]['mae']
            self.assertAlmostEqual(actual, min(brute[repeat]), places=10)
        self.assertEqual(info['evaluated_per_repeat'], space.total)
        self.assertEqual(info['optimality_gap'], 0.)

    def test_empty_action_space_has_one_calendar_and_constant_timing(self):
        rec, w, stats, cfg, actions = self.case()
        initial = actions[0][:1]
        space = ScheduleSpace((), public_opportunities(rec))
        self.assertEqual(space.total, 1)
        np.testing.assert_array_equal(space.enumerate(), [[1]])
        ev = Evaluator(rec, w, stats, cfg)
        scores = ev.score_many_repeats([initial], np.array([[1]]))
        metrics = ev.metrics(initial, np.array([[1]]))
        self.assertAlmostEqual(scores[0, 0], metrics[0]['mae'], places=10)
        self.assertEqual(metrics[0]['refresh_count'], 0)


if __name__ == '__main__':
    unittest.main()
