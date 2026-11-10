import copy
import unittest
import numpy as np

from data_mine.finite_trajectories import admission_stream
from experiments.workflow_extensions.finite_cache.data import PublicBatch, replay, split, significant_roots
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.finite_cache.public_stats import fit
from experiments.workflow_extensions.finite_cache.mechanisms import Cache, paired_uniforms
from experiments.workflow_extensions.finite_cache.analyze import first_reached, select


def fixture(sid='s0'):
    return {'template': 'ANEMIA', 'subject_id': sid, 'trajectory_id': f'ANEMIA:{sid}', 'T': 6,
            'initial_state': {'hb': 10., 'hct': 30., 'rbc': 3.},
            'events': [{'turn': 3, 'batch_id': 'b1', 'available_time': 'time1',
                        'measured_roots': ['hb', 'hct'], 'values': {'hb': 11., 'hct': 30.}},
                       {'turn': 5, 'batch_id': 'b2', 'available_time': 'time2',
                        'measured_roots': ['rbc'], 'values': {'rbc': 4.}}],
            'queries': [{'turn': 2, 'node': 'mchc'}, {'turn': 4, 'node': 'mchc'}]}


class Protocol(unittest.TestCase):
    def make(self, **kw):
        args = dict(roots=['a', 'b', 'unused'], h=[1., .25, 0.], C=[.1, .2, 0.], B=100., method='M1-a', N_max=100)
        args.update(kw)
        return Cache(**args)

    def test_pathwise_budget_and_positive_publications(self):
        for method in ['B0', 'B2', 'B2p', 'M1-a', 'M1-g', 'M1-b', 'M1-a_fixed']:
            c = self.make(method=method)
            c.initialize(np.array([.2, .3, .4]), np.array([.6, .4, .7]))
            for turn in range(2, 102):
                decision = c.decide(PublicBatch(turn, ('a', 'b')))
                if decision.trigger not in ('none', 'empty', 'frozen'):
                    c.publish(decision, np.array([.7, .5, .1]), np.array([.3, .8, .7]))
                self.assertLessEqual(c.spent, c.B + 1e-9)
                self.assertAlmostEqual(sum(c.charges), c.spent)
                self.assertTrue(all(e > 0 for r in c.transcript for e in r['eps']))
            self.assertTrue(all('unused' not in r['roots'] for r in c.transcript))

    def test_schedule_invariant_to_true_values_and_noise(self):
        a, b = self.make(), self.make()
        a.initialize(np.array([.1, .1, .1]), np.array([.1, .2, .3]))
        b.initialize(np.array([.9, .8, .2]), np.array([.8, .6, .2]))
        for turn in range(2, 32):
            batch = PublicBatch(turn, ('a',) if turn % 2 else ('a', 'b'))
            da, db = a.decide(batch), b.decide(batch)
            self.assertEqual(da, db)
            if da.trigger not in ('none', 'empty', 'frozen'):
                a.publish(da, np.array([.1, .4, .2]), np.array([.5, .4, .1]))
                b.publish(db, np.array([.9, .3, .6]), np.array([.2, .6, .9]))
            self.assertEqual(a.snapshot(), b.snapshot())

    def test_skip_and_reanchor_overwrite(self):
        c = Cache(['a'], [1.], [0.], 20., 'M1-g', 5)
        c.initialize([.3], [.75])
        spent, cache = c.spent, c.cache.copy()
        decision = c.decide(PublicBatch(2, ('a',)))
        self.assertEqual(decision.trigger, 'none')
        self.assertEqual(c.spent, spent)
        np.testing.assert_equal(c.cache, cache)
        c = Cache(['a'], [1.], [.1], 100., 'B2p', 4, period=1)
        c.initialize([.1], [.9])
        d = c.decide(PublicBatch(2, ('a',)))
        c.publish(d, [.7], [.5])
        self.assertAlmostEqual(c.cache[0], .7)
        self.assertAlmostEqual(c.last_eps[0], sum(d.eps))

    def test_empty_Czero_quota_and_freeze(self):
        c = self.make(C=[0., 0., 0.])
        c.initialize([.1, .2, .3], [.5, .5, .5])
        self.assertTrue(np.all(np.isinf(c.G)))
        self.assertEqual(c.decide(PublicBatch(2, ('unused',))).trigger, 'empty')
        c.root_remaining[:] = -1000
        d = c.decide(PublicBatch(3, ('a', 'b')))
        self.assertTrue(all(e > 0 for e in d.eps))
        f = self.make(B=.001)
        f.initialize([.1, .2, .3], [.5, .5, .5])
        self.assertTrue(f.frozen)
        self.assertEqual(f.decide(PublicBatch(2, ('a',))).trigger, 'frozen')

    def test_sparse_ages_and_unmeasured_cache(self):
        c = self.make(C=[.05, .2, 0.])
        c.initialize([.1, .2, .3], [.5, .5, .5])
        for turn in range(2, 8):
            d = c.decide(PublicBatch(turn, ('a',)))
            old = c.cache[1]
            if d.trigger not in ('none', 'empty', 'frozen'):
                c.publish(d, [.8, .9, .4], [.5, .5, .5])
            self.assertEqual(c.cache[1], old)
        d = c.decide(PublicBatch(8, ('b',)))
        self.assertTrue(d.age_force)
        self.assertGreater(d.ages_before[1], c.G[1])
        c.publish(d, [.8, .9, .4], [.5, .5, .5])
        self.assertEqual(c.ages[1], 0)

    def test_public_Nmax_zero_future_batches(self):
        c = self.make(N_max=0)
        c.initialize([.1, .2, .3], [.5, .5, .5])
        self.assertAlmostEqual(c.spent, c.B)
        with self.assertRaises(ValueError):
            c.decide(PublicBatch(2, ('a',)))

    def test_age_ablation_always_disables_forcing(self):
        for mode in ['global', 'per_root', 'none']:
            c = self.make(method='M1-g', age_mode=mode)
            self.assertTrue(np.all(np.isinf(c.G)))

    def test_local_budget_refinement(self):
        from experiments.workflow_extensions.finite_cache.run import refinement_points
        budgets = [100., 110., 121., 133.1]
        extra = refinement_points(budgets, [121., None, 100.])
        self.assertEqual(len(extra), 1)
        self.assertAlmostEqual(extra[0], 115.5)
        self.assertEqual(refinement_points(budgets, [None, 100.]), [])
        refined = sorted([110., 121.] + extra)
        self.assertTrue(all(b / a <= 1.05 + 1e-10 for a, b in zip(refined, refined[1:])))

    def test_paired_random_stream(self):
        a = paired_uniforms(1, 'p', 'trajectory', 100, 0, 4, 3)
        np.testing.assert_array_equal(a, paired_uniforms(1, 'p', 'trajectory', 100, 0, 4, 3))
        self.assertFalse(np.array_equal(a, paired_uniforms(1, 'p', 'trajectory', 100, 1, 4, 3)))
        self.assertFalse(np.array_equal(a, paired_uniforms(1, 'p', 'trajectory', 100, 0, 5, 3)))

    def test_replay_and_train_only_stats(self):
        r = fixture()
        x = replay(r)
        np.testing.assert_array_equal(x[2], [11, 30, 3])
        np.testing.assert_array_equal(x[4], [11, 30, 4])
        s = fit([r], Workload('ANEMIA'), [.5])
        self.assertEqual(s['drift_samples'], [1, 1, 1])
        self.assertEqual(s['C']['0.5'][1], 0.)
        self.assertEqual(s['h'][2], 0.)
        self.assertEqual(s['train_subject_ids'], ['s0'])

    def test_patient_group_split(self):
        records = [fixture(str(i)) for i in range(12)]
        second = copy.deepcopy(records)
        for r in second:
            r['template'] = 'FIB4'
        s = split(records + second)
        self.assertEqual(len(s), 12)
        self.assertEqual(set(s.values()), {'train', 'validation', 'test'})
        self.assertEqual(s, split(records))

    def test_formulas_and_global_derivative_bounds(self):
        self.assertAlmostEqual(float(Workload('ANEMIA').evaluate([10, 30, 3])['mchc']), 100 / 3)
        self.assertAlmostEqual(float(Workload('FIB4').evaluate([55, 80, 64, 150])['fib4']), 55 * 80 / 1200)
        rng = np.random.default_rng(41)
        for template in ['ANEMIA', 'FIB4', 'NLR', 'TYG', 'MELD_NA']:
            w = Workload(template)
            bound = w.derivative_bounds()
            for _ in range(30):
                x = w.denormalize(rng.uniform(.05, .95, len(w.roots)))
                self.assertTrue(np.all(np.abs(w.jacobian(x)) <= bound * (1 + 1e-4) + 1e-5))
            self.assertTrue(all(np.isfinite(v) for v in w.from_cache(np.full(len(w.roots), -100.)).values()))

    def test_finite_grid_nonmonotonic(self):
        curve = [{'B': 10, 'mean': 4}, {'B': 20, 'mean': 6}, {'B': 40, 'mean': 3}]
        self.assertEqual(first_reached(curve, 5), 10)
        self.assertIsNone(first_reached(curve, 2))
        rows = [dict(r, split='validation', template='ANEMIA', method='B2', parameter='P=2') for r in curve]
        self.assertEqual(select(rows, 'ANEMIA', 'B2', 5)['parameter'], 'P=2')

    def test_strict_significant_threshold_excludes_decimal_equality(self):
        self.assertGreater(.55 - .5, .05)  # The actual numerical pitfall.
        self.assertEqual(significant_roots([.5], [.55], [1.], ['na'], .05), set())
        self.assertEqual(significant_roots([.5], [.551], [1.], ['na'], .05), {'na'})
        self.assertEqual(significant_roots([.5], [.549], [1.], ['na'], .05), set())

    def test_retain_measured_unchanged_batch(self):
        def row(value, time):
            return {'valuenum': str(value), 'charttime': time, 'storetime': time}
        batches = [('1', {k: row(v, '1') for k, v in [('hb', 10), ('hct', 30), ('rbc', 3)]}),
                   ('2', {'hb': row(10, '2')})]
        stream = admission_stream('ANEMIA', 'p', batches, {})
        self.assertEqual(len(stream), 2)
        self.assertEqual(stream[0]['state'], stream[1]['state'])
        self.assertEqual(set(stream[1]['rows']), {'hb'})

    def test_document_single_root_noise_comparison(self):
        c = Cache(['a'], [1.], [0.], 20., 'M1-g', 10, c=.6)
        c.initialize([.3], [.75])
        d = c.decide(PublicBatch(2, ('a',)))
        self.assertAlmostEqual(c.last_eps[0], 10.)
        self.assertAlmostEqual(d.L_keep, .1)
        self.assertAlmostEqual(d.L_refresh, 1 / 6)
        self.assertEqual(d.trigger, 'none')

    def test_baseline_and_gate_share_candidate_allocation(self):
        a, b = self.make(method='B2p', period=1), self.make(C=[1., 1., 0.])
        for c in [a, b]:
            c.initialize([.2, .3, .4], [.5, .5, .5])
        batch = PublicBatch(2, ('a', 'b'))
        da, db = a.decide(batch), b.decide(batch)
        self.assertEqual(da.eps, db.eps)
        self.assertNotEqual(da.trigger, 'none')
        self.assertNotEqual(db.trigger, 'none')

    def test_future_values_do_not_affect_prefix(self):
        from experiments.workflow_extensions.finite_cache.run import simulate
        r = fixture('test')
        future = copy.deepcopy(r)
        future['events'][1]['values']['rbc'] = 8.
        stats = fit([fixture('train')], Workload('ANEMIA'), [.5])
        config = {'seed': 1, 'quantiles': [.5], 'significant_delta': .02, 'mechanism': {}}
        _, a = simulate(r, Workload('ANEMIA'), stats, 'M1-a', 'q=0.5', 100., 0, config, detailed=True)
        _, b = simulate(future, Workload('ANEMIA'), stats, 'M1-a', 'q=0.5', 100., 0, config, detailed=True)
        self.assertEqual(a['rounds'][:4], b['rounds'][:4])

    def test_aoi_uses_dialogue_rounds_not_batches(self):
        from experiments.workflow_extensions.finite_cache.run import simulate
        r = fixture('test')
        stats = fit([fixture('train')], Workload('ANEMIA'), [.5])
        config = {'seed': 1, 'quantiles': [.5], 'significant_delta': .02, 'mechanism': {}}
        metrics, _ = simulate(r, Workload('ANEMIA'), stats, 'B0', 'static', 100., 0, config)
        self.assertEqual(metrics['aoi_mean'], 2.5)  # Age 0,1,2,3,4,5 over six rounds.
        self.assertGreaterEqual(metrics['p95_abs_error'], 0.)

    def test_periodic_refresh_is_not_attributed_to_age_force(self):
        from experiments.workflow_extensions.finite_cache.run import simulate
        stats = fit([fixture('train')], Workload('ANEMIA'), [.5])
        stats['C']['0.5'] = [1., 1., 1.]  # Age predicate is true, but periodic policy ignores it.
        config = {'seed': 1, 'quantiles': [.5], 'significant_delta': .02, 'mechanism': {}}
        metric, _ = simulate(fixture('test'), Workload('ANEMIA'), stats, 'B2p', 'P=1', 100., 0, config)
        self.assertGreater(metric['refresh_count'], 0)
        self.assertEqual(metric['age_refreshes'], 0)

    def test_batched_repeats_match_reference_metrics_and_releases(self):
        from experiments.workflow_extensions.finite_cache.run import simulate
        from experiments.workflow_extensions.finite_cache.batch import simulate_repeats
        for template in ['ANEMIA', 'MELD_NA']:
            rec = fixture('test')
            if template == 'MELD_NA':
                rec.update(template=template, initial_state={'bil': 1.2, 'inr': 1.3, 'cr': 1., 'na': 135.})
                rec['events'][0].update(measured_roots=['bil', 'na'], values={'bil': 2., 'na': 135.})
                rec['events'][1].update(measured_roots=['cr'], values={'cr': 2.5})
                rec['queries'] = [{'turn': 2, 'node': 'meld_na'}, {'turn': 4, 'node': 'meld'}]
            train = copy.deepcopy(rec)
            train['subject_id'] = 'train'
            w = Workload(template)
            stats = fit([train], w, [.5, .95])
            config = {'seed': 1, 'quantiles': [.5, .95], 'significant_delta': .05, 'mechanism': {}, 'save_logs': True}
            for method, param in [('B0', 'static'), ('B2', 'P=2'), ('B2p', 'P=2'), ('M1-a', 'q=0.5'), ('M1-g', 'q=0.5')]:
                for B in [1., 320.]:
                    batch = simulate_repeats(rec, w, stats, method, param, B, 3, config)
                    for r, (actual, log) in enumerate(batch):
                        expected, reference = simulate(rec, w, stats, method, param, B, r, config, detailed=True)
                        self.assertEqual(set(actual), set(expected))
                        for key, v in expected.items():
                            if isinstance(v, float):
                                np.testing.assert_allclose(actual[key], v, rtol=1e-10, atol=1e-10, err_msg=key)
                            else:
                                self.assertEqual(actual[key], v, key)
                        self.assertEqual(log['decisions'], reference['decisions'])
                        self.assertEqual(len(log['transcript']), len(reference['transcript']))
                        for a, b in zip(log['transcript'], reference['transcript']):
                            self.assertEqual(a['roots'], b['roots'])
                            self.assertEqual(a['eps'], b['eps'])
                            np.testing.assert_allclose(a['values'], b['values'], rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
