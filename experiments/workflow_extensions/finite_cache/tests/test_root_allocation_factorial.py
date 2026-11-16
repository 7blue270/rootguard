import unittest
import numpy as np

from experiments.workflow_extensions.finite_cache.public_stats import fit
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.finite_cache.matched_timing import run_pair as matched_pair
from experiments.workflow_extensions.finite_cache.pure_timing import run_pair as pure_pair
from experiments.workflow_extensions.finite_cache.root_allocation_factorial import (
    allocate_package, replay_cube, CELLS, label)
from experiments.workflow_extensions.finite_cache.root_allocation_effects import effect_definitions, weight_vector
from .test_protocol import fixture


class RootAllocationFactorial(unittest.TestCase):
    def case(self):
        rec = fixture('test')
        rec['T'] = 18
        rec['events'] = []
        for j, t in enumerate([2, 4, 7, 9, 12, 15, 17]):
            root = ['hb', 'hct', 'rbc'][j % 3]
            rec['events'].append({'turn': t, 'batch_id': str(j), 'available_time': str(j),
                                  'measured_roots': [root], 'values': {root: 4. + j}})
        w = Workload('ANEMIA')
        stats = fit([fixture('train')], w, [.5])
        stats['h'] = [1., 1., 1.]
        stats['C']['0.5'] = [.007, .007, .007]
        cfg = {'seed': 17, 'quantiles': [.5], 'significant_delta': .05, 'mechanism': {}}
        return rec, w, stats, cfg

    def test_all_eight_cells_reproduce_baselines_and_fix_cost_noise(self):
        rec, w, stats, cfg = self.case()
        for repeat in range(2):
            old_metrics, matched = matched_pair(rec, w, stats, 'q=0.5', 320., repeat, cfg)
            pure_metrics, pure, failure = pure_pair(rec, w, stats, 'q=0.5', 320., repeat, cfg)
            self.assertIsNone(failure)
            rows, logs, audit = replay_cube(rec, w, stats, 'q=0.5', 320., repeat, cfg, pure, matched)
            self.assertEqual(len(rows), 8)
            self.assertEqual({(r['T'],r['S'],r['A']) for r in rows}, set(CELLS))
            bycell = {(r['T'], r['S'], r['A']): r for r in rows}
            self.assertAlmostEqual(bycell[('M1_time','M1_roots','quota')]['mae'], pure_metrics[0]['mae'], places=10)
            self.assertAlmostEqual(bycell[('uniform_time','M1_roots','quota')]['mae'], pure_metrics[1]['mae'], places=10)
            self.assertAlmostEqual(bycell[('uniform_time','uniform_roots','quota')]['mae'], old_metrics[1]['mae'], places=10)
            self.assertLess(max(r['spent'] for r in rows)-min(r['spent'] for r in rows), 1e-10)
            self.assertEqual(len({r['refresh_count'] for r in rows}), 1)
            for s in ['M1_roots', 'uniform_roots']:
                for a in ['quota', 'sqrt_h']:
                    pair = [r for r in logs if r['S'] == s and r['A'] == a]
                    self.assertEqual([v['eps'] for v in pair[0]['transcript']], [v['eps'] for v in pair[1]['transcript']])

    def test_sqrt_h_reduces_formula_and_ledgers_are_independent(self):
        rec, w, stats, cfg = self.case()
        stats['h'] = [1., 4., 9.]
        roots = [('hb','hct','rbc'), ('hb','hct'), ('hb','rbc'), ('hb','hct')]
        totals = [160., 20., 20., 20.]
        eps_h, ledger_h, _ = allocate_package(rec,w,stats,'q=0.5',320.,cfg,roots,totals,'sqrt_h')
        eps_q, ledger_q, _ = allocate_package(rec,w,stats,'q=0.5',320.,cfg,roots,totals,'quota')
        self.assertEqual(eps_h[0], eps_q[0])
        np.testing.assert_allclose(eps_h[1], [20/3,40/3])
        np.testing.assert_allclose(eps_h[2], [5.,15.])
        self.assertFalse(np.allclose(eps_h[-1], eps_q[-1]))
        again, _, _ = allocate_package(rec,w,stats,'q=0.5',320.,cfg,roots,totals,'quota')
        self.assertEqual(again,eps_q)
        self.assertNotEqual(ledger_h[-1]['root_remaining_after'],ledger_q[-1]['root_remaining_after'])

    def test_analytic_interaction_and_shapley_endpoint_identities(self):
        # Known polynomial checks signs, interactions and all three endpoint schemes.
        values=[]
        for t,s,a in CELLS:
            t=int(t=='M1_time');s=int(s=='M1_roots');a=int(a=='quota')
            values.append(t+2*s+3*a+4*s*a+5*t*s+6*t*a+7*t*s*a)
        values=np.array(values)
        effects={d['effect']:float(values@weight_vector(d)) for d in effect_definitions()}
        self.assertAlmostEqual(effects['S_main'],8.25)
        self.assertAlmostEqual(effects['A_main'],9.75)
        self.assertAlmostEqual(effects['S_A_interaction|T=uniform_time'],4.)
        self.assertAlmostEqual(effects['S_A_interaction|T=M1_time'],11.)
        self.assertAlmostEqual(effects['S_A_interaction_mean'],7.5)
        self.assertAlmostEqual(effects['T_S_A_interaction'],7.)
        self.assertAlmostEqual(effects['original_M1_minus_matched_uniform'],25.)
        self.assertAlmostEqual(effects['direct_quota_shapley_T'],13.)
        self.assertAlmostEqual(effects['direct_quota_shapley_S'],12.)
        self.assertEqual(effects['direct_quota_shapley_A'],0.)
        self.assertAlmostEqual(effects['full_shapley_T'],1+2.5+3+7/3)
        self.assertAlmostEqual(effects['contrast_shapley_A'],2+3+7/3)
        self.assertAlmostEqual(sum(effects['contrast_shapley_'+f] for f in ['T','S','A']),25.)
        self.assertAlmostEqual(sum(effects['full_shapley_'+f] for f in ['T','S','A']),28.)

    def test_constant_cube_has_zero_effects_and_preserves_pairing(self):
        for d in effect_definitions():
            self.assertAlmostEqual(float(np.full(8,7.)@weight_vector(d)),0.,places=12)


if __name__=='__main__':
    unittest.main()
