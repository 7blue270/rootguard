import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from data_mine.build_bun_cr import chart_stream
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.finite_cache.run import run, refinement_points
from experiments.workflow_extensions.finite_cache.round2_report import crossing_draws, ratios
from .test_protocol import fixture


class Round2(unittest.TestCase):
    def test_frontier_scans_nonmonotone_curves(self):
        result=crossing_draws([80.,88.,96.],[[1.,3.],[3.,2.],[0.,1.]],1.5)
        self.assertEqual(result,[80.,96.])

    def test_conditional_ratio_cannot_certify_win_when_most_pairs_censored(self):
        a={'B_min':80.,'patient_draws':[80.,None],'hierarchical_draws':[80.,None]}
        b={'B_min':160.,'patient_draws':[160.,160.],'hierarchical_draws':[160.,160.]}
        result=ratios(a,b)
        self.assertEqual(result['ratio'],.5)
        self.assertEqual(result['significance'],'insufficient_or_censored')
        self.assertEqual(result['patient_both_reached_fraction'],.5)

    def test_bun_formula_and_global_bounds(self):
        w = Workload('BUN_CR')
        self.assertEqual(w.roots, ['bun','cr'])
        self.assertAlmostEqual(float(w.evaluate([40.,2.])['ratio']),20.)
        self.assertIsNone(w.classes([20.]))
        for z in [[.1,.1],[.4,.6],[.8,.2]]:
            self.assertTrue(np.all(np.abs(w.jacobian(w.denormalize(z))) <= w.derivative_bounds()*(1+1e-4)))

    def test_charttime_stream_waits_for_storetime_and_counts_real_changes(self):
        def row(v,c,s): return {'valuenum':str(v),'charttime':c,'storetime':s}
        batches=[('2',{'bun':row(20,'2','2')}),('1',{'bun':row(10,'1','5'),'cr':row(1,'1','5')}),
                 ('3',{'bun':row(20,'3','3')}),('4',{'cr':row(2,'4','4')})]
        states=chart_stream(batches)
        self.assertEqual([r['measurement_time'] for r in states],['1','2','4'])
        self.assertEqual([r['available_time'] for r in states],['5','5','5'])
        self.assertEqual(states[-1]['state'],{'bun':20.,'cr':2.})

    def test_temporal_median_is_constant_predictor_lower_bound_not_initial_floor(self):
        truth=np.array([1.,5.,5.,7.])
        floor=np.abs(truth-np.median(truth)).mean()
        self.assertLess(floor,np.abs(truth-truth[0]).mean())
        for constant in np.linspace(-10,20,100):
            self.assertGreaterEqual(np.abs(truth-constant).mean()+1e-12,floor)

    def test_runner_sensitivity_is_separate_from_validation_and_primary_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inp=root/'input'; inp.mkdir()
            records=[]
            for i in range(10):
                r=fixture(str(i));r.update(schema_version='finite-cache-v1',initial_available_time='0')
                records.append(r)
            (inp/'ANEMIA.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            (inp/'manifest.json').write_text('{}')
            cfg={'templates':['ANEMIA'],'seed':17,'split_fractions':[.5,.25],'repeats':2,'bootstrap':20,
                 'budgets':[80.,160.],'periods':[2],'quantiles':[.5],'targets':{'ANEMIA':[1.,2.,3.]},
                 'mechanism':{},'significant_delta':.05,'refine':True,'save_logs':True,'log_repeats':1}
            out=root/'output'
            run(cfg,out,inp,refinement_methods=['B0','B2','B2p','M1-a','M1-g'],sensitivity_periods=[2,3])
            metrics=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
            diagnostic=[r for r in metrics if r['split']=='test_period_sensitivity']
            self.assertTrue(diagnostic)
            self.assertEqual({r['parameter'] for r in diagnostic},{'P=2','P=3'})
            selected=json.loads((out/'selections.json').read_text())
            self.assertTrue(all(s['B2']['parameter']=='P=2' for s in selected['ANEMIA'].values()))


if __name__=='__main__':unittest.main()
