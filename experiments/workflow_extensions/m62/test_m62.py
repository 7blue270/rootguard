import copy
import json
import unittest
from pathlib import Path
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from .protocol import ROOT,config,calendars,fold_validation,prior_bounds
from .engine import execute,schedule,real_replay
from .predict import generate,predict_grid
from .calibrate import estimate_delta
from .analysis import bootstrap


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=config(True);cls.records=data.load(ROOT/'data_mine/finite_cache',cls.cfg['templates'])
        cls.rec=next(r for r in cls.records if r['template']=='ANEMIA');cls.w=Workload('ANEMIA')

    def test_fixed_protocol_and_queries(self):
        self.assertEqual(self.cfg['refresh_turns'],[14,27]);self.assertEqual(schedule('D',[],self.cfg),[(1,.5),(14,.25),(27,.25)])
        self.assertEqual(self.cfg['budgets'],[80,160,320,480,640,960]);self.assertIsNone(self.cfg['eta'])

    def test_common_executor_zero_noise_constant_state(self):
        states=np.full((3,40,3),.3);noise=np.zeros((41,3,3));qw=np.ones((3,40))
        for m in ['S','D']:
            r=execute(states,80,schedule(m,[],self.cfg),noise,self.w,qw,detailed=True)
            np.testing.assert_allclose(r['mae'],0);np.testing.assert_allclose(r['spent'],80)
            np.testing.assert_allclose(r['histories'][0]['epsilon'].sum(),80 if m=='S' else 40)

    def test_early_stop_charges_only_executed_releases(self):
        states=np.full((2,40,3),.3);noise=np.zeros((41,2,3));qw=np.ones((2,40))
        for stop,total in [(10,40),(20,60),(40,80)]:
            r=execute(states,80,schedule('D',[],self.cfg),noise,self.w,qw,stop=stop)
            np.testing.assert_allclose(r['spent'],total)
        with self.assertRaises(AssertionError):execute(states,80,schedule('D',[],self.cfg),noise,self.w,qw,stop=41)
        r=execute(states,80,schedule('D',[],self.cfg),noise,self.w,np.zeros((2,40)),stop=2,detailed=True)
        self.assertTrue(np.isnan(r['mae']).all());np.testing.assert_allclose(r['spent'],40)

    def test_causal_future_change(self):
        rec=copy.deepcopy(self.rec);future=copy.deepcopy(rec)
        event=next(e for e in future['events'] if e['turn']>=27)
        for r in event['values']:event['values'][r]*=2
        a=real_replay(rec,320,'D',self.w,self.cfg,'unit',2,True)
        b=real_replay(future,320,'D',self.w,self.cfg,'unit',2,True)
        np.testing.assert_array_equal(a['caches'][:,:event['turn']-1],b['caches'][:,:event['turn']-1])

    def test_actual_actions_and_paired_initial_noise(self):
        a=real_replay(self.rec,320,'S',self.w,self.cfg,'unit',2,True)
        b=real_replay(self.rec,320,'D',self.w,self.cfg,'unit',2,True)
        z=self.w.normalize(data.replay(self.rec))[0]
        for r in range(2):
            # Recover unit noise only where both kernels are not clipped.
            x=a['histories'][0]['cache'][r];y=b['histories'][0]['cache'][r]
            ok=(x>0)&(x<1)&(y>0)&(y<1)
            np.testing.assert_allclose((x-z)[ok]*320/3,(y-z)[ok]*160/3,atol=1e-12)
        self.assertEqual(len(b['histories']),3)

    def test_reference_prior_does_not_shrink_privacy_domain(self):
        before=self.w.width.copy();lo,hi=prior_bounds(self.w,'L1')
        self.assertTrue(np.all(lo>=0)&np.all(hi<=1));np.testing.assert_array_equal(self.w.width,before)
        self.assertTrue(np.all(hi-lo<1))

    def test_public_generation_does_not_use_values(self):
        pub=calendars(self.records,'ANEMIA');changed=copy.deepcopy(self.records)
        for rec in changed:
            rec['initial_state']={r:999 for r in rec['initial_state']}
            for e in rec['events']:e['values']={r:999 for r in e['values']}
        self.assertEqual(pub,calendars(changed,'ANEMIA'))
        a=generate(self.w,pub,self.cfg,'L0',.01,0,'linear')[0]
        b=generate(self.w,calendars(changed,'ANEMIA'),self.cfg,'L0',.01,0,'linear')[0]
        np.testing.assert_array_equal(a,b)

    def test_linear_and_redraw_are_distinct_and_correlation_recorded(self):
        pub=calendars(self.records,'ANEMIA')
        a=generate(self.w,pub,self.cfg,'L0',.01,0,'linear')
        b=generate(self.w,pub,self.cfg,'L0',.01,0,'redraw')
        self.assertFalse(np.array_equal(a[0],b[0]))
        c=generate(self.w,pub,self.cfg,'L0',.01,.9,'linear')
        self.assertGreater(c[-1],.7)

    def test_whole_and_local_use_identical_input_losses(self):
        rows,raw=predict_grid(self.w,calendars(self.records,'ANEMIA'),self.cfg,'L0',.01,0,'linear',[320.],True)
        r=rows[0];v=raw['320.0']
        self.assertAlmostEqual(r['G_session'],np.mean(v['S']-v['D']))
        self.assertAlmostEqual(r['G_local'],np.mean(v['local_S']-v['local_D']))
        self.assertEqual(r['choice'],'D' if r['G_session']>0 else 'S')

    def test_deduplication_across_templates(self):
        rs=self.records[:2];a,ea=estimate_delta(rs)
        b,eb=estimate_delta(rs+copy.deepcopy(rs))
        self.assertEqual(a['delta'],b['delta']);self.assertEqual(a['valid_differences'],b['valid_differences'])
        self.assertEqual(a['unique_source_rows'],b['unique_source_rows'])

    def test_excluded_subject_mutation_cannot_change_L2(self):
        held=str(self.records[0]['subject_id']);a,_=estimate_delta(self.records,held);changed=copy.deepcopy(self.records)
        for rec in changed:
            if str(rec['subject_id'])==held:
                for ss in [rec['initial_source_rows']]+[e['source_rows'] for e in rec['events']]:
                    for row in ss.values():row['valuenum']='9999999'
        b,_=estimate_delta(changed,held);self.assertEqual(a['delta'],b['delta']);self.assertNotIn(held,a['subjects'])

    def test_subject_folds_globally_bound(self):
        subjects=sorted({str(r['subject_id']) for r in self.records})
        for s in subjects:
            tr,va=fold_validation(subjects,s,self.cfg)
            self.assertNotIn(s,tr);self.assertNotIn(s,va);self.assertTrue(set(va)<=set(tr))
            for t in self.cfg['templates']:
                if any(r['template']==t and str(r['subject_id'])==s for r in self.records):
                    self.assertTrue(any(r['template']==t and str(r['subject_id']) in va for r in self.records))

    def test_bootstrap_binds_patients_and_repeats_across_methods(self):
        x=np.arange(12.,dtype=float).reshape(1,1,1,3,4)
        x=np.concatenate([x,x+7],axis=0)
        patients=np.array([[0,0,2],[2,1,0]])
        repeats=np.array([[[0,0,1,1],[3,3,2,2],[1,2,3,0]],[[3,2,1,0],[0,0,0,0],[2,2,2,2]]])
        p,n=bootstrap(x,patients,repeats)
        np.testing.assert_allclose(p[1]-p[0],7)
        np.testing.assert_allclose(n[1]-n[0],7)
        for d,idx in enumerate(patients):
            expected=np.mean([np.mean(x[0,0,0,s,repeats[d,j]]) for j,s in enumerate(idx)])
            self.assertAlmostEqual(n[0,0,0,d],expected)

if __name__=='__main__':unittest.main()
