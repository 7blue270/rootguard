import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import original,old,classes
from data_cases import stress,inventory,split
from publishing import quota_period,oracle_quota,scheduled,deployment,frozen_schedule_pair
from event_metrics import measure


def fixture(truth_classes,pred_classes):
    values=np.array([30.,34.,38.]);gt=np.array(truth_classes);pc=np.array(pred_classes)
    x=np.stack([values[gt]*.4,np.full(len(gt),40.),np.full(len(gt),4.)],axis=-1)[None]
    z=np.stack([values[pc]*.4,np.full(len(pc),40.),np.full(len(pc),4.)],axis=-1)[None,None]
    d=dict(name='ANEMIA',x=x)
    l=dict(flags=np.ones((1,1,len(gt)),bool),spent=np.zeros((1,1)),bound=0,dp=False)
    return measure(d,z,l)[0]


class ProtocolTests(unittest.TestCase):
    def test_strict_initial_error_not_hidden(self):
        a=fixture([0,0,1,1],[1,1,1,1]);self.assertEqual(a['recall1'],0);self.assertEqual(a['recovery1'],100)
    def test_delay_and_next_event_boundary(self):
        a=fixture([0,0,1,1],[0,0,0,1]);self.assertEqual(a['recall0'],0);self.assertEqual(a['recall1'],100);self.assertEqual(a['delay_success2'],1)
        self.assertEqual(a['false_change'],0);self.assertGreater(a['stable_output_change_rate'],0)
        a=fixture([0,1,0],[0,0,1]);self.assertEqual(a['recall2'],0)
    def test_duplicate_reports_are_false_events(self):
        a=fixture([0,1,1,1],[0,1,0,1]);self.assertEqual(a['recall2'],100);self.assertAlmostEqual(a['precision2'],100/3);self.assertEqual(a['false_events'],2)
    def test_no_events_and_metric_formula(self):
        a=fixture([0,0],[0,0]);self.assertIsNone(a['recall1']);self.assertIsNone(a['f1_2'])
        a=fixture([0,1,2],[1,1,0]);self.assertEqual(a['rce'],50);self.assertAlmostEqual(a['cmr'],200/3)
    def test_stress_rules_and_domains(self):
        for name in ['ANEMIA','FIB4','HOMA']:
            for T in [8,16]:
                for p in ['single','spaced','adjacent','chatter','stable']:
                    d=stress(name,T,p);lo,hi=old.bounds(name)
                    self.assertTrue(np.all(d['x']>=lo-1e-10));self.assertTrue(np.all(d['x']<=hi+1e-10))
    def test_class_vectorization_matches_original(self):
        for name in ['ANEMIA','FIB4','HOMA']:
            a=np.r_[np.linspace(0,50,1000),[1,2.5,1.3,2.67,32,36]]
            np.testing.assert_array_equal(classes(name,a),original.classes(name,a))
    def test_quota_count_initial_cache_and_oracle_status(self):
        d=stress('ANEMIA',8,'single');d=dict(d,x=d['x'][:4],ids=d['ids'][:4])
        rng=np.random.default_rng(13);u=rng.random((4,12,5,3))
        for K in [1,2,4]:
            a,l=scheduled(d,.3,K,quota_period(4,8,K),u)
            b,m=scheduled(d,.3,K,oracle_quota(d,K),u,oracle=True)
            np.testing.assert_array_equal(a[:,:,0],b[:,:,0]);self.assertTrue(np.all(l['flags'].sum(axis=2)==K+1));self.assertFalse(m['dp'])
            for ie,re in [(True,True),(True,False),(False,True),(False,False)]:
                z,ll=scheduled(d,.3,K,oracle_quota(d,K),u,ie,re,oracle=True)
                self.assertFalse(ll['dp'])
                if ie:np.testing.assert_array_equal(z[:,:,0],np.repeat(d['x'][:,None,0],12,axis=1))
    def test_svt_budget_cap(self):
        d=stress('HOMA',8,'chatter');d=dict(d,x=d['x'][:4],ids=d['ids'][:4])
        rng=np.random.default_rng(4);u=rng.random((4,12,8,2));q=rng.laplace(size=(4,12,8));th=rng.laplace(size=(4,12,4))
        for mode in ['attribute','task']:
            z,l=deployment(d,.3,dict(method='svt',mode=mode,C=4,fraction=.5,threshold=.03),u,q,th)
            self.assertLessEqual(l['spent'].max(),.3+1e-8)
            a,b=frozen_schedule_pair(d,l)
            np.testing.assert_array_equal(a[1]['flags'].sum(axis=2),b[1]['flags'].sum(axis=2))
            self.assertFalse(a[1]['dp']);self.assertFalse(b[1]['dp'])
    def test_exact_oracle_quadrant_and_budget_reservation(self):
        d=stress('ANEMIA',8,'single');d=dict(d,x=d['x'][:4],ids=d['ids'][:4])
        u=np.random.default_rng(11).random((4,12,5,3));plan=oracle_quota(d,4)
        z,l=scheduled(d,.3,4,plan,u,True,True,oracle=True)
        a,_=measure(d,z,l);self.assertEqual(a['recall1'],100);self.assertEqual(a['rce'],0);self.assertIsNone(a['spent_max'])
        z,l=scheduled(d,.3,4,plan,u,oracle=True,publish_fraction=.8)
        self.assertFalse(l['dp']);self.assertTrue(np.allclose(l['spent'],.24))


if __name__=='__main__':unittest.main()
