import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapter import old,original
from dataset import make,split
from release import sensitivity,score,run,metrics


class ProtocolTests(unittest.TestCase):
    def test_exact_cancellation_and_no_clipping(self):
        for name in original.ROOTS:
            d=make(name,16,'cancel');y=original.compute(name,d['x'])
            np.testing.assert_allclose(y,np.repeat(y[:,:1],16,axis=1),rtol=1e-12)
            lo,hi=old.bounds(name);self.assertTrue(np.all(d['x']>=lo));self.assertTrue(np.all(d['x']<=hi))
    def test_split_and_phases(self):
        d=make('FIB4',16,'amplify');v,t=split(d)
        self.assertFalse(set(v['ids'])&set(t['ids']));self.assertGreater(len(set(tuple(p) for p in d['audit']['phases'])),20)
    def test_task_sensitivity(self):
        rng=np.random.default_rng(4)
        for name in original.ROOTS:
            lo,hi=old.bounds(name);a=rng.uniform(lo,hi,(200,len(lo)));b=rng.uniform(lo,hi,(200,len(lo)));p=(hi+lo)/2
            dist=np.max(abs(a-b)/(hi-lo)*999,axis=-1)
            for mode in ['task','attribute']:
                diff=abs(score(name,a,p,mode)-score(name,b,p,mode))
                self.assertTrue(np.all(diff<=sensitivity(name,mode)*dist+1e-12))
    def test_above_threshold_budget_cap_and_oracle_label(self):
        d=make('ANEMIA',8,'amplify');d=dict(d,x=d['x'][:4],ids=d['ids'][:4])
        rng=np.random.default_rng(4);u=rng.random((4,10,8,3));q=rng.laplace(size=(4,10,8));t=rng.laplace(size=(4,10,4))
        for C in [2,4]:
            p=dict(method='svt',mode='task',C=C,fraction=.5,threshold=.03)
            z,l=run(d,.3,p,u,q,t);self.assertLessEqual(l['spent'].max(),.3+1e-9)
            self.assertTrue(np.all(l['refresh_flags'].sum(axis=2)<=C+1))
        z,l=run(d,.3,dict(method='oracle',C=4),u,q,t);self.assertFalse(l['certified'])
    def test_no_events_recall_is_undefined(self):
        d=make('ANEMIA',8,'cancel');d=dict(d,x=d['x'][:4],ids=d['ids'][:4])
        exact=np.repeat(d['x'][:,None],2,axis=1)
        a,p=metrics(d,exact,dict(spent=np.zeros((4,2)),refresh_flags=np.zeros((4,2,8)),release_roots=0,certified=True))
        self.assertIsNone(a['tcr0']);self.assertIsNone(a['tcr2']);self.assertEqual(a['false_change'],0)


if __name__=='__main__':unittest.main()
