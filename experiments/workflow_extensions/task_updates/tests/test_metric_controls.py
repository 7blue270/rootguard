"""Positive/negative controls for response metrics, separate from mechanism tests."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset import make
from release import metrics


class MetricControls(unittest.TestCase):
    def ledger(self,n,R,H):
        return dict(spent=np.zeros((n,R)),refresh_flags=np.zeros((n,R,H)),release_roots=0,certified=True)
    def test_truth_response_and_static_negative_control(self):
        d=make('ANEMIA',8,'amplify');d=dict(d,x=d['x'][:12],ids=d['ids'][:12])
        truth=np.repeat(d['x'][:,None],2,axis=1)
        a,_=metrics(d,truth,self.ledger(12,2,8))
        self.assertEqual(a['tcr0'],100.);self.assertEqual(a['tcr2'],100.);self.assertEqual(a['censored_pct'],0.)
        constant=np.repeat(truth[:,:,:1,:],8,axis=2)
        a,_=metrics(d,constant,self.ledger(12,2,8))
        self.assertEqual(a['tcr0'],0.);self.assertEqual(a['tcr2'],0.)
    def test_noisy_stable_task_is_false_change(self):
        d=make('ANEMIA',8,'cancel');d=dict(d,x=d['x'][:12],ids=d['ids'][:12])
        cache=np.repeat(d['x'][:,None],2,axis=1).copy()
        cache[:,:,1::2,0]*=1.2
        a,_=metrics(d,cache,self.ledger(12,2,8))
        self.assertIsNone(a['tcr2']);self.assertGreater(a['false_change'],90.)


if __name__=='__main__':unittest.main()
