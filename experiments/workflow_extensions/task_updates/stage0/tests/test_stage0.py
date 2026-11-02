import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metrics import classification_metrics
from audit import audit_cohort,original


class Stage0Tests(unittest.TestCase):
    def test_paper_distance_is_not_mismatch(self):
        m=classification_metrics([0,1,2],[1,1,0],3)
        self.assertEqual(m['rce'],50);self.assertAlmostEqual(m['cmr'],200/3)
        self.assertEqual(classification_metrics([0],[1],3)['rce'],50)
        self.assertEqual(classification_metrics([0],[2],3)['rce'],100)
        m=classification_metrics([0,1],[1,1],2);self.assertEqual(m['rce'],m['cmr'])
    def test_invalid_classes_rejected(self):
        for t,p,C in [([0],[3],3),([0],[.5],3),([],[],3),([0],[1],1)]:
            with self.assertRaises(ValueError):classification_metrics(t,p,C)
    def test_event_rounds_and_intervals_have_no_offset(self):
        x=np.array([[[12,40,4],[13,40,4],[15,40,4],[13,40,4]]],float)
        d=dict(name='ANEMIA',x=x,ids=['fixture'],audit={'timestamps':[['2100-01-01 00:00:00','2100-01-01 02:00:00','2100-01-01 05:00:00','2100-01-01 06:00:00']]})
        a,p,e,g,th,pa=audit_cohort('fixture',d,'fixture')
        self.assertEqual([r['round'] for r in e],[2,3,4])
        self.assertEqual([r['direction'] for r in e],['0->1','1->2','2->1'])
        self.assertEqual([r['round_gap'] for r in g],[1,1]);self.assertEqual([r['hours_gap'] for r in g],[3,1])
        self.assertEqual(a['event_patients'],1);self.assertFalse(a['pass_event_gate'])
    def test_original_threshold_endpoint_conventions(self):
        np.testing.assert_array_equal(original.classes('ANEMIA',[31.999,32,36,36.001]),[0,1,1,2])
        np.testing.assert_array_equal(original.classes('FIB4',[1.299,1.30,2.67,2.671]),[0,1,1,2])
        np.testing.assert_array_equal(original.classes('HOMA',[.999,1.,2.499,2.5]),[0,1,1,2])
    def test_no_events_is_not_a_positive_gate(self):
        d=dict(name='ANEMIA',x=np.array([[[13,40,4],[13,40,4]]]),ids=['fixture'],audit={})
        a,p,e,g,th,pa=audit_cohort('fixture',d,'fixture')
        self.assertEqual(a['events'],0);self.assertEqual(a['event_patients'],0);self.assertIsNone(a['event_round_gaps'])


if __name__=='__main__':unittest.main()
