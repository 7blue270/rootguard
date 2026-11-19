import unittest
import numpy as np
from .core import Plan,realize
from .test_core import record,Scripted,CFG
from .periodic_skeleton import aligned_pair


class SkeletonTests(unittest.TestCase):
    def test_exact_schedule_anchor_and_no_skip_equivalence(self):
        plan=Plan(.5,sets=((0,1,2),),period=1)
        b,c=aligned_pair(record(),320.,plan,Scripted([[True]*3]*3),CFG,'supplement-unit',2)
        self.assertEqual([h['turn'] for h in c['histories']],[4,8,20])
        self.assertEqual(c['K'],3)
        np.testing.assert_allclose(b['caches'],c['caches'],atol=1e-14)
        np.testing.assert_allclose(b['spent'],c['spent'])

    def test_same_opportunities_can_skip_and_carry(self):
        plan=Plan(.5,sets=((0,1,2),),period=1)
        b,c=aligned_pair(record(),320.,plan,Scripted([[False]*3,[True]*3,[True]*3]),CFG,'supplement-unit',2)
        self.assertEqual(b['K'],c['K'])
        np.testing.assert_allclose(c['histories'][1]['eps_can'],80)
        np.testing.assert_allclose(c['m'],2)
        np.testing.assert_allclose(b['m'],3)
        np.testing.assert_allclose(c['spent'],320)


if __name__=='__main__':unittest.main()
