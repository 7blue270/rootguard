from __future__ import annotations
import inspect
import json
from pathlib import Path
import unittest
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from .core import Plan, Policy, PrivateSource, allocate, evaluate, paired_noise, simulate, task
from .planning import public_population
from .run import budget_grid
from .analysis import bmin, bootstrap_curves

CFG=json.loads((Path(__file__).parent/'configs/real_v1.json').read_text())


def record():
    return dict(template='ANEMIA',subject_id=1,trajectory_id='unit',T=40,
                initial_state=dict(hb=12.,hct=36.,rbc=4.),
                events=[dict(turn=t,measured_roots=roots,values=values) for t,roots,values in [
                    (4,['hb'],dict(hb=11.)),(8,['hct'],dict(hct=35.)),(20,['hb','hct'],dict(hb=10.,hct=33.))]])


def policy():
    w=Workload('ANEMIA')
    stats=dict(h=[1.,4.,1.],raw_h=[1.,4.,0.],cbar=[.01,.02,0.],mu=[.4,.4,.4])
    return Policy(w,stats,CFG),stats


class Scripted(Policy):
    def __init__(self, decisions):
        p,stats=policy()
        super().__init__(p.w,stats,CFG)
        self.decisions=iter(decisions)
        self.seen=[]

    def decide(self,cache,eps_last,ages,candidate_mask,candidate_epsilon):
        self.seen.append((ages.copy(),candidate_epsilon.copy()))
        select=np.broadcast_to(next(self.decisions),candidate_mask.shape).copy() & candidate_mask
        return select,allocate(candidate_epsilon,candidate_mask,self.h)


class CoreTests(unittest.TestCase):
    def test_task_exact_existing_formula(self):
        for template in CFG['templates']:
            w=Workload(template)
            z=np.random.default_rng(3).uniform(size=(5,7,len(w.roots)))
            np.testing.assert_allclose(task(w,z),w.from_cache(z)[w.target],rtol=1e-13)

    def test_current_carry_budget_after_skip(self):
        p=Scripted([[False]*3,[True]*3])
        plan=Plan(.5,(.1,.2),((0,1,2),(0,1,2)))
        r=simulate(record(),320.,plan,'carry',p,CFG,'unit',2,True)
        np.testing.assert_allclose(p.seen[0][1],80)
        np.testing.assert_allclose(p.seen[1][1],160)
        np.testing.assert_allclose(r['spent'],320)
        np.testing.assert_allclose(r['histories'][0]['left_after'],160)
        self.assertEqual(r['histories'][0]['n_after'],1)
        np.testing.assert_allclose(r['m'],1)

    def test_public_planner_scores_actual_carry_executor(self):
        from .planning import public_select
        plan=Plan(.5,(.1,.2),((0,1,2),)*2)
        fixed,_,_=public_select([(plan,0.)],[record()],320.,'fixed',Scripted([[False]*3,[True]*3]),CFG)
        carry,_,_=public_select([(plan,0.)],[record()],320.,'carry',Scripted([[False]*3,[True]*3]),CFG)
        self.assertAlmostEqual(fixed['spend'],.75)
        self.assertAlmostEqual(carry['spend'],1.)
        self.assertNotEqual(fixed['mae'],carry['mae'])

    def test_fixed_skip_does_not_increase_package(self):
        p=Scripted([[False]*3,[True]*3])
        r=simulate(record(),320.,Plan(.5,(.1,.2),((0,1,2),)*2),'fixed',p,CFG,'unit',2,True)
        np.testing.assert_allclose(p.seen[1][1],80)
        np.testing.assert_allclose(r['spent'],240)

    def test_all_veto_and_final_veto(self):
        p=Scripted([[False]*3,[False]*3])
        r=simulate(record(),320.,Plan(.5,(.1,.2),((0,1,2),)*2),'carry',p,CFG,'unit',2,True)
        np.testing.assert_allclose(r['spent'],160)
        np.testing.assert_allclose(r['left'],160)
        np.testing.assert_allclose(r['m'],0)
        self.assertEqual(r['histories'][-1]['n_after'],0)

    def test_no_veto_fixed_equals_carry(self):
        plan=Plan(.5,(.1,.2),((0,1,2),)*2)
        a=simulate(record(),320.,plan,'fixed',Scripted([[True]*3]*2),CFG,'unit',2,True)
        b=simulate(record(),320.,plan,'carry',Scripted([[True]*3]*2),CFG,'unit',2,True)
        np.testing.assert_allclose(a['caches'],b['caches'],atol=1e-14)
        np.testing.assert_allclose(a['spent'],b['spent'])

    def test_partial_batch_age_and_reallocation(self):
        p=Scripted([[False]*3,[True,False,False]])
        r=simulate(record(),320.,Plan(.5,(.1,.2),((0,1,2),)*2),'carry',p,CFG,'unit',2,True)
        np.testing.assert_array_equal(p.seen[0][0],[[1,0,0]]*2)
        np.testing.assert_array_equal(p.seen[1][0],[[1,1,0]]*2)
        np.testing.assert_allclose(r['histories'][1]['actual_epsilon'],[[160,0,0]]*2)
        self.assertFalse(np.array_equal(r['histories'][1]['candidate_epsilon'],r['histories'][1]['actual_epsilon']))

    def test_k_zero_static_and_poststop(self):
        p,s=policy()
        rec=record()
        r=simulate(rec,320.,Plan(1.),'carry',p,CFG,'unit',2,True)
        self.assertEqual(r['stop'],1)
        self.assertEqual(r['release_reads'],[1])
        np.testing.assert_allclose(r['spent'],320)
        m=evaluate(rec,r,p.w,s,{**CFG,'_B':320},True)
        truth=p.w.evaluate(data.replay(rec))[p.w.target]
        err=abs(task(p.w,r['caches'])-truth)
        np.testing.assert_allclose(m['post_stop_mae'],err[:,1:].sum(axis=1)/39)
        self.assertTrue(np.isnan(m['frac_veto']).all())
        self.assertTrue(np.isnan(m['harmful_refresh']).all())

    def test_truth_free_policy_and_no_future_release(self):
        self.assertEqual(list(inspect.signature(Policy.decide).parameters),['self','cache','eps_last','ages','candidate_mask','candidate_epsilon'])
        p,_=policy()
        rec=record()
        future=record()
        future['events'][-1]['values']['hb']=24.
        plan=Plan(.5,(.1,.2),((0,1,2),)*2)
        a=simulate(rec,320.,plan,'periodic',p,CFG,'unit',2,True)
        b=simulate(future,320.,plan,'periodic',p,CFG,'unit',2,True)
        np.testing.assert_array_equal(a['caches'],b['caches'])
        # Releasing hct at turn4 must use initial hct (36), not turn8 hct (35).
        noise=paired_noise(CFG['noise_seed'],'unit','1','unit',320.,2,40,tuple(p.w.roots))
        eps=a['histories'][0]['actual_epsilon']
        z=p.w.normalize([11.,36.,4.])
        expected=np.clip(z+noise[4]/eps,0,1)
        np.testing.assert_allclose(a['histories'][0]['cache_after'],expected)

    def test_paired_noise_and_phase_budget_separation(self):
        a=paired_noise(7,'test','1','traj',80.,2,40,('hb',))
        np.testing.assert_array_equal(a,paired_noise(7,'test','1','traj',80.,2,40,('hb',)))
        self.assertFalse(np.array_equal(a,paired_noise(7,'public_calibration','1','traj',80.,2,40,('hb',))))
        self.assertFalse(np.array_equal(a,paired_noise(7,'test','1','traj',81.,2,40,('hb',))))

    def test_public_alignment_and_grid(self):
        from .core import realize
        plan=Plan(.5,(.1,.2),((0,),(1,)))
        self.assertEqual(realize(plan,[4,8,20],40),[(4,(0,)),(8,(1,))])
        grid,coarse=budget_grid(CFG)
        self.assertEqual(len(coarse),28)
        self.assertEqual(grid[-1],960)
        self.assertLessEqual(max(b/a for a,b in zip(grid,grid[1:])),1.0501)
        self.assertEqual(len(grid),len(set(grid)))
        with self.assertRaises(ValueError):
            realize(plan,[4],40)

    def test_bootstrap_and_nonmonotonic_censoring(self):
        curves=np.array([[4.,1.,3.],[4.,3.,2.]])
        np.testing.assert_array_equal(bmin(curves,[80,160,320],1.5),[160,np.inf])
        x=np.random.default_rng(2).uniform(size=(1,3,4,5))
        x=np.concatenate([x,x],axis=0)
        p,n=bootstrap_curves(x,100,99)
        self.assertEqual(p.shape,(2,3,100))
        np.testing.assert_array_equal(p[0],p[1])
        np.testing.assert_array_equal(n[0],n[1])

    def test_public_generator_train_only_and_constant_age(self):
        # No access to validation/test arguments is possible in this interface.
        self.assertEqual(list(inspect.signature(public_population).parameters),['train','w','cfg','delta'])
        rec=record()
        pop=public_population([rec],Workload('ANEMIA'),CFG,.02)
        self.assertEqual(len(pop),4)
        self.assertEqual([e['measured_roots'] for e in pop[0]['events']],[['hb'],['hct'],['hb','hct']])
        self.assertTrue(all(r['source_kind']=='train-only-public-synthetic' for r in pop))


if __name__=='__main__':
    unittest.main()
