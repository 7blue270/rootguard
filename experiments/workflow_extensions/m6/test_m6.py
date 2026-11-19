import copy
import inspect
import unittest
import numpy as np
from scipy.special import logsumexp
from experiments.workflow_extensions.m4_carry.core import Plan
from experiments.workflow_extensions.m5.test_m5 import record
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.run import ROOT
from .run import config
from .public import fit_public,ast_groups,allocate
from .kernels import grid_logZ,log_likelihood,emit,danger
from .core import Policy,simulate

CFG=config(ROOT/'experiments/workflow_extensions/outputs/m5/real_v2_20261117',True)


class Tests(unittest.TestCase):
    def setUp(self):
        self.rec=record();self.w=Workload('ANEMIA');self.p=fit_public([self.rec],self.w,CFG)

    def test_exact_grid_normalizer_and_privacy(self):
        x=np.linspace(0,1,51)
        for lo,hi,m in [(.2,.7,7),(.001,.02,128),(.3,.3,128)]:
            grid=np.linspace(lo,hi,m);eps=31.
            exact=logsumexp(-eps*abs(x[:,None]-grid)/2,axis=1)
            # Coincident grid entries are equivalent to one constant output.
            if hi==lo:exact-=np.log(m)
            np.testing.assert_allclose(grid_logZ(x,eps,lo,hi,m),exact,atol=1e-12)
            for y in grid[::max(1,m//4)]:
                ll=-eps*abs(x-y)/2-exact
                self.assertLessEqual(np.max(abs(np.diff(ll))/np.diff(x)),eps+1e-9)

    def test_emit_laplace_and_grid(self):
        z=np.array([[.2,.5,.4],[.8,.1,.6]]);e=np.ones_like(z)*10;noise=np.array([[1.,-3.,.1],[-4.,0.,1.]])
        out,pr=emit(z,e,noise,np.zeros_like(z,bool),self.p)
        np.testing.assert_allclose(out,np.clip(z+noise/e,0,1))
        out,pr=emit(z,e,noise,np.ones_like(z,bool),self.p)
        self.assertTrue(np.all(out>=self.p['work_lo']));self.assertTrue(np.all(out<=self.p['work_hi']))
        self.assertFalse(pr.any())

    def test_high_gradient_branch_independent_of_small_denominator(self):
        p=copy.deepcopy(self.p);p['denominator_roots']=[False]*3
        p['a_quantile']=[1e-6]*3
        self.assertTrue(danger(np.array([p['mu']]),p).all())
        p['a_quantile']=[1e30]*3
        self.assertFalse(danger(np.array([p['mu']]),p).any())
        self.assertFalse(danger(np.array([p['mu']]),p,False).any())

    def test_ast_groups_not_weights(self):
        self.assertEqual(ast_groups(self.w)[0],[[0,1,2]])
        self.assertEqual(ast_groups(Workload('FIB4'))[0],[[0,1],[2,3]])
        self.assertEqual(ast_groups(Workload('BUN_CR'))[0],[[0,1]])
        self.assertEqual(ast_groups(Workload('MELD_NA'))[0],[[0],[1],[2],[3]])

    def test_weights_and_regime_train_only(self):
        p=self.p
        self.assertEqual(p['exponent'],.5)
        e=allocate(80,np.ones(3,bool),p);self.assertAlmostEqual(e.sum(),80);self.assertTrue(np.all(e>0))
        expected='STATIC' if p['cancellation']>.7 or p['nonlinearity']>3 else 'DYNAMIC'
        self.assertEqual(p['regime'],expected)
        self.assertNotIn('truth',inspect.signature(Policy.decide).parameters)
        self.assertNotIn('rec',inspect.signature(Policy.decide).parameters)

    def test_future_mutation_does_not_change_prefix(self):
        b=copy.deepcopy(self.rec);b['events'][-1]['values']={'hb':24.,'hct':70.}
        plan=Plan(.5,(.1,.75),((0,1,2),)*2)
        a=simulate(self.rec,320,plan,self.p,CFG,'unit',2,detailed=True)
        c=simulate(b,320,plan,self.p,CFG,'unit',2,detailed=True)
        np.testing.assert_array_equal(a['caches'][:,:29],c['caches'][:,:29])

    def test_carry_conservation_and_skips(self):
        p=self.p;seen=[]
        class Script:
            def decide(self,turn,cache,last,ages,mask,total):
                seen.append(total.copy());sel=mask if len(seen)==2 else np.zeros_like(mask)
                return sel,allocate(total,sel,p),{}
        result=simulate(self.rec,320,Plan(.5,(.1,.75),((0,1,2),)*2),p,CFG,'unit',2,policy=Script(),detailed=True)
        np.testing.assert_allclose(seen,[[80,80],[160,160]])
        np.testing.assert_allclose(result['spent'],320)
        self.assertEqual(result['histories'][0]['n_after'],1)

    def test_iso_actual_refreshes_fixed(self):
        plan=Plan(.5,(.1,.75),((0,1,2),)*2)
        for groups,horizon in [(False,False),(False,True),(True,True)]:
            res=simulate(self.rec,320,plan,self.p,CFG,'unit',2,groups=groups,horizon=horizon,force=True)
            np.testing.assert_allclose(res['m'],2);np.testing.assert_allclose(res['spent'],320)

    def test_static_no_extra_truth_reads(self):
        res=simulate(self.rec,80,Plan(1.),self.p,CFG,'unit',2)
        self.assertEqual(res['release_reads'],[1]);np.testing.assert_allclose(res['spent'],80)
        np.testing.assert_array_equal(res['caches'],np.repeat(res['anchor_cache'][:,None,:],40,axis=1))

    def test_real_current_latent_not_noised_cache(self):
        # The source of simulated refreshes is the public latent path, verified
        # with a scalar jump example; noise is never added directly to cache.
        from unittest.mock import patch
        policy=Policy(self.w,self.p,CFG,force=True)
        sources=[]
        from .kernels import emit as actual
        def spy(z,*args):sources.append(z.copy());return actual(z,*args)
        with patch('experiments.workflow_extensions.m6.core.emit',spy):
            policy.decide(8,np.full((2,3),.9),np.full((2,3),100.),np.ones((2,3)),np.ones((2,3),bool),np.full(2,100.))
        for z in sources:np.testing.assert_array_equal(z,np.broadcast_to(policy.paths[None,:,7,:],z.shape))

    def test_mixed_attack_normalizer(self):
        from .attack import reconstruct
        p=self.p;p['work_lo'][0]=.1;p['work_hi'][0]=.5;p['grid_points'][0]=7
        ll=log_likelihood(.3,30,True,p,0,51);x=np.linspace(0,1,51);grid=np.linspace(.1,.5,7)
        expected=-15*abs(x-.3)-logsumexp(-15*abs(x[:,None]-grid),axis=1)
        np.testing.assert_allclose(ll,expected,atol=1e-12)
        res=simulate(self.rec,80,Plan(1.),p,CFG,'unit',2)
        view=dict(T=40,events=[dict(turn=e['turn'],measured_roots=e['measured_roots']) for e in self.rec['events']])
        z=reconstruct(view,[],res['anchor_cache'],res['anchor_epsilon'],p,res['anchor_kernels'],CFG)
        self.assertEqual(z.shape,(2,40,3));self.assertTrue(np.isfinite(z).all())

if __name__=='__main__':unittest.main()
