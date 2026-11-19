import ast
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from .rootguard_mopt import MOpt,ORIGINAL,exp_pmf,simulate_baseline,uniforms
from .run import ROOT
from .test_core import CFG,record
from experiments.workflow_extensions.finite_cache.workload import Workload


def allocator():
    w=Workload('ANEMIA')
    return MOpt(w,{'mu':w.normalize([10.,30.,4.]).tolist()})


class OriginalMOptTests(unittest.TestCase):
    def test_verbatim_original_solver_allocation(self):
        a=allocator();em,eg=a.allocate(320.)
        ref=ORIGINAL['optimal_budget_allocation_exact'](a.info['sensitivities'],320/999,epsilon_min=.001,
                    domain_widths=dict(zip(a.w.roots,a.w.width)),num_candidates=1000,mode='abs')
        np.testing.assert_array_equal(eg,[ref[r] for r in a.w.roots])
        np.testing.assert_allclose(em,eg*999)
        self.assertAlmostEqual(em.sum(),320.)
        self.assertTrue(a.logs[-1]['solver']['success'])
        self.assertEqual(a.info['derivative_zero_roots'],['rbc'])

    def test_native_pmf_exactly_matches_original_sanitizer(self):
        tree=ast.parse((ROOT/'preempt/sanitizer.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Sanitizer')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='M_exponential_discrete')
        ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original kernel>','exec'),ns)
        for z in [0.,.001,.27,.5,.999,1.]:
            captured={}
            def choose(n,p):captured['p']=p;return 0
            with patch('numpy.random.choice',choose):
                ns[fn.name](None,2+23*z,.12,2.,25.,1000)
            np.testing.assert_allclose(exp_pmf(z,.12),captured['p'],rtol=1e-13,atol=1e-15)

    def test_normalized_privacy_conversion(self):
        metric_epsilon=40.
        for a,b in [(0.,.001),(.2,.7),(.999,1.),(.1,.10001)]:
            pa=exp_pmf(a,metric_epsilon/999);pb=exp_pmf(b,metric_epsilon/999)
            self.assertLessEqual(float(np.max(abs(np.log(pa)-np.log(pb)))),metric_epsilon*abs(a-b)+1e-10)

    def test_floor_fallback_is_original_and_budget_valid(self):
        a=allocator();em,eg=a.allocate(1.)
        self.assertTrue(a.logs[-1]['insufficient_floor_fallback'])
        np.testing.assert_allclose(em,np.full(3,1/3))
        self.assertTrue((eg>0).all())

    def test_static_periodic_ledger_and_native_support(self):
        a=allocator()
        for kernel in ['native_exp','laplace']:
            for P in [None,1,2]:
                r=simulate_baseline(record(),320.,P,kernel,a,CFG,'mopt-unit',3,True)
                np.testing.assert_allclose(r['spent'],320)
                actual=r['anchor_epsilon'].sum(axis=1)+sum(h['actual_epsilon'].sum(axis=1) for h in r['histories'])
                np.testing.assert_allclose(actual,r['spent'])
                self.assertTrue((r['caches']>=0).all() and (r['caches']<=1).all())
                if kernel=='native_exp':
                    np.testing.assert_allclose(r['caches']*999,np.rint(r['caches']*999),atol=1e-10)
                    self.assertTrue((r['n_proj']==0).all())

    def test_release_is_causal(self):
        a=allocator();first=record();changed=record();changed['events'][-1]['values']['hb']=24.
        for kernel in ['native_exp','laplace']:
            x=simulate_baseline(first,320.,1,kernel,a,CFG,'mopt-unit',3,True)
            y=simulate_baseline(changed,320.,1,kernel,a,CFG,'mopt-unit',3,True)
            np.testing.assert_array_equal(x['caches'][:,:19],y['caches'][:,:19])
            self.assertFalse(np.array_equal(x['caches'][:,19:],y['caches'][:,19:]))

    def test_uniform_pairing_transform(self):
        from .core import paired_noise,key_seed
        noise=paired_noise(7,'test','1','a',320.,2,40,('hb',))
        u=uniforms(noise)
        for t in [1,4,20]:
            for r in range(2):
                v=key_seed(7,'test','1','a',320.,r,t,'hb')
                expected=((v>>11)+.5)/2**53
                self.assertAlmostEqual(u[t,r,0],expected)


if __name__=='__main__':unittest.main()
