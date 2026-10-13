import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanisms import original,run,metrics,indices,cap_allocation
from data import synthetic,real

class DynamicTests(unittest.TestCase):
    def setup_case(self,name='ANEMIA',scenario='sparse'):
        d=synthetic(name,5,8,scenario,.01,101)
        rng=np.random.default_rng(42)
        u=rng.random((5,10,8,len(original.ROOTS[name])))
        lap=rng.laplace(size=(5,10,8))
        return d,u,lap

    def test_public_schedule_budget_and_current_roots(self):
        d,u,lap=self.setup_case()
        for m in ['static','full_uniform','full_opt','root_uniform','root_opt','period2','period4','predictive30']:
            period=2 if m=='period2' else 4 if m=='period4' else 1
            cache,l=run(d['name'],d['x'],d['plan'],d['mu'],.3,m,u,lap,period=period)
            self.assertTrue(l['certified']);self.assertLessEqual(l['spent'].max(),.3+1e-8)
            if m=='root_opt':
                for t in range(1,8):
                    unchanged=~d['plan'][t]
                    np.testing.assert_array_equal(cache[:,:,t,unchanged],cache[:,:,t-1,unchanged])

    def test_real_panels_and_units(self):
        d=real(8,10)
        self.assertEqual(d['x'].shape,(10,8,3));self.assertEqual(len(set(d['ids'])),10)
        self.assertEqual(d['audit'].get('wrong_unit_rows',0),0)
        for ts in d['audit']['timestamps']:self.assertEqual(len(set(ts)),8);self.assertEqual(ts,sorted(ts))

    def test_no_update_identity(self):
        d,u,lap=self.setup_case(scenario='no_updates')
        a,_=run(d['name'],d['x'],d['plan'],d['mu'],.3,'static',u,lap)
        b,_=run(d['name'],d['x'],d['plan'],d['mu'],.3,'root_opt',u,lap)
        np.testing.assert_array_equal(a,b)

    def test_shared_noise_reveals_private_differences(self):
        d,u,lap=self.setup_case(scenario='drift')
        z,l=run(d['name'],d['x'],d['plan'],d['mu'],.3,'unsafe_shared_noise',u,lap)
        self.assertFalse(l['certified'])
        np.testing.assert_allclose(z[:,:,1:]-z[:,:,:-1],np.repeat(np.diff(d['x'],axis=1)[:,None],10,axis=1),atol=1e-12)

    def test_class_metrics_do_not_count_repeats_as_patients(self):
        d,u,lap=self.setup_case()
        z,l=run(d['name'],d['x'],d['plan'],d['mu'],.3,'root_opt',u,lap)
        a=metrics(d['name'],d['x'],z)
        self.assertEqual(a['miss_num'].shape,(5,));self.assertEqual(a['truth'].shape,(5,8))
        # Complete transcript consists of cached values; emitting q copies
        # does not create q independent releases or change the task result.
        for q in [1,4,8]:
            copies=np.repeat(z[:,:,:,None,:],q,axis=3)
            np.testing.assert_array_equal(copies[:,:,:,0],copies[:,:,:,-1])
            np.testing.assert_array_equal(original.compute(d['name'],copies).mean(axis=3),original.compute(d['name'],z))

    def test_single_root_exponential_metric_bound(self):
        grid=np.arange(31)
        for e in [.0001,.01,.3]:
            p=np.exp(-e*np.abs(grid[:,None]-grid[None,:])/2);p/=p.sum(axis=1,keepdims=True)
            for t in range(30):
                self.assertLessEqual(np.max(np.abs(np.log(p[t]/p[t+1]))),e+1e-12)

    def test_private_test_sensitivity_and_cap(self):
        d,u,lap=self.setup_case('FIB4','jump')
        # Score max |z-prediction| is 1-Lipschitz in the same trajectory metric.
        a=indices('FIB4',d['x'][:,0]);b=indices('FIB4',d['x'][:,4]);p=a+7
        lhs=np.abs(np.max(np.abs(a-p),axis=1)-np.max(np.abs(b-p),axis=1))
        self.assertTrue(np.all(lhs<=np.max(np.abs(a-b),axis=1)+1e-10))
        z,l=run('FIB4',d['x'],d['plan'],d['mu'],.3,'predictive30',u,lap)
        self.assertLessEqual(l['spent'].max(),.3+1e-9)

    def test_standard_trajectory_calibration(self):
        d,u,lap=self.setup_case()
        E=4.;B=E/999
        z,l=run(d['name'],d['x'],d['plan'],d['mu'],B,'full_opt',u,lap)
        self.assertLessEqual(l['spent'].max()*999,E+1e-9)

if __name__=='__main__':unittest.main()
