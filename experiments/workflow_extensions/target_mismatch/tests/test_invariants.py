import ast
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import REPO, ROOTS, DOMAINS, OPT, compute, sensitivities, proxy, allocate, noisy_roots, classes, source_functions
from run_pilot import load_data

class Invariants(unittest.TestCase):
    def test_true_formula_and_zero_proxy(self):
        for name in ROOTS:
            x, y, mu, _ = load_data(name, 200)
            for j in range(len(ROOTS[name])):
                yh, s = proxy(name, x, mu, 'power', j, 0.)
                np.testing.assert_allclose(yh, y, rtol=1e-12)
                np.testing.assert_allclose(s, sensitivities(name, mu), rtol=1e-12)
                np.testing.assert_allclose(allocate(name, s, .3), allocate(name, sensitivities(name, mu), .3), atol=1e-12)

    def test_analytic_gradient(self):
        for name in ROOTS:
            x, _, mu, _ = load_data(name, 5)
            for j in range(len(mu)):
                for level in [-1., -.1, .25, 1.]:
                    _, s = proxy(name, x, mu, 'power', j, level)
                    for z in range(len(mu)):
                        d = mu[z] * 1e-5
                        plus, minus = mu.copy(), mu.copy()
                        plus[z] += d
                        minus[z] -= d
                        yp, _ = proxy(name, plus, mu, 'power', j, level)
                        ym, _ = proxy(name, minus, mu, 'power', j, level)
                        self.assertAlmostEqual(abs((yp-ym)/(2*d)), s[z], places=6)

    def test_original_dag_gradients(self):
        sys.path.insert(0, str(REPO))
        from utils.med_domain.all_templates import template_expressions, template_nodes
        ns = source_functions(REPO / 'utils/utils.py', ['compute_target_sensitivities', 'compute_population_mean_sensitivities'])
        derived = {'FIB4': ['fib4_prod', 'fib4_denom', 'fib4'],
                   'HOMA': ['homa_prod', 'homa'], 'ANEMIA': ['mcv', 'mch', 'mchc']}
        targets = {'FIB4': 'fib4', 'HOMA': 'homa', 'ANEMIA': 'mchc'}
        for name in ROOTS:
            _, _, mu, _ = load_data(name, 5)
            s = ns['compute_population_mean_sensitivities'](targets[name], ROOTS[name], template_expressions,
                 [], ROOTS[name] + derived[name], template_nodes, dict(zip(ROOTS[name], mu)))
            np.testing.assert_allclose([s[r] for r in ROOTS[name]], sensitivities(name, mu), rtol=1e-12)

    def test_budget_and_zero_sensitive_floor(self):
        for name in ROOTS:
            _, _, mu, _ = load_data(name, 5)
            for B in [.1, .3, .7]:
                e = allocate(name, sensitivities(name, mu), B)
                self.assertAlmostEqual(e.sum(), B, places=6)
                self.assertTrue(np.all(e >= .001 - 1e-7))
        self.assertAlmostEqual(allocate('ANEMIA', sensitivities('ANEMIA', load_data('ANEMIA', 5)[2]), .3)[2], .001, places=6)

    def test_original_sampler_same_distribution_and_draw(self):
        tree = ast.parse((REPO / 'preempt/sanitizer.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Sanitizer')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'M_exponential_discrete')
        class NumpyAdapter:
            def __getattr__(self, key):
                return getattr(np, key)
        for name in ROOTS:
            x, _, _, _ = load_data(name, 5)
            u = np.random.default_rng(42).random((5, 4, len(ROOTS[name])))
            got = noisy_roots(name, x, np.full(len(ROOTS[name]), .1), u)
            # Force the unchanged original choice to use the same uniform draw.
            for i in range(5):
                for t in range(4):
                    for j, root in enumerate(ROOTS[name]):
                        adapter = NumpyAdapter()
                        class Random:
                            @staticmethod
                            def choice(n, p):
                                return np.searchsorted(np.cumsum(p), u[i,t,j], side='right')
                        adapter.random = Random
                        ns = {'np': adapter}
                        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<original sampler>', 'exec'), ns)
                        lo, hi = DOMAINS[name][root]
                        expected = ns['M_exponential_discrete'](None, x[i,j], .1, lo, hi)
                        self.assertAlmostEqual(got[i,t,j], expected, places=12)

    def test_fixed_cache_postprocessing(self):
        sys.path.insert(0, str(REPO / 'rq4_agent_eval'))
        from sanitizers.rootguard import RootGuard
        name = 'FIB4'
        x, _, mu, _ = load_data(name, 3)
        u = np.random.default_rng(8).random((3, 2, 4))
        z = noisy_roots(name, x, allocate(name, sensitivities(name, mu), .3), u)
        values = dict(zip(ROOTS[name], z[0, 0]))
        template = SimpleNamespace(compute=lambda vals: float(compute(name, np.array([vals[r] for r in ROOTS[name]]))))
        cache = RootGuard(cached=values, template=template)
        first = cache.get('ast')
        copy = cache.all_cached()
        copy['ast'] = -99
        self.assertEqual(cache.get('ast'), first)
        self.assertAlmostEqual(cache.compute_target(), float(compute(name, z)[0,0]))
        before = cache.compute_target()
        proxy(name, x, mu, 'power', 0, 1.)
        self.assertEqual(cache.compute_target(), before)

if __name__ == '__main__':
    unittest.main()
