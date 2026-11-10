"""Continuous workload, public-domain derivative bounds and nominal signed M.

All queries use ONE released root cache. Domain projection is explicit shared
postprocessing, not an uncharged access to the current true roots.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
from data_mine.finite_trajectories import ROOTS, TARGETS, NODES
from utils.risk_class import get_risk_class

# Finite positive public benchmark domains, not fitted to the test cohort.
# These define the protected input after explicit clipping; audits report its effect.
DOMAINS = {
    'ANEMIA': {'hb': [2., 25.], 'hct': [5., 75.], 'rbc': [.3, 10.]},
    'FIB4': {'age': [18., 110.], 'ast': [1., 20000.], 'alt': [1., 20000.], 'plt': [1., 1500.]},
    'NLR': {'neu': [.001, 100.], 'lym': [.001, 100.]},
    'TYG': {'tg': [1., 3000.], 'glu': [1., 1000.]},
    'MELD_NA': {'bil': [.2, 40.], 'inr': [.8, 9.], 'cr': [.2, 15.], 'na': [115., 155.]},
}
# (coefficient, exponents) are exactly the original positive-root formulas.
MONOMIALS = {
    'ANEMIA': {'mcv': (10., [0, 1, -1]), 'mch': (10., [1, 0, -1]), 'mchc': (100., [1, -1, 0])},
    'FIB4': {'fib4_prod': (1., [1, 1, 0, 0]), 'fib4_denom': (1., [0, 0, .5, 1]),
             'fib4': (1., [1, 1, -.5, -1])},
    'TYG': {'tyg_prod': (1., [1, 1]), 'tyg_half': (.5, [1, 1])},
    'NLR': {'nlr': (1., [1, -1])},
}
FORMULA_VERSION = 'rootguard-continuous-v1_meld-pilot-always-na-v1'


@dataclass
class Workload:
    template: str
    domain_override: dict | None = None

    def __post_init__(self):
        self.roots = list(ROOTS[self.template])
        self.nodes = list(NODES[self.template])
        self.target = TARGETS[self.template]
        domain = self.domain_override or DOMAINS[self.template]
        self.lo = np.array([domain[k][0] for k in self.roots], float)
        self.hi = np.array([domain[k][1] for k in self.roots], float)
        if not np.all(np.isfinite(self.lo)) or not np.all(np.isfinite(self.hi)) or np.any(self.lo <= 0) or np.any(self.hi <= self.lo):
            raise ValueError('Finite positive ordered public domains are required')
        self.width = self.hi - self.lo

    def normalize(self, x):
        return np.clip((np.asarray(x) - self.lo) / self.width, 0., 1.)

    def denormalize(self, z):
        return self.lo + np.asarray(z) * self.width

    def evaluate(self, x):
        x = np.asarray(x, float)
        if np.any(x <= 0) or not np.all(np.isfinite(x)):
            raise ValueError('Task evaluation expects finite positive roots')
        out = {}
        for node, (coef, exponents) in MONOMIALS.get(self.template, {}).items():
            out[node] = coef * np.prod(x ** np.asarray(exponents), axis=-1)
        if self.template == 'TYG':
            out['tyg'] = np.log(out['tyg_half'])
        elif self.template == 'NLR':
            out['nlr_sum'] = x[..., 0] + x[..., 1]
            out['nlr_diff'] = np.abs(x[..., 0] - x[..., 1])
        elif self.template == 'MELD_NA':
            bil, inr, cr, na = np.moveaxis(x, -1, 0)
            out['bil_term'] = 3.78 * np.log(np.maximum(bil, 1.))
            out['inr_term'] = 11.2 * np.log(np.maximum(inr, 1.))
            out['cr_term'] = 9.57 * np.log(np.clip(cr, 1., 4.))
            out['meld'] = out['bil_term'] + out['inr_term'] + out['cr_term'] + 6.43
            delta = 137. - np.clip(na, 125., 137.)
            out['meld_na'] = np.clip(out['meld'] + 1.32 * delta - .033 * out['meld'] * delta, 6., 40.)
        return out

    def from_cache(self, cache):
        return self.evaluate(self.denormalize(np.clip(cache, 0., 1.)))

    def classes(self, target):
        if self.template == 'MELD_NA':
            return None  # No invented clinical categories for this numerical workload.
        a = np.asarray(target)
        return np.array([get_risk_class(self.template, v) for v in a.ravel()]).reshape(a.shape)

    def node_bounds(self):
        corners = np.array(list(product(*zip(self.lo, self.hi))))
        values = self.evaluate(corners)
        return {node: [float(np.min(values[node])), float(np.max(values[node]))] for node in self.nodes}

    def derivative_bounds(self):
        """Global coordinatewise bounds in normalized-root coordinates.

        Monomial suprema are exact. MELD uses conservative analytic chain bounds,
        including the clamp breakpoints; no empirical gradient is called a bound.
        """
        k = len(self.roots)
        rows = {}
        for node, (coef, powers) in MONOMIALS.get(self.template, {}).items():
            p = np.array(powers, float)
            b = np.zeros(k)
            for i in range(k):
                if p[i] != 0:
                    q = p.copy()
                    q[i] -= 1
                    b[i] = abs(coef * p[i]) * np.prod(np.where(q >= 0, self.hi, self.lo) ** q) * self.width[i]
            rows[node] = b
        if self.template == 'TYG':
            rows['tyg'] = self.width / self.lo
        elif self.template == 'NLR':
            rows['nlr_sum'] = self.width.copy()
            rows['nlr_diff'] = self.width.copy()
        elif self.template == 'MELD_NA':
            rows['bil_term'] = np.array([3.78 * self.width[0] / max(self.lo[0], 1.), 0, 0, 0])
            rows['inr_term'] = np.array([0, 11.2 * self.width[1] / max(self.lo[1], 1.), 0, 0])
            rows['cr_term'] = np.array([0, 0, 9.57 * self.width[2] / max(self.lo[2], 1.), 0])
            rows['meld'] = rows['bil_term'] + rows['inr_term'] + rows['cr_term']
            mb = self.node_bounds()['meld']
            b = rows['meld'].copy()  # |1 - .033*(137-Na)| <= 1 on [125,137].
            b[3] = max(abs(1.32 - .033 * mb[0]), abs(1.32 - .033 * mb[1])) * self.width[3]
            rows['meld_na'] = b
        return np.array([rows[node] for node in self.nodes])

    def jacobian(self, mu):
        """Signed local numerical Jacobian; evaluation metric only."""
        z = self.normalize(mu)
        result = np.empty((len(self.nodes), len(self.roots)))
        for i in range(len(self.roots)):
            a, b = z.copy(), z.copy()
            a[i] = max(0., z[i] - 1e-6)
            b[i] = min(1., z[i] + 1e-6)
            va, vb = self.evaluate(self.denormalize(a)), self.evaluate(self.denormalize(b))
            result[:, i] = [(vb[n] - va[n]) / (b[i] - a[i]) for n in self.nodes]
        return result
