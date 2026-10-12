"""Lightweight adapters to unchanged RootGuard sources and experiment primitives."""
from __future__ import annotations
import ast
import hashlib
import math
import random
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]

def source_functions(path, names, namespace=None):
    """Execute selected original definitions, avoiding unrelated NER/model imports.

    Definitions are not rewritten; every requested function must exist. Source
    hashes are recorded by the runner. Only repository-owned Python is accepted.
    """
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise RuntimeError(f"Missing original definitions: {names}")
    ns = dict(np=np, math=math, random=random, **(namespace or {}))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), ns)
    return ns

OPT_NAMES = ['compute_sums', 'objective_absolute', 'gradient_absolute',
             'objective_variance', 'gradient_variance', '_compute_phi',
             '_solve_eps_for_lambda', 'optimal_budget_allocation_exact']
OPT = source_functions(REPO / 'utils/utils.py', OPT_NAMES)
RISK = source_functions(REPO / 'utils/risk_class.py', ['get_risk_class'])['get_risk_class']

def original_domains():
    tree = ast.parse((REPO / 'preempt/sanitizer.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TEMPLATE_DOMAINS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError('TEMPLATE_DOMAINS not found')

DOMAINS = original_domains()
ROOTS = {'FIB4': ['age', 'ast', 'alt', 'plt'], 'HOMA': ['glu', 'ins'],
         'ANEMIA': ['hb', 'hct', 'rbc']}
EXPONENTS = {'FIB4': [1., 1., -.5, -1.], 'HOMA': [1., 1.],
             'ANEMIA': [1., -1., 0.]}
TARGETS = {'FIB4': 'fib4', 'HOMA': 'homa', 'ANEMIA': 'mchc'}

def compute(name, x):
    """Real task, vectorized over the last root axis."""
    if name == 'FIB4':
        return x[..., 0] * x[..., 1] / (np.sqrt(x[..., 2]) * x[..., 3])
    if name == 'HOMA':
        return x[..., 0] * x[..., 1] / 405.
    return 100. * x[..., 0] / x[..., 1]

def classes(name, values):
    a = np.asarray(values)
    return np.asarray([RISK(name, v) for v in a.ravel()]).reshape(a.shape)

def sensitivities(name, mu):
    return float(compute(name, mu)) * np.abs(EXPONENTS[name]) / mu

def allocate(name, sens, B, m=1000, floor=.001):
    roots = ROOTS[name]
    widths = {r: DOMAINS[name][r][1] - DOMAINS[name][r][0] for r in roots}
    out = OPT['optimal_budget_allocation_exact'](
        dict(zip(roots, sens)), B, floor, domain_widths=widths,
        num_candidates=m, mode='abs')
    e = np.asarray([out[r] for r in roots])
    if not np.all(np.isfinite(e)) or np.any(e < floor - 1e-7) or not np.isclose(e.sum(), B, atol=1e-6):
        raise RuntimeError(f'Invalid allocation {name}: {e}, B={B}')
    delta = np.asarray(list(widths.values())) / (m - 1)
    center = np.full(len(roots), (m - 1) / 2.)
    objective = lambda v: OPT['objective_absolute'](v, sens, delta, center, m)
    if objective(e) > objective(np.full(len(roots), B / len(roots))) + 1e-6:
        raise RuntimeError('Original optimizer worse than uniform on its own objective')
    return e

def noisy_roots(name, x, eps, uniforms, m=1000):
    """Batch inverse-CDF sampling of original unrounded index-space PMF.

    uniforms has shape (patients, repeats, roots). Shared uniforms provide
    paired comparisons; roots/repeats/patients remain independent within a cell.
    Returned roots are one cache per patient/repeat, never refreshed per turn.
    """
    output = np.empty(uniforms.shape)
    grid = np.arange(m)
    for j, root in enumerate(ROOTS[name]):
        lo, hi = DOMAINS[name][root]
        t = np.clip((x[:, j] - lo) / (hi - lo) * (m - 1), 0, m - 1)
        logp = -eps[j] * np.abs(t[:, None] - grid) / 2.
        p = np.exp(logp - logp.max(axis=1, keepdims=True))
        cdf = np.cumsum(p / p.sum(axis=1, keepdims=True), axis=1)
        cdf[:, -1] = 1.
        s = np.stack([np.searchsorted(cdf[i], uniforms[i, :, j], side='right') for i in range(len(x))])
        output[..., j] = lo + s * (hi - lo) / (m - 1)
    return output

def proxy(name, x, mu, family, root_index, level):
    """Return proxy output and public-mean absolute gradients.

    power: ghat=g(mu)*prod((x_i/mu_i)**q_i). Change selected nonzero
    exponent p to p*(1+level); for zero p (RBC), q=level.
    weights: diagnostic-only selected sensitivity factor 1+level; no ghat.
    """
    s = sensitivities(name, mu).copy()
    if family == 'weights':
        if s[root_index] == 0:
            # False relevance as a fraction of largest domain-scaled importance.
            widths = np.array([DOMAINS[name][r][1] - DOMAINS[name][r][0] for r in ROOTS[name]])
            s[root_index] = max(level, 0.) * np.max(s * widths) / widths[root_index]
        else:
            s[root_index] *= 1 + level
        return None, s
    q = np.array(EXPONENTS[name])
    q[root_index] = q[root_index] * (1 + level) if q[root_index] != 0 else max(level, 0.)
    gmu = float(compute(name, mu))
    y = gmu * np.prod((x / mu) ** q, axis=-1)
    return y, gmu * np.abs(q) / mu

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
