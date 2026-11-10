"""Round-2: does any refresh policy beat a frozen cache at equal standard epsilon?

All methods use the same session budget B on the normalized root domain, with
Laplace noise.  Released quantity:
  'incr'   : clipped increment, sensitivity C (public drift bound)
  'anchor' : full re-anchor, sensitivity 1

Methods:
  static      all budget on the initial anchor, never refresh (RootGuard semantics)
  periodic-k  anchor 0.3B + k evenly spaced refreshes sharing 0.7B
  gated-k     same budget, but the k updates with the largest workload gain G are refreshed
              (oracle gate: knows the true drift; upper bound for a private gate)
  always      refresh at every update
  oracle      noiseless, always current (utility upper bound, not DP)
"""
import json, itertools
from pathlib import Path
import numpy as np

ROOT = Path('/home/rootguard-research')
DOM = {'hb': (7.20, 19.90), 'hct': (23.00, 58.80), 'rbc': (2.32, 7.84)}
ROOTS = ['hb', 'hct', 'rbc']
A = {'mcv': np.array([0., 1., -1.]), 'mch': np.array([1., 0., -1.]),
     'mchc': np.array([1., -1., 0.]), 'anemia_class': np.array([1., -1., 0.])}
F = {'mcv': lambda z: 10*z[1]/z[2], 'mch': lambda z: 10*z[0]/z[2],
     'mchc': lambda z: 100*z[0]/z[1]}
LBL = {0: 'Hypochromic', 1: 'Normochromic (Normal Color)', 2: 'Hyperchromic'}
def cls(v): return 0 if v < 32 else (2 if v > 36 else 1)
def norm(x): return np.array([(x[r]-DOM[r][0])/(DOM[r][1]-DOM[r][0]) for r in ROOTS])
def denorm(z): return [DOM[r][0] + z[i]*(DOM[r][1]-DOM[r][0]) for i, r in enumerate(ROOTS)]

def load():
    recs = json.load(open(ROOT/'data_mine'/'ANEMIA.json'))['ANEMIA']
    out = []
    for r in recs:
        turns = []
        for t in r['turns']:
            pv = t['private_value']
            if set(pv) <= set(ROOTS):
                turns.append(('root', {k: float(v) for k, v in pv.items()}))
            else:
                k = next(iter(pv)); turns.append(('query', (k, pv[k])))
        out.append({'id': r['id'], 'turns': turns})
    return out

WEIGHTS = None
def weights(recs):
    global WEIGHTS
    if WEIGHTS is not None: return WEIGHTS
    cnt = {n: 0 for n in A}
    for r in recs:
        for kind, v in r['turns']:
            if kind == 'query': cnt[v[0]] += 1
    tot = sum(cnt.values()); WEIGHTS = {n: cnt[n]/tot for n in cnt}
    return WEIGHTS

def Mmat():
    M = np.zeros((3, 3))
    for n, a in A.items(): M += weights(load())[n]*np.outer(a, a)
    return M

def Jval(b, S, M, C, eps_r):
    if not S: return float(b @ M @ b)
    bb = b.copy(); bb[list(S)] = 0.
    k = len(S)
    return float(bb @ M @ bb + sum((min(C, 1.)/(eps_r/k))**2*M[i, i] for i in S))

def gain(b, M, C, eps_r):
    j0 = Jval(b, (), M, C, eps_r); best = j0; bestS = ()
    for k in (1, 2, 3):
        for S in itertools.combinations(range(3), k):
            v = Jval(b, S, M, C, eps_r)
            if v < best: best, bestS = v, S
    return j0-best, bestS

def prep(recs):
    """Per patient: list of (kind, payload) with true normalized roots maintained."""
    out = []
    for r in recs:
        seq = []; z = None
        anchor = None
        for i, (kind, v) in enumerate(r['turns']):
            if kind == 'root':
                if z is None:
                    z = norm(v); anchor = i
                    seq.append(('root', z.copy(), None))
                else:
                    d = np.zeros(3)
                    for j, rr in enumerate(ROOTS):
                        if rr in v:
                            zn = (v[rr]-DOM[rr][0])/(DOM[rr][1]-DOM[rr][0])
                            d[j] = zn-z[j]; z[j] = zn
                    seq.append(('root', z.copy(), d))
            else:
                seq.append(('query', v, None))
        out.append((anchor, seq))
    return out

def simulate(P, method, B, C, release, k, rng):
    M = Mmat(); errs = []; ok = []
    for anchor, seq in P:
        upd_after = [i for i in range(anchor+1, len(seq)) if seq[i][0] == 'root']
        n_upd = len(upd_after)
        if method == 'static':
            eps0, m, epsr = B, 0, None
        elif method == 'oracle':
            eps0, m, epsr = None, 0, None
        else:
            eps0 = 0.3*B
            m = n_upd if method == 'always' else min(k, n_upd)
            epsr = 0.7*B/m if m else None
        pick = set()
        if method == 'periodic' and m:
            for j in range(m):
                pick.add(upd_after[int(round(j*(len(upd_after)-1)/max(m-1, 1)))])
        elif method == 'gated' and m:
            scored = []
            for i in upd_after:
                g, _ = gain(seq[i][2], M, C, epsr)
                scored.append((g, i))
            pick = set(i for g, i in sorted(scored, reverse=True)[:m])
        zhat = None
        for i in range(anchor, len(seq)):
            kind, pay, d = seq[i]
            if kind == 'root':
                if i == anchor:
                    if method == 'oracle': zhat = pay.copy()
                    else: zhat = pay + rng.laplace(0, 1/eps0, 3)
                else:
                    do = (method == 'always') or (i in pick)
                    if do:
                        if release == 'anchor':
                            zhat = pay + rng.laplace(0, 1/epsr, 3)
                        else:
                            zhat = zhat + np.clip(d, -C, C) + rng.laplace(0, C/epsr, 3)
            else:
                node, ref = pay
                x = denorm(zhat)
                if node == 'anemia_class':
                    ok.append(int(LBL[cls(F['mchc'](x))] == ref))
                else:
                    est = F[node](x); errs.append(abs(est-ref)/abs(ref))
    return float(np.mean(errs)), float(np.mean(ok))

def main():
    recs = load(); P = prep(recs)
    print(f"{'B':>4}  {'method':<12} {'rel%':>7} {'cls%':>7}")
    for B in (10, 20, 40, 80):
        for method, lab, k in [('static', 'static', 0), ('periodic', 'periodic-4', 4),
                               ('gated', 'gated-4', 4), ('periodic', 'periodic-8', 8),
                               ('gated', 'gated-8', 8), ('always', 'always', 0),
                               ('oracle', 'oracle', 0)]:
            res = [simulate(P, method, B, 0.15, 'incr', k, np.random.default_rng(1000+s))
                   for s in range(8)]
            rel = np.mean([r[0] for r in res]); c = np.mean([r[1] for r in res])
            print(f"{B:>4}  {lab:<12} {rel*100:>7.2f} {c*100:>7.2f}")
        print()

if __name__ == '__main__':
    main()
