"""Round-1: premise check on ANEMIA trajectories.

Questions:
  Q1 how large is the pure staleness error of a frozen cache (no DP at all)?
  Q2 what is the empirical distribution of per-update normalized drift?
  Q3 for each update, does *any* refresh subset reduce the workload error J?
  Q4 how often does the M-metric "cancellation" case appear (roots move, task doesn't)?
"""
import json, math, itertools, statistics
from pathlib import Path
import numpy as np

ROOT = Path('/home/rootguard-research')
DOM = {'hb': (7.20, 19.90), 'hct': (23.00, 58.80), 'rbc': (2.32, 7.84)}
ROOTS = ['hb', 'hct', 'rbc']
# log-elasticities in (hb,hct,rbc) order
A = {
    'mcv':  np.array([0.0,  1.0, -1.0]),
    'mch':  np.array([1.0,  0.0, -1.0]),
    'mchc': np.array([1.0, -1.0,  0.0]),
    'anemia_class': np.array([1.0, -1.0, 0.0]),
}
F = {
    'mcv': lambda z: 10*z[1]/z[2],
    'mch': lambda z: 10*z[0]/z[2],
    'mchc': lambda z: 100*z[0]/z[1],
}
def cls(v):
    return 0 if v < 32 else (2 if v > 36 else 1)

def norm(x):
    return np.array([(x[r]-DOM[r][0])/(DOM[r][1]-DOM[r][0]) for r in ROOTS])
def denorm(z):
    return {r: DOM[r][0] + z[i]*(DOM[r][1]-DOM[r][0]) for i, r in enumerate(ROOTS)}

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

def workload_weights(recs):
    cnt = {n: 0 for n in A}
    for r in recs:
        for kind, v in r['turns']:
            if kind == 'query':
                cnt[v[0]] += 1
    tot = sum(cnt.values())
    return {n: cnt[n]/tot for n in cnt}, cnt

def Mmat(w):
    M = np.zeros((3, 3))
    for n, a in A.items():
        M += w[n] * np.outer(a, a)
    return M

def J(b, S, M, C, eps_r):
    k = len(S)
    if k == 0:
        return float(b @ M @ b)
    bb = b.copy(); bb[list(S)] = 0.0
    var = 0.0
    for i in S:
        var += (min(C, 1.0) / (eps_r/k))**2 * M[i, i]
    return float(bb @ M @ bb + var)

def main():
    recs = load()
    w, cnt = workload_weights(recs)
    M = Mmat(w)
    print('query counts', cnt)
    print('weights', {k: round(v, 3) for k, v in w.items()})
    print('M =\n', np.round(M, 3))

    # ---- Q1 staleness: freeze roots at session start, no DP at all ----
    stale_rel = []      # relative error of using frozen roots for queried node
    stale_cls = []
    all_true = []
    for r in recs:
        z0 = None; last = None
        for kind, v in r['turns']:
            if kind == 'root':
                if set(v) == set(ROOTS):
                    z = norm(v)
                    if z0 is None:
                        z0 = z.copy()
                    last = z.copy()
            else:
                node, ref = v
                x0 = [denorm(z0)[k] for k in ROOTS]
                if node == 'anemia_class':
                    label = {0: 'Hypochromic', 1: 'Normochromic (Normal Color)', 2: 'Hyperchromic'}
                    stale_cls.append(int(label[cls(F['mchc'](x0))] == ref))
                else:
                    est = F[node](x0)
                    stale_rel.append(abs(est-ref)/abs(ref))
    print(f'\nQ1 frozen-cache (no DP) relative error on queried nodes: '
          f'mean={np.mean(stale_rel)*100:.2f}%  p95={np.percentile(stale_rel,95)*100:.2f}%  n={len(stale_rel)}')
    if stale_cls:
        print(f'   frozen-cache class accuracy: {np.mean(stale_cls)*100:.1f}%  n={len(stale_cls)}')

    # ---- Q2 drift distribution ----
    drift = []
    for r in recs:
        prev = None
        for kind, v in r['turns']:
            if kind == 'root' and set(v) == set(ROOTS):
                z = norm(v)
                if prev is not None:
                    drift.append(np.abs(z-prev))
                prev = z.copy()
    drift = np.array(drift)
    print(f'\nQ2 per-root |dz| over {len(drift)} roots-updated: '
          f'mean={drift.mean():.4f} max={drift.max():.4f}')
    for C in (0.02, 0.05, 0.10, 0.15, 0.20):
        print(f'   C={C:.2f}: fraction of root-updates exceeding C = {(drift>C).mean()*100:.1f}%')

    # ---- Q3 does any refresh reduce J? ----
    C = 0.10
    for eps_r in (1.0, 2.0, 4.0):
        beneficial = 0; tot = 0; best_sizes = []
        for r in recs:
            prev = None
            for kind, v in r['turns']:
                if kind == 'root' and set(v) == set(ROOTS):
                    z = norm(v)
                    if prev is not None:
                        d = z - prev
                        tot += 1
                        vals = [J(d, S, M, C, eps_r) for S in
                                [s for k in range(4) for s in itertools.combinations(range(3), k)]]
                        j0 = vals[0]
                        if min(vals[1:]) < j0:
                            beneficial += 1
                        best_sizes.append(int(np.argmin(vals)).bit_count())
                    prev = z.copy()
        print(f'Q3 eps_r={eps_r}: refresh strictly better in {beneficial}/{tot} = '
              f'{beneficial/tot*100:.0f}% of updates; mean best |S|={np.mean(best_sizes):.2f}')

    # ---- Q4 cancellation: mchc unchanged while hb,hct move ----
    canc = 0; tot = 0; big = 0
    for r in recs:
        prev = None
        for kind, v in r['turns']:
            if kind == 'root' and set(v) == set(ROOTS):
                z = norm(v)
                if prev is not None:
                    d = z - prev
                    tot += 1
                    dmchc = abs(A['mchc'] @ d)
                    if np.linalg.norm(d) > 0.05:
                        big += 1
                        if dmchc < 0.005:
                            canc += 1
                prev = z.copy()
    print(f'\nQ4 of {big} updates with ||dz||>0.05, {canc} ({canc/big*100:.0f}%) leave MCHC within 0.005 '
          f'(i.e. roots move, task does not)')

if __name__ == '__main__':
    main()
