"""MELD-Na pilot: does a dynamic cache beat a frozen cache on *value* queries?

Data  : MIMIC-IV demo 2.2 labevents (local), patients with >=8 complete
        (bilirubin, INR, creatinine, sodium) panels.
Task  : MELD-Na = f(bil, inr, cr, na)  -- roots -> intermediate MELD -> final score.
        Values 6..40 points.
Budget: fixed session budget B (standard Laplace on the normalised root domain).
        static    : all B on the first panel, never refresh
        periodic-P: 0.5B anchor + 0.5B over T/P refreshes
        gated-th  : same budget, refresh when max root drift since last refresh > th
Release   : clipped increment, sensitivity C=0.15 (public bound), or re-anchor.
"""
import csv, gzip, collections, math, sys
from pathlib import Path
import numpy as np

BASE = Path('/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2')
IDS = {'bil': {'50885', '53089'}, 'inr': {'51237', '51675'},
       'cr': {'50912', '52546'}, 'na': {'50983', '52623'}}
REV = {i: k for k, v in IDS.items() for i in v}
R = ['bil', 'inr', 'cr', 'na']
DOM = {'bil': (0.2, 40.0), 'inr': (0.8, 9.0), 'cr': (0.2, 15.0), 'na': (115.0, 155.0)}
C = 0.15

def load():
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    with gzip.open(BASE / 'labevents.csv.gz', 'rt') as f:
        for r in csv.DictReader(f):
            it = r['itemid']
            if it in REV and r['valuenum']:
                try: v = float(r['valuenum'])
                except ValueError: continue
                rows[r['subject_id']][r['charttime']][REV[it]] = v
    P = {}
    for s, ts in rows.items():
        seq = [d for t, d in sorted(ts.items())
               if set(R) <= set(d) and all(0 < d[k] < 1e6 for k in R)]
        if len(seq) >= 8: P[s] = seq
    return P

def z(d): return np.array([(d[k]-DOM[k][0])/(DOM[k][1]-DOM[k][0]) for k in R])
def den(z): return {k: DOM[k][0]+z[i]*(DOM[k][1]-DOM[k][0]) for i, k in enumerate(R)}

def meld(bil, inr, cr, na):
    bil = max(bil, 1.0); inr = max(inr, 1.0); cr = min(max(cr, 1.0), 4.0)
    na = min(max(na, 125.0), 137.0)
    m = 3.78*math.log(bil) + 11.2*math.log(inr) + 9.57*math.log(cr) + 6.43
    return max(6.0, min(40.0, m + 1.32*(137-na) - 0.033*m*(137-na)))

def run(Z, method, B, P, th, mode, seed):
    rng = np.random.default_rng(seed); out = {}
    for s, Zs in Z.items():
        T = len(Zs)
        if method == 'static':
            eps0, m = B, 0
        else:
            eps0 = 0.5*B
            m = max(1, T//P) if method == 'periodic' else max(1, T//3)
            epsr = 0.5*B/m
        zhat = Zs[0] + rng.laplace(0, 1/eps0, 4)
        t0 = 0; errs = []
        for t in range(1, T):
            do = (method == 'periodic' and t % P == 0) or \
                 (method == 'gated' and np.max(np.abs(Zs[t]-Zs[t0])) > th)
            if do:
                d = Zs[t]-Zs[t0]
                if mode == 'anchor':
                    zhat = Zs[t] + rng.laplace(0, 1/epsr, 4)
                else:
                    zhat = zhat + np.clip(d, -C, C) + rng.laplace(0, C/epsr, 4)
                t0 = t
            errs.append(abs(meld(**den(zhat)) - meld(**den(Zs[t]))))
        out[s] = float(np.mean(errs))
    return out

def main():
    P = load()
    Z = {s: [z(d) for d in seq] for s, seq in P.items()}
    print(f"patients={len(P)} panels={sum(len(v) for v in P.values())} "
          f"median T={int(np.median([len(v) for v in P.values()]))}")
    print(f"{'B':>5} {'method':<18} {'mean MAE':>9} {'wins/20':>8} {'p95':>6}")
    for B in (40, 80, 160, 320):
        st = run(Z, 'static', B, 5, 0, 'incr', 0)
        mae = np.mean(list(st.values()))
        print(f"{B:>5} {'static':<18} {mae:>9.2f} {'-':>8} {'-':>6}")
        for lab, meth, par, mode in [
                ('periodic-5 anchor', 'periodic', 5, 'anchor'),
                ('periodic-5 incr', 'periodic', 5, 'incr'),
                ('periodic-10 incr', 'periodic', 10, 'incr'),
                ('gated0.10 incr', 'gated', 5, 'incr')]:
            res = run(Z, meth, B, par, 0.10 if meth == 'gated' else 0, mode, 0)
            wins = sum(res[s] < st[s] for s in st)
            diffs = sorted(res[s]-st[s] for s in st)
            print(f"{B:>5} {lab:<18} {np.mean(list(res.values())):>9.2f} {wins:>8} {diffs[len(diffs)//20]:>6.2f}")

if __name__ == '__main__':
    main()
