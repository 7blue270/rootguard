"""MELD-Na: does task-aware allocation beat uniform? does the gate beat periodic?

Adds to round1:
  * allocation across the 4 roots: uniform vs M-Opt (public, from the cohort mean)
    vs adaptive M-Opt (uses the current value; upper bound, costs privacy)
  * release: re-anchor vs clipped increment
  * anchor/refresh budget split mu sweep
"""
import csv, gzip, collections, math, itertools
from pathlib import Path
import numpy as np

BASE = Path('/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2')
IDS = {'bil': {'50885','53089'}, 'inr': {'51237','51675'},
       'cr': {'50912','52546'}, 'na': {'50983','52623'}}
REV = {i:k for k,v in IDS.items() for i in v}
R = ['bil','inr','cr','na']
DOM = {'bil': (0.2,40.0), 'inr': (0.8,9.0), 'cr': (0.2,15.0), 'na': (115.0,155.0)}
C = 0.15

def load():
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    with gzip.open(BASE/'labevents.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            it=r['itemid']
            if it in REV and r['valuenum']:
                try: v=float(r['valuenum'])
                except ValueError: continue
                rows[r['subject_id']][r['charttime']][REV[it]]=v
    P={}
    for s,ts in rows.items():
        seq=[d for t,d in sorted(ts.items()) if set(R)<=set(d) and all(0<d[k]<1e6 for k in R)]
        if len(seq)>=8: P[s]=seq
    return P

def z(d): return np.array([(d[k]-DOM[k][0])/(DOM[k][1]-DOM[k][0]) for k in R])
def den(z): return {k: DOM[k][0]+z[i]*(DOM[k][1]-DOM[k][0]) for i,k in enumerate(R)}

def meld(bil,inr,cr,na):
    bil=max(bil,1.0); inr=max(inr,1.0); cr=min(max(cr,1.0),4.0); na=min(max(na,125.0),137.0)
    m=3.78*math.log(bil)+11.2*math.log(inr)+9.57*math.log(cr)+6.43
    return max(6.0,min(40.0,m+1.32*(137-na)-0.033*m*(137-na)))

def grad(x):
    """|dMELDNa/dx_i| * domain width, numerically"""
    h=[]
    for i,k in enumerate(R):
        e=(DOM[k][1]-DOM[k][0])*1e-4
        a=dict(x); b=dict(x); a[k]+=e; b[k]-=e
        h.append(abs(meld(**a)-meld(**b))/(2*e))
    return np.array(h)

def weights(kind, x=None, mu=None):
    if kind=='uniform': return np.ones(4)
    h = grad(mu) if kind=='mopt' else grad(x)
    D = np.array([DOM[k][1]-DOM[k][0] for k in R])
    return np.maximum(h*D, 1e-12)**(2/3)

def run(Z, mu_mean, method, B, P, mode, alloc, mu_split, seed):
    rng=np.random.default_rng(seed); out={}
    for s,Zs in Z.items():
        T=len(Zs)
        if method=='static': eps0=B; m=0
        else:
            eps0=mu_split*B
            m=max(1,T//P) if method=='periodic' else max(1,T//3)
            epsr=(1-mu_split)*B/m
        # anchor
        w=weights(alloc, x=den(Zs[0]), mu=mu_mean); w=w/w.sum()
        zhat=Zs[0]+rng.laplace(0,1/(eps0*w),4)
        t0=0; errs=[]
        for t in range(1,T):
            do=(method=='periodic' and t%P==0) or \
               (method=='gated' and np.max(np.abs(Zs[t]-Zs[t0]))>0.10)
            if do:
                w=weights(alloc, x=den(Zs[t]), mu=mu_mean); w=w/w.sum()
                if mode=='anchor':
                    zhat=Zs[t]+rng.laplace(0,1/(epsr*w),4)
                else:
                    d=Zs[t]-Zs[t0]
                    zhat=zhat+np.clip(d,-C,C)+rng.laplace(0,C/(epsr*w),4)
                t0=t
            errs.append(abs(meld(**den(zhat))-meld(**den(Zs[t]))))
        out[s]=float(np.mean(errs))
    return out

def main():
    P=load(); Z={s:[z(d) for d in seq] for s,seq in P.items()}
    mu={k: float(np.mean([den(zv)[k] for zs in Z.values() for zv in zs])) for k in R}
    print('cohort mean', {k:round(v,2) for k,v in mu.items()})
    h=grad(mu); D=np.array([DOM[k][1]-DOM[k][0] for k in R])
    print('sensitivity h_i*D_i:', dict(zip(R,np.round(h,2))))
    print('M-Opt weights:', dict(zip(R,np.round(weights('mopt',mu=mu)/weights('mopt',mu=mu).sum(),3))))
    print()
    for B in (80,160,320,640):
        base=run(Z,mu,'static',B,5,'anchor','mopt',0.5,0)
        print(f"B={B}  static {np.mean(list(base.values())):.2f}")
        for lab,meth,par,mode,alloc in [
            ('static unif          ','static',5,'anchor','uniform'),
            ('periodic-5 mopt anchor','periodic',5,'anchor','mopt'),
            ('periodic-5 mopt incr  ','periodic',5,'incr','mopt'),
            ('periodic-10 mopt anchor','periodic',10,'anchor','mopt'),
            ('periodic-10 mopt incr ','periodic',10,'incr','mopt'),
            ('gated mopt incr       ','gated',5,'incr','mopt')]:
            res=run(Z,mu,meth,B,par,mode,alloc,0.5,0)
            wins=sum(res[s]<base[s] for s in base)
            print(f"          {lab} {np.mean(list(res.values())):>6.2f}  wins {wins}/20")
        print()

if __name__=='__main__': main()
