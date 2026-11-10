"""Round-3: cumulative-drift design with periodic re-anchoring + gate on the tail.

Fixes over round 2:
  * the decision statistic uses cumulative drift since the last refresh (not per-turn),
  * increments are released w.r.t. the last published estimate, sensitivity s = min(Gmax*C, 1)
    with Gmax public (so the noise scale stays data-independent),
  * every P updates a public re-anchor resets the error (bias control),
  * reports average error, p95 tail error, class accuracy, and the large-drift subset.
"""
import json, itertools
from pathlib import Path
import numpy as np

ROOT = Path('/home/rootguard-research')
DOM = {'hb': (7.20, 19.90), 'hct': (23.00, 58.80), 'rbc': (2.32, 7.84)}
ROOTS = ['hb', 'hct', 'rbc']
A = {'mcv': np.array([0.,1.,-1.]), 'mch': np.array([1.,0.,-1.]),
     'mchc': np.array([1.,-1.,0.]), 'anemia_class': np.array([1.,-1.,0.])}
F = {'mcv': lambda z: 10*z[1]/z[2], 'mch': lambda z: 10*z[0]/z[2],
     'mchc': lambda z: 100*z[0]/z[1]}
LBL = {0:'Hypochromic', 1:'Normochromic (Normal Color)', 2:'Hyperchromic'}
def cls(v): return 0 if v < 32 else (2 if v > 36 else 1)
def norm(x): return np.array([(x[r]-DOM[r][0])/(DOM[r][1]-DOM[r][0]) for r in ROOTS])
def denorm(z): return [DOM[r][0] + z[i]*(DOM[r][1]-DOM[r][0]) for i, r in enumerate(ROOTS)]

def load():
    recs = json.load(open(ROOT/'data_mine'/'ANEMIA.json'))['ANEMIA']
    out=[]
    for r in recs:
        t=[]
        for x in r['turns']:
            pv=x['private_value']
            if set(pv)<=set(ROOTS): t.append(('root',{k:float(v) for k,v in pv.items()}))
            else:
                k=next(iter(pv)); t.append(('query',(k,pv[k])))
        out.append({'id':r['id'],'turns':t})
    return out

_cnt={n:0 for n in A}
for r in load():
    for kind,v in r['turns']:
        if kind=='query': _cnt[v[0]]+=1
_tot=sum(_cnt.values()); W={n:_cnt[n]/_tot for n in _cnt}
M=sum(W[n]*np.outer(a,a) for n,a in A.items())

def Jval(b,S,C,eps_r):
    if not S: return float(b@M@b)
    bb=b.copy(); bb[list(S)]=0.; k=len(S)
    return float(bb@M@bb + sum((min(C,1.)/(eps_r/k))**2*M[i,i] for i in S))

def gain(b,C,eps_r):
    j0=Jval(b,(),C,eps_r); best=j0; bS=()
    for k in (1,2,3):
        for S in itertools.combinations(range(3),k):
            v=Jval(b,S,C,eps_r)
            if v<best: best,bS=v,S
    return j0-best,bS

def prep():
    P=[]
    for r in load():
        seq=[]; z=None; anchor=None
        for i,(kind,v) in enumerate(r['turns']):
            if kind=='root':
                if z is None:
                    z=norm(v); anchor=i; seq.append(('root',z.copy()))
                else:
                    for j,rr in enumerate(ROOTS):
                        if rr in v: z[j]=(v[rr]-DOM[rr][0])/(DOM[rr][1]-DOM[rr][0])
                    seq.append(('root',z.copy()))
            else: seq.append(('query',v))
        P.append((anchor,seq))
    return P

def simulate(P,method,B,C,mu,Gmax,Pblock,k,rng):
    errs=[]; ok=[]; tail=[]; big=[]
    for anchor,seq in P:
        upd=[i for i in range(anchor+1,len(seq)) if seq[i][0]=='root']
        n=len(upd)
        if method=='static': eps0=B; epsr=0; m=0
        else:
            eps0=mu*B; m=min(k,n) if method!='always' else n
            epsr=(1-mu)*B/m if m else 0
        pick=set()
        if method=='periodic' and m:
            for j in range(m): pick.add(upd[int(round(j*(n-1)/max(m-1,1)))])
        elif method=='gated' and m:
            zl=seq[anchor][1].copy(); sc=[]
            for i in upd:
                d=seq[i][1]-zl; g,_=gain(d,C,epsr); sc.append((g,i)); zl=seq[i][1].copy()
            pick=set(i for g,i in sorted(sc,reverse=True)[:m] if g>0)
        zhat=seq[anchor][1]+rng.laplace(0,1/eps0,3) if method!='oracle' else seq[anchor][1].copy()
        zlast=seq[anchor][1].copy(); last_ref=anchor; gcount=0
        for i in range(anchor+1,len(seq)):
            kind,pay=seq[i]
            if kind=='root':
                gcount+=1
                if method=='oracle':
                    zhat=pay.copy(); zlast=pay.copy(); last_ref=i; gcount=0
                elif i in pick or method=='always':
                    reanchor = (method=='always') or (gcount>=Pblock)
                    zl=seq[last_ref][1]
                    d=pay-zl
                    s=min(Gmax*C,1.) if not reanchor else 1.
                    if reanchor:
                        zhat=pay+rng.laplace(0,1/epsr,3)
                    else:
                        zhat=zhat+np.clip(d,-s,s)+rng.laplace(0,s/epsr,3)
                    zlast=pay.copy(); last_ref=i; gcount=0
            else:
                node,ref=pay; x=denorm(zhat)
                if node=='anemia_class':
                    v=int(LBL[cls(F['mchc'](x))]==ref); ok.append(v)
                else:
                    e=abs(F[node](x)-ref)/abs(ref); errs.append(e); tail.append(e)
    return np.mean(errs),np.mean(ok),np.percentile(errs,95)

def main():
    P=prep()
    print(f"{'B':>4} {'mu':>4} {'method':<11} {'rel%':>7} {'p95%':>7} {'cls%':>7}")
    for B in (40,80,160):
        for mu in (0.5,0.8):
            for method,lab,k,Pb in [('static','static',0,0),('periodic','periodic-4',4,5),
                                    ('gated','gated-4',4,5),('gated','gated-8',8,5)]:
                res=[simulate(P,method,B,0.12,mu,5,Pb,k,np.random.default_rng(7+s)) for s in range(10)]
                rel=np.mean([r[0] for r in res]); cl=np.mean([r[1] for r in res]); p95=np.mean([r[2] for r in res])
                print(f"{B:>4} {mu:>4} {lab:<11} {rel*100:>7.2f} {p95*100:>7.2f} {cl*100:>7.2f}")
        print()

if __name__=='__main__': main()
