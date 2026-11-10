"""Round-4: stratified comparison on the real (unamplified) ANEMIA trajectory.

Methods (same anchor budget, same per-refresh budget, same m_max=8):
  static        no refresh (RootGuard cache semantics)
  periodic      fixed public schedule, 8 evenly spaced refreshes
  root_drift    top-8 by max_i |dz_i|          (no task structure)
  final_only    top-8 by |a_mchc . dz|         (only the final target)
  workload_M    top-8 by G(S), full M          (final + intermediate, ORACLE gate, no eps_g)
  private_gate  top-8 by G + Lap(dG/eps_g)     (pays 20% of B for the gate)

Strata (pre-defined, multi-label on the update preceding each query):
  persistent_jump, transient_spike, cancellation, intermediate_only, class_cross
"""
import json, itertools, math
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
def denorm(z): return [DOM[r][0]+z[i]*(DOM[r][1]-DOM[r][0]) for i,r in enumerate(ROOTS)]

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

RECS=load()
cnt={n:0 for n in A}
for r in RECS:
    for kind,v in r['turns']:
        if kind=='query': cnt[v[0]]+=1
tot=sum(cnt.values()); W={n:cnt[n]/tot for n in cnt}
M=sum(W[n]*np.outer(a,a) for n,a in A.items())
LAM=float(np.max(np.linalg.eigvalsh(M)))

def prep():
    P=[]
    for r in RECS:
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
P=prep()

def tag_updates(seq, anchor):
    up=[i for i in range(anchor+1,len(seq)) if seq[i][0]=='root']
    tags={}; prev=None
    for n,i in enumerate(up):
        z=seq[i][1]; zp=prev if prev is not None else seq[anchor][1]
        prev=z
        d=z-zp; t=set()
        mz=100*z[0]/max(1e-9,z[1]); mp=100*zp[0]/max(1e-9,zp[1])
        if cls(mz)!=cls(mp): t.add('class_cross')
        if np.linalg.norm(d)>0.05 and abs(A['mchc']@d)/max(1e-9,mz)<0.001: t.add('cancellation')
        if abs(d[2])>0.02 and abs(d[2])>3*max(abs(d[0]),abs(d[1])): t.add('intermediate_only')
        if np.max(np.abs(d))>0.08:
            if n+1<len(up):
                dn=seq[up[n+1]][1]-z
                if dn@d < -0.5*(d@d): t.add('transient_spike')
                else: t.add('persistent_jump')
            else: t.add('persistent_jump')
        tags[i]=t
    return tags

def Jval(b,S,C,eps_r):
    if not S: return float(b@M@b)
    bb=b.copy(); bb[list(S)]=0.; k=len(S)
    return float(bb@M@bb+sum((min(C,1.)/(eps_r/k))**2*M[i,i] for i in S))
def Gval(b,C,eps_r):
    j0=Jval(b,(),C,eps_r); best=j0
    for k in (1,2,3):
        for S in itertools.combinations(range(3),k): best=min(best,Jval(b,S,C,eps_r))
    return j0-best

def scores(seq,anchor,method,C,eps_r):
    up=[i for i in range(anchor+1,len(seq)) if seq[i][0]=='root']
    sc={}; prev=None
    for i in up:
        zp=prev if prev is not None else seq[anchor][1]
        prev=seq[i][1]
        d=seq[i][1]-zp
        if method=='root_drift': sc[i]=float(np.max(np.abs(d)))
        elif method=='final_only': sc[i]=float(abs(A['mchc']@d))
        else: sc[i]=Gval(d,C,eps_r)
    return sc

def run(method,B,mu,C,Gmax,m_max,seed,gate_share=0.0):
    rng=np.random.default_rng(seed)
    recs=[]; nref=[]
    dG=2*C*LAM*math.sqrt(3)
    for anchor,seq in P:
        up=[i for i in range(anchor+1,len(seq)) if seq[i][0]=='root']
        n=len(up)
        if method=='static':
            eps0=B; epsr=None; pick=set()
        else:
            eps0=(mu-gate_share)*B; epsr=(1-mu)*B/m_max; pick=set()
            if method=='periodic':
                m=min(m_max,n)
                for j in range(m): pick.add(up[int(round(j*(n-1)/max(m-1,1)))])
            else:
                sc=scores(seq,anchor,method,C,epsr)
                if method=='private_gate':
                    eps_g=gate_share*B/n
                    sc={i:v+rng.laplace(0,dG/eps_g) for i,v in sc.items()}
                pick=set(i for _,i in sorted(((v,i) for i,v in sc.items()),reverse=True)[:m_max])
        tags=tag_updates(seq,anchor)
        zhat=seq[anchor][1]+rng.laplace(0,1/eps0,3)
        last_ref=anchor; last_upd=anchor; nref.append(len(pick))
        for i in range(anchor+1,len(seq)):
            kind,pay=seq[i]
            if kind=='root':
                last_upd=i
                if i in pick:
                    g=i-last_ref; s=min(g*C,1.); d=pay-seq[last_ref][1]
                    if g>Gmax: zhat=pay+rng.laplace(0,1/epsr,3)
                    else: zhat=zhat+np.clip(d,-s,s)+rng.laplace(0,s/epsr,3)
                    last_ref=i
            else:
                node,ref=pay; x=denorm(zhat); tg=frozenset(tags.get(last_upd,set()))
                if node=='anemia_class':
                    recs.append((tg,None,int(LBL[cls(F['mchc'](x))]==ref)))
                else:
                    recs.append((tg,abs(F[node](x)-ref)/abs(ref),None))
    return recs,float(np.mean(nref))

def summarize(recs):
    e=[r[1] for r in recs if r[1] is not None]; o=[r[2] for r in recs if r[2] is not None]
    return np.mean(e),np.mean(o)

def main():
    print(f"M=\n{np.round(M,3)}\nlambda_max={LAM:.3f}\n")
    STRATA=['persistent_jump','transient_spike','cancellation','intermediate_only','class_cross']
    # count strata
    cnts={s:0 for s in STRATA}
    for anchor,seq in P:
        for i,t in tag_updates(seq,anchor).items():
            for s in t: cnts[s]+=1
    print('stratum update counts:',cnts,'\n')
    for B in (160,320,640):
        print(f'=== B={B} ===')
        print(f"{'method':<14} {'rel%':>7} {'cls%':>7} {'#ref':>5} | "+
              " ".join(f"{s[:9]:>10}" for s in STRATA))
        for method,lab in [('static','static'),('periodic','periodic'),
                           ('root_drift','root_drift'),('final_only','final_only'),
                           ('workload_M','workload_M'),('private_gate','private_gate')]:
            allrecs=[]; nrs=[]
            for s in range(10):
                recs,nr=run(method,B,0.5,0.12,5,8,s,gate_share=0.2 if method=='private_gate' else 0.0)
                allrecs+=recs; nrs.append(nr)
            rel,cl=summarize(allrecs)
            cells=[]
            for st in STRATA:
                sub=[r for r in allrecs if st in r[0]]
                if not sub: cells.append(f"{'-':>10}"); continue
                e=[r[1] for r in sub if r[1] is not None]; o=[r[2] for r in sub if r[2] is not None]
                cells.append(f"{(np.mean(e)*100 if e else float('nan')):>5.1f}/{(np.mean(o)*100 if o else float('nan')):>4.0f}")
            print(f"{lab:<14} {rel*100:>7.2f} {cl*100:>7.2f} {np.mean(nrs):>5.1f} | "+" ".join(cells))
        print()

if __name__=='__main__': main()
