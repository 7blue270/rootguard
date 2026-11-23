"""Paired uncertainty, partial attribution and right-censored iso-utility."""
from collections import defaultdict
from itertools import product
import csv
import json
import numpy as np

from .appointment_study import OUT,OLD,REF,DAT,BUDGETS,SEED,setup,read,write,Context
from .analysis import derived,quantile,mean_valid
from .refresh_study import bmin,finite
from ..m4_carry.core import key_seed

def main(out=OUT):
    parts,_,grid=setup(DAT)
    tau=read(REF/'targets.json')
    strata={(r['template'],r['subject']):int(r['high']) for r in csv.DictReader((out/'strata.csv').open())}
    panels=defaultdict(dict)
    with (out/'paths.csv').open() as f:
        for r in csv.DictReader(f):
            t,B,m,s,i=r.pop('template'),float(r.pop('B')),r.pop('method'),r.pop('subject'),int(r.pop('repeat'))
            panels[t,B,m][s,i]={k:float(v) for k,v in r.items() if v}
    with (OLD/'paths.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            m={'R-new':'oldroute','S-cal':'static','P-cal':'periodic'}.get(r['method'])
            if m and r['B'] not in BUDGETS:
                panels[r['template'],r['B'],m][str(r['subject']),r['repeat']]={'mae':r['mae']}
    summaries=[]; pairs=[]; iso=[]; sub=[]; tests=[]
    contrasts=[('route','oldroute'),('route','static'),('route','periodic'),('scheduled','oldroute'),
               ('scheduled','periodic'),('scheduled','static'),('scheduled','full'),('pack','full'),('scheduled','pack')]
    for t,p in parts.items():
        ctx=Context(t,p['train'],SEED)
        subjects=sorted(str(r['subject_id']) for r in p['test'])
        n=len(subjects)
        rng=np.random.default_rng(key_seed('appointments',t))
        pat=rng.integers(n,size=(1000,n,1)); noise=rng.integers(20,size=(1000,n,20))
        curves=defaultdict(list); anchors=defaultdict(list); raw=defaultdict(list)
        for B in grid:
            methods=[m for a,b,m in panels if a==t and b==B]
            for m in methods:
                tab=panels[t,B,m]
                keys=sorted({k for r in tab.values() for k in r})
                x=np.array([[[tab[s,i].get(k,np.nan) for k in keys] for i in range(20)] for s in subjects])
                center=dict(zip(keys,mean_valid(x,(0,1))))
                draws={}
                # Bound peak memory across metric families.
                for j in range(0,len(keys),8):
                    draws.update(zip(keys[j:j+8],mean_valid(x[:,:,j:j+8][pat,noise],(1,2)).T))
                curves[m].append((center['mae'],draws['mae']))
                if B not in BUDGETS: continue
                raw[m].append(x[:,:,keys.index('mae')])
                for fields in (center,draws):
                    fields.update(derived(fields,ctx))
                    fields['spend_pct']=100*fields['spent']/B
                    den=fields['refreshes']
                    fields['harm_pct']=np.divide(100*fields['harmful'],den,out=np.full_like(den,np.nan),where=den>0)
                keep=[k for k in center if ':' not in k and k not in ('harmful','final_truth_abs')]
                anchors[m].append({k:(center[k],draws[k]) for k in keep})
                for k in keep:
                    lo,hi=quantile(draws[k])
                    summaries.append(dict(template=t,B=B,method=m,metric=k,estimate=finite(center[k]),lo=lo,hi=hi))
        for a,b in contrasts:
            for k in anchors[a][0]:
                vals=[(x[k][0]-y[k][0],x[k][1]-y[k][1]) for x,y in zip(anchors[a],anchors[b])]
                avg=np.mean([v for v,_ in vals])
                if not np.isfinite(avg):continue
                lo,hi=quantile(np.mean([d for _,d in vals],axis=0))
                pairs.append(dict(template=t,method=a,reference=b,metric=k,delta=avg,lo=lo,hi=hi))
            diff=np.mean(np.array(raw[a])-raw[b],axis=0)
            for level in (0,1):
                ids=[i for i,s in enumerate(subjects) if strata[t,s]==level]
                if not ids: continue
                ns=len(ids); sp=rng.choice(ids,size=(1000,ns,1)); sn=rng.integers(20,size=(1000,ns,20))
                lo,hi=quantile(diff[sp,sn].mean(axis=(1,2)))
                sub.append(dict(template=t,method=a,reference=b,high=level,n=ns,mae_delta=diff[ids].mean(),lo=lo,hi=hi))
            if a=='route' and b=='oldroute' and t in ('MELD_NA','BUN_CR'):
                d=diff.mean(axis=1)
                null=np.array(list(product((-1,1),repeat=n)))@d/n
                tests.append(dict(template=t,n=n,delta=d.mean(),p=float(np.mean(abs(null)>=abs(d.mean())-1e-12))))
        for scenario,tg in tau[t].items():
            for label in ('lo','hi'):
                threshold=tg['tau_'+label]
                hits={}
                for m,curve in curves.items():
                    if len(curve)!=len(grid):continue
                    mean,d=zip(*curve); mat=np.array(d).T
                    hits[m]=(bmin(np.array(mean),grid,threshold),bmin(mat,grid,threshold),bmin(np.quantile(mat,.95,axis=0),grid,threshold))
                for a in ('route','scheduled'):
                    for b in ('oldroute','static','periodic'):
                        v,x,u=hits[a]; w,y,z=hits[b]
                        both=np.isfinite(x)&np.isfinite(y)
                        lo,hi=quantile(1-x[both]/y[both])
                        iso.append(dict(template=t,scenario=scenario,target=label,tau=threshold,method=a,reference=b,
                            B_min=finite(v),base_B_min=finite(w),saving=finite(1-v/w) if np.isfinite(w) and np.isfinite(v) else None,
                            B_UCB=finite(u),base_UCB=finite(z),saving_UCB=1-u/z if np.isfinite(u) and np.isfinite(z) else None,
                            reached_candidate=np.isfinite(x).mean(),both_reached=both.mean(),saving_lo=lo,saving_hi=hi))
    tests.sort(key=lambda r:r['p'])
    prev=0
    for i,r in enumerate(tests):
        prev=max(prev,min(1,(len(tests)-i)*r['p']));r['holm']=prev
    for name,rows in [('summary',summaries),('paired',pairs),('bmin',iso),('stratified',sub),('primary',tests)]:
        write(out/(name+'.csv'),rows)
    print(tests,flush=True)

if __name__=='__main__':
    main()
