"""Patient-clustered paired analysis for frozen source-B predictions."""
from collections import defaultdict
import csv
import numpy as np
from .original_study import OUT,REF,read,write,BUDGETS

def load(file):
    with (OUT/file).open() as f:return list(csv.DictReader(f))

def main():
    cfg=read(OUT/'protocol.json')
    rows=load('paths.csv')
    ids=sorted({r['subject'] for r in rows})
    idx={s:i for i,s in enumerate(ids)}
    budgets=list(BUDGETS)
    panels={}
    for r in rows:
        key=r['method']
        if key not in panels:panels[key]={k:np.zeros((len(ids),4)) for k in r if k not in ('subject','B','method')}
        for k,mat in panels[key].items():mat[idx[r['subject']],budgets.index(float(r['B']))]=float(r[k])
    groups={'all':np.ones(len(ids),bool),'high':panels['route']['span'][:,0]>cfg['threshold'],'low':panels['route']['span'][:,0]<=cfg['threshold'],'dense16':panels['route']['batches'][:,0]>=16}
    stats=[]; contrasts=[]
    rng=np.random.default_rng(20261120)
    for group,mask in groups.items():
        n=int(mask.sum())
        draws=rng.integers(n,size=(1000,n))
        for B,cols in [('avg',range(4))]+[(b,[i]) for i,b in enumerate(budgets)]:
            estimates={}
            for m,f in panels.items():
                vals={k:v[mask][:,cols].mean(axis=1) for k,v in f.items()}
                for k in ('mae','p95','refresh','roots','spent','batch4_ms','wmape','spend_pct'):
                    if k=='wmape':
                        v=100*vals['mae'].mean()/vals['truth_abs'].mean()
                        samples=100*vals['mae'][draws].mean(axis=1)/vals['truth_abs'][draws].mean(axis=1)
                    else:
                        a=(100*f['spent'][mask][:,cols]/np.array(budgets)[list(cols)]).mean(axis=1) if k=='spend_pct' else vals[k]
                        v=a.mean();samples=a[draws].mean(axis=1)
                    estimates[m,k]=v,samples
                    stats.append(dict(group=group,n=n,B=B,method=m,metric=k,value=v))
            for m,ref in [('route',r) for r in ('periodic','static','oldroute','native_static','native_periodic','dynamic')]+[('dynamic','time'),('time','count'),('count','periodic'),('dynamic','static')]:
                for k in ('mae','wmape','p95','refresh','roots','spend_pct'):
                    x,u=estimates[m,k];y,v=estimates[ref,k];lo,hi=np.quantile(u-v,[.025,.975])
                    contrasts.append(dict(group=group,n=n,B=B,method=m,baseline=ref,metric=k,delta=x-y,lo=lo,hi=hi))
    write(OUT/'summary.csv',stats);write(OUT/'paired.csv',contrasts)
    grid=cfg['grid']; curves=defaultdict(lambda:np.full((len(grid),len(ids)),np.nan))
    with (OUT/'grid.csv').open() as f:
        for r in csv.DictReader(f):
            if r['method'] in ('periodic','static','oldroute','route'):
                curves[r['method']][grid.index(float(r['B'])),idx[r['subject']]]=float(r['mae'])
    thresholds=read(REF/'targets.json')['BUN_CR']
    weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=1000)/len(ids)
    iso=[]
    for regime,tg in thresholds.items():
        for side in ('lo','hi'):
            tau=tg['tau_'+side]
            def first(c):
                return np.min(np.where(c.T<=tau,np.array(grid),np.inf),axis=-1)
            hits={}
            for m,c in curves.items():
                assert np.isfinite(c).all()
                boot=c@weights.T
                hits[m]=first(c.mean(1)),first(boot),first(np.quantile(boot,.95,axis=1))
            x,a,u=hits['route']
            for ref in ('static','periodic','oldroute'):
                y,b,v=hits[ref];both=np.isfinite(a)&np.isfinite(b)
                ci=np.quantile(1-a[both]/b[both],[.025,.975]) if both.any() else [None,None]
                iso.append(dict(regime=regime,side=side,tau=tau,ref=ref,B_min=x,baseline_B_min=y,saving=1-x/y if np.isfinite([x,y]).all() else None,
                    candidate_UCB=u,baseline_UCB=v,saving_UCB=1-u/v if np.isfinite([u,v]).all() else None,both=both.mean(),lo=ci[0],hi=ci[1]))
    write(OUT/'iso.csv',iso)
    attacks=load('attacks.csv')
    by=defaultdict(lambda:defaultdict(list))
    for r in attacks:
        for k,v in r.items():
            if k not in ('B','method','subject'):by[r['method'],k][r['subject']].append(float(v))
    privacy=[]
    for (m,k),table in by.items():
        ss=sorted(table);base=by['periodic',k]
        x=np.array([np.mean(table[s]) for s in ss]); y=np.array([np.mean(base[s]) for s in ss])
        d=x-y;boot=rng.integers(len(d),size=(1000,len(d)))
        lo,hi=np.quantile(d[boot].mean(axis=1),[.025,.975])
        privacy.append(dict(method=m,metric=k,n=len(ss),estimate=x.mean(),delta_vs_periodic=d.mean(),lo=lo,hi=hi))
    write(OUT/'privacy.csv',privacy)

if __name__=='__main__':main()
