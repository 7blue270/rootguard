"""Same raw releases, recalibrated with exactly the candidate's local reference stays."""
from .study import (np,json,Context,SEED,references,prep,order,data,DATA,OUT,PERIOD,STATIC,
                    sim,BUDGETS,write,windows)
def main(t):
    rec=sorted(data.load(DATA,[t]),key=order)
    split=json.loads((DATA/'split.json').read_text())
    train=[r for r in rec if split[r['subject_id']]=='train']
    test=[r for r in rec if split[r['subject_id']]=='test']
    origin=Context(t,references(t,t=='BUN_CR'),SEED)
    ctx=Context(t,train[:192],SEED)
    prep(origin,test+train)
    cut=np.median([r['_span'] for r in train])
    change=np.concatenate([abs(np.diff(r['_y'])) for r in train])
    jump=np.median(change[change>0]);rows=[]
    for r in test:
        y=r['_y'];events=np.flatnonzero(abs(np.diff(y))>=jump)+1
        turns=np.unique(np.minimum(len(y)-1,np.r_[events,events+1]))
        for B in BUDGETS:
            for name,plan in [('local_joint',PERIOD),('local_static',STATIC)]:
                ps=sim(origin,r,B,plan,4,True)
                ps['anchor_service']=ctx.calibrated(ps['anchor_cache'],np.zeros_like(ps['anchor_cache'],int),ps['anchor_epsilon'],plan)
                for h in ps['histories']:
                    h['service']=ctx.calibrated(h['cache_after'],h['last'],h['last_eps'],plan)
                c=np.broadcast_to(ps['anchor_service'][:,None],ps['caches'].shape).copy()
                for st,en,x,_ in windows(ctx,ps):c[:,st:en]=x[:,None]
                err=abs(ctx.work.from_cache(c)[ctx.work.target]-y)
                rows.append(dict(subject=r['subject_id'],B=B,method=name,high=int(r['_span']>cut),dense=int(len(r['events'])>=4),mae=err.mean(),den=abs(y).mean(),response=err[:,turns].mean() if len(turns) else None,p95=np.quantile(err,.95,axis=1).mean(),refreshes=ps['refreshes'].mean(),spent=ps['spent'].mean()))
    write(OUT/(t+'_local.csv'),rows)
if __name__=='__main__':
    for t in ('BUN_CR','FIB4'):main(t)
