"""Patient-level new-plan costs and paired freshness decomposition."""
import numpy as np
from .physionet_study import OUT,DIR,read,order,simulate,noise_for,BUDGETS,SEED,STATIC,write
from .physionet_policy import choose,run
from .engine import Context,public_view,Branch
from .audit import ledger
from ..finite_cache import data

def main():
    cfg=read(OUT/'protocol.json'); plans=read(OUT/'frozen.json')
    split=read(DIR/'split.json'); records=sorted(data.load(DIR,['BUN_CR']),key=order)
    refs={r['subject_id']:r for r in records}
    ctx=Context('BUN_CR',[refs[s] for s in cfg['references']],SEED)
    rows=[]
    for r in records:
        if split[r['subject_id']]!='test':continue
        raw=data.replay(r)
        r['_raw']=ctx.work.normalize(raw);r['_public']=public_view(r)
        truth=ctx.work.evaluate(raw)['ratio']
        for B in BUDGETS:
            d=int(len(r['events'])>=8)
            n=noise_for(r,B,4)
            dyn=Branch(**plans[f'{B}/{d}']['dynamic'])
            route=choose(plans,B,r['_public'])
            res=simulate(ctx,r,B,dyn,n,True)
            ledger(res,B,dyn)
            if not rows:
                live=run(ctx,plans,B,r['_public'],lambda t,names:[raw[t-1,ctx.work.roots.index(name)] for name in names])
                assert live['spent']<=B+1e-9
            static=simulate(ctx,r,B,STATIC,n,False)
            y0=ctx.work.from_cache(res['anchor_service'])['ratio']
            y=ctx.work.from_cache(res['caches'])['ratio']
            sy=ctx.work.from_cache(static['caches'])['ratio']
            err=lambda x:abs(x-truth).mean()
            anchor_cost=err(y0[:,None])-err(sy)
            freshness=err(y0[:,None])-err(y)
            traces=res['histories']; eps=[res['anchor_epsilon'][0]]
            harm=0
            prev=y0
            for j,h in enumerate(traces):
                start=h['turn']-1;end=traces[j+1]['turn']-1 if j+1<len(traces) else len(truth)
                future=truth[start:end]
                after=y[:,start]
                harm+=np.sum(abs(after[:,None]-future).mean(axis=1)>abs(prev[:,None]-future).mean(axis=1)+1e-12)
                prev=after;eps.append(h['actual_epsilon'][0])
            charges=np.stack(eps)
            rows.append(dict(subject=r['subject_id'],B=B,high=np.ptp(truth)>cfg['threshold'],
                routed_dynamic=route.period is not None,anchor_cost=anchor_cost,freshness=freshness,
                forced_minus_static=err(y)-err(sy),forced_refresh=len(traces),harmful=harm/4,
                bun_share=charges[:,0].sum()/B,cr_share=charges[:,1].sum()/B,
                eps_min=charges[charges>0].min(),eps_max=charges.max()))
    write(OUT/'freshness.csv',rows)

if __name__=='__main__':main()
