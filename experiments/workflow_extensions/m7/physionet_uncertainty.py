"""Two-level paired intervals for the two predeclared high-volatility controls."""
import numpy as np
from .physionet_study import OUT,DIR,read,order,simulate,noise_for,BUDGETS,SEED,write
from .engine import Context,public_view,Branch
from ..finite_cache import data

def main():
    cfg=read(OUT/'protocol.json'); plans=read(OUT/'frozen.json');split=read(DIR/'split.json')
    by={r['subject_id']:r for r in data.load(DIR,['BUN_CR'])}
    ctx=Context('BUN_CR',[by[s] for s in cfg['references']],SEED)
    records=sorted([r for r in by.values() if split[r['subject_id']]=='test'],key=order)
    methods=['route','periodic','static','dynamic','matched_latest','oldroute']
    from .physionet_study import OLD
    old=read(OLD/'frozen.json')['BUN_CR']
    errors=np.empty((len(records),4,4,len(methods)))
    flags=[]
    for i,r in enumerate(records):
        x=data.replay(r);r['_public']=public_view(r);r['_raw']=ctx.work.normalize(x)
        y=ctx.work.evaluate(x)['ratio'];flags.append(np.ptp(y)>cfg['threshold'])
        for j,B in enumerate(BUDGETS):
            d=int(len(r['events'])>=8); n=noise_for(r,B,4)
            plan=plans[f'{B}/{d}']; memo={}
            for k,m in enumerate(methods):
                b=Branch(**(old[str(B)]['branches']['R-new'] if m=='oldroute' else plan[m]))
                if b not in memo:
                    sim=simulate(ctx,r,B,b,n,False)
                    memo[b]=abs(ctx.work.from_cache(sim['caches'])['ratio']-y).mean(axis=1)
                errors[i,j,:,k]=memo[b]
    np.savez_compressed(OUT/'paired_mae.npz',errors=errors,methods=methods,subjects=[r['subject_id'] for r in records],high=flags)
    rows=[];rng=np.random.default_rng(SEED)
    for group,mask in [('all',np.ones(len(flags),bool)),('high',np.array(flags)),('low',~np.array(flags))]:
        e=errors[mask];draw=[]
        for _ in range(1000):
            sub=rng.integers(len(e),size=(len(e),1,1))
            noise=rng.integers(4,size=(len(e),4,4))
            draw.append(e[sub,np.arange(4)[None,:,None],noise].mean(axis=(0,2)))
        draw=np.array(draw)
        for m,ref in [('route','periodic'),('route','static'),('route','oldroute'),('dynamic','periodic'),('dynamic','static'),('dynamic','matched_latest')]:
            diff=e[:,:,:,methods.index(m)]-e[:,:,:,methods.index(ref)]
            z=draw[:,:,methods.index(m)]-draw[:,:,methods.index(ref)]
            for B,v,u in [('avg',diff.mean(),z.mean(axis=1))]+[(b,diff[:,j].mean(),z[:,j]) for j,b in enumerate(BUDGETS)]:
                l,h=np.quantile(u,[.025,.975]);a,c=np.quantile(u,[.0125,.9875])
                rows.append(dict(group=group,B=B,method=m,ref=ref,delta=v,lo=l,hi=h,bonf2_lo=a,bonf2_hi=c))
    write(OUT/'two_level.csv',rows)

if __name__=='__main__':main()
