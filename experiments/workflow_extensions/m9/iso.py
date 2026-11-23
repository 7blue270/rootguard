"""Full original 54-point grid; batch independent stays with identical public actions."""
import json
from collections import defaultdict
import numpy as np
from ..m7.engine import cdf
from ..m4_carry.rootguard_mopt import uniforms
from ..m4_carry.core import key_seed
from .study import OUT,COHORTS,setup,Branch,choose,ORIGINAL,SEED,BUDGETS,write
from .policy import plans,pick,estimate

def draw(ctx,turn,roots,eps,x,u,branch):
    raw=np.empty(u.shape);cal=np.empty_like(raw)
    for j,(i,e) in enumerate(zip(roots,eps)):
        values,inv=np.unique(x[:,i],return_inverse=True)
        indices=np.empty(u.shape[:2],int)
        for k,v in enumerate(values):
            ids=np.flatnonzero(inv==k)
            indices[ids]=np.searchsorted(cdf(float(v),float(e)),u[ids,:,j],side='right')
        raw[:,:,j]=indices/999
        cal[:,:,j]=ctx.marginal_grid(i,turn,float(e),branch.shrink)[indices] if branch.calibration=='marginal' else raw[:,:,j]
    return raw.reshape(-1,len(roots)),cal.reshape(-1,len(roots))

def batch(ctx,b,turn,B,ids,x,y,u,co,median):
    n=len(ids);R=u.shape[2];T=y.shape[1];d=x.shape[-1]
    eps=ctx.allocation(B*b.anchor,ctx.roots,b.weights)
    raw,anchor=draw(ctx,0,ctx.roots,eps,x[ids,0],u[ids,1],b)
    hist=[];shape=(n*R,T,d)
    ps=dict(anchor_cache=raw,anchor_service=anchor,anchor_epsilon=np.broadcast_to(eps,(n*R,d)),histories=hist,caches=np.empty(shape))
    stream=np.broadcast_to(anchor[:,None],shape)
    if turn:
        mask=b.fixed_roots or ctx.roots
        e=ctx.allocation(B-B*b.anchor,mask,b.refresh_weights)
        z,fresh=draw(ctx,turn-1,mask,e,x[ids,turn-1],u[ids,turn][:,:,list(mask)],b)
        after=raw.copy();obs=anchor.copy();after[:,mask]=z;obs[:,mask]=fresh
        ev=np.zeros(d);ev[list(mask)]=e
        hist.append(dict(turn=turn,cache_after=after,service=obs,actual_epsilon=np.broadcast_to(ev,(n*R,d)),selected=np.broadcast_to(ev>0,(n*R,d))))
        if co is None:
            if b.fusion:
                var=np.minimum(.25,8/eps**2)
                pred=var+b.drift_multiplier*ctx.drift(turn-1,tuple([0]*d))
                gain=pred[list(mask)]/(pred[list(mask)]+np.minimum(.25,8/e**2))
                obs[:,mask]=anchor[:,mask]+gain*(fresh-anchor[:,mask])
            stream=stream.copy();stream[:,turn-1:]=obs[:,None]
    if co is not None:stream=estimate(ctx,ps,co,median)
    pred=ctx.work.from_cache(stream)[ctx.work.target].reshape(n,R,T)
    return abs(pred-y[ids,None]).mean(axis=(1,2))

def run(name):
    ctx,(_,_,test),_=setup(name);dest=OUT/name
    cfg=json.loads((dest/'frozen.json').read_text());matrix=plans(ctx,cfg['package'])
    grid=json.loads((OUT.parents[1]/'m8/original_protocol.json').read_text())['grid']
    x=np.stack([r['_x'] for r in test]);y=np.stack([r['_y'] for r in test])
    high=np.array([r['_span']>cfg['span'] for r in test]);rr=[]
    for B in grid:
        near=min(BUDGETS,key=lambda v:abs(np.log(v/B)));key=str(near)
        u=uniforms(np.stack([np.random.default_rng(key_seed(SEED,'physionet2020',r['subject_id'],B)).laplace(size=(r['T']+1,4,len(ctx.roots))) for r in test]))
        methods=('rootguard','m8','static','full','package','m9')
        buckets=defaultdict(lambda:defaultdict(list))
        for i,r in enumerate(test):
            pub=r['_public'];events=[e['turn'] for e in pub['events'] if e['turn']>1]
            for m in methods:
                a=pick(cfg['route'],B,pub) if m=='m9' else m
                b=Branch(1.) if m=='rootguard' else choose(ORIGINAL,B,pub) if m=='m8' else matrix[a]
                op=events if b.appointment else [t for t in events if t<r['T']]
                clock=b.appointment or (r['T']+2)/2
                turn=min(op,key=lambda t:(abs(t-clock),t)) if b.period and op else 0
                typ=None if m in ('rootguard','m8') else 'static' if a=='static' else 'period'
                buckets[b,turn,typ][m].append(i)
        vals={m:np.empty(len(test)) for m in methods}
        for (b,turn,typ),mapping in buckets.items():
            ids=np.unique(np.concatenate(list(mapping.values())))
            co=None if typ is None else {bool(int(k)):np.asarray(v) for k,v in cfg['models'][key][typ].items()}
            for start in range(0,len(ids),192):
                group=ids[start:start+192]
                values=batch(ctx,b,turn,B,group,x,y,u,co,cfg['median'])
                for m,subset in mapping.items():
                    ok=np.isin(group,subset);vals[m][group[ok]]=values[ok]
        for m,v in vals.items():
            for g,subset in [('all',np.ones(len(test),bool)),('high',high),('low',~high)]:
                if subset.any():rr.append(dict(B=B,method=m,group=g,mae=v[subset].mean()))
    write(dest/'grid54.csv',rr)
if __name__=='__main__':
    import sys
    for name in sys.argv[1:] or COHORTS:run(name)
