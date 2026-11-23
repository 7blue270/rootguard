"""54-budget scans of the other ablations, including actual repeated full-root releases."""
import json,sys
from collections import defaultdict
import numpy as np
from .iso import batch,draw
from .study import OUT,COHORTS,setup,pick,SEED,BUDGETS,write
from .policy import estimate,plans
from ..m4_carry.rootguard_mopt import uniforms
from ..m4_carry.core import key_seed

def periodic(ctx,b,B,x,y,u,schedules,co,median):
    n,T,d=x.shape;R=u.shape[2]
    eps=ctx.allocation(B*b.anchor,ctx.roots,b.weights)
    raw,anchor=draw(ctx,0,ctx.roots,eps,x[:,0],u[:,1],b)
    buckets=defaultdict(list)
    for i,s in enumerate(schedules):buckets[s].append(i)
    result=np.empty(n)
    for times,members in buckets.items():
        for start in range(0,len(members),128):
            ids=np.asarray(members[start:start+128]);nr=len(ids)*R
            a=anchor.reshape(n,R,d)[ids].reshape(nr,d);hist=[]
            ps=dict(anchor_cache=raw.reshape(n,R,d)[ids].reshape(nr,d),anchor_service=a,anchor_epsilon=np.broadcast_to(eps,(nr,d)),histories=hist,caches=np.empty((nr,T,d)))
            for turn in times:
                e=ctx.allocation((1-b.anchor)*B/len(times),ctx.roots,b.weights)
                z,f=draw(ctx,turn-1,ctx.roots,e,x[ids,turn-1],u[ids,turn],b)
                hist.append(dict(turn=turn,cache_after=z,service=f,selected=np.ones((nr,d),bool),actual_epsilon=np.broadcast_to(e,(nr,d))))
            c=estimate(ctx,ps,co,median)
            pred=ctx.work.from_cache(c)[ctx.work.target].reshape(len(ids),R,T)
            result[ids]=abs(pred-y[ids,None]).mean(axis=(1,2))
    return result

def run(name):
    ctx,(_,_,test),_=setup(name);dest=OUT/name
    cfg=json.loads((dest/'frozen.json').read_text());matrix=plans(ctx,cfg['package'])
    grid=json.loads((OUT.parents[1]/'m8/original_protocol.json').read_text())['grid']
    x=np.stack([r['_x'] for r in test]);y=np.stack([r['_y'] for r in test])
    high=np.array([r['_span']>cfg['span'] for r in test]);rr=[]
    events=[[e['turn'] for e in r['events']] for r in test]
    sched=[tuple(ts[1::2] or ts[-1:]) for ts in events]
    for B in grid:
        near=min(BUDGETS,key=lambda v:abs(np.log(v/B)));key=str(near)
        u=uniforms(np.stack([np.random.default_rng(key_seed(SEED,'physionet2020',r['subject_id'],B)).laplace(size=(r['T']+1,4,len(ctx.roots))) for r in test]))
        coef={k:{bool(int(i)):np.asarray(c) for i,c in v.items()} for k,v in cfg['models'][key].items()}
        methods=('early','anchor75','equal','combined','pooled','force_dynamic','no_estimator')
        buckets=defaultdict(lambda:defaultdict(list))
        for i,r in enumerate(test):
            pub=r['_public']
            for m in methods:
                a=cfg['pooled'][key] if m=='pooled' else pick(cfg['forced'],B,pub) if m=='force_dynamic' else pick(cfg['route'],B,pub) if m=='no_estimator' else m
                b=matrix[a];turn=min(events[i],key=lambda t:(abs(t-b.appointment),t)) if b.appointment and events[i] else 0
                typ=None if m=='no_estimator' else 'static' if a=='static' else 'period'
                buckets[b,turn,typ][m].append(i)
        vals={m:np.empty(len(test)) for m in methods}
        for (b,turn,typ),mapping in buckets.items():
            ids=np.unique(np.concatenate(list(mapping.values())))
            for start in range(0,len(ids),192):
                group=ids[start:start+192];v=batch(ctx,b,turn,B,group,x,y,u,None if typ is None else coef[typ],cfg['median'])
                for m,subset in mapping.items():
                    ok=np.isin(group,subset);vals[m][group[ok]]=v[ok]
        vals['period']=periodic(ctx,matrix['period'],B,x,y,u,sched,coef['period'],cfg['median'])
        for m,v in vals.items():
            for g,mask in [('all',np.ones(len(test),bool)),('high',high),('low',~high)]:
                if mask.any():rr.append(dict(B=B,method=m,group=g,mae=v[mask].mean()))
    write(dest/'extra54.csv',rr)
if __name__=='__main__':
    for name in sys.argv[1:] or COHORTS:run(name)
