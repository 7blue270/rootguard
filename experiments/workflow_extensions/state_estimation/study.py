"""AutoResearch matched transcript experiment on frozen RootGuard cohorts."""
from pathlib import Path
import json
import numpy as np
from ..m8.study import (SEED,DEFAULT_SPLIT,OLDPLANS,ORIGINAL,BUDGETS,Branch,Context,data,
                         STATIC,order,prep,sim,ledger,utility,write,dump,HERE)
from .filters import streams

OUT=HERE.parent/'outputs/state_estimation/frozen_v1'
def load(template,large):
    path=Path('data_mine/finite_cache_physionet2019' if large else 'data_mine/finite_cache')
    records=sorted(data.load(path,[template]),key=order)
    split=json.loads((path/'split.json' if large else DEFAULT_SPLIT).read_text())
    train,val,test=[[r for r in records if split.get(r['subject_id'])==role] for role in ('train','validation','test')]
    if large:
        refs=json.loads((HERE/'original_protocol.json').read_text())['references']
        by={r['subject_id']:r for r in train}
        train=[by[s] for s in refs]
    ctx=Context(template,train,SEED)
    prep(ctx,train+val+test)
    spans=[r['_span'] for r in train]
    changes=np.concatenate([abs(np.diff(r['_y'])) for r in train])
    return ctx,val,test,float(np.median(spans)),float(np.median(changes[changes>1e-12]))

def branches(r,t,B,large):
    plan=ORIGINAL[f"{B}/{int(len(r['events'])>=8)}"]
    p=plan['periodic'] if large else OLDPLANS[t][str(B)]['branches']['P-cal']
    return Branch(**p),Branch(**plan['route'])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'protocol.json',dict(flow='AutoResearch plan -> bounded validation -> frozen transfer',
        primary='patient-equal four-anchor mean MAE task vs joint Gaussian; >=5% relative reduction; AST-specific gain vs same posterior root median; positive in >=2 structures',
        shared='same anchor service, strong periodic actions, raw grid releases and charges for all filters',
        uncertainty='paired patient bootstrap; pointwise intervals; no patient pooling across templates',
        response='MAE at first two turns following train-median nonzero target jump inclusive, overlapping windows union; jump amplitude error on event turn',
        tuning='two carry probabilities, rank on 80 PhysioNet A validation patients only; same scalar on every other template',
        guard='source B and MIMIC test previously used by M8, no adjustment this round'))
    ctx,val,test,span,jump=load('BUN_CR',True)
    logs=[]
    for memory in (.5,.9):
        scores=[]
        for r in val[:80]:
            for B in BUDGETS:
                periodic,_=branches(r,'BUN_CR',B,True)
                ps=sim(ctx,r,B,periodic,4,True)
                service=streams(ctx,ps,memory,False)['task']
                scores.append(abs(ctx.work.from_cache(service)['ratio']-r['_y']).mean())
        logs.append(dict(memory=memory,validation_mae=np.mean(scores)))
    write(OUT/'iterations.csv',logs)
    memory=min(logs,key=lambda x:x['validation_mae'])['memory']
    dump(OUT/'frozen.json',dict(memory=memory,development_ids=[r['subject_id'] for r in val[:80]]))
    print('Frozen memory',memory,flush=True)
    for name,t in [('PhysioNet','BUN_CR'),('MIMIC_BUN','BUN_CR'),('MELD_NA','MELD_NA'),('ANEMIA','ANEMIA'),('FIB4','FIB4')]:
        large=name=='PhysioNet'
        if not large:ctx,val,test,span,jump=load(t,False)
        rows=[]
        for r in test:
            y=r['_y'];event=np.flatnonzero(abs(np.diff(y))>=jump)+1
            response=np.unique(np.minimum(len(y)-1,np.r_[event,event+1]))
            for B in BUDGETS:
                periodic,m8=branches(r,t,B,large)
                R=4 if large else 20
                ps=sim(ctx,r,B,periodic,R,True);ledger(ps,B,periodic)
                caches=streams(ctx,ps,memory)
                caches['overwrite']=ps['caches']
                for label,b in [('rootguard',Branch(1.)),('static',STATIC),('m8',m8)]:
                    caches[label]=sim(ctx,r,B,b,R,False)['caches']
                for k,c in caches.items():
                    u=utility(ctx,r,{**ps,'caches':c})
                    pred=ctx.work.from_cache(c)[ctx.work.target]
                    row=dict(subject=r['subject_id'],B=B,method=k,high=int(r['_span']>span),
                        response=abs(pred[:,response]-y[response]).mean() if len(event) else None,
                        jump_error=abs(np.diff(pred)[:,event-1]-np.diff(y)[event-1]).mean() if len(event) else None)
                    row.update({key:v.mean() for key,v in u.items() if key not in ('spent','roots','refreshes')})
                    rows.append(row)
        write(OUT/(name+'.csv'),rows)
        print(name,len(test),'finished',flush=True)
if __name__=='__main__':main()
