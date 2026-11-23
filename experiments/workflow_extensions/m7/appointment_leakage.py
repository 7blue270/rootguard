"""Incremental reconstruction benefit of older observations on the same transcript."""
from collections import defaultdict
import numpy as np
from .appointment_study import (OUT,OLD,DAT,BUDGETS,SEED,STATIC,Branch,Context,
                                setup,read,write,execute,cumulative)
from .analysis import quantile
from ..m4_carry.core import key_seed

def main():
    parts,_,_=setup(DAT)
    follow=OUT.parent/'appointment_gate_20261119'
    plans=read(OUT/'frozen.json'); gates=read(follow/'frozen.json'); old=read(OLD/'frozen.json')
    rows=[]
    for t,p in parts.items():
        ctx=Context(t,p['train'],SEED)
        store=defaultdict(list)
        for B in BUDGETS:
            slot=plans[t][str(B)]; gate=gates[t][str(B)]
            win=Branch(**slot['win']); gated=Branch(**gate['win'])
            prior=Branch(**old[t][str(B)]['branches']['R-new'])
            methods=dict(static=STATIC,periodic=Branch(**old[t][str(B)]['branches']['P-cal']),
                         oldroute=prior,scheduled=win,route=win if slot['dynamic'] else STATIC,
                         gate=gated,gateroute=gated if gate['dynamic'] else prior)
            at=defaultdict(list)
            for rec in sorted(p['test'],key=lambda r:str(r['subject_id'])):
                cache={}
                for n,b in methods.items():
                    if b not in cache:
                        res=execute(rec,B,b,ctx,'test',20,True)
                        cache[b]=cumulative(ctx,rec,res)
                    for model in ('mean','kalman','changepoint'):
                        at[n,model].append(cache[b][model+'_history_minus_latest_hit'])
            for k,v in at.items():store[k].append(v)
        rng=np.random.default_rng(key_seed('history-only',t)); size=len(p['test'])
        pat=rng.integers(size,size=(1000,size,1)); noise=rng.integers(20,size=(1000,size,20))
        for (m,model),arr in store.items():
            x=np.mean(arr,axis=0);lo,hi=quantile(x[pat,noise].mean(axis=(1,2)))
            rows.append(dict(template=t,method=m,observer=model,extra_success_pp=x.mean(),lo=lo,hi=hi))
    write(OUT/'incremental_leakage.csv',rows)

if __name__=='__main__':
    main()
