"""Fixed follow-up check of continuous corrections on disjoint source-A patients."""
import json
import numpy as np
from .window_study import data, order, references, load, prep, branches, BUDGETS, write, dump
from .calibrated_window import DEST,predict
from ..m7.engine import _execute
from ..m4_carry.core import key_seed
from pathlib import Path

def main():
    ctx,_,_,_,_=load('BUN_CR',True)
    path=Path('data_mine/finite_cache_physionet2019')
    split=json.loads((path/'split.json').read_text())
    ref={r['subject_id'] for r in references('BUN_CR',True)}
    rows=sorted((r for r in data.load(path,['BUN_CR']) if split.get(r['subject_id'])=='train' and r['subject_id'] not in ref),key=order)[:512]
    dump(DEST/'replication_protocol.json',dict(n=len(rows),design='post-hoc independent-of-this-fit source A check; first 512 hash-ordered train IDs outside the 48 fixed references, new seed, fixed v3 coefficients and shrink, four repeats; no selection'))
    prep(ctx,rows)
    models=json.loads((DEST/'PhysioNet_coefficients.json').read_text())
    shrink=json.loads((DEST/'frozen.json').read_text())['shrink']
    out=[]
    for r in rows:
        losses={k:[] for k in ('joint','root_window','task_window','task_now')}
        for B in BUDGETS:
            b,_=branches(r,'BUN_CR',B,True)
            rng=np.random.default_rng(key_seed(712,'continuous-check',r['subject_id'],B))
            ps=_execute(r['_public'],B,b,ctx,lambda turn,ii:r['_x'][turn-1,list(ii)],rng.laplace(size=(r['T']+1,4,len(ctx.roots))),True)
            model={k:np.asarray(v) for k,v in models[str(B)].items()}
            for k,c in predict(ctx,ps,model,shrink).items():
                y=ctx.work.from_cache(c)[ctx.work.target]
                losses[k].append(abs(y-r['_y']).mean())
        out.append(dict(subject=r['subject_id'],**{k:np.mean(v) for k,v in losses.items()}))
    write(DEST/'replication_patients.csv',out)
    ix=np.random.default_rng(113).integers(len(out),size=(2000,len(out)))
    a=np.array([r['task_window'] for r in out])
    effects={}
    for k in ('joint','root_window','task_now'):
        b=np.array([r[k] for r in out])
        interval=np.quantile(100*(1-a[ix].mean(axis=1)/b[ix].mean(axis=1)),[.025,.975])
        effects[k]=[100*(1-a.mean()/b.mean()),*interval]
    dump(DEST/'replication_result.json',effects)
if __name__=='__main__':main()
