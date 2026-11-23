"""Patient-paired BUN/Cr study; source A chooses policy, source B evaluates it."""
import csv
import hashlib
import json
import os
from dataclasses import asdict,replace
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np

from ..finite_cache import data
from ..m4_carry.core import key_seed
from ..m7.engine import Branch,Context,_execute,public_view
from ..m7.audit import ledger
from ..m7.metrics import attack_result
from ..m7.appointment_study import cumulative,write,STATIC,OLD,REF,SEED,BUDGETS
from ..m7.analysis import combine_attack

DIR=Path('data_mine/finite_cache_physionet2019')
FUSION=os.environ.get('RG_PHYSIONET_FUSION','1')=='1'
OUT=OLD.parent/('physionet_fusion_20261120' if FUSION else 'physionet_20261120')
def read(p): return json.loads(p.read_text())
def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2))
def order(r):return hashlib.sha256(r['subject_id'].encode()).digest()
def dense(r):return int(len(r['events'])>=8)
def simulate(ctx,r,B,b,noise,trace):
    raw=r['_raw']
    return _execute(r['_public'],B,b,ctx,lambda t,ids:raw[t-1,list(ids)],noise,trace)

def noise_for(r,B,R):
    rng=np.random.default_rng(key_seed(SEED,'physionet2020',r['subject_id'],B))
    return rng.laplace(size=(41,R,2))

def main():
    OUT.mkdir(exist_ok=True)
    split=read(DIR/'split.json')
    records=sorted(data.load(DIR,['BUN_CR']),key=order)
    train,val,test=[[r for r in records if split[r['subject_id']]==role] for role in ('train','validation','test')]
    ctx=Context('BUN_CR',train[:48],SEED)
    for r in records:
        x=data.replay(r)
        r['_raw']=ctx.work.normalize(x)
        r['_truth']=ctx.work.evaluate(x)['ratio']
        r['_public']=public_view(r)
    old=read(OLD/'frozen.json')['BUN_CR']
    grid=sorted(map(float,old))
    dump('protocol.json',dict(dataset=data.input_hashes(DIR,['BUN_CR']),primary='source B paired MAE vs same-calibration strong periodic and static on train-median upper half of raw task ranges',anchors=BUDGETS,grid=grid,
        references=[r['subject_id'] for r in train[:48]],validation='first 120 hash-ranked A validation patients per <8 or >=8 batches',test=len(test),repeat=4,attack_patients=80,threshold=float(np.median([np.ptp(r['_truth']) for r in train]))))
    plans={}; logs=[]
    for B in BUDGETS:
        for g in (0,1):
            panel=[r for r in val if dense(r)==g][:120]
            noises=[noise_for(r,B,4) for r in panel]
            @lru_cache(None)
            def score(b):
                means=[]
                for r,n in zip(panel,noises):
                    sim=simulate(ctx,r,B,b,n,False)
                    y=ctx.work.from_cache(sim['caches'])['ratio']
                    means.append(np.abs(y-r['_truth']).mean())
                return np.array(means)
            def pick(stage,bs):
                for b in bs:logs.append(dict(B=B,density=g,stage=stage,candidate=json.dumps(asdict(b)),mae=score(b).mean(),n=len(panel)))
                return min(bs,key=lambda b:score(b).mean())
            periodic=pick('strong periodic',[replace(STATIC,anchor=.5,period=p) for p in (2,3,4,5,6,8,10,12)])
            count=pick('count',[replace(STATIC,anchor=.5,period=1,refresh_count=k) for k in (1,2,3,4)])
            anchor=pick('time budget',[replace(count,anchor=a) for a in (.25,.5,.7)])
            win=pick('root package and weights',[anchor,replace(anchor,refresh_weights='equal'),replace(anchor,refresh_weights='rms'),replace(anchor,fixed_roots=(0,)),replace(anchor,fixed_roots=(1,))])
            if FUSION:
                proposals=[win]
                for k in (1,2,3):
                    for a in (.4,.7):
                        proposals.append(replace(win,refresh_count=k,anchor=a,fusion=True))
                win=pick('fusion and refresh frequency',proposals)
            gain=score(STATIC)-score(win)
            use=gain.mean()>gain.std(ddof=1)/np.sqrt(len(panel))
            plans[B,g]=dict(static=STATIC,periodic=periodic,count=count,time=anchor,dynamic=win,route=win if use else STATIC)
            if FUSION: plans[B,g]['matched_latest']=replace(win,fusion=False)
        print('freeze',B,flush=True)
    dump('frozen.json',{f'{B}/{g}':{k:asdict(b) for k,b in p.items()} for (B,g),p in plans.items()})
    write(OUT/'iterations.csv',logs)
    # Evaluate old reference and new plans on common, training-only calibration.
    attacks=[]; paths=[]; costs=[]
    with (OUT/'grid.csv').open('w') as handle:
        writer=csv.writer(handle);writer.writerow(['B','subject','method','mae'])
        for B in grid:
            near=min(BUDGETS,key=lambda a:abs(np.log(B/a)))
            for i,r in enumerate(test):
                p=plans[near,dense(r)].copy()
                p['oldroute']=Branch(**old[str(B)]['branches']['R-new'])
                if B not in BUDGETS:p={k:p[k] for k in ('static','periodic','route','oldroute')}
                else:p.update(native_static=Branch(1.),native_periodic=replace(p['periodic'],calibration='none',shrink=0.))
                n=noise_for(r,B,4)
                memo={}
                for m,b in p.items():
                    if b not in memo:
                        tic=perf_counter()
                        sim=simulate(ctx,r,B,b,n,B in BUDGETS)
                        ms=(perf_counter()-tic)*1000
                        y=ctx.work.from_cache(sim['caches'])['ratio']
                        e=np.abs(y-r['_truth'])
                        row=dict(mae=e.mean(),p95=np.quantile(e,.95,axis=1).mean(),truth_abs=np.abs(r['_truth']).mean(),roots=sim['roots'].mean(),refresh=sim['refreshes'].mean(),spent=sim['spent'].mean(),batch4_ms=ms)
                        assert np.all(sim['spent']<=B+1e-8)
                        privacy={}
                        if B in BUDGETS:
                            ledger(sim,B,b)
                            if i<80 and b.calibration=='marginal':
                                privacy={k:v.mean() for k,v in cumulative(ctx,r,sim).items()}
                                for z in attack_result(ctx,r,sim,b):
                                    tag=z['attack']+'_'+z['prior']
                                    for k in ('success_005','normalized_mae'):
                                        key=tag+'_'+k
                                        privacy[key]=privacy.get(key,0)+z[k]/4
                                    for scope,num,den in [('all','root_mae','root_truth_abs'),('initial','initial_root_mae','initial_root_truth_abs'),('release','root_release_mae','root_release_truth_abs')]:
                                        for j in (0,1):
                                            for typ,col in [('num',num),('den',den)]:
                                                key=f'{tag}_{scope}_{j}_{typ}'
                                                privacy[key]=privacy.get(key,0)+z[col][j]/4
                        memo[b]=row,privacy
                    row,privacy=memo[b]
                    sid=r['subject_id']; writer.writerow([B,sid,m,row['mae']])
                    if B in BUDGETS:
                        meta=dict(B=B,subject=sid,method=m)
                        paths.append(dict(**meta,**row,span=np.ptp(r['_truth']),batches=len(r['events'])))
                        if privacy:attacks.append(dict(**meta,**privacy))
            handle.flush()
            if B in BUDGETS:print('test',B,flush=True)
    write(OUT/'paths.csv',paths);write(OUT/'attacks.csv',attacks)

if __name__=='__main__':main()
