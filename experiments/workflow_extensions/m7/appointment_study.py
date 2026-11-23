"""One paid appointment at the nearest public measurement, with train-selected roots."""
import argparse
from dataclasses import asdict, replace
from itertools import combinations, product
from pathlib import Path
from time import process_time, perf_counter
import csv
import json
import numpy as np

from .refresh_study import (setup, read, ledger, harm, REF, ROOT, BUDGETS, SEED,
                            Context, execute, utility, route_choice, dump)
from .engine import Branch, public_view
from .metrics import attack_result
from .analysis import combine_attack
from ..finite_cache import data

OLD=REF.parent.parent/'m7/refresh_iso_20261119'
OUT=OLD.parent/'appointment_20261119'
STATIC=Branch(1.,calibration='marginal',shrink=.85)
DAT=ROOT/'data_mine/finite_cache'

def write(path, rows):
    keys=dict.fromkeys(k for row in rows for k in row)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(keys))
        w.writeheader()
        w.writerows(rows)

def risk(ctx,records,B,b,phase):
    vals=[utility(ctx,r,execute(r,B,b,ctx,phase,8)) for r in records]
    return {k:np.array([x[k].mean() for x in vals]) for k in ('mae','query_objective')}

def develop():
    OUT.mkdir(exist_ok=True)
    parts,hashes,grid=setup(DAT)
    plans={}
    logs=[]
    times=[]
    for t,p in parts.items():
        start=process_time()
        ctx=Context(t,p['train'],SEED)
        plans[t]={}
        for B in BUDGETS:
            base=risk(ctx,p['train'],B,STATIC,'appointment-train')
            def score(b,stage):
                r=risk(ctx,p['train'],B,b,'appointment-train')
                v=sum(w*r[k].mean()/max(base[k].mean(),1e-12) for k,w in [('mae',.8),('query_objective',.2)])
                logs.append(dict(template=t,B=B,stage=stage,score=v,branch=json.dumps(asdict(b))))
                return v
            arms=[replace(STATIC,period=1,appointment=turn,anchor=a,fixed_roots=ctx.roots)
                  for turn,a in product((10,20,30),(.35,.65,.8))]
            full=min(arms,key=lambda b:score(b,'time'))
            masks=[m for k in range(1,len(ctx.roots)+1) for m in combinations(ctx.roots,k)]
            pack=min((replace(full,fixed_roots=m) for m in masks),key=lambda b:score(b,'package'))
            arms=[pack]+[replace(pack,refresh_weights=w) for w in ('equal','rms') if len(pack.fixed_roots)>1]
            win=min(arms,key=lambda b:score(b,'allocation'))
            sv=risk(ctx,p['validation'],B,STATIC,'m7-refresh-validation')
            dv=risk(ctx,p['validation'],B,win,'m7-refresh-validation')
            plans[t][B]=dict(full=asdict(full),pack=asdict(pack),win=asdict(win),
                              dynamic=route_choice(sv,dv),gain=float((sv['mae']-dv['mae']).mean()))
        times.append(dict(template=t,planning_cpu_s=process_time()-start,candidates=sum(r['template']==t for r in logs)))
        print(t,times[-1],flush=True)
    dump(OUT/'frozen.json',plans)
    dump(OUT/'protocol.json',dict(input_hashes=hashes,grid=grid,targets=str(REF/'targets.json'),
         primary='four-anchor paired routed MAE vs old routed M7 in MELD_NA and BUN_CR, exact paired sign-flip Holm',
         secondary='train-median whole-trajectory final-score range split, only for offline stratification',
         assignment='nearest log B anchor, train rank .8 relative MAE + .2 relative workload, validation half-SE vs static',
         selection_noise=8,test_noise=20,targets_fixed=True))
    write(OUT/'iterations.csv',logs)
    write(OUT/'planning.csv',times)

def cumulative(ctx,rec,res):
    # Use only fresh observer-visible marginal outputs. Precision and the training
    # process variance are public. Each observer filters a prefix causally.
    from .metrics import invert_marginal
    truth=ctx.work.normalize(data.replay(rec))
    repeats,_,k=res['caches'].shape
    selected=np.ones((repeats,k),bool)
    eps=res['anchor_epsilon']
    initial=invert_marginal(ctx,res['anchor_service'],np.zeros_like(eps,int),eps,STATIC,selected)
    mu=ctx.bank[:,0].mean(axis=0)
    var0=np.maximum(ctx.bank[:,0].var(axis=0),1e-8)
    variance=1/(1/var0+eps**2/8)
    kal=variance*(mu/var0+initial*eps**2/8)
    avg=initial.copy()
    cp=avg.copy()
    count=np.ones_like(avg)
    ncp=count.copy()
    cpvar=8/eps**2
    prev=res['anchor_service']
    single=initial.copy()
    single_kal=kal.copy()
    marginal=var0.copy()
    changes={h['turn']-1:h for h in res['histories']}
    tracks={n:[] for n in ('mean','kalman','changepoint')}
    released=np.zeros_like(res['caches'],bool)
    released[:,0]=True
    increments={n:[] for n in tracks}
    for turn in range(len(truth)):
        if turn:
            process=np.mean((ctx.bank[:,turn]-ctx.bank[:,turn-1])**2,axis=0)
            variance+=process
            marginal+=process
        if turn in changes:
            h=changes[turn]
            m=h['selected']
            obs=h['service']
            if h['fusion_gain'] is not None:
                obs=prev+(obs-prev)/h['fusion_gain']
            obs=invert_marginal(ctx,obs,h['last'],h['last_eps'],STATIC,m)
            prev=h['service']
            released[:,turn]=m
            v=8/h['last_eps']**2
            single=np.where(m,obs,single)
            single_kal=np.where(m,mu+(obs-mu)*marginal/(marginal+v),single_kal)
            gain=variance/(variance+v)
            kal=np.where(m,kal+gain*(obs-kal),kal)
            variance=np.where(m,(1-gain)*variance,variance)
            count+=m
            avg+=m*(obs-avg)/count
            jump=m&(abs(obs-cp)>2*np.sqrt(cpvar+v))
            ncp=np.where(jump,1,ncp+m)
            cp=np.where(jump,obs,cp+m*(obs-cp)/ncp)
            cpvar=np.where(m,np.where(jump,v,((ncp-1)**2*cpvar+v)/ncp**2),cpvar)
        for n,x in [('mean',avg),('kalman',kal),('changepoint',cp)]:
            tracks[n].append(x.copy())
            single_estimate=single_kal if n=='kalman' else single
            diff=(abs(x-truth[turn])<=.05).astype(float)-(abs(single_estimate-truth[turn])<=.05)
            increments[n].append(diff)
    rows={}
    for n,v in tracks.items():
        err=abs(np.stack(v,axis=1)-truth)
        for scope,mask in [('all',np.ones_like(released)),('fresh',released)]:
            size=mask.sum(axis=(1,2))
            rows[n+'_'+scope+'_mae']=(err*mask).sum(axis=(1,2))/size
            rows[n+'_'+scope+'_hit']=100*((err<=.05)*mask).sum(axis=(1,2))/size
        rows[n+'_history_minus_latest_hit']=100*np.stack(increments[n],axis=1).mean(axis=(1,2))
        rows[n+'_p95']=np.quantile(err,.95,axis=(1,2))
    return rows

def confirm(out=OUT, fallback=False):
    parts,_,grid=setup(DAT)
    plans=read(out/'frozen.json')
    old=read(OLD/'frozen.json')
    rows=[]
    strata=[]
    clock=[]
    for t,p in parts.items():
        ctx=Context(t,p['train'],SEED)
        ranges=np.ptp(ctx.bank_tasks[:,:,ctx.target],axis=1)
        threshold=np.median(ranges)
        for rec in p['test']:
            x=ctx.work.evaluate(data.replay(rec))[ctx.work.target]
            strata.append(dict(template=t,subject=str(rec['subject_id']),score_range=np.ptp(x),train_cut=threshold,high=int(np.ptp(x)>threshold)))
        for B in grid:
            anchor=min(BUDGETS,key=lambda a:abs(np.log(a/B)))
            plan=plans[t][str(anchor)]
            win=Branch(**plan['win'])
            control=Branch(**old[t][str(B)]['branches']['R-new']) if fallback else STATIC
            arms={'scheduled':win,'route':win if plan['dynamic'] else control}
            if B in BUDGETS:
                arms.update(full=Branch(**plan['full']),pack=Branch(**plan['pack']),
                    static=STATIC,periodic=Branch(**old[t][str(B)]['branches']['P-cal']),
                    oldroute=Branch(**old[t][str(B)]['branches']['R-new']))
            for rec in p['test']:
                cache={}
                for arm,b in arms.items():
                    if b not in cache:
                        start=perf_counter()
                        res=execute(rec,B,b,ctx,'test',20,True)
                        elapsed=perf_counter()-start
                        ledger(res,B,b)
                        u=utility(ctx,rec,res)
                        if B in BUDGETS:
                            u.update(cumulative(ctx,rec,res))
                            u['harmful']=harm(ctx,rec,res)
                            epslist=[res['anchor_epsilon']]+[h['actual_epsilon'] for h in res['histories']]
                            eps=np.concatenate(epslist,axis=1)
                            u['eps_min']=np.where(eps>0,eps,np.inf).min(axis=1)
                            u['eps_max']=eps.max(axis=1)
                            u['eps_median']=np.nanmedian(np.where(eps>0,eps,np.nan),axis=1)
                            attacks=attack_result(ctx,rec,res,b)
                            per=[{} for _ in range(20)]
                            for z in attacks:
                                combine_attack(per[z['repeat']],[z],ctx.work.roots)
                            for key in per[0]: u[key]=np.array([v[key] for v in per])
                            clock.append(dict(template=t,B=B,method=arm,batch20_ms=1000*elapsed))
                        cache[b]=u
                    u=cache[b]
                    for rep in range(20):
                        row=dict(template=t,B=B,subject=str(rec['subject_id']),repeat=rep,method=arm)
                        row.update({k:float(v[rep]) for k,v in u.items() if B in BUDGETS or k=='mae'})
                        rows.append(row)
        print(t,'test complete',flush=True)
    write(out/'paths.csv',rows)
    write(out/'strata.csv',strata)
    write(out/'timing.csv',clock)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['develop','confirm'])
    args=parser.parse_args()
    (develop if args.stage=='develop' else confirm)()
