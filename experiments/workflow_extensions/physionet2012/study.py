"""AutoResearch: locked Challenge 2012 external replay and local anchor ablations."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize
from data_mine.build_physionet2012 import DEST as DATA
from ..state_estimation.window_study import references,library,estimates,windows
from ..state_estimation.calibrated_window import DEST as OLD,predict
from ..m8.study import Context,prep,order,SEED,STATIC,Branch,BUDGETS,data,sim,ledger,write,dump
from ..finite_cache.workload import MONOMIALS

OUT=Path(__file__).resolve().parent.parent/'outputs/physionet2012'
PERIOD=Branch(.5,period=2,calibration='marginal',shrink=.85)
VARIANTS=('task','root_loss','own_only','snapshot')
STRENGTH=(0.,.5,1.)
def segments(ctx,ps):
    anchor=ps['anchor_service']
    win=list(windows(ctx,ps))
    first=win[0][0] if win else ps['caches'].shape[1]
    yield 0,first,anchor,anchor,ps['anchor_cache']
    for h,(st,en,x,_) in zip(ps['histories'],win):
        yield st,en,x,h['service'],h['cache_after']

def inputs(ctx,seg,anchor,median,snap):
    st,en,x,fresh,raw=seg
    if snap:x=fresh
    log=lambda z:np.log(ctx.work.denormalize(z))
    base=log(x)
    history=np.zeros_like(base) if snap else log(anchor)-base
    f=np.column_stack([np.ones(len(x)),np.clip(base-median,-2,2),np.clip(log(raw)-base,-2,2),np.clip(history,-2,2)])
    return base,f

def fit_corrections(ctx,recs,B,plan,median):
    samples=defaultdict(list)
    for r in recs:
        ps=sim(ctx,r,B,plan,8,True)
        for seg in segments(ctx,ps):
            st,en=seg[:2];R=len(seg[2])
            turns=st+np.arange(R)*(en-st)//R
            for snap in (False,True):
                base,f=inputs(ctx,seg,ps['anchor_service'],median,snap)
                samples[bool(st),snap].append((base,f,data.replay(r)[turns],np.full(R,en-st)))
    models={}
    p=np.array(MONOMIALS[ctx.work.template][ctx.work.target][1])
    lo=np.log(ctx.work.lo);hi=np.log(ctx.work.hi)
    for (phase,snap),sample in samples.items():
        base,f,truth,weight=[np.concatenate(z) for z in zip(*sample)]
        weight=weight/weight.sum()
        yt=ctx.work.evaluate(truth)[ctx.work.target]
        for kind in (('snapshot',) if snap else ('task','root_loss','own_only')):
            n,d=base.shape;shape=(f.shape[1],d)
            mask=np.ones(shape)
            if kind=='own_only':mask[1:]=np.tile(np.eye(d),(3,1))
            scale=np.maximum(np.median(truth,axis=0),.001)
            def evaluate(v):
                c=v.reshape(shape)*mask
                raw=base+f@c;l=np.clip(raw,lo,hi)
                if kind=='root_loss':
                    pred=np.exp(l);diff=(pred-truth)/scale
                    loss=np.mean(abs(diff),axis=1)
                    dl=np.sign(diff)*pred/scale/d
                else:
                    pred=np.exp(l@p);diff=pred-yt
                    loss=abs(diff)
                    dl=(np.sign(diff)*pred)[:,None]*p
                grad=f.T@((weight[:,None]*dl)*((raw>=lo)&(raw<=hi)))
                return weight@loss,grad*mask
            zero=np.zeros(np.prod(shape));unit=max(evaluate(zero)[0],1e-8)
            def obj(v):
                val,g=evaluate(v)
                return val/unit+.06*(v@v),g.ravel()/unit+.12*v
            res=minimize(obj,zero,jac=True,method='L-BFGS-B',bounds=[(-.6,.6)]*len(zero),options={'maxiter':90})
            models[phase,kind]=(res.x.reshape(shape)*mask)
    return models

def serve(ctx,ps,model,median,kind,strength,parts,mode='both'):
    out=ps['caches'].copy()
    for seg in parts:
        st,en=seg[:2];base,f=inputs(ctx,seg,ps['anchor_service'],median,kind=='snapshot')
        gain=strength if mode=='both' or (mode=='anchor')==(st==0) else 0.
        out[:,st:en]=ctx.work.normalize(np.exp(base+gain*f@model[bool(st),kind]))[:,None]
    return out

def run(t):
    large=t=='BUN_CR';refs=references(t,large)
    ctx=Context(t,refs,SEED);prep(ctx,refs)
    recs=sorted(data.load(DATA,[t]),key=order)
    split=json.loads((DATA/'split.json').read_text())
    tr,va,te=[[r for r in recs if split[r['subject_id']]==k] for k in ('train','validation','test')]
    prep(ctx,tr+va+te)
    train=tr[:192];dev=va[:160]
    median=np.median(np.log(ctx.work.denormalize(np.stack([r['_x'] for r in train]))),axis=(0,1))
    cut=np.median([r['_span'] for r in tr])
    change=np.concatenate([abs(np.diff(r['_y'])) for r in tr]);jump=np.median(change[change>0])
    old=json.loads((OLD/(('PhysioNet' if large else 'FIB4')+'_coefficients.json')).read_text())
    banks={B:library(ctx,refs,t,large,B) for B in BUDGETS}
    def prev(ps,B):
        c={k:np.asarray(v) for k,v in old[str(B)].items()}
        return predict(ctx,ps,c,1.)['task_window'],estimates(ctx,ps,banks[B],[(128,.5)])[('task_window',128,.5)]
    # Freeze external baselines and measure their validation performance before fitting.
    frozen=[]
    for r in dev:
        for B in BUDGETS:
            ps=sim(ctx,r,B,PERIOD,4,True)
            for m,c in zip(('continuous','analog'),prev(ps,B)):
                frozen.append(dict(subject=r['subject_id'],B=B,method=m,mae=abs(ctx.work.from_cache(c)[ctx.work.target]-r['_y']).mean()))
    write(OUT/(t+'_old_validation.csv'),frozen)
    model={};scores=defaultdict(list)
    for B in BUDGETS:
        for label,b in [('period',PERIOD),('static',STATIC)]:
            mod=fit_corrections(ctx,train,B,b,median);model[B,label]=mod
            for r in dev:
                ps=sim(ctx,r,B,b,4,True);parts=list(segments(ctx,ps))
                for s in STRENGTH:
                    c=serve(ctx,ps,mod,median,'task',s,parts)
                    scores[label,s].append(abs(ctx.work.from_cache(c)[ctx.work.target]-r['_y']).mean())
    best={l:min(STRENGTH,key=lambda s:np.mean(scores[l,s])) for l in ('period','static')}
    dump(OUT/(t+'_frozen.json'),dict(strength=best,median=median.tolist(),span=float(cut),jump=float(jump),train=[r['subject_id'] for r in train],validation=[r['subject_id'] for r in dev],models={f'{B}/{l}':{f'{int(ph)}/{k}':v.tolist() for (ph,k),v in mod.items()} for (B,l),mod in model.items()}))
    write(OUT/(t+'_iterations.csv'),[dict(policy=l,strength=s,mae=np.mean(v)) for (l,s),v in scores.items()])
    rows=[]
    for r in te:
        y=r['_y'];events=np.flatnonzero(abs(np.diff(y))>=jump)+1
        turns=np.unique(np.minimum(len(y)-1,np.r_[events,events+1]))
        for B in BUDGETS:
            ps=sim(ctx,r,B,PERIOD,4,True);ledger(ps,B,PERIOD)
            ss=sim(ctx,r,B,STATIC,4,True);raw=sim(ctx,r,B,Branch(1.),4,True)
            parts=list(segments(ctx,ps));strength=best['period']
            oldc,olda=prev(ps,B)
            caches={'raw_static':(raw['caches'],raw),'cal_static':(ss['caches'],ss),'new_static':(serve(ctx,ss,model[B,'static'],median,'task',best['static'],list(segments(ctx,ss))),ss)}
            dyn={'old_continuous':oldc,'old_analog':olda,'overwrite':ps['caches']}
            for kind in VARIANTS:dyn[kind]=serve(ctx,ps,model[B,'period'],median,kind,strength,parts)
            for mode in ('anchor','refresh'):dyn[mode]=serve(ctx,ps,model[B,'period'],median,'task',strength,parts,mode)
            dyn['joint']=serve(ctx,ps,model[B,'period'],median,'task',0,parts)
            caches.update({k:(c,ps) for k,c in dyn.items()})
            for k,(c,run) in caches.items():
                pred=ctx.work.from_cache(c)[ctx.work.target];err=abs(pred-y)
                rows.append(dict(subject=r['subject_id'],B=B,method=k,high=int(r['_span']>cut),dense=int(len(r['events'])>=4),mae=err.mean(),den=abs(y).mean(),response=err[:,turns].mean() if len(turns) else None,p95=np.quantile(err,.95,axis=1).mean(),refreshes=run['refreshes'].mean(),spent=run['spent'].mean()))
    write(OUT/(t+'.csv'),rows)
    print(t,len(te),best,flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True)
    dump(OUT/'protocol.json',dict(rule='set A fit/validation only, set B paired replay. Each template independently selects strength on four-budget average validation MAE; no test-based selection.',primary='full-path patient-equal MAE vs joint; target 5% and positive across both ASTs; compare with matched root loss and restricted features.',plans='same original frozen prior, half anchor and refresh every second real batch (fallback last), m-opt allocations; static all at anchor. Calendar mapped to 40 rounds.',variants=list(VARIANTS),strength=STRENGTH))
    for task in ('BUN_CR','FIB4'):run(task)
