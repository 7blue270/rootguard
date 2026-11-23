"""Train bounded per-root corrections to the continuous joint estimate."""
import csv
import numpy as np
from .window_study import (DEST as ANALOG, OUT, COHORTS, references, load, windows, sim, branches,
                           ledger, prep, write, dump, BUDGETS, utility)

DEST=OUT.parent/'continuous_v3'
KINDS=('root_window','task_now','task_window')

def design(ctx,x,anchor,raw):
    # Log physical roots keep multiplicative corrections bounded on positive domains.
    base=np.log(ctx.work.denormalize(x))
    median=np.median(np.log(ctx.work.denormalize(ctx.bank)),axis=(0,1))
    features=np.stack([np.ones_like(x)*.2,np.clip(base-median,-2,2),
                       np.clip(np.log(ctx.work.denormalize(anchor))-base,-2,2),
                       np.clip(np.log(ctx.work.denormalize(raw))-base,-2,2)],axis=-1)
    return base,features

def corrected(ctx,base,f,coef):
    logx=base+np.einsum('...df,df->...d',f,coef)
    return ctx.work.normalize(np.exp(logx))

def fit(ctx,records,t,large,B):
    bases=[];features=[];current=[];future=[]
    for r in records:
        b,_=branches(r,t,B,large)
        ps=sim(ctx,r,B,b,24,True)
        for j,(st,en,x,_) in enumerate(windows(ctx,ps)):
            base,f=design(ctx,x,ps['anchor_service'],ps['histories'][j]['cache_after'])
            # Include every future turn; all training trajectories have the same length.
            for turn in range(st,en):
                bases.append(base);features.append(f)
                current.append(np.broadcast_to(r['_x'][st],x.shape))
                future.append(np.broadcast_to(r['_x'][turn],x.shape))
    base,f,now,later=map(np.concatenate,(bases,features,current,future))
    coef0=np.zeros(f.shape[1:])
    results={}
    for kind in KINDS:
        target=now if kind=='task_now' else later
        y=ctx.work.from_cache(target)[ctx.work.target]
        root_scale=np.maximum(.001,ctx.bank.std(axis=(0,1)))
        def error(coef):
            roots=corrected(ctx,base,f,coef)
            if kind=='root_window':return np.mean(abs(roots-target)/root_scale)
            return np.mean(abs(ctx.work.from_cache(roots)[ctx.work.target]-y))
        coef=coef0.copy();initial=max(1e-12,error(coef))
        def loss(c):return error(c)/initial+.04*np.sum(c**2)
        best=loss(coef)
        for step in (.3,.12):
            for i,k in np.ndindex(coef.shape):
                original=coef[i,k];winner=original
                for move in (-step,step):
                    coef[i,k]=original+move
                    candidate=loss(coef)
                    if candidate<best:
                        winner=coef[i,k];best=candidate
                coef[i,k]=winner
        results[kind]=coef
    return results

def predict(ctx,ps,models,shrink):
    caches={k:np.broadcast_to(ps['anchor_service'][:,None],ps['caches'].shape).copy() for k in ('joint',*KINDS)}
    for j,(st,en,x,_) in enumerate(windows(ctx,ps)):
        caches['joint'][:,st:en]=x[:,None]
        base,f=design(ctx,x,ps['anchor_service'],ps['histories'][j]['cache_after'])
        for k,coef in models.items():
            caches[k][:,st:en]=corrected(ctx,base,f,shrink*coef)[:,None]
    return caches

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    dump(DEST/'protocol.json',dict(hypothesis='continuous joint state, fit bounded log-root corrections using complete training windows, preserve high-precision support',
        decision='choose between analog and continuous by source first-80 validation only; freeze before opening either test score',
        shrink=[.5,1.],loss='relative in-sample loss + 0.04 sum of squared coefficients, one coordinate pass at each of two fixed step sizes',
        attribution='same correction family/training features, root-window vs terminal-window, terminal-now vs terminal-window'))
    for label,t in COHORTS:
        large=label=='PhysioNet'
        ctx,val,test,span,jump=load(t,large)
        refs=references(t,large);prep(ctx,refs)
        models={B:fit(ctx,refs,t,large,B) for B in BUDGETS}
        dump(DEST/(label+'_coefficients.json'),{str(B):{k:c.tolist() for k,c in model.items()} for B,model in models.items()})
        if large:
            risks={s:[] for s in (.5,1.)}
            for r in val[:80]:
                for B in BUDGETS:
                    b,_=branches(r,t,B,large);ps=sim(ctx,r,B,b,4,True)
                    for s in risks:
                        c=predict(ctx,ps,models[B],s)['task_window']
                        risks[s].append(abs(ctx.work.from_cache(c)[ctx.work.target]-r['_y']).mean())
            shrink=min(risks,key=lambda s:np.mean(risks[s]))
            write(DEST/'iterations.csv',[dict(shrink=s,mae=np.mean(v)) for s,v in risks.items()])
            with (ANALOG/'iterations.csv').open() as file:analog=min(float(r['mae']) for r in csv.DictReader(file))
            winner='continuous' if np.mean(risks[shrink])<analog else 'analog'
            dump(DEST/'frozen.json',dict(shrink=shrink,selected_family=winner,analog_val=analog,continuous_val=np.mean(risks[shrink])))
        for dataset,subset in ([(label+'_A_check',val[80:])] if large else [])+[(label,test)]:
            rows=[]
            for r in subset:
                events=np.flatnonzero(abs(np.diff(r['_y']))>=jump)+1
                turns=np.unique(np.minimum(len(r['_y'])-1,np.r_[events,events+1]))
                for B in BUDGETS:
                    b,_=branches(r,t,B,large)
                    ps=sim(ctx,r,B,b,4 if large else 20,True);ledger(ps,B,b)
                    for k,c in predict(ctx,ps,models[B],shrink).items():
                        u=utility(ctx,r,{**ps,'caches':c})
                        y=ctx.work.from_cache(c)[ctx.work.target]
                        row=dict(subject=r['subject_id'],B=B,method=k,high=int(r['_span']>span),
                                 response=abs(y[:,turns]-r['_y'][turns]).mean() if len(turns) else None,
                                 jump_error=abs(np.diff(y)[:,events-1]-np.diff(r['_y'])[events-1]).mean() if len(events) else None)
                        row.update({key:v.mean() for key,v in u.items() if key not in ('spent','roots','refreshes')})
                        rows.append(row)
            write(DEST/(dataset+'.csv'),rows)
            print(dataset,len(subset),len(refs),flush=True)
if __name__=='__main__':main()
