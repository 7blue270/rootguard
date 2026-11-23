"""Frozen-transcript analog forecasting for the entire cache reuse window."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.spatial import cKDTree
from .study import load, branches, sim, ledger, utility, write, dump, OUT, BUDGETS, HERE, DEFAULT_SPLIT, data, order, prep

DEST=OUT.parent/'window_v2'
COHORTS=[('PhysioNet','BUN_CR'),('MIMIC_BUN','BUN_CR'),('MELD_NA','MELD_NA'),('ANEMIA','ANEMIA'),('FIB4','FIB4')]
OPTIONS=[(k,a) for k in (32,128) for a in (.5,1.)]

def references(template,large):
    path=Path('data_mine/finite_cache_physionet2019' if large else 'data_mine/finite_cache')
    records=sorted(data.load(path,[template]),key=order)
    if large:
        ids=json.loads((HERE/'original_protocol.json').read_text())['references']
        by={r['subject_id']:r for r in records}
        return [by[i] for i in ids]
    split=json.loads(DEFAULT_SPLIT.read_text())
    return [r for r in records if split.get(r['subject_id'])=='train']

def windows(ctx,ps):
    # Mirror frozen v1 joint filter. Return only features available at refresh.
    x=ps['anchor_service'].copy()
    dim=x.shape[1]
    cov=np.tile(np.diag(np.minimum(.25,8/ps['anchor_epsilon'][0]**2)),(len(x),1,1))
    last=0
    history=ps['histories']
    for j,h in enumerate(history):
        start=h['turn']-1
        end=history[j+1]['turn']-1 if j+1<len(history) else ps['caches'].shape[1]
        assert h['selected'].all()
        delta=ctx.bank[:,start]-ctx.bank[:,last]
        q=delta.T@delta/len(delta)
        p=cov+.85*q+.15*np.diag(np.diag(q))
        R=np.diag(np.minimum(.25,8/h['actual_epsilon'][0]**2))
        innovation=h['service']-x
        score=np.einsum('ri,rij,rj->r',innovation,np.linalg.inv(p+R),innovation)/dim
        p=p*np.clip(score,1,4)[:,None,None]
        gain=p@np.linalg.inv(p+R)
        x=np.clip(x+np.einsum('rij,rj->ri',gain,innovation),0,1)
        rem=np.eye(dim)-gain
        cov=rem@p@rem.transpose(0,2,1)+gain@R@gain.transpose(0,2,1)
        features=np.column_stack([x,ps['anchor_cache'],h['cache_after'],np.full(len(x),start),np.full(len(x),end-start)])
        yield start,end,x,features
        last=start

def library(ctx,records,t,large,B):
    features=[];current=[];future=[]
    for r in records:
        plan,_=branches(r,t,B,large)
        ps=sim(ctx,r,B,plan,48,True)
        for start,end,_,f in windows(ctx,ps):
            features.extend(f)
            current.extend(np.repeat(r['_x'][None,start],len(f),axis=0))
            # Stratified uniform time in each training window across noise replicates.
            positions=start+np.arange(len(f))*(end-start)//len(f)
            future.extend(r['_x'][positions])
    feat=np.asarray(features)
    scale=np.maximum(np.std(feat,axis=0),1e-5)
    scale[-2:]*=3
    return cKDTree(feat/scale),scale,np.array(current),np.array(future)

def readouts(ctx,points):
    center=np.median(points,axis=1)
    y=ctx.work.from_cache(points)[ctx.work.target]
    # Upper sample median is an absolute-loss minimizer.
    idx=np.argsort(y,axis=1)[:,y.shape[1]//2]
    target=y[np.arange(len(points)),idx]
    side=ctx.work.from_cache(center)[ctx.work.target]<target
    valid=np.where(side[:,None],y>=target[:,None],y<=target[:,None])
    dist=(((points-center[:,None])/np.maximum(.001,ctx.bank.std(axis=(0,1))))**2).sum(axis=-1)
    end=points[np.arange(len(points)),np.argmin(np.where(valid,dist,np.inf),axis=1)]
    lo=np.zeros(len(points));hi=np.ones(len(points))
    for _ in range(16):
        mid=(lo+hi)/2
        val=ctx.work.from_cache(center+(end-center)*mid[:,None])[ctx.work.target]
        advance=np.where(side,val<target,val>target)
        lo=np.where(advance,mid,lo);hi=np.where(advance,hi,mid)
    return center,center+(end-center)*hi[:,None]

def estimates(ctx,ps,model,opts):
    tree,scale,current,future=model
    def initial():return np.broadcast_to(ps['anchor_service'][:,None],ps['caches'].shape).copy()
    caches={'joint':initial()}
    for k,a in opts:
        for m in ('root_window','task_now','task_window'):
            caches[m,k,a]=initial()
    for start,end,joint,f in windows(ctx,ps):
        caches['joint'][:,start:end]=joint[:,None]
        for k in sorted({pair[0] for pair in opts}):
            idx=tree.query(f/scale,k=min(k,tree.n))[1]
            roots,forecast=readouts(ctx,future[idx])
            _,instant=readouts(ctx,current[idx])
            for nb,a in opts:
                if nb!=k:continue
                for m,v in [('root_window',roots),('task_now',instant),('task_window',forecast)]:
                    caches[m,k,a][:,start:end]=((1-a)*joint+a*v)[:,None]
    return caches

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    dump(DEST/'protocol.json',dict(primary='patient-equal all-turn MAE, four budget average, target 5%, positive in >=2 ASTs; historic test only',
        training='same fixed references as joint, 48 paired-noise simulations per patient/budget; fixed anchor; no extra private observations',
        candidates=OPTIONS,select='minimum source validation task_window MAE on first 80; lock before all evaluations',
        checks='remaining source validation separate from selection, historic test, four MIMIC transfers; same readout vs root-window / task-now'))
    for label,t in COHORTS:
        large=label=='PhysioNet'
        ctx,val,test,span,jump=load(t,large)
        refs=references(t,large);prep(ctx,refs)
        models={B:library(ctx,refs,t,large,B) for B in BUDGETS}
        if large:
            risks={o:[] for o in OPTIONS}
            for r in val[:80]:
                for B in BUDGETS:
                    b,_=branches(r,t,B,large)
                    pred=estimates(ctx,sim(ctx,r,B,b,4,True),models[B],OPTIONS)
                    for o in OPTIONS:
                        y=ctx.work.from_cache(pred[('task_window',*o)])[ctx.work.target]
                        risks[o].append(abs(y-r['_y']).mean())
            write(DEST/'iterations.csv',[dict(neighbors=k,blend=a,mae=np.mean(v)) for (k,a),v in risks.items()])
            chosen=min(risks,key=lambda o:np.mean(risks[o]))
            dump(DEST/'frozen.json',dict(neighbors=chosen[0],blend=chosen[1]))
        for dataset,subset in ([(label+'_A_check',val[80:])] if large else [])+[(label,test)]:
            rows=[]
            for r in subset:
                for B in BUDGETS:
                    b,_=branches(r,t,B,large)
                    ps=sim(ctx,r,B,b,4 if large else 20,True);ledger(ps,B,b)
                    caches=estimates(ctx,ps,models[B],[chosen])
                    for key,c in caches.items():
                        u=utility(ctx,r,{**ps,'caches':c})
                        y=ctx.work.from_cache(c)[ctx.work.target]
                        events=np.flatnonzero(abs(np.diff(r['_y']))>=jump)+1
                        turns=np.unique(np.minimum(len(r['_y'])-1,np.r_[events,events+1]))
                        row=dict(subject=r['subject_id'],B=B,method=key if isinstance(key,str) else key[0],high=int(r['_span']>span),
                                 response=abs(y[:,turns]-r['_y'][turns]).mean() if len(turns) else None,
                                 jump_error=abs(np.diff(y)[:,events-1]-np.diff(r['_y'])[events-1]).mean() if len(events) else None)
                        row.update({k:v.mean() for k,v in u.items() if k not in ('roots','spent','refreshes')})
                        rows.append(row)
            write(DEST/(dataset+'.csv'),rows)
            print(dataset,len(subset),len(refs),flush=True)
if __name__=='__main__':main()
