"""Causal postprocessing of fixed, full-root periodic DP releases."""
import numpy as np
from ..m6.kernels import grid_logZ

def likelihood(bank,z,eps):
    logp=(-eps*abs(bank[None]-z[:,None])/2-grid_logZ(bank,eps,0,1,1000)).sum(-1)
    return np.exp(logp-logp.max(axis=1,keepdims=True))

def weights(bank,z,eps,prior):
    w=prior*likelihood(bank,z,eps)
    return w/w.sum(axis=1,keepdims=True)

def median_index(values,w):
    indices=np.argsort(values,axis=1)
    ordered=np.take_along_axis(w,indices,axis=1)
    idx=np.argmax(np.cumsum(ordered,axis=1)>=.5,axis=1)
    return indices[np.arange(len(values)),idx]

def empirical(work,bank,raw,eps,prior):
    w=weights(bank,raw,eps,prior)
    points=.85*bank[None]+.15*raw[:,None]
    mean=(w[:,:,None]*points).sum(axis=1)
    rows=np.arange(len(raw))
    root_med=np.stack([points[rows,median_index(points[:,:,i],w),i] for i in range(raw.shape[1])],axis=1)
    tasks=work.from_cache(points)[work.target]
    idx=median_index(tasks,w)
    target=tasks[rows,idx]
    # Pick a nearby joint root state on the same side of the posterior task median;
    # bisect the continuous AST along this segment to attain that median.
    y=work.from_cache(root_med)[work.target]
    up=y<target
    admissible=np.where(up[:,None],tasks>=target[:,None],tasks<=target[:,None])
    dist=(((points-root_med[:,None])/np.maximum(.001,bank.std(axis=0)))**2).sum(-1)
    end=points[rows,np.argmin(np.where(admissible,dist,np.inf),axis=1)]
    lo=np.zeros(len(y)); hi=np.ones(len(y))
    for _ in range(14):
        mid=(lo+hi)/2
        z=root_med+mid[:,None]*(end-root_med)
        below=work.from_cache(z)[work.target]<target
        move=np.where(up,below,~below)
        lo=np.where(move,mid,lo);hi=np.where(move,hi,mid)
    task=np.where((y==target)[:,None],root_med,root_med+hi[:,None]*(end-root_med))
    return w,mean,root_med,task

def streams(ctx,result,memory,full=True):
    work=ctx.work
    records=ctx.bank
    anchor=result['anchor_service']; eps=result['anchor_epsilon'][0]
    raw=result['anchor_cache']
    names=['task'] if not full else ['independent','joint','emp_mean','emp_median','task','no_history','half_refs','shuffled']
    output={k:np.repeat(anchor[:,None],result['caches'].shape[1],axis=1) for k in names}
    banks={'task':records}
    if full:
        banks['no_history']=records
        banks['half_refs']=records[::2]
        banks['shuffled']=np.stack([np.roll(records[:,:,j],j*max(1,len(records)//records.shape[-1]),axis=0)
                                    for j in range(records.shape[-1])],axis=-1)
    probs={k:weights(bank[:,0],raw,eps,1.) for k,bank in banks.items()}
    eye=np.eye(raw.shape[1])
    loc={k:anchor.copy() for k in ('independent','joint')}
    cov={k:np.tile(np.diag(np.minimum(.25,8/eps**2)),(len(raw),1,1)) for k in loc}
    last=0
    for j,h in enumerate(result['histories']):
        turn=h['turn']-1
        end=result['histories'][j+1]['turn']-1 if j+1<len(result['histories']) else result['caches'].shape[1]
        assert h['selected'].all()
        obs=h['service'];raw=h['cache_after'];eps=h['actual_epsilon'][0]
        if full:
            changes=records[:,turn]-records[:,last]
            q=changes.T@changes/len(records)
            q=.85*q+.15*np.diag(np.diag(q))
            R=np.diag(np.minimum(.25,8/eps**2))
            for k in loc:
                P=cov[k]+(np.diag(np.diag(q)) if k=='independent' else q)
                residual=obs-loc[k]
                score=np.einsum('ri,rij,rj->r',residual,np.linalg.inv(P+R),residual)/raw.shape[1]
                P=P*np.clip(score,1,4)[:,None,None]
                gain=P@np.linalg.inv(P+R)
                loc[k]=np.clip(loc[k]+np.einsum('rij,rj->ri',gain,residual),0,1)
                cov[k]=(eye-gain)@P@(eye-gain).transpose(0,2,1)+gain@R@gain.transpose(0,2,1)
                output[k][:,turn:end]=loc[k][:,None]
        for k,bank in banks.items():
            retain=0. if k=='no_history' else memory
            prior=retain*probs[k]+(1-retain)/len(bank)
            probs[k],mean,median,task=empirical(work,bank[:,turn],raw,eps,prior)
            output[k][:,turn:end]=task[:,None]
            if k=='task' and full:
                output['emp_mean'][:,turn:end]=mean[:,None]
                output['emp_median'][:,turn:end]=median[:,None]
        last=turn
    return output
