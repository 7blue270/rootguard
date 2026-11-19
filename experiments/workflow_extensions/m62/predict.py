"""Whole-session and legacy local ratios from identical public simulations."""
from functools import lru_cache
import time
import numpy as np
from experiments.workflow_extensions.m4_carry.core import key_seed,task
from .protocol import prior_bounds
from .engine import execute,schedule,release


@lru_cache(maxsize=12)
def inputs(seed,template,N,T,k):
    rng=np.random.default_rng(key_seed(seed,template,'m62-public-independent-of-actual'))
    uniform=rng.uniform(size=(N,k));shared=rng.choice([-1.,1.],(N,T,1));individual=rng.choice([-1.,1.],(N,T,k))
    noise=rng.laplace(size=(T+1,N,k))
    return uniform,shared,individual,noise


def generate(w,calendars,cfg,level,delta,component,model):
    N=cfg['n_sim'];T=cfg['T'];k=len(w.roots)
    u,shared,individual,noise=inputs(cfg['public_seed'],w.template,N,T,k)
    lo,hi=prior_bounds(w,level);state=lo+u*(hi-lo)
    idx=np.arange(N)%len(calendars);mask=np.zeros((len(calendars),T,k));qw=np.zeros((len(calendars),T))
    for j,c in enumerate(calendars):
        for event in c['events']:mask[j,event['turn']-1,event['roots']]=1
        for turn in c['query_turns']:qw[j,turn-1]+=1
    paths=np.empty((N,T,k));paths[:,0]=state
    for t in range(1,T):
        clock=0 if model=='linear' else t
        step=delta*(component*shared[:,clock]+(1-component)*individual[:,clock])
        state=np.clip(state+step*mask[idx,t],0,1);paths[:,t]=state
    d=np.diff(paths,axis=1).reshape(-1,k);cov=np.cov(d,rowvar=False);sd=np.sqrt(np.diag(cov))
    corr=np.divide(cov,sd[:,None]*sd[None,:],out=np.full_like(cov,np.nan),where=sd[:,None]*sd[None,:]>1e-20)
    off=corr[np.triu_indices(k,1)];valid=off[np.isfinite(off)]
    return paths,qw[idx],noise,corr,float(valid.mean()) if len(valid) else None


def predict_grid(w,calendars,cfg,level,delta,component,model,budgets=None,keep_losses=False):
    started=time.perf_counter();paths,qw,noise,corr,mean_corr=generate(w,calendars,cfg,level,delta,component,model)
    rows=[];losses={};N=len(paths);k=len(w.roots);t=cfg['refresh_turns'][0]
    for B in budgets or cfg['budgets']:
        a=execute(paths,B,schedule('S',[],cfg),noise,w,qw)
        b=execute(paths,B,schedule('D',[],cfg),noise,w,qw)
        old,_,_=release(paths[:,0],.5,B,k,noise[1]);new,_,_=release(paths[:,t-1],.25,B,k,noise[t])
        truth=task(w,paths[:,t-1]);local_S=abs(task(w,old)-truth);local_D=abs(task(w,new)-truth)
        S=float(a['mae'].mean());D=float(b['mae'].mean());gain=a['mae']-b['mae']
        row=dict(template=w.template,B=B,level=level,delta=delta,component=component,model=model,n_sim=N,
                 loss_S=S,loss_D=D,R_session=S/D if D>0 else None,G_session=S-D,choice='D' if S>D else 'S',
                 mc_gain_se=float(gain.std(ddof=1)/np.sqrt(N)),
                 local_loss_S=float(local_S.mean()),local_loss_D=float(local_D.mean()),
                 R_local=float(local_S.mean()/local_D.mean()) if local_D.mean()>0 else None,
                 G_local=float(local_S.mean()-local_D.mean()),local_choice='D' if local_S.mean()>local_D.mean() else 'S',
                 root_loss_S=float(a['root_loss'].mean()),root_loss_D=float(b['root_loss'].mean()),
                 no_ast_choice='D' if a['root_loss'].mean()>b['root_loss'].mean() else 'S',
                 empirical_correlation=corr,mean_empirical_correlation=mean_corr,
                 model_source='public calendar metadata, independent uniform/reference prior, fixed signs/noise; no private evaluation')
        rows.append(row)
        if keep_losses:losses[str(B)]=dict(S=a['mae'],D=b['mae'],local_S=local_S,local_D=local_D)
    elapsed=time.perf_counter()-started
    for row in rows:row['offline_seconds_per_configuration']=elapsed/len(rows)
    return rows,losses
