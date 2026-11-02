import numpy as np
from common import original,old,previous_release,events


def quota_period(N,T,K,offset=0):
    times=np.clip(np.rint(np.linspace(0,T-1,K+2)[1:-1]).astype(int)+offset,1,T-1)
    assert len(set(times))==K
    return np.repeat(times[None,:],N,axis=0)


def oracle_quota(d,K):
    _,_,e=events(d);N,T=e.shape;default=quota_period(N,T,K)
    result=[]
    for i in range(N):
        selected=list(np.flatnonzero(e[i]))[:K]
        for t in list(default[i])+list(range(1,T)):
            if len(selected)==K:break
            if int(t) not in selected:selected.append(int(t))
        result.append(sorted(selected))
    result=np.array(result)
    assert result.shape==(N,K) and np.all(result>0)
    return result


def scheduled(d,B,K,schedule,u,initial_exact=False,refresh_exact=False,oracle=False,publish_fraction=1.):
    N,T,k=d['x'].shape;R=u.shape[1];cache=np.empty((N,R,T,k))
    beta=old.cap_allocation(d['name'],d['mu'],B*publish_fraction/(K+1),np.ones(k),'opt')
    first=old.sample(d['name'],d['x'][:,0],beta,u[:,:,0])
    cache[:,:,0]=d['x'][:,None,0] if initial_exact else first
    sampled=[]
    for j in range(K):
        xj=d['x'][np.arange(N),schedule[:,j]]
        sampled.append(np.repeat(xj[:,None],R,axis=1) if refresh_exact else old.sample(d['name'],xj,beta,u[:,:,j+1]))
    flags=np.zeros((N,R,T),bool);flags[:,:,0]=True
    for t in range(1,T):
        cache[:,:,t]=cache[:,:,t-1]
        for j in range(K):
            mask=schedule[:,j]==t
            cache[mask,:,t]=sampled[j][mask];flags[mask,:,t]=True
    assert np.all(flags.sum(axis=2)==K+1)
    dp=not oracle and not initial_exact and not refresh_exact
    spent=np.full((N,R),B*publish_fraction)
    return cache,dict(flags=flags,spent=spent,bound=B if dp else None,dp=dp)


def deployment(d,B,param,u,q,th):
    cache,l=previous_release.run(d,B,param,u,q,th)
    assert l['certified'] and np.max(l['spent'])<=B+1e-8
    return cache,dict(flags=l['refresh_flags'],spent=l['spent'],bound=B,dp=True)


def frozen_schedule_pair(d,svt_ledger):
    """Exact-output non-DP probe on frozen realized SVT times/counts.

    Compare to a truth-event schedule with exactly the SAME number of actual
    refreshes per patient/repeat. Not a deployable detector, not A1's fixed
    preassigned K, and not independent additive attribution of detector error.
    """
    flags=svt_ledger['flags'];N,R,T=flags.shape;counts=flags.sum(axis=2)-1;k=d['x'].shape[2]
    actual=np.empty((N,R,T,k));ideal=np.empty_like(actual)
    actual[:,:,0]=d['x'][:,None,0];ideal[:,:,0]=d['x'][:,None,0]
    alternative=np.zeros_like(flags);alternative[:,:,0]=True
    for K in range(1,int(counts.max())+1):
        plan=oracle_quota(d,K);mask=counts==K
        for j in range(K):
            for t in range(1,T):alternative[:,:,t]|=mask&(plan[:,j,None]==t)
    assert np.array_equal(flags.sum(axis=2),alternative.sum(axis=2))
    for t in range(1,T):
        actual[:,:,t]=np.where(flags[:,:,t,None],d['x'][:,None,t],actual[:,:,t-1])
        ideal[:,:,t]=np.where(alternative[:,:,t,None],d['x'][:,None,t],ideal[:,:,t-1])
    ledger=lambda f:dict(flags=f,spent=np.zeros((N,R)),bound=None,dp=False)
    return (actual,ledger(flags)),(ideal,ledger(alternative))


def candidates(T):
    rows=[dict(method='period',period=P) for P in [1,2,4,8,16] if P<=T]
    for mode in ['attribute','task']:
        for C in [2,4]:
            for f in [.2,.5]:
                for threshold in [.01,.03,.1]:rows.append(dict(method='svt',mode=mode,C=C,fraction=f,threshold=threshold))
    return rows
