"""Finite-trajectory mechanisms with an explicit public layout and budget cap.

For z_it=(m-1)*(x_it-lo_i)/(hi_i-lo_i), the protected distance is
d_inf(Z,Z')=max_it |z_it-z'_it|. Each exponential release is eps-mDP;
sum of all release charges <= B proves B-mDP for the complete transcript.
Since d_inf <= m-1, B=E/(m-1) also suffices for standard E-trajectory LDP.
Public schedules/layouts agree across neighboring trajectories. No private
change detector, value-dependent allocation, raw prompt or raw routing allowed.
"""
from __future__ import annotations
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'target_mismatch'))
import core as original

def bounds(name):
    a = np.array([original.DOMAINS[name][r] for r in original.ROOTS[name]])
    return a[:, 0], a[:, 1]

def indices(name, x, m=1000):
    lo, hi = bounds(name)
    return (x-lo)/(hi-lo)*(m-1)

def cap_allocation(name, mu, B, counts, mode, m=1000):
    return _allocation(name,tuple(mu),float(B),tuple(counts),mode,m).copy()

@lru_cache(maxsize=4096)
def _allocation(name, mu, B, counts, mode, m):
    """Public-count baseline; optimize the original absolute-noise surrogate.

    Beta_i is longitudinal root cap. eps_i=beta_i/count_i. This is a
    straightforward public-schedule extension, not a proposed new algorithm.
    """
    k = len(counts)
    mu=np.array(mu)
    if mode == 'uniform':
        return np.full(k, B/k)
    s = original.sensitivities(name, mu)
    lo, hi = bounds(name)
    delta = (hi-lo)/(m-1)
    center = np.full(k, (m-1)/2)
    counts = np.asarray(counts)
    # Solve in fractions of the total cap for numerically consistent bounds.
    floor = 1e-6/k
    obj = lambda w: original.OPT['objective_absolute'](B*w/counts, s, delta, center, m)
    jac = lambda w: original.OPT['gradient_absolute'](B*w/counts, s, delta, center, m)*B/counts
    initial = np.sqrt(np.maximum(s*delta*counts, 1e-12))
    initial /= initial.sum()
    result = minimize(obj, initial, jac=jac, method='SLSQP',
        bounds=[(floor, 1.)]*k, constraints={'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones(k)},
        options={'ftol':1e-10,'maxiter':500})
    if not result.success:
        raise RuntimeError('Public-count optimizer failed: '+result.message)
    w = result.x
    if abs(w.sum()-1.)>1e-6 or np.min(w)<floor-1e-8 or obj(w)>obj(np.full(k,1/k))+1e-6:
        raise RuntimeError('Allocation invariant failed')
    return B*w

def sample(name, x, eps, uniforms, m=1000):
    # Matches the original unrounded exponential PMF (already validated in
    # target_mismatch). Its interface expects patients x roots and N x R x k.
    return original.noisy_roots(name, x, eps, uniforms, m)

def run(name, x, update_plan, mu, B, method, uniforms, test_laplace,
        m=1000, period=1, test_fraction=.15, threshold=30.):
    """x=N,H,k; public update_plan=H,k; uniforms=N,R,H,k.

    A cache is one root vector per patient/repeat. Every task uses this vector
    coherently; repeated queries do not cause sampling. Private test decisions
    are part of the transcript and charged even when a cache is reused.
    """
    n,H,k=x.shape
    R=uniforms.shape[1]
    cache=np.empty((n,R,H,k))
    spent=np.zeros((n,R))
    calls=np.zeros((n,R),int)
    refresh_flags=np.zeros((n,R,H),bool)
    if method == 'unsafe_shared_noise':
        beta=cap_allocation(name,mu,B,np.ones(k),'opt',m)
        first=sample(name,x[:,0],beta,uniforms[:,:,0],m)
        cache=first[:,:,None,:]+(x[:,None,:,:]-x[:,None,0:1,:])
        return cache,dict(certified=False,budget_bound=None,spent=None,release_roots=k,
            exact_change_disclosure=True, refresh_flags=np.ones((n,R,H),bool), cap=beta.tolist())
    if method.startswith('predictive'):
        # Adapted private-test carry-forward baseline (PETS 2014 idea).
        # Predict previous released roots; query private max index discrepancy.
        # Laplace scale 1/eps_test because max discrepancy is 1-Lipschitz in
        # d_inf. Reserve the maximum branch cost at every scheduled time.
        step=B/H
        beta0=cap_allocation(name,mu,step,np.ones(k),'opt',m)
        beta_new=cap_allocation(name,mu,step*(1-test_fraction),np.ones(k),'opt',m)
        cache[:,:,0]=sample(name,x[:,0],beta0,uniforms[:,:,0],m)
        spent[:]=step; calls[:]=k; refresh_flags[:,:,0]=True
        for t in range(1,H):
            score=np.max(np.abs(indices(name,x[:,None,t],m)-indices(name,cache[:,:,t-1],m)),axis=-1)
            refresh=(score+test_laplace[:,:,t]/(step*test_fraction))>threshold
            candidate=sample(name,x[:,t],beta_new,uniforms[:,:,t],m)
            cache[:,:,t]=np.where(refresh[:,:,None],candidate,cache[:,:,t-1])
            spent+=step*test_fraction+refresh*step*(1-test_fraction)
            calls+=refresh*k
            refresh_flags[:,:,t]=refresh
        return cache,dict(certified=True,budget_bound=B,spent=spent,release_roots=calls,
            refresh_flags=refresh_flags,cap=None,test_fraction=test_fraction,threshold=threshold)
    plan=np.zeros((H,k),bool)
    if method == 'static':
        plan[0]=True
    elif method.startswith('root_'):
        plan=update_plan.copy()
    else:
        plan[::period]=True
    counts=plan.sum(axis=0)
    if not np.all(counts>=1):raise ValueError('Initial roots must all be initialized')
    mode='uniform' if method.endswith('uniform') else 'opt'
    beta=cap_allocation(name,mu,B,counts,mode,m)
    eps=beta/counts
    for t in range(H):
        if t:cache[:,:,t]=cache[:,:,t-1]
        if plan[t].any():
            sampled=sample(name,x[:,t],eps,uniforms[:,:,t],m)
            cache[:,:,t,plan[t]]=sampled[:,:,plan[t]]
            cost=float(eps[plan[t]].sum())
            spent+=cost; calls+=int(plan[t].sum())
            refresh_flags[:,:,t]=True
    if np.any(spent>B+1e-8):raise RuntimeError('Trajectory cap exceeded')
    return cache,dict(certified=True,budget_bound=B,spent=spent,release_roots=calls,
                     refresh_flags=refresh_flags,cap=beta.tolist(),counts=counts.tolist())

def metrics(name, truth_roots, cache):
    truth=original.compute(name,truth_roots)
    est=original.compute(name,cache)
    gt=original.classes(name,truth)
    pred=original.classes(name,est)
    error=np.abs(est-truth[:,None,:])
    wrong=(pred!=gt[:,None,:])
    changes=(gt[:,1:]!=gt[:,:-1])
    pred_changes=(pred[:,:,1:]!=pred[:,:,:-1])
    misses=changes[:,None,:] & ~pred_changes
    false=~changes[:,None,:] & pred_changes
    # Root freshness error in fractions of fixed public domains.
    lo,hi=bounds(name)
    re=np.abs(cache-truth_roots[:,None,:,:])/(hi-lo)
    delays=[]
    for i,t in zip(*np.nonzero(changes)):
        newclass=gt[i,t+1]
        stop=t+2
        while stop<truth.shape[1] and gt[i,stop]==newclass:stop+=1
        for r in range(pred.shape[1]):
            candidates=np.flatnonzero(pred[i,r,t+1:stop]==newclass)
            delays.append(int(candidates[0]) if len(candidates) else stop-(t+1))
    return dict(truth=truth,error=error,wrong=wrong,root_error=re,
        miss_num=misses.sum(axis=(1,2)),change_den=changes.sum(axis=1)*pred.shape[1],
        false_num=false.sum(axis=(1,2)),stable_den=(~changes).sum(axis=1)*pred.shape[1],
        delay_mean=None if not delays else float(np.mean(delays)))
