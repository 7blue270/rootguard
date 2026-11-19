"""Metric epsilon always multiplies NORMALIZED VALUE distance, not grid index.

Output grids may be strict subsets of the input domain [0,1]. Their normalizer
depends on the candidate input. No local sensitivity or narrowed input domain.
"""
import numpy as np
from scipy.special import logsumexp
from experiments.workflow_extensions.m4_carry.rootguard_mopt import uniforms


def grid_for(public,i):
    return np.linspace(public['work_lo'][i],public['work_hi'][i],public['grid_points'][i])


def danger(cache,public,enabled=True):
    if not enabled:return np.zeros_like(cache,dtype=bool)
    from experiments.workflow_extensions.finite_cache.workload import Workload
    from experiments.workflow_extensions.m5.public import jacobian
    # Both branches in the specification: small denominator OR high workload
    # gradient. The latter also covers logs and cross-root noise amplification.
    w=Workload(public['template']);weights=np.array(public['query_weights']);scales=np.array(public['node_scales'])
    gradients=np.sum(weights[:,None]*abs(jacobian(w,cache))/scales[:,None],axis=-2)
    high=gradients>np.array(public['a_quantile'])+1e-12
    small=np.asarray(public['denominator_roots'])[None,:] & (cache<=np.asarray(public['danger_threshold'])[None,:])
    return high|small


def emit(z,eps,unit_noise,kernels,public):
    """Vectorized over arbitrary leading dimensions. kernel True means grid EM."""
    z,eps,unit_noise,kernels=np.broadcast_arrays(z,eps,unit_noise,kernels)
    safe=np.where(eps>0,eps,1.)
    raw=z+unit_noise/safe;out=np.clip(raw,0,1);proj=((raw<0)|(raw>1))&~kernels
    u=uniforms(unit_noise)
    for i in range(z.shape[-1]):
        active=kernels[...,i]&(eps[...,i]>0)
        if not active.any():continue
        grid=grid_for(public,i)
        scores=-eps[...,i][active,None]*abs(grid-z[...,i][active,None])/2
        pmf=np.exp(scores-logsumexp(scores,axis=-1,keepdims=True));cdf=np.cumsum(pmf,axis=-1);cdf[:,-1]=1.
        out[...,i][active]=grid[np.minimum(np.sum(cdf<u[...,i][active,None],axis=-1),len(grid)-1)]
    return out,proj


def grid_logZ(x,epsilon,lo,hi,m):
    """Exact uniform-output-grid partition function for arbitrary INPUT x."""
    x=np.asarray(x,float)
    if m==1 or hi-lo<1e-15:return -epsilon*abs(x-lo)/2
    step=(hi-lo)/(m-1);a=epsilon/2
    n=np.clip(np.floor((x-lo)/step).astype(int)+1,0,m)
    denom=-np.expm1(-a*step)
    left=np.exp(-a*np.maximum(0,x-(lo+(n-1)*step)))*(-np.expm1(-a*step*n))/denom
    right=np.exp(-a*np.maximum(0,lo+n*step-x))*(-np.expm1(-a*step*(m-n)))/denom
    return np.log(left+right)


def log_likelihood(observation,epsilon,is_grid,public,i,m):
    from experiments.workflow_extensions.m5.attack import log_likelihood as lap
    if not is_grid:return lap(observation,epsilon,'laplace',m)
    x=np.linspace(0,1,m)
    return -epsilon*abs(x-observation)/2-grid_logZ(x,epsilon,public['work_lo'][i],public['work_hi'][i],public['grid_points'][i])
