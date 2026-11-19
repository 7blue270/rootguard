"""Native RootGuard M-Opt allocation and static/periodic baselines.

Loads the original solver and, for existing templates, sensitivity propagation
verbatim by AST to avoid unrelated model-loading imports. The native kernel has
exactly the original continuous-index discrete-exponential PMF, with inverse-CDF
sampling only to pair randomness. All charges use normalized trajectory metric
units: eps_metric=(m-1)*eps_grid; delta=0. No change to M4 core.
"""
from __future__ import annotations
import ast
from functools import lru_cache
import hashlib
import math
import random
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize as original_minimize
from experiments.workflow_extensions.finite_cache import data
from .core import Plan, realize, paired_noise, task
from .run import ROOT


def original_functions(names):
    path=ROOT/'utils/utils.py'
    tree=ast.parse(path.read_text())
    defs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    required=set(names)
    while True:
        more={n.id for k in required for n in ast.walk(defs[k]) if isinstance(n,ast.Name) and n.id in defs}
        if more<=required:break
        required|=more
    ns={'np':np,'math':math,'random':random}
    exec(compile(ast.Module(body=[defs[k] for k in sorted(required)],type_ignores=[]),str(path),'exec'),ns)
    return ns,sorted(required)


ORIGINAL,LOADED_FUNCTIONS=original_functions(['optimal_budget_allocation_exact','compute_population_mean_sensitivities'])


def local_sensitivities(w,stats):
    mu=w.denormalize(np.asarray(stats['mu']))
    if w.template in ('ANEMIA','FIB4'):
        from utils.med_domain.all_templates import template_nodes,template_expressions
        topology=w.roots+w.nodes
        sens=ORIGINAL['compute_population_mean_sensitivities'](w.target,w.roots,template_expressions,[],topology,template_nodes,dict(zip(w.roots,mu)))
        grad=np.array([sens[r] for r in w.roots])
        origin='original compute_population_mean_sensitivities + original analytic DAG chain rule'
    elif w.template=='BUN_CR':
        bun,cr=mu;grad=np.abs([1/cr,-bun/cr**2])
        origin='formula extension: analytic derivatives of bun/cr at train mu'
    else:
        bil,inr,cr,na=mu
        m=3.78*np.log(max(bil,1))+11.2*np.log(max(inr,1))+9.57*np.log(np.clip(cr,1,4))+6.43
        d=137-np.clip(na,125,137);y=m+1.32*d-.033*m*d
        grad=np.array([(1-.033*d)*3.78/bil if bil>1 else 0.,
                       (1-.033*d)*11.2/inr if inr>1 else 0.,
                       (1-.033*d)*9.57/cr if 1<cr<4 else 0.,
                       -(1.32-.033*m) if 125<na<137 else 0.]) if 6<y<40 else np.zeros(4)
        grad=abs(grad)
        origin='formula extension: analytic derivatives of the frozen MELD-Na function at train mu'
    numerical=abs(w.jacobian(mu)[w.nodes.index(w.target)])/w.width
    np.testing.assert_allclose(grad,numerical,rtol=1e-4,atol=1e-7)
    assert np.all(np.isfinite(grad)) and np.all(grad>=0)
    return dict(mu_raw=mu.tolist(),roots=w.roots,sensitivities=dict(zip(w.roots,grad.tolist())),
                normalized_local= (grad*w.width).tolist(),source=origin,
                finite_difference_check='passed',derivative_zero_roots=[r for r,g in zip(w.roots,grad) if g==0])


def exp_pmf(z,epsilon_grid,m=1000):
    center=float(np.clip(z,0,1))*(m-1)
    lp=-epsilon_grid*abs(center-np.arange(m))/2
    lp-=lp.max();p=np.exp(lp);p/=p.sum()
    return p


def uniforms(laplace):
    # Same underlying uniform stream as M4; unlike common Laplace, the native
    # output is not an additive copy of M4's unit noise.
    return np.where(laplace<=0,.5*np.exp(np.minimum(laplace,0)),1-.5*np.exp(-np.maximum(laplace,0)))


class MOpt:
    def __init__(self,w,stats,m=1000,epsilon_min_grid=.001):
        self.w,self.m,self.floor=w,m,epsilon_min_grid
        self.info=local_sensitivities(w,stats)
        self.logs=[]
        self.cache={}

    def allocate(self,total_metric):
        total_metric=float(total_metric)
        if total_metric in self.cache:return self.cache[total_metric]
        total_grid=total_metric/(self.m-1)
        diagnostic={}
        def recording(*args,**kwargs):
            result=original_minimize(*args,**kwargs)
            diagnostic.update(success=bool(result.success),status=int(result.status),message=str(result.message),nit=int(result.nit),objective=float(result.fun))
            return result
        with patch('scipy.optimize.minimize',recording):
            alloc=ORIGINAL['optimal_budget_allocation_exact'](self.info['sensitivities'],total_grid,
                       epsilon_min=self.floor,domain_widths=dict(zip(self.w.roots,self.w.width)),num_candidates=self.m,mode='abs')
        eg=np.array([alloc[r] for r in self.w.roots])
        em=eg*(self.m-1)
        assert np.all(np.isfinite(eg)) and np.all(eg>0)
        assert abs(em.sum()-total_metric)<1e-10
        assert em.sum()<=total_metric+1e-10
        fallback=len(eg)*self.floor>total_grid
        self.logs.append(dict(template=self.w.template,total_metric=total_metric,total_grid=total_grid,epsilon_grid=eg.tolist(),
                              epsilon_metric=em.tolist(),insufficient_floor_fallback=fallback,
                              solver=diagnostic or dict(success=True,message='original edge-case branch')))
        self.cache[total_metric]=(em,eg)
        return em,eg


class ReleaseOwner:
    def __init__(self,w,rec,adapter):
        self.__z=w.normalize(data.replay(rec))
        self.m=adapter.m

    def release(self,t,eps_metric,eps_grid,noise,kernel):
        z=self.__z[t-1]
        if kernel=='laplace':
            raw=z+noise/eps_metric
            return np.clip(raw,0,1),((raw<0)|(raw>1))
        assert kernel=='native_exp'
        u=uniforms(noise)
        out=np.empty_like(noise)
        for i in range(len(z)):
            cdf=np.cumsum(exp_pmf(z[i],eps_grid[i],self.m));cdf[-1]=1
            idx=np.searchsorted(cdf,u[:,i],side='right').clip(0,self.m-1)
            out[:,i]=idx/(self.m-1)
        return out,np.zeros_like(out,dtype=bool)


def simulate_baseline(rec,B,period,kernel,adapter,cfg,phase,repeats,detailed=False):
    w=adapter.w;k=len(w.roots);T=rec['T'];R=repeats
    roots=tuple(range(k))
    plan=Plan(1.) if period is None else Plan(.5,sets=(roots,),period=period)
    schedule=realize(plan,[e['turn'] for e in rec['events']],T)
    K=len(schedule)
    anchor_metric,anchor_grid=adapter.allocate(B if period is None else B/2)
    refresh_eps=adapter.allocate(B/(2*K)) if K else None
    noise=paired_noise(cfg['noise_seed'],phase,str(rec['subject_id']),str(rec['trajectory_id']),float(B),R,T,tuple(w.roots))
    owner=ReleaseOwner(w,rec,adapter)
    cache,proj=owner.release(1,anchor_metric,anchor_grid,noise[1],kernel)
    anchor_cache=cache.copy();spent=np.full(R,anchor_metric.sum())
    n_proj=proj.sum(axis=1);hist=[];caches=np.empty((R,T,k));mapping={t:j for j,(t,_) in enumerate(schedule)}
    for t in range(1,T+1):
        if t in mapping:
            eps,eg=refresh_eps
            before=cache.copy();left_before=B-spent.copy()
            cache,proj=owner.release(t,eps,eg,noise[t],kernel)
            n_proj+=proj.sum(axis=1);spent+=eps.sum()
            if detailed:
                hist.append(dict(turn=t,action_index=mapping[t]+1,planned_roots=roots,selected=np.ones((R,k),bool),
                                 candidate_epsilon=np.tile(eps,(R,1)),actual_epsilon=np.tile(eps,(R,1)),
                                 epsilon_grid=np.tile(eg,(R,1)),eps_can=np.full(R,eps.sum()),left_before=left_before,left_after=B-spent.copy(),
                                 cache_before=before,cache_after=cache.copy()))
        caches[:,t-1]=cache
        assert np.all(spent<=B+1e-10)
    assert np.all(abs(spent-B)<1e-10)
    return dict(caches=caches,spent=spent,left=B-spent,m=np.full(R,K),n_pub=np.full(R,(K+1)*k),n_proj=n_proj,
                empty=np.zeros(R),root_veto=np.zeros(R),root_count=np.full(R,K*k),K=K,stop=schedule[-1][0] if K else 1,
                histories=hist,anchor_epsilon=np.tile(anchor_metric,(R,1)),anchor_epsilon_grid=np.tile(anchor_grid,(R,1)),
                anchor_cache=anchor_cache,stop_reason='static' if not K else 'opportunities_exhausted')
