"""Public policy, native RootGuard releases, and calibrated service caches."""
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations
from secrets import randbits

import numpy as np

from ..finite_cache import data
from ..finite_cache.workload import Workload
from ..m4_carry.core import Plan, key_seed, paired_noise, realize
from ..m4_carry.rootguard_mopt import MOpt, exp_pmf, uniforms
from ..m5.public import jacobian
from ..m6.kernels import grid_logZ

GRID = 1000
NATIVE = ('MELD_NA', 'ANEMIA', 'FIB4', 'BUN_CR')


@dataclass(frozen=True)
class Branch:
    anchor: float
    period: int | None = None
    gate: str = 'off'
    packages: str = 'all'
    carry: bool = False
    weights: str = 'mopt'
    calibration: str = 'none'
    shrink: float = 0.
    fusion: bool = False
    drift_multiplier: float = 1.
    appointment: int | None = None
    fixed_roots: tuple = ()
    refresh_weights: str = 'mopt'
    refresh_count: int | None = None

    def __post_init__(self):
        object.__setattr__(self,'fixed_roots',tuple(self.fixed_roots))


BASE_STATIC = Branch(1.)
BASE_PERIODIC = Branch(.5, 8)


def branches(profile, template, objective):
    static = replace(BASE_STATIC, calibration='marginal', shrink=.85)
    dynamic = replace(BASE_PERIODIC, gate=objective, packages='singles', carry=True,
                      calibration='marginal', shrink=.85)
    if profile == 'historical' and template == 'ANEMIA':
        static = replace(static, weights='rms', calibration='joint', shrink=.35)
        dynamic = replace(dynamic, packages='all', carry=False, calibration='joint', shrink=.35)
    if profile == 'historical' and template == 'BUN_CR':
        static = replace(static, calibration='joint', shrink=.35)
        dynamic = replace(dynamic, anchor=.65, carry=False, calibration='joint', shrink=.35)
    return static, dynamic


class FormulaMOpt(MOpt):
    def __init__(self, work, center):
        if work.template in NATIVE:
            super().__init__(work, {'mu': center})
            return
        self.w, self.m, self.floor = work, GRID, .001
        gradient = abs(work.jacobian(work.denormalize(center))[work.nodes.index(work.target)])/work.width
        self.info = {'sensitivities': dict(zip(work.roots, gradient))}
        self.logs, self.cache = [], {}


@lru_cache(maxsize=48000)
def cdf(value, epsilon):
    curve = np.cumsum(exp_pmf(value, epsilon/(GRID-1), GRID))
    curve[-1] = 1.
    return curve


def release(state, epsilon, noise):
    """Only the trusted release step receives current-session root values."""
    u = uniforms(noise)
    output = []
    for j,(value, e) in enumerate(zip(state, epsilon)):
        indices=np.searchsorted(cdf(float(value),float(e)),u[...,j],side='right')
        output.append(np.minimum(indices,GRID-1)/(GRID-1))
    return np.stack(output, axis=-1)


class Context:
    def __init__(self, template, references, seed):
        if not references:
            raise ValueError('At least one reference trajectory is required')
        self.work = Workload(template)
        self.seed = seed
        self.roots = tuple(range(len(self.work.roots)))
        self.bank = np.stack([self.work.normalize(data.replay(record)) for record in references])
        self.center = self.bank.mean(axis=(0,1))
        self.mopt = FormulaMOpt(self.work, self.center)
        self.bank_values = self.work.from_cache(self.bank)
        self.nodes = tuple(self.work.nodes)
        self.target = self.nodes.index(self.work.target)
        self.bank_tasks = np.stack([self.bank_values[n] for n in self.nodes],axis=-1)
        abs_query_values = {node:[] for node in self.nodes}
        for index,record in enumerate(references):
            for q in record['queries']:
                abs_query_values[q['node']].append(abs(self.bank_values[q['node']][index,q['turn']-1]))
        train_query_count = np.array([len(abs_query_values[node]) for node in self.nodes],float)
        self.query_weights = train_query_count/max(1,train_query_count.sum())
        self.scales = np.array([max(1e-8,np.mean(abs_query_values[node]) if abs_query_values[node]
                                  else np.mean(abs(self.bank_values[node]))) for node in self.nodes])
        grad = jacobian(self.work,self.bank.reshape(-1,len(self.roots)))[:,self.target]
        weight = np.maximum(np.mean(grad**2,axis=0)**(1/3),1e-12)
        self.rms = weight/weight.sum()
        rng = np.random.default_rng(key_seed(seed,template,'m7-policy-probe'))
        self.probe = rng.laplace(size=(len(references),4,len(self.roots)))
        jumps = []
        for record, values in zip(references,self.bank):
            jumps.extend(abs(values[e['turn']-1]-values[e['turn']-2]) for e in record['events'])
        self.attack_public = dict(roots=self.work.roots, mu=self.center, root_std=self.bank.std(axis=(0,1)),
                                  max_train_jump=np.max(jumps,axis=0) if jumps else np.zeros(len(self.roots)),
                                  work_lo=np.zeros(len(self.roots)),work_hi=np.ones(len(self.roots)),
                                  grid_points=np.full(len(self.roots),GRID))

    def allocation(self, total, mask, weights):
        original=self.mopt.allocate(float(total))[0]
        if weights=='mopt' and mask==self.roots:
            return original
        shares=original[list(mask)]/total
        if weights=='equal':
            return np.full(len(mask),total/len(mask))
        if weights=='rms':
            shares=.25*shares+.75*self.rms[list(mask)]
        return total*shares/shares.sum()

    @lru_cache(maxsize=8192)
    def marginal_grid(self, root, turn, epsilon, alpha):
        values=self.bank[:,turn,root]
        grid=np.linspace(0,1,GRID)
        weight=-epsilon*abs(values[:,None]-grid)/2-grid_logZ(values,epsilon,0,1,GRID)[:,None]
        public_weights=np.exp(weight-weight.max(axis=0))
        target=(public_weights*values[:,None]).sum(axis=0)/public_weights.sum(axis=0)
        return (1-alpha)*grid+alpha*target

    def calibrated(self, cache, last, eps, branch):
        if branch.calibration=='marginal' and last.ndim==2 and cache.ndim==2 and np.all(last==last[0]) and np.all(eps==eps[0]):
            return self.calibrated(cache,last[0],eps[0],branch)
        if branch.calibration=='none':
            return cache
        if branch.calibration=='marginal' and np.ndim(last)==1:
            grid_indices=np.rint(cache*(GRID-1)).astype(int)
            return np.stack([self.marginal_grid(i,int(last[i]),float(eps[i]),branch.shrink)[grid_indices[...,i]]
                             for i in self.roots],axis=-1)
        # The empirical likelihood is evaluated separately from the raw gate.
        values=self.bank[:,last,np.array(self.roots)]
        likelihood=-abs(values-cache[None,...])*eps/2-grid_logZ(values,eps,0,1,GRID)
        if branch.calibration=='joint':
            likelihood=likelihood.sum(axis=-1,keepdims=True)
        w=np.exp(likelihood-likelihood.max(axis=0,keepdims=True))
        estimate=(w*values).sum(axis=0)/w.sum(axis=0)
        return (1-branch.shrink)*cache+branch.shrink*estimate

    @lru_cache(maxsize=4096)
    def forecasts(self, turn, mask, eps):
        selected=list(mask)
        return np.stack([release(z[selected],eps,u[:,selected])
                         for z,u in zip(self.bank[:,turn],self.probe)])

    def task_weights(self, length, queries, start=0, end=None):
        end=length if end is None else end
        weights=np.zeros((end-start,len(self.nodes)))
        weights[:,self.target]=.5/(end-start)
        active=[(turn,node) for turn,node in queries if start<=turn<end]
        if active:
            for turn,node in active:
                weights[turn-start,self.nodes.index(node)]+=.5/len(active)
        else:
            weights[:,self.target]*=2
        return weights/self.scales

    @lru_cache(maxsize=4096)
    def drift(self, turn, last):
        """Empirical second moment of elapsed root change, in normalized units."""
        changes=self.bank[:,turn,:]-self.bank[:,np.asarray(last),np.array(self.roots)]
        return np.mean(changes**2,axis=0)

    def fuse(self, current, variance, fresh, indices, turn, last, epsilon, drift_multiplier):
        predicted=variance[indices]+drift_multiplier*self.drift(turn,tuple(last))[indices]
        released=np.minimum(.25,8/epsilon**2)
        gain=predicted/(predicted+released)
        return (current[indices]+gain*(fresh-current[indices]),(1-gain)*predicted,gain)

    def gate(self, turn, end, queries, cache, last, prior_eps, total, branch, current=None, variance=None):
        """The gate has no access to the current user's root values."""
        old_values=self.bank[:,last,np.array(self.roots)]
        likelihood=-abs(cache-old_values)*prior_eps/2
        if branch.gate!='final':
            likelihood-=grid_logZ(old_values,prior_eps,0,1,GRID)
        log_weight=likelihood.sum(axis=1)
        weight=np.exp(log_weight-log_weight.max())
        weight/=weight.sum()
        future=self.bank_tasks[:,turn:end]
        window_weights=self.task_weights(len(self.bank_tasks[0]),queries,turn,end)
        if branch.gate!='workload':
            window_weights[:]=0.
            window_weights[:,self.target]=1/(end-turn)

        def loss(raw,last_release,epsilon,*,indices=None):
            service=self.calibrated(raw,last_release,epsilon,branch) if branch.gate!='final' else raw
            if branch.fusion:
                if indices is None:
                    service=current
                else:
                    merged=np.broadcast_to(current,service.shape).copy()
                    merged[...,indices],_,_=self.fuse(current,variance,service[...,indices],indices,turn,last,epsilon[indices],branch.drift_multiplier)
                    service=merged
            predictions=self.work.from_cache(service)
            values=np.stack([predictions[n] for n in self.nodes],axis=-1)
            if values.ndim==1:
                error=abs(values[None,None,:]-future)
                return float(weight @ (error*window_weights).sum(axis=(1,2)))
            error=abs(values[:,:,None,:]-future[:,None])
            return float(weight @ (error*window_weights).sum(axis=(2,3)).mean(axis=1))

        reuse=loss(cache,last,prior_eps)
        best,chosen= reuse, ()
        masks=(branch.fixed_roots,) if branch.appointment is not None else package_masks(self.roots,branch.packages)
        allocation_weights=branch.refresh_weights if branch.appointment is not None or branch.refresh_count is not None else branch.weights
        for mask in masks:
            indices=list(mask)
            eps=self.allocation(total,mask,allocation_weights)
            planned=np.broadcast_to(cache,self.probe.shape).copy()
            planned[...,indices]=self.forecasts(turn,mask,tuple(eps))
            new_last=last.copy()
            new_last[indices]=turn
            new_eps=prior_eps.copy()
            new_eps[indices]=eps
            risk=loss(planned,new_last,new_eps,indices=indices)
            if risk<best:
                best,chosen=risk,mask
        return chosen,reuse-best


def package_masks(roots, mode):
    if mode=='all':
        return (roots,)
    if mode=='singles':
        return tuple((i,) for i in roots if len(roots)>1)+(roots,)
    if mode=='subsets':
        return tuple(combo for k in range(1,len(roots)+1) for combo in combinations(roots,k))
    raise ValueError(f'Unknown root package mode {mode}')


def public_view(record):
    return dict(T=record['T'], events=[dict(turn=e['turn'],measured_roots=e['measured_roots']) for e in record['events']],
                queries=tuple((q['turn']-1,q['node']) for q in record['queries']))


def execute(record, B, branch, ctx, phase, repeats, trace=False):
    """Offline replay with paired experimental noise."""
    public=public_view(record)
    T=public['T']
    values=ctx.work.normalize(data.replay(record))
    noise=paired_noise(ctx.seed,phase,str(record['subject_id']),record['trajectory_id'],float(B),repeats,T,
                       tuple(ctx.work.roots))
    return _execute(public,B,branch,ctx,lambda turn,ids:values[turn-1,list(ids)],noise,trace)


def run_live(ctx, route, budget, public, read_roots):
    """Run one chosen branch with secret randomness and a trusted current-value reader.

    read_roots(turn, names) must return the latest authorized values at that turn,
    in the supplied order. The fixed public calendar and queries have already
    been supplied. Service output contains calibrated cache and action metadata.
    """
    branch=route.choose(budget)
    T=public['T']
    size=(T+1,1,len(ctx.roots))
    u=np.fromiter(((randbits(52)+.5)/2**52-.5 for _ in range(np.prod(size))),
                  dtype=float,count=int(np.prod(size))).reshape(size)
    noise=-np.sign(u)*np.log1p(-2*abs(u))

    def normalized_read(turn,ids):
        names=tuple(ctx.work.roots[i] for i in ids)
        values=np.asarray(read_roots(turn,names),float)
        if values.shape!=(len(ids),) or not np.isfinite(values).all():
            raise ValueError('Reader must return one finite current value per requested root')
        return np.clip((values-ctx.work.lo[list(ids)])/ctx.work.width[list(ids)],0,1)

    result=_execute(public,budget,branch,ctx,normalized_read,noise,True)
    actions=[dict(turn=1,roots=ctx.work.roots,epsilon=result['anchor_epsilon'][0])]
    for row in result['histories']:
        mask=row['selected'][0]
        if mask.any():
            actions.append(dict(turn=row['turn'],roots=[root for root,selected in zip(ctx.work.roots,mask) if selected],
                                epsilon=row['actual_epsilon'][0,mask]))
    return dict(cache=result['caches'][0],actions=actions,spent=float(result['spent'][0]))


def _execute(public, B, branch, ctx, read, noise, trace):
    T=public['T']
    repeats=noise.shape[1]
    if B<=0 or repeats<=0 or not 0<branch.anchor<=1:
        raise ValueError('Positive budget, repeats and anchor fraction required')
    if branch.fusion and (branch.calibration!='marginal' or branch.gate=='final'):
        raise ValueError('Fusion needs marginal calibration and a calibrated or inactive gate')
    if (branch.anchor==1)!=(branch.period is None):
        raise ValueError('A full anchor has no period, a partial anchor requires a period')
    schedule=[] if branch.period is None else [
        t for t,_ in realize(Plan(.5,sets=(ctx.roots,),period=branch.period),
                             [e['turn'] for e in public['events']],T)]
    if branch.refresh_count is not None:
        if branch.refresh_count < 1 or branch.period is None or branch.appointment is not None or branch.gate!='off':
            raise ValueError('Invalid fixed-count refresh plan')
        events=[e['turn'] for e in public['events'] if 1<e['turn']<T]
        # Each target uses its nearest public opportunity. Shared matches are paid once.
        schedule=sorted({min(events,key=lambda t:(abs(t-target),t))
                         for target in np.linspace(1,T+1,branch.refresh_count+2)[1:-1]}) if events else []
    if branch.appointment is not None:
        if branch.gate not in ('off','workload') or branch.fusion or branch.period is None:
            raise ValueError('Scheduled packages require a fixed dynamic release')
        if not branch.fixed_roots or tuple(sorted(set(branch.fixed_roots)))!=branch.fixed_roots or not set(branch.fixed_roots)<=set(ctx.roots):
            raise ValueError('Invalid scheduled root package')
        events=[e['turn'] for e in public['events'] if e['turn']>1]
        schedule=[min(events,key=lambda t:(abs(t-branch.appointment),t))] if events else []
    if branch.period is not None and not schedule and branch.appointment is None and branch.refresh_count is None:
        raise ValueError('Dynamic branch needs a public opportunity')
    epsilon=ctx.allocation(B*branch.anchor,ctx.roots,branch.weights)
    raw=release(read(1,ctx.roots),epsilon,noise[1])
    last=np.zeros_like(raw,dtype=int)
    last_eps=np.tile(epsilon,(repeats,1))
    static_observations=last_eps.copy()
    anchor_raw=raw.copy()
    service=ctx.calibrated(raw,last,last_eps,branch)
    anchor_service=service.copy()
    variance=np.minimum(.25,8/last_eps**2) if branch.fusion else None
    spent=last_eps.sum(axis=1)
    refreshes=np.zeros(repeats,int)
    root_count=np.full(repeats,len(ctx.roots))
    histories=[]
    stream=np.empty((repeats,T,len(ctx.roots)))
    boundaries=[1,*schedule,T+1]
    stream[:,:boundaries[1]-1]=service[:,None,:]
    for j,(turn,end) in enumerate(zip(schedule,boundaries[2:])):
        K=len(schedule)
        budget=(B-spent)/(K-j) if branch.carry else np.full(repeats,(1-branch.anchor)*B/K)
        selected=np.zeros_like(raw,dtype=bool)
        action_epsilon=np.zeros_like(raw)
        predicted=np.zeros(repeats)
        fusion_gain=np.ones_like(raw) if branch.fusion else None
        if branch.gate=='off' and not branch.fusion:
            # Full periodic refresh has the same action and charge for every replay.
            mask=branch.fixed_roots or ctx.roots
            indices=list(mask)
            weights=branch.refresh_weights if branch.appointment is not None or branch.refresh_count is not None else branch.weights
            eps=ctx.allocation(budget[0],mask,weights)
            raw[:,indices]=release(read(turn,mask),eps,noise[turn][:,indices])
            selected[:,indices]=True
            action_epsilon[:,indices]=eps
            last[:,indices]=turn-1
            last_eps[:,indices]=eps
        else:
            for r in range(repeats):
                if branch.gate=='off':
                    mask=branch.fixed_roots or ctx.roots
                else:
                    mask,predicted[r]=ctx.gate(turn-1,end-1,public['queries'],raw[r],last[r],last_eps[r],budget[r],branch,
                                               service[r] if branch.fusion else None,
                                               variance[r] if branch.fusion else None)
                if not mask:
                    continue
                ids=list(mask)
                eps=ctx.allocation(budget[r],mask,branch.refresh_weights if branch.appointment is not None or branch.refresh_count is not None else branch.weights)
                raw[r,ids]=release(read(turn,ids),eps,noise[turn,r,ids])
                if branch.fusion:
                    indices=np.rint(raw[r,ids]*(GRID-1)).astype(int)
                    fresh=np.array([ctx.marginal_grid(i,turn-1,float(e),branch.shrink)[j]
                                    for i,e,j in zip(ids,eps,indices)])
                    service[r,ids],variance[r,ids],fusion_gain[r,ids]=ctx.fuse(
                        service[r],variance[r],fresh,ids,turn-1,last[r],eps,branch.drift_multiplier)
                last[r,ids]=turn-1
                last_eps[r,ids]=eps
                selected[r,ids]=True
                action_epsilon[r,ids]=eps
        charge=action_epsilon.sum(axis=1)
        spent+=charge
        if np.any(spent>B+1e-9) or not np.allclose(charge[selected.any(axis=1)],budget[selected.any(axis=1)],atol=1e-9,rtol=0):
            raise ArithmeticError('Invalid prefix budget')
        refreshes+=selected.any(axis=1)
        root_count+=selected.sum(axis=1)
        if not branch.fusion:
            service=ctx.calibrated(raw,last,last_eps,branch)
        stream[:,turn-1:end-1]=service[:,None,:]
        if trace:
            histories.append(dict(turn=turn,selected=selected,actual_epsilon=action_epsilon,
                                  cache_after=raw.copy(),service=service.copy(),last=last.copy(),
                                  last_eps=last_eps.copy(),budget=budget,predicted_gain=predicted,
                                  kernels=np.ones_like(selected,dtype=bool),fusion_gain=fusion_gain))
    return dict(caches=stream,spent=spent,refreshes=refreshes,roots=root_count,histories=histories,
                anchor_cache=anchor_raw,anchor_service=anchor_service,anchor_epsilon=static_observations,
                anchor_kernels=np.ones_like(raw,dtype=bool),schedule=schedule,public=public)
