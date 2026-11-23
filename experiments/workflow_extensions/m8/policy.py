"""M8 fixed-clock policy with an exact bounded exponential sampler."""
from fractions import Fraction
from secrets import randbelow
import math

import numpy as np
from ..m7.engine import Branch,GRID

def choose(plans,budget,public):
    density=int(len(public['events'])>=8)
    budgets=sorted({float(k.split('/')[0]) for k in plans})
    nearest=min(budgets,key=lambda b:abs(math.log(b/budget)))
    return Branch(**plans[f'{nearest}/{density}']['route'])

def bernoulli(p):
    return randbelow(p.denominator)<p.numerator

def exp_coin(x):
    """Exact Bernoulli(exp(-x)) for nonnegative rational x."""
    while x>0:
        unit=min(x,1)
        k=1
        while bernoulli(Fraction(unit,k)):
            k+=1
        if k%2==0:
            return False
        x-=unit
    return True

def sample(value,epsilon):
    while True:
        j=randbelow(GRID)
        if exp_coin(epsilon*abs(value-Fraction(j,GRID-1))/2):
            return j

def run(ctx,plans,budget,public,read_roots):
    """Trusted reader returns finite in-domain or clippable real root values."""
    B=Fraction(budget)
    if B<=0:raise ValueError('Positive budget required')
    branch=choose(plans,float(B),public)
    if not 0<branch.anchor<=1 or (branch.refresh_count is None)!=(branch.anchor==1):
        raise ValueError('Invalid initial budget or refresh count')
    if branch.refresh_count is not None and not 1<=branch.refresh_count<=public['T']:
        raise ValueError('Invalid refresh count')
    if tuple(sorted(set(branch.fixed_roots)))!=branch.fixed_roots or not set(branch.fixed_roots)<=set(ctx.roots):
        raise ValueError('Invalid root package')
    if branch.gate!='off' or branch.appointment is not None or branch.calibration!='marginal':
        raise ValueError('Unsupported frozen M8 branch')
    T=public['T']
    events=[e['turn'] for e in public['events'] if 1<e['turn']<T]
    targets=np.linspace(1,T+1,(branch.refresh_count or 0)+2)[1:-1]
    times=sorted({min(events,key=lambda t:(abs(t-v),t)) for v in targets}) if events else []
    if branch.period is not None and branch.refresh_count is None:
        raise ValueError('Expected a fixed-count plan')
    total=B*Fraction(branch.anchor)
    service=np.empty(len(ctx.roots)); variance=np.empty_like(service)
    last=np.zeros(len(service),int); actions=[]; stream=np.empty((T,len(service)))
    for idx,turn in enumerate([1,*times]):
        mask=(branch.fixed_roots or ctx.roots) if idx else ctx.roots
        amount=(B-total)/len(times) if idx else total
        weights=branch.refresh_weights if idx else branch.weights
        ratios=[Fraction(float(a)) for a in ctx.allocation(float(amount),mask,weights)]
        if any(r<=0 for r in ratios):raise ValueError('Nonpositive public root allocation')
        eps=[amount*r/sum(ratios) for r in ratios]
        names=[ctx.work.roots[i] for i in mask]
        values=read_roots(turn,names)
        if len(values)!=len(mask):raise ValueError('One measurement per requested root required')
        fresh=[]
        for i,e,v in zip(mask,eps,values):
            lo,hi=map(Fraction,[float(ctx.work.lo[i]),float(ctx.work.hi[i])])
            x=max(0,min(1,(Fraction(v)-lo)/(hi-lo)))
            j=sample(x,e)
            fresh.append(ctx.marginal_grid(i,turn-1,float(e),branch.shrink)[j])
        ids=list(mask); ef=np.array(list(map(float,eps)))
        if idx and branch.fusion:
            service[ids],variance[ids],_=ctx.fuse(service,variance,np.array(fresh),ids,turn-1,last,ef,branch.drift_multiplier)
        else:
            service[ids]=fresh;variance[ids]=np.minimum(.25,8/ef**2)
        last[ids]=turn-1
        end=times[idx] if idx<len(times) else T+1
        stream[turn-1:end-1]=service
        actions.append(dict(turn=turn,roots=names,epsilon=eps))
    spent=sum(sum(a['epsilon']) for a in actions)
    assert spent<=B
    return dict(cache=stream,actions=actions,spent=spent)
