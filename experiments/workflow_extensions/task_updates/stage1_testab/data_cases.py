import json
import numpy as np
from common import old,old_data,original,previous_data,events,THRESHOLDS,HERE


def anchor(name,base,target):
    """Bounded log-space projection onto the original exact monomial task.

    Project roots inside their public log box, solve target equality by
    bisection. This is construction of feasible stress states, not clipping
    a completed trajectory and hoping that the required class survives.
    """
    lo,hi=old.bounds(name);p=np.array(original.EXPONENTS[name],float)
    if name=='FIB4':p[0]=0
    z=np.log(base);low=np.full(len(base),-100.);high=np.full(len(base),100.)
    for _ in range(65):
        m=(low+high)/2;value=np.exp(np.clip(z+m[:,None]*p,np.log(lo),np.log(hi)))
        too_low=original.compute(name,value)<target
        low=np.where(too_low,m,low);high=np.where(too_low,high,m)
    result=np.exp(np.clip(z+((low+high)/2)[:,None]*p,np.log(lo),np.log(hi)))
    assert np.allclose(original.compute(name,result),target,rtol=1e-11)
    assert np.all(result>=lo-1e-10) and np.all(result<=hi+1e-10)
    return result


def stress(name,T,pattern):
    base,mu,ids=old_data.nhanes(name,200);N,k=base.shape
    boundary=np.array(THRESHOLDS[name]);b=boundary[np.arange(N)%2]
    margin=.003*b if pattern=='chatter' else np.minimum(.1*b,.4*(boundary[1]-boundary[0]))
    a=anchor(name,base,b-margin);h=anchor(name,base,b+margin)
    states=np.zeros((N,T),int);rng=np.random.default_rng(310013+T+['ANEMIA','FIB4','HOMA'].index(name))
    for i in range(N):
        if pattern=='single':times=[int(rng.integers(1,T-1))]
        elif pattern=='spaced':
            shift=int(rng.integers(-1,2)) if T==8 else int(rng.integers(-2,3))
            times=[T//4+shift,T//2+shift,3*T//4+shift]
        elif pattern=='adjacent':
            first=int(rng.integers(1,T-1));times=[first,first+1]
        elif pattern=='chatter':times=list(range(1,T))
        else:times=[]
        for j,t in enumerate(times):states[i,t:]=(j+1)%2
    x=np.where(states[:,:,None]==0,a[:,None,:],h[:,None,:])
    mutable=np.array(original.EXPONENTS[name])!=0
    if name=='FIB4':mutable[0]=False
    if pattern=='stable':
        p=np.array(original.EXPONENTS[name]);js=np.flatnonzero(mutable);v=np.zeros(k)
        v[js[0]]=p[js[1]];v[js[1]]=-p[js[0]]
        lo,hi=old.bounds(name);room=np.minimum(np.log(a/lo),np.log(hi/a))
        scale=np.minimum(.04,.98*np.min(np.where(abs(v)>0,room/np.maximum(abs(v),1e-30),np.inf),axis=1))
        wave=(-1.)**np.arange(T);wave[0]=0
        x=a[:,None,:]*np.exp(scale[:,None,None]*wave[None,:,None]*v[None,None,:])
    plan=np.zeros((T,k),bool);plan[0]=True;plan[1:]=mutable
    d=dict(name=name,x=x,mu=mu,ids=ids,plan=plan,audit=dict(source='controlled_risk_event_stress_not_clinical_longitudinal',pattern=pattern,
                margin_rule='0.3% of threshold for chatter; otherwise min(10% threshold,40% inter-threshold gap)',no_posthoc_clipping=True))
    y,c,e=events(d)
    required={'single':1,'spaced':3,'adjacent':2,'chatter':T-1,'stable':0}[pattern]
    assert np.all(e.sum(axis=1)==required)
    if pattern=='stable':assert np.allclose(y,y[:,:1],rtol=1e-11)
    return d


def inventory(cfg):
    rows=[]
    for name in cfg['templates']:
        for T in cfg['horizons']:
            for pattern in ['cancel','near_cancel','amplify','independent']:
                rows.append((f'legacy_{name}_{pattern}_H{T}',previous_data.make(name,T,pattern,.12),'legacy_synthetic'))
            if name=='ANEMIA':rows.append((f'legacy_{name}_irrelevant_H{T}',previous_data.make(name,T,'irrelevant',.12),'legacy_synthetic'))
    for name in cfg['templates']:
        for T in cfg['horizons']:
            for pattern in cfg['stress_patterns']:rows.append((f'stress_{name}_{pattern}_H{T}',stress(name,T,pattern),'stress'))
    ids=json.loads((HERE.parent.parent/'outputs/task_updates/exploration_20261013/metadata.json').read_text())['datasets']['real_ANEMIA_H8']
    for T in cfg['horizons']:
        d=old_data.real(T,100);d['validation_ids']=ids['validation_ids'];d['test_ids']=ids['test_ids']
        rows.append((f'real_ANEMIA_H{T}',d,'observed'))
    return rows


def split(d,source):
    if source=='observed':
        sets=[set(d['validation_ids']),set(d['test_ids'])]
        indices=[np.array([i for i,p in enumerate(d['ids']) if p in s]) for s in sets]
    else:indices=[np.arange(60),np.arange(60,200)]
    result=[dict(d,x=d['x'][idx],ids=[d['ids'][i] for i in idx]) for idx in indices]
    assert not set(result[0]['ids'])&set(result[1]['ids'])
    assert sum(len(r['ids']) for r in result)==len(d['ids'])
    return result
