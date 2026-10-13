import numpy as np
from adapter import old, old_data, original


def make(name,H,scenario,amplitude=.12):
    base,mu,ids=old_data.nhanes(name,200)
    n,k=base.shape; rng=np.random.default_rng(1013+['FIB4','HOMA','ANEMIA'].index(name))
    # Same phase and waveform for all control conditions; never align all
    # patients' change times with a particular public refresh period.
    wave=np.zeros((n,H)); phases=[]
    for i in range(n):
        times=np.sort(rng.choice(np.arange(1,H),3,replace=False));phases.append(times.tolist())
        levels=rng.choice([-1.,1.],3)
        for t,v in zip(times,levels):wave[i,t:]=v
    if name=='FIB4': pattern=np.array([0.,1.,0.,1.]); j=3
    elif name=='HOMA': pattern=np.array([1.,-1.]);j=1
    else: pattern=np.array([1.,1.,0.]);j=1
    if scenario=='near_cancel':pattern[j]*=.9
    if scenario=='amplify':pattern[j]*=-1
    if scenario=='irrelevant':pattern=np.array([0.,0.,1.])
    if scenario=='independent':
        pattern=np.ones(k);pattern[0]=0 if name=='FIB4' else 1
        signs=rng.choice([-1.,1.],(n,k));coef=signs*pattern
    else:coef=np.repeat(pattern[None,:],n,axis=0)
    lo,hi=old.bounds(name)
    margin=np.minimum(np.log(base/lo),np.log(hi/base))
    room=np.min(np.where(abs(coef)>0,margin/np.maximum(abs(coef),1e-30),np.inf),axis=1)
    scale=np.minimum(amplitude,.98*room)
    x=base[:,None,:]*np.exp(scale[:,None,None]*wave[:,:,None]*coef[:,None,:])
    assert np.all(x>=lo-1e-10) and np.all(x<=hi+1e-10)
    plan=np.zeros((H,k),bool);plan[0]=True;plan[1:]=np.any(coef!=0,axis=0)
    return dict(name=name,x=x,mu=mu,ids=ids,plan=plan,
        audit=dict(source='synthetic_from_static_NHANES',scenario=scenario,amplitude=amplitude,
                   min_effective_amplitude=float(scale.min()),zero_amplitude_patients=int((scale<1e-12).sum()),
                   phases=phases,clipping=False))


def observed():
    return old_data.real(8,100)


def split(d):
    cut=20 if d['audit']['source'].startswith('MIMIC') else 60
    parts=[]
    for sl in [slice(None,cut),slice(cut,None)]:
        parts.append(dict(d,x=d['x'][sl],ids=d['ids'][sl]))
    assert not set(parts[0]['ids'])&set(parts[1]['ids'])
    return parts


def diagnosis(d):
    name=d['name'];x=d['x'];p=np.array(original.EXPONENTS[name])
    dx=np.diff(np.log(x),axis=1)
    contributions=dx*p
    gross=np.abs(contributions).sum(axis=-1)
    net=np.abs(contributions.sum(axis=-1))
    attr=np.max(np.abs(dx),axis=-1)
    changed=attr>1e-8; relevant=gross>1e-8
    y=original.compute(name,x);cl=original.classes(name,y)
    return dict(n=len(x),attribute_events=int(changed.sum()),relevant_events=int(relevant.sum()),
        cancellation_ratio_mean=None if not relevant.any() else float(np.mean(net[relevant]/gross[relevant])),
        small_task_given_changed_pct=None if not changed.any() else float(100*(net[changed]<=np.log1p(.02)).mean()),
        continuous_task_events=int((net>np.log1p(.02)).sum()),class_events=int((np.diff(cl,axis=1)!=0).sum()),
        attribute_log_step_mean=float(attr.mean()),task_log_step_mean=float(net.mean()))
