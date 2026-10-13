"""Finite public horizon, d_inf metric DP; restarted AboveThreshold reference.

Original exponential releases are charged by composition. AboveThreshold
uses threshold Lap(2*Delta/eps) and query Lap(4*Delta/eps), stops at one
positive, restarts at most C times; reserve eps_test/C per block. Published
roots receive (B-eps_test)/(C+1) per vector. Oracle is explicitly non-DP.
"""
import numpy as np
from adapter import old,original


def sensitivity(name,mode):
    if mode=='attribute':return 1/999
    lo,hi=old.bounds(name)
    return float(np.sum(np.abs(original.EXPONENTS[name])*(hi-lo)/(999*lo)))


def score(name,x,pred,mode):
    if mode=='attribute':
        lo,hi=old.bounds(name)
        return np.max(np.abs(x-pred)/(hi-lo),axis=-1)
    return np.abs(np.log(original.compute(name,x))-np.log(original.compute(name,pred)))


def run(d,B,param,u,qnoise,tnoise):
    name=d['name'];x=d['x'];N,H,k=x.shape;R=u.shape[1]
    method=param['method']
    if method in ['static','root','period']:
        m='root_opt' if method=='root' else 'static' if method=='static' else 'full_opt'
        z,l=old.run(name,x,d['plan'],d['mu'],B,m,u,qnoise,period=param.get('period',1))
        l['certified']=True;return z,l
    C=param['C'];fraction=param.get('fraction',0.)
    pub=B*(1-fraction)/(C+1)
    beta=old.cap_allocation(name,d['mu'],pub,np.ones(k),'opt')
    cache=np.empty((N,R,H,k));flags=np.zeros((N,R,H),bool)
    cache[:,:,0]=old.sample(name,x[:,0],beta,u[:,:,0]);flags[:,:,0]=True
    positives=np.zeros((N,R),int);spent=np.full((N,R),pub)
    if method=='oracle':
        anchor=x[:,0].copy()
        for t in range(1,H):
            cache[:,:,t]=cache[:,:,t-1]
            event=np.abs(np.log(original.compute(name,x[:,t]))-np.log(original.compute(name,anchor)))>np.log1p(.02)
            refresh=event[:,None]&(positives<C)
            new=old.sample(name,x[:,t],beta,u[:,:,t])
            cache[:,:,t]=np.where(refresh[:,:,None],new,cache[:,:,t])
            flags[:,:,t]=refresh;positives+=refresh;spent+=refresh*pub
            anchor[event]=x[event,t]
        return cache,dict(certified=False,spent=spent,refresh_flags=flags,release_roots=(positives+1)*k)
    eps=B*fraction/C;delta=sensitivity(name,param['mode'])
    block=np.zeros((N,R),int)
    noisy_threshold=param['threshold']+2*delta/eps*tnoise[:,:,0]
    spent+=eps  # start the first AboveThreshold block, even with no positives
    for t in range(1,H):
        cache[:,:,t]=cache[:,:,t-1]
        active=positives<C
        query=score(name,x[:,None,t],cache[:,:,t-1],param['mode'])
        refresh=active&(query+4*delta/eps*qnoise[:,:,t]>=noisy_threshold)
        new=old.sample(name,x[:,t],beta,u[:,:,t])
        cache[:,:,t]=np.where(refresh[:,:,None],new,cache[:,:,t])
        flags[:,:,t]=refresh;spent+=refresh*pub;positives+=refresh
        restart=refresh&(positives<C)&(t<H-1)
        block+=restart
        fresh=param['threshold']+2*delta/eps*np.take_along_axis(tnoise,block[:,:,None],axis=2)[:,:,0]
        noisy_threshold=np.where(restart,fresh,noisy_threshold)
        spent+=restart*eps
    assert np.max(spent)<=B+1e-8
    return cache,dict(certified=True,spent=spent,refresh_flags=flags,release_roots=(positives+1)*k,
                      test_delta=delta,test_block_epsilon=eps)


def metrics(d,cache,ledger):
    name=d['name'];truth=original.compute(name,d['x']);est=original.compute(name,cache)
    gt=original.classes(name,truth);pred=original.classes(name,est)
    step=np.abs(np.diff(np.log(truth),axis=1));events=step>np.log1p(.02)
    predstep=np.diff(np.log(est),axis=2);true_signed=np.diff(np.log(truth),axis=1)
    reports=np.abs(predstep)>np.log1p(.02)
    # Correct response: actual report, correct direction, and <=5% log error
    # in the new current task value. Existing class is auxiliary only.
    good=reports&(np.sign(predstep)==np.sign(true_signed[:,None,:]))&(np.abs(np.log(est[:,:,1:]/truth[:,None,1:]))<=np.log1p(.05))
    N,R,H=est.shape
    hit0=(good&events[:,None]).sum(axis=(1,2));hit2=np.zeros(N);den=events.sum(axis=1)*R
    delays=[];censored=0
    for i,t0 in zip(*np.nonzero(events)):
        later=np.flatnonzero(events[i,t0+1:]);stop=min(H-1,t0+1+int(later[0])) if len(later) else H-1
        target=truth[i,t0+1]
        direction=np.sign(true_signed[i,t0])
        for r in range(R):
            valid=[]
            for j in range(t0,stop):
                if reports[i,r,j] and np.sign(predstep[i,r,j])==direction and abs(np.log(est[i,r,j+1]/target))<=np.log1p(.05):valid.append(j-t0)
            delay=min(valid) if valid else stop-t0
            hit2[i]+=bool(valid and delay<=2);delays.append(delay);censored+=not bool(valid)
    stable=~events
    false=(reports&stable[:,None]).sum(axis=(1,2));false_den=stable.sum(axis=1)*R
    patient=dict(error=np.abs(est-truth[:,None]).mean(axis=(1,2)),truth=np.abs(truth).mean(axis=1),
        wrong=(pred!=gt[:,None]).mean(axis=(1,2)),hit0=hit0,hit2=hit2,event_den=den,
        false=false,false_den=false_den)
    agg=dict(wmape=float(100*patient['error'].sum()/patient['truth'].sum()),rce=float(100*patient['wrong'].mean()),
        tcr0=None if den.sum()==0 else float(100*hit0.sum()/den.sum()),
        tcr2=None if den.sum()==0 else float(100*hit2.sum()/den.sum()),
        false_change=None if false_den.sum()==0 else float(100*false.sum()/false_den.sum()),
        delay_capped=None if not delays else float(np.mean(delays)),
        censored_pct=None if not delays else float(100*censored/len(delays)),
        vector_refreshes=float(ledger['refresh_flags'].sum(axis=2).mean()),
        root_releases=float(np.mean(ledger['release_roots'])),spent_max=float(np.max(ledger['spent'])),
        certified=ledger['certified'])
    return agg,patient
