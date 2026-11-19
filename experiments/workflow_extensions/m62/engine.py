"""One release/cache executor shared by public simulation and real replay."""
from functools import lru_cache
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.m4_carry.core import task,paired_noise


def schedule(method,public_events,cfg):
    if method=='S':return [(1,1.)]
    if method=='D':return [(1,.5)]+[(t,.25) for t in cfg['refresh_turns']]
    assert method.startswith('P')
    P=int(method[1:]);times=sorted(public_events)[P-1::P]
    times=times or sorted(public_events)[-1:]
    assert times
    return [(1,.5)]+[(t,.5/len(times)) for t in times]


def release(values,total,B,k,unit_noise):
    epsilon=np.full(k,total*B/k)
    assert np.all(epsilon>0)
    raw=values+unit_noise/epsilon
    return np.clip(raw,0,1),epsilon,(raw<0)|(raw>1)


def execute(states,B,actions,noise,w,query_weights,stop=None,detailed=False):
    """states [N,T,k], noise [T+1,N,k]. Reads only states[:,turn-1].

    Public decisions (actions) must be frozen before this function. This trusted
    executor never returns a private value to a policy/selector.
    """
    N,T,k=states.shape;stop=T if stop is None else stop
    assert 1<=stop<=T
    assert actions[0] in [(1,1.),(1,.5)]
    assert all(0<rho<=1 for t,rho in actions)
    assert all(a[0]<b[0] for a,b in zip(actions,actions[1:]));assert sum(rho for _,rho in actions)<=1+1e-12
    cache=np.empty((N,T,k)) if detailed else None
    pred=np.empty((N,T));root_loss=np.empty((N,T));projected=np.zeros(N);publications=np.zeros(N);spent=0.;hist=[]
    for j,(t,rho) in enumerate(actions):
        if t>stop:break
        after,epsilon,proj=release(states[:,t-1],rho,B,k,noise[t])
        end=min(stop,actions[j+1][0]-1 if j+1<len(actions) else T)
        pred[:,t-1:end]=task(w,after)[:,None]
        root_loss[:,t-1:end]=abs(after[:,None,:]-states[:,t-1:end]).mean(axis=-1)
        if detailed:cache[:,t-1:end]=after[:,None,:]
        spent+=epsilon.sum();projected+=proj.sum(axis=1);publications+=k
        assert spent<=B+1e-10
        hist.append(dict(turn=t,rho=rho,epsilon=epsilon,cache=after,projection=proj))
    truth=task(w,states[:,:stop]);qw=query_weights[:,:stop];den=qw.sum(axis=1)
    error=abs(pred[:,:stop]-truth)
    def average(values):return np.divide((values*qw).sum(axis=1),den,out=np.full(N,np.nan),where=den>0)
    mae=average(error);truth_abs=average(abs(truth))
    result=dict(mae=mae,truth_abs=truth_abs,root_loss=average(root_loss[:,:stop]),
                spent=np.full(N,spent),n_actions=np.full(N,len(hist)),n_root_releases=publications,
                projection_rate=projected/publications,p95=np.array([np.quantile(error[n,qw[n]>0],.95) if den[n]>0 else np.nan for n in range(N)]) if detailed else None,
                histories=hist,caches=cache)
    if detailed:
        classes=w.classes(truth)
        result['rce']=np.full(N,np.nan) if classes is None else average(100*abs(w.classes(pred[:,:stop])-classes)/2)
    else:result['rce']=None
    return result


def real_replay(rec,B,method,w,cfg,phase,R,detailed=False):
    z=w.normalize(data.replay(rec));states=np.broadcast_to(z,(R,)+z.shape)
    noise=paired_noise(cfg['noise_seed'],phase,str(rec['subject_id']),rec['trajectory_id'],float(B),R,rec['T'],tuple(w.roots))
    qw=np.zeros((R,rec['T']))
    for q in rec['queries']:
        if q['node']==w.target:qw[:,q['turn']-1]+=1
    actions=schedule(method,[e['turn'] for e in rec['events']],cfg)
    result=execute(states,B,actions,noise,w,qw,detailed=detailed)
    return result
