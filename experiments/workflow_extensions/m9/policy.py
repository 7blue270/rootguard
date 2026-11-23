"""Fixed-clock M9 policy. Its estimator only accepts released observations."""
from fractions import Fraction
from functools import lru_cache
import numpy as np
from ..m7.engine import Branch, GRID
from ..m8.policy import sample
from ..m7.appointment_study import STATIC
from ..physionet2012.study import inputs

def plans(ctx,package):
    def one(mask,clock=21,anchor=.5,weights='mopt'):
        return Branch(anchor,period=1,appointment=clock,fixed_roots=tuple(mask),
                      calibration='marginal',shrink=.85,refresh_weights=weights)
    return dict(static=STATIC,period=Branch(.5,period=2,calibration='marginal',shrink=.85),
                full=one(ctx.roots),package=one(package),early=one(package,11),
                anchor75=one(package,anchor=.75),equal=one(package,weights='equal'),
                combined=one(package,11,.75,'equal'))

def pick(table,B,public):
    near=min((float(k) for k in table),key=lambda v:abs(np.log(v/B)))
    return table[str(near)][str(int(len(public['events'])>=8))]

@lru_cache(maxsize=8192)
def process_noise(ctx,start,end):
    dx=ctx.bank[:,end]-ctx.bank[:,start]
    q=dx.T@dx/len(dx)
    return .85*q+.15*np.diag(np.diag(q))

def estimate(ctx,ps,coef,median):
    anchor=ps['anchor_service'];x=anchor.copy();d=x.shape[-1]
    cov=np.tile(np.diag(np.minimum(.25,8/ps['anchor_epsilon'][0]**2)),(len(x),1,1))
    out=np.empty_like(ps['caches']);last=0
    history=ps['histories']
    for j in range(-1,len(history)):
        if j<0:
            st=0;raw=ps['anchor_cache']
        else:
            h=history[j];st=h['turn']-1;ids=np.flatnonzero(h['selected'][0])
            assert len(ids) and np.all(h['selected']==h['selected'][0])
            P=cov+process_noise(ctx,last,st)
            R=np.diag(np.minimum(.25,8/h['actual_epsilon'][0,ids]**2))
            delta=h['service'][:,ids]-x[:,ids]
            S=P[:,ids][:,:,ids]+R
            scale=np.einsum('ri,rij,rj->r',delta,np.linalg.inv(S),delta)/len(ids)
            P=P*np.clip(scale,1,4)[:,None,None]
            gain=P[:,:,ids]@np.linalg.inv(P[:,ids][:,:,ids]+R)
            x=np.clip(x+np.einsum('rij,rj->ri',gain,delta),0,1)
            rem=np.broadcast_to(np.eye(d),P.shape).copy();rem[:,:,ids]-=gain
            cov=rem@P@rem.transpose(0,2,1)+gain@R@gain.transpose(0,2,1)
            raw=x.copy();raw[:,ids]=h['cache_after'][:,ids]
            last=st
        en=history[j+1]['turn']-1 if j+1<len(history) else out.shape[1]
        base,f=inputs(ctx,(st,en,x,x,raw),anchor,median,False)
        value=ctx.work.normalize(np.exp(base+f@coef[bool(st)])) if coef is not None else x
        out[:,st:en]=value[:,None]
    return out

def run(ctx,config,budget,public,read_roots):
    """Exact live sampler; reader returns physical units only for paid coordinates."""
    B=Fraction(str(budget))
    if B<=0:raise ValueError('positive budget required')
    near=min(map(float,config['models']),key=lambda v:abs(np.log(v/float(B))))
    key=str(near);arm=pick(config['route'],float(B),public)
    branch=plans(ctx,config['package'])[arm]
    T=public['T'];events=[e['turn'] for e in public['events'] if 1<e['turn']<=T]
    times=[min(events,key=lambda t:(abs(t-branch.appointment),t))] if branch.appointment and events else []
    actions=[];hist=[];anchor=B*Fraction(str(branch.anchor))
    for j,turn in enumerate([1,*times]):
        mask=branch.fixed_roots if j else ctx.roots
        total=(B-anchor) if j else anchor
        shares=list(map(Fraction,ctx.allocation(float(total),mask,branch.refresh_weights if j else branch.weights)))
        if min(shares)<=0:raise ValueError('nonpositive allocation')
        eps=[total*s/sum(shares) for s in shares]
        names=[ctx.work.roots[i] for i in mask];values=read_roots(turn,names)
        if len(values)!=len(mask):raise ValueError('incorrect reader shape')
        raw=np.zeros((1,len(ctx.roots)));cal=raw.copy();evec=raw.copy()
        for i,v,e in zip(mask,values,eps):
            lo,hi=map(Fraction,[float(ctx.work.lo[i]),float(ctx.work.hi[i])])
            z=max(0,min(1,(Fraction(v)-lo)/(hi-lo)))
            ix=sample(z,e);raw[0,i]=ix/(GRID-1);evec[0,i]=float(e)
            cal[0,i]=ctx.marginal_grid(i,turn-1,float(e),branch.shrink)[ix]
        actions.append(dict(turn=turn,roots=names,epsilon=eps))
        if not j:
            ps=dict(anchor_cache=raw,anchor_service=cal,anchor_epsilon=evec,histories=hist,caches=np.empty((1,T,len(ctx.roots))))
        else:hist.append(dict(turn=turn,cache_after=raw,service=cal,selected=evec>0,actual_epsilon=evec))
    c=config['models'][key]['static' if arm=='static' else 'period']
    co={bool(int(k)):np.asarray(v) for k,v in c.items()}
    spent=sum(sum(a['epsilon']) for a in actions)
    assert spent<=B
    return dict(cache=estimate(ctx,ps,co,np.asarray(config['median']))[0],actions=actions,spent=spent,arm=arm)
