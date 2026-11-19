"""RootGuard-style grid MAP with exact kernel likelihood and train-only priors.

History extension uses a factorized bounded random-walk model on public measured
roots; exact Viterbi on that declared grid/model, not exact clinical inference.
Cached answers never become extra independent observations. Seeds are not input.
"""
from .kernels import log_likelihood
from functools import lru_cache
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.special import logsumexp


def prior_scores(public,prior,m):
    grid=np.linspace(0,1,m);mu=np.array(public['mu']);std=np.maximum(public['root_std'],1/(m-1))
    score=np.zeros((len(mu),m)) if prior=='uniform' else -.5*((grid[None,:]-mu[:,None])/std[:,None])**2
    return score-logsumexp(score,axis=1)[:,None]


def reconstruct(rec,histories,anchor_cache,anchor_eps,public,kernels,cfg,prior='informed',history=False):
    """Inputs are public metadata plus actual releases; no truth/data replay."""
    m=cfg['attack_grid_points'];T=rec['T'];R,k=anchor_cache.shape;grid=np.linspace(0,1,m)
    initial=prior_scores(public,prior,m);releases={1:[(np.ones((R,k),bool),anchor_cache,anchor_eps,kernels)]}
    for h in histories:releases.setdefault(h['turn'],[]).append((h['selected'],h['cache_after'],h['actual_epsilon'],h['kernels']))
    public_events={e['turn']:e['measured_roots'] for e in rec['events']}
    out=np.empty((R,T,k));radius=np.ceil(np.array(public['max_train_jump'])*(m-1)).astype(int)
    radius=np.clip(radius,0,m-1)
    for i,root in enumerate(public['roots']):
        if not history:
            current=np.full(R,grid[np.argmax(initial[i])])
            for t in range(1,T+1):
                for selected,cache,eps,is_grid in releases.get(t,[]):
                    for r in np.where(selected[:,i])[0]:
                        ll=initial[i]+log_likelihood(cache[r,i],eps[r,i],is_grid[r,i],public,i,m)
                        current[r]=grid[int(np.argmax(ll))]
                out[:,t-1,i]=current
            continue
        scores=np.tile(initial[i],(R,1));stored=[];rad=int(radius[i])
        idx=np.arange(m);length=np.minimum(m-1,idx+rad)-np.maximum(0,idx-rad)+1
        for t in range(1,T+1):
            if t>1 and root in public_events.get(t,[]) and rad:
                scores=maximum_filter1d(scores-np.log(length)[None,:],size=2*rad+1,axis=1,mode='constant',cval=-np.inf)
            for selected,cache,eps,is_grid in releases.get(t,[]):
                for r in np.where(selected[:,i])[0]:scores[r]+=log_likelihood(cache[r,i],eps[r,i],is_grid[r,i],public,i,m)
            scores-=np.max(scores,axis=1)[:,None];stored.append(scores.copy())
        states=np.argmax(scores,axis=1)
        for t in range(T,0,-1):
            out[:,t-1,i]=grid[states]
            if t>1 and root in public_events.get(t,[]) and rad:
                for r in range(R):
                    lo=max(0,states[r]-rad);hi=min(m,states[r]+rad+1)
                    states[r]=lo+int(np.argmax(stored[t-2][r,lo:hi]-np.log(length[lo:hi])))
    return out


def evaluate_attack(rec,result,w,public,cfg):
    from experiments.workflow_extensions.finite_cache import data
    # Public view strips all values, available times and private source records.
    view=dict(T=rec['T'],events=[dict(turn=e['turn'],measured_roots=e['measured_roots']) for e in rec['events']])
    truth=data.replay(rec);rows=[]
    for prior,history in [('uniform',False),('informed',False),('informed',True)]:
        z=reconstruct(view,result['histories'],result['anchor_cache'],result['anchor_epsilon'],public,result['anchor_kernels'],cfg,prior,history)
        x=w.denormalize(z);err=abs(x-truth[None,:,:]);normalized=err/w.width
        for r in range(len(z)):
            rows.append(dict(repeat=r,attack='history_map' if history else 'snapshot_map',prior=prior,
                             root_mae=err[r].mean(axis=0).tolist(),root_truth_abs=np.mean(abs(truth),axis=0).tolist(),
                             initial_root_mae=err[r,0].tolist(),initial_root_truth_abs=abs(truth[0]).tolist(),
                             normalized_mae=float(normalized[r].mean()),
                             success_005=float(np.mean(normalized[r]<=.05)),
                             root_release_truth_abs=[float(np.mean([abs(truth[h['turn']-1,i]) for h in result['histories'] if h['selected'][r,i]]+[abs(truth[0,i])])) for i in range(len(w.roots))],
                             root_release_mae=[float(np.mean([err[r,h['turn']-1,i] for h in result['histories'] if h['selected'][r,i]]+[err[r,0,i]])) for i in range(len(w.roots))]))
    return rows
