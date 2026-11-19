"""RootGuard-style grid MAP with exact kernel likelihood and train-only priors.

History extension uses a factorized bounded random-walk model on public measured
roots; exact Viterbi on that declared grid/model, not exact clinical inference.
Cached answers never become extra independent observations. Seeds are not input.
"""
from functools import lru_cache
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.special import logsumexp


@lru_cache(maxsize=6000)
def exp_logZ(epsilon,m):
    # Candidate roots at integer index positions; original PMF normalization.
    a=epsilon/(2*(m-1));idx=np.arange(m)
    def geom(n):
        return np.exp(-a)*(-np.expm1(-a*n))/(-np.expm1(-a))
    return np.log1p(geom(idx)+geom(m-1-idx))


def log_likelihood(observation,epsilon,kernel,m):
    grid=np.linspace(0,1,m)
    if kernel=='native_exp':
        return -epsilon/2*abs(grid-observation)-exp_logZ(float(epsilon),m)
    if kernel!='laplace':raise ValueError(kernel)
    # Clip masses at 0/1 and interior density differ by constants independent of
    # the candidate root. Include them nevertheless for likelihood tests.
    if observation<=0:return np.log(.5)-epsilon*grid
    if observation>=1:return np.log(.5)-epsilon*(1-grid)
    return np.log(epsilon/2)-epsilon*abs(grid-observation)


def prior_scores(public,prior,m):
    grid=np.linspace(0,1,m);mu=np.array(public['mu']);std=np.maximum(public['root_std'],1/(m-1))
    score=np.zeros((len(mu),m)) if prior=='uniform' else -.5*((grid[None,:]-mu[:,None])/std[:,None])**2
    return score-logsumexp(score,axis=1)[:,None]


def reconstruct(rec,histories,anchor_cache,anchor_eps,public,kernel,cfg,prior='informed',history=False):
    """Inputs are public metadata plus actual releases; no truth/data replay."""
    m=cfg['attack_grid_points'];T=rec['T'];R,k=anchor_cache.shape;grid=np.linspace(0,1,m)
    initial=prior_scores(public,prior,m);releases={1:[(np.ones((R,k),bool),anchor_cache,anchor_eps)]}
    for h in histories:releases.setdefault(h['turn'],[]).append((h['selected'],h['cache_after'],h['actual_epsilon']))
    public_events={e['turn']:e['measured_roots'] for e in rec['events']}
    out=np.empty((R,T,k));radius=np.ceil(np.array(public['max_train_jump'])*(m-1)).astype(int)
    radius=np.clip(radius,0,m-1)
    for i,root in enumerate(public['roots']):
        if not history:
            current=np.full(R,grid[np.argmax(initial[i])])
            for t in range(1,T+1):
                for selected,cache,eps in releases.get(t,[]):
                    for r in np.where(selected[:,i])[0]:
                        ll=initial[i]+log_likelihood(cache[r,i],eps[r,i],kernel,m)
                        current[r]=grid[int(np.argmax(ll))]
                out[:,t-1,i]=current
            continue
        scores=np.tile(initial[i],(R,1));stored=[];rad=int(radius[i])
        idx=np.arange(m);length=np.minimum(m-1,idx+rad)-np.maximum(0,idx-rad)+1
        for t in range(1,T+1):
            if t>1 and root in public_events.get(t,[]) and rad:
                scores=maximum_filter1d(scores-np.log(length)[None,:],size=2*rad+1,axis=1,mode='constant',cval=-np.inf)
            for selected,cache,eps in releases.get(t,[]):
                for r in np.where(selected[:,i])[0]:scores[r]+=log_likelihood(cache[r,i],eps[r,i],kernel,m)
            scores-=np.max(scores,axis=1)[:,None];stored.append(scores.copy())
        states=np.argmax(scores,axis=1)
        for t in range(T,0,-1):
            out[:,t-1,i]=grid[states]
            if t>1 and root in public_events.get(t,[]) and rad:
                for r in range(R):
                    lo=max(0,states[r]-rad);hi=min(m,states[r]+rad+1)
                    states[r]=lo+int(np.argmax(stored[t-2][r,lo:hi]-np.log(length[lo:hi])))
    return out


def evaluate_attack(rec,result,w,public,kernel,cfg):
    from experiments.workflow_extensions.finite_cache import data
    # Public view strips all values, available times and private source records.
    view=dict(T=rec['T'],events=[dict(turn=e['turn'],measured_roots=e['measured_roots']) for e in rec['events']])
    truth=data.replay(rec);rows=[]
    for prior,history in [('uniform',False),('informed',False),('informed',True)]:
        z=reconstruct(view,result['histories'],result['anchor_cache'],result['anchor_epsilon'],public,kernel,cfg,prior,history)
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
