"""RootGuard-compatible task and transcript reconstruction metrics."""
import numpy as np

from ..finite_cache import data
from ..m6.attack import evaluate_attack
from .engine import GRID


def utility(ctx, record, result):
    work=ctx.work
    truth=work.evaluate(data.replay(record))
    service=work.from_cache(result['caches'])
    error=np.stack([abs(service[node]-truth[node]) for node in ctx.nodes],axis=-1)
    out=dict(mae=error[...,ctx.target].mean(axis=1),
             final_truth_abs=np.full(len(error),np.mean(abs(truth[work.target]))),
             p95=np.quantile(error[...,ctx.target],.95,axis=1),
             query_objective=(error*ctx.task_weights(record['T'],result['public']['queries'])).sum(axis=(1,2)),
             refreshes=result['refreshes'],roots=result['roots'],spent=result['spent'])
    labels=work.classes(truth[work.target])
    if labels is not None:
        predicted=work.classes(service[work.target])
        # RootGuard's NLR and TYG label sets are binary. The historical
        # ANEMIA and FIB4 comparisons retain the ordinal three-class metric.
        max_distance=1 if work.template in ('NLR','TYG') else 2
        out['rce']=100*abs(predicted-labels).mean(axis=1)/max_distance
        out['cmr']=100*(predicted!=labels).mean(axis=1)
    for j,node in enumerate(ctx.nodes):
        turns=[turn for turn,n in result['public']['queries'] if n==node]
        if turns:
            out['num:'+node]=error[:,turns,j].mean(axis=1)
            out['den:'+node]=np.full(len(error),np.mean(abs(truth[node][turns])))
    return out


def invert_marginal(ctx, service, last, eps, branch, selected):
    """Decode calibrated grid points from observer-visible roots and public eps."""
    observed=np.zeros_like(service)
    for r,i in zip(*np.nonzero(selected)):
        mapped=ctx.marginal_grid(i,int(last[r,i]),float(eps[r,i]),branch.shrink)
        if np.any(np.diff(mapped)<=0):
            raise ArithmeticError('Marginal grid mapping is not injective')
        y=service[r,i]
        index=int(np.searchsorted(mapped,y))
        pair=np.clip([index-1,index],0,GRID-1)
        j=pair[np.argmin(abs(mapped[pair]-y))]
        if abs(mapped[j]-y)>1e-8:
            raise ArithmeticError('Released value is not on the calibrated grid')
        observed[r,i]=j/(GRID-1)
    return observed


def attack_result(ctx, record, result, branch):
    """Reuse RootGuard's passive snapshot/history MAP on native grid likelihoods."""
    mapped={**result}
    if branch.calibration=='marginal':
        anchor=np.ones_like(result['anchor_cache'],dtype=bool)
        mapped['anchor_cache']=invert_marginal(ctx,result['anchor_service'],
            np.zeros_like(anchor,dtype=int),result['anchor_epsilon'],branch,anchor)
        np.testing.assert_allclose(mapped['anchor_cache'],result['anchor_cache'],atol=1e-14)
        mapped['histories']=[]
        previous=result['anchor_service']
        for h in result['histories']:
            observation=h['service']
            if branch.fusion:
                gain=h['fusion_gain']
                observation=previous+(observation-previous)/gain
            recovered=invert_marginal(ctx,observation,h['last'],h['last_eps'],branch,h['selected'])
            previous=h['service']
            np.testing.assert_allclose(recovered[h['selected']],h['cache_after'][h['selected']],atol=1e-14)
            mapped['histories'].append({**h,'cache_after':recovered})
        model='invertible marginal transcript with history fusion' if branch.fusion else 'invertible marginal transcript'
    elif branch.calibration=='joint':
        model='stronger raw transcript adversary for joint calibration'
    else:
        model='native observable transcript'
    attack_rows=evaluate_attack(record,mapped,ctx.work,ctx.attack_public,{'attack_grid_points':GRID})
    return [dict(row,observation_model=model) for row in attack_rows]
