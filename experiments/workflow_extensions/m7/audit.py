"""Mechanism review: budget, causal reads, native parity, and MAP observations."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..finite_cache import data
from ..m4_carry.rootguard_mopt import simulate_baseline
from .engine import BASE_PERIODIC, BASE_STATIC, Context, branches, execute, public_view, run_live
from .metrics import attack_result, utility
from .routing import Route


def ledger(result, B, branch):
    spent=result['anchor_epsilon'].sum(axis=1)
    assert np.all(result['anchor_epsilon']>0)
    np.testing.assert_allclose(spent,B*branch.anchor,atol=1e-9,rtol=0)
    assert [h['turn'] for h in result['histories']]==result['schedule']
    releases=np.zeros(len(spent),int)
    roots=np.full(len(spent),result['anchor_epsilon'].shape[1])
    previous=result['anchor_cache']
    for i,h in enumerate(result['histories']):
        mask=h['selected']
        actual=h['actual_epsilon']
        K=len(result['schedule'])
        maximum=(B-spent)/(K-i) if branch.carry else (1-branch.anchor)*B/K
        np.testing.assert_allclose(actual.sum(axis=1),np.where(mask.any(axis=1),maximum,0.),rtol=0,atol=1e-9)
        assert np.all(actual[mask]>0) and np.all(actual[~mask]==0)
        np.testing.assert_array_equal(previous[~mask],h['cache_after'][~mask])
        spent+=actual.sum(axis=1)
        assert np.all(spent <= B+1e-9)
        releases+=mask.any(axis=1)
        roots+=mask.sum(axis=1)
        previous=h['cache_after']
    np.testing.assert_allclose(spent,result['spent'],atol=1e-9,rtol=0)
    np.testing.assert_array_equal(releases,result['refreshes'])
    np.testing.assert_array_equal(roots,result['roots'])


def audit(data_dir):
    templates=('MELD_NA','ANEMIA','FIB4','BUN_CR')
    source=data.load(data_dir,templates)
    for template in templates:
        examples=[rec for rec in source if rec['template']==template]
        refs, record=examples[:4], examples[4]
        ctx=Context(template,refs,99361116)
        # Exact native M-Opt on a common phase and deterministic paired stream.
        for branch in (BASE_STATIC,BASE_PERIODIC):
            native=execute(record,318.99,branch,ctx,'m7-audit',3,True)
            legacy=simulate_baseline(record,318.99,branch.period,'native_exp',ctx.mopt,
                                     {'noise_seed':ctx.seed},'m7-audit',3)
            np.testing.assert_array_equal(native['caches'],legacy['caches'])
            ledger(native,318.99,branch)
        for profile in ('portable','historical'):
            s,d=branches(profile,template,'workload')
            results=[execute(record,318.99,b,ctx,'m7-audit',3,True) for b in (s,d)]
            for b,res in zip((s,d),results):
                ledger(res,318.99,b)
                scores=utility(ctx,record,res)
                assert np.isfinite(scores['mae']).all() and np.isfinite(scores['query_objective']).all()
            # Modify only the final event. No earlier release, scoring, or cache
            # may change. The metadata and common RNG stream are held fixed.
            event=record['events'][-1]
            altered={**record,'events':[*record['events'][:-1],
                        {**event,'values':{k:2*v for k,v in event['values'].items()}}]}
            changed=execute(altered,318.99,d,ctx,'m7-audit',3,True)
            np.testing.assert_array_equal(results[1]['caches'][:,:event['turn']-1],
                                          changed['caches'][:,:event['turn']-1])
        # Full marginal observer inversion and RootGuard MAP reconstruction.
        probe=execute(record,318.99,replace(d,calibration='marginal',shrink=.85),ctx,'m7-audit',3,True)
        attacks=attack_result(ctx,record,probe,replace(d,calibration='marginal',shrink=.85))
        assert len(attacks)==9 and all(np.isfinite(row['normalized_mae']) for row in attacks)
        # A deployed static route reads only the anchor, never the dynamic side.
        known=data.replay(record)

        def read(turn,names):
            if turn!=1:
                raise AssertionError('Static routing read a later private value')
            return [known[turn-1,ctx.work.roots.index(name)] for name in names]

        route=Route({318.99:(s,d)},{318.99:False},'audit')
        served=run_live(ctx,route,318.99,public_view(record),read)
        assert len(served['actions'])==1 and abs(served['spent']-318.99)<1e-9
        print(template,'native parity, accounting, future invariance, MAP and single-branch release: pass',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data_mine/finite_cache'))
    audit(parser.parse_args().data)
