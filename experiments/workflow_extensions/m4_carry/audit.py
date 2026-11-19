"""Independent reconciliation of saved action ledgers against per-repeat metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from .run import ROOT, dump


def audit_output(out, data_dir=None):
    out=Path(out)
    cfg=json.loads((out/'config.json').read_text())
    split=json.loads((out/'split.json').read_text())
    selections=json.loads((out/'selections.json').read_text())
    records=data.load(data_dir or ROOT/'data_mine/finite_cache',cfg['templates'])
    bytraj={r['trajectory_id']:r for r in records}
    metrics={}
    for line in (out/'metrics.jsonl').open():
        r=json.loads(line)
        key=(r['template'],r['subject'],r['trajectory'],r['B'],r['method'],r['repeat'])
        assert key not in metrics
        metrics[key]=r
        assert split[r['subject']]=='test'
    groups={}
    checks=0
    for line in (out/'actions.jsonl').open():
        r=json.loads(line)
        key=(r['template'],r['subject'],r['trajectory'],r['B'],r['method'],r['repeat'])
        assert key in metrics
        rec=bytraj[r['trajectory']]
        # Stored vectors use Workload root order (not dict serialization order).
        from experiments.workflow_extensions.finite_cache.workload import Workload
        roots=Workload(r['template']).roots
        av=set(rec['initial_state'])
        legal={e['turn'] for e in rec['events']}
        assert set(r['selected_roots'])<=av
        if r['action_index']:
            assert r['turn'] in legal
            assert set(r['selected_roots'])<=set(r['planned_roots'])
        else:
            assert r['turn']==1 and set(r['selected_roots'])==av
        eps=np.array(r['actual_epsilon'])
        mask=np.array([root in r['selected_roots'] for root in roots])
        assert np.all(eps[mask]>0) and np.all(np.isfinite(eps))
        assert np.all(eps[~mask]==0)
        if r['action_index']:
            assert r['n_after']==r['n_before']-1
            if r['method'].startswith('M4-carry'):
                assert abs(r['eps_can']-r['R_before']/r['n_before'])<1e-10
            if mask.any():
                assert abs(eps.sum()-r['eps_can'])<1e-10
            assert abs(r['R_before']-eps.sum()-r['R_after'])<1e-10
        assert r['R_after']>=-1e-10
        g=groups.setdefault(key,dict(spent=0.,m=0,K=0,last_turn=0,last_R=None))
        assert g['last_turn']<r['turn']
        if g['last_R'] is not None:
            assert abs(g['last_R']-r['R_before'])<1e-10
        g['spent']+=float(eps.sum())
        g['m']+=bool(mask.any()) and r['action_index']>0
        g['K']+=r['action_index']>0
        g['last_turn'],g['last_R']=r['turn'],r['R_after']
        checks+=1
    assert groups.keys()==metrics.keys()
    for key,g in groups.items():
        r=metrics[key]
        assert abs(g['spent']-r['used'])<1e-10
        assert abs(r['spend_fraction']*r['B']-r['used'])<1e-10
        assert g['spent']<=r['B']+1e-10
        assert g['m']==r['m'] and g['K']==r['K']
        assert abs(g['last_R']+g['spent']-r['B'])<1e-10
    for t,bs in selections.items():
        for B,s in bs.items():
            assert s['M4']==s['M4-carry']
    stats=json.loads((out/'public_stats.json').read_text())
    for t,s in stats.items():
        assert all(split[str(subject)]=='train' for subject in s['train_subjects'])
    p=json.loads((out/'phase_a_complete.json').read_text())
    assert p['selections_complete'] and not p['test_errors_evaluated']
    max_pair_difference=0.
    paired_paths=0
    # This is an observation to audit, not a universal mechanism requirement:
    # for this run carry and fixed happen to coincide at Bmax. Always report
    # the measured difference instead of asserting coincidence for other configs.
    for key,r in metrics.items():
        if r['method']!='M4' or r['B']!=cfg['budget_grid']['high']:
            continue
        other=metrics[(*key[:4],'M4-carry',key[5])]
        for metric in ('mae','p95','used','m','projection_rate','post_stop_mae'):
            max_pair_difference=max(max_pair_difference,abs(r[metric]-other[metric]))
        paired_paths+=1
    dump(out/'paired_path_audit.json',dict(B=cfg['budget_grid']['high'],paired_paths=paired_paths,
                                         max_metric_difference=max_pair_difference,
                                         equal_within_1e10=max_pair_difference<1e-10))
    result=dict(passed=True,mechanism_paths=len(groups),action_records=checks,
                checks=['independent epsilon sum','positive released epsilon','exact vector mask','per-path total budget',
                        'carry current R/n','skip does not spend','opportunity decrements','strict legal times',
                        'roots causally available','M4/carry same plan','train-only stats','selection before test'])
    dump(out/'independent_audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    print(audit_output(p.parse_args().output))
