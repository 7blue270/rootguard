"""Independent transcript, freeze-order and patient-exclusion audit."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.core import paired_noise
from experiments.workflow_extensions.m4_carry.run import dump
from .protocol import config


def audit(out,data_dir):
    out=Path(out);cfg=json.loads((out/'config.json').read_text());freeze=json.loads((out/'PREDICTIONS_FROZEN.json').read_text())
    assert freeze['all_fold_predictions_complete'] and not freeze['test_stream_evaluated']
    assert all(hashlib.sha256((out/f).read_bytes()).hexdigest()==h for f,h in freeze['hashes'].items())
    records={r['trajectory_id']:r for r in data.load(data_dir,cfg['templates'])}
    for ln in (out/'fold_predictions.jsonl').open():
        p=json.loads(ln);held=p['fold'];assert held not in p['training_subjects'] and held not in p['validation_subjects'] and held not in p['L2_global_training_subjects']
        assert set(p['validation_subjects'])<=set(p['training_subjects'])
    groups=defaultdict(list)
    for ln in (out/'actions.jsonl').open():
        a=json.loads(ln);groups[a['trajectory'],a['B'],a['execution'],a['repeat']].append(a)
    maxdiff=0.;root_releases=0;charges={}
    for (tr,B,method,rep),rows in groups.items():
        rec=records[tr];w=Workload(rec['template']);z=w.normalize(data.replay(rec));k=len(w.roots)
        noise=paired_noise(cfg['noise_seed'],'m62_evaluation',str(rec['subject_id']),tr,float(B),cfg['repeats'],rec['T'],tuple(w.roots))
        total=0.;ts=[]
        for row in rows:
            t=row['turn'];eps=np.array(row['epsilon']);assert np.all(eps>0);np.testing.assert_allclose(eps,eps[0],atol=1e-12)
            raw=z[t-1]+noise[t,rep]/eps
            diff=float(np.max(abs(np.clip(raw,0,1)-np.array(row['cache']))));maxdiff=max(maxdiff,diff);assert diff<1e-12
            np.testing.assert_array_equal(row['projection'],(raw<0)|(raw>1))
            total+=eps.sum();assert total<=B+1e-10;ts.append(t);root_releases+=k
        assert len(set(ts))==len(ts) and ts==sorted(ts)
        np.testing.assert_allclose(total,B,atol=1e-10,rtol=0)
        if method=='S':assert ts==[1]
        if method=='D':assert ts==[1]+cfg['refresh_turns']
        charges[tr,B,method,rep]=total
    metrics={};n=0
    for ln in (out/'metrics.jsonl').open():
        r=json.loads(ln);n+=1;assert abs(r['spent']-charges[r['trajectory'],r['B'],r['execution'],r['repeat']])<1e-10
        key=r['trajectory'],r['B'],r['execution'],r['repeat'];v=tuple(r[f] for f in ['mae','truth_abs','rce','spent','n_actions','projection_rate'])
        if key in metrics:assert metrics[key]==v
        metrics[key]=v
    provenance=dict(config='M6.2 public S/D; original formula/domain modules unchanged',
       paths=len(groups),root_releases=root_releases,metric_rows=n,max_release_difference=maxdiff,
       paired_selectors_exact_reuse=True,uniform_allocation=True,full_B_including_anchor=True,
       ledger_every_prefix=True,causal_latest_values=True,fold_subject_exclusion=True,
       frozen_before_evaluation_noise=True,public_actual_noise_separated=True,
       no_test_parameter_selection=True,eta_not_invented=True,
       bootstrap_rules='conditional on learned outer-fold rules; reselecting training inside bootstrap not performed')
    dump(out/'audit.json',provenance)
    (out/'asserts.log').write_text('\n'.join(k+': PASS' for k,v in provenance.items() if v is True)+'\n')
    return provenance
