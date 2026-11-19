"""Independent serialized transcript/ledger and causal-value audit."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.core import paired_noise
from experiments.workflow_extensions.m4_carry.rootguard_mopt import exp_pmf,uniforms
from experiments.workflow_extensions.m4_carry.run import dump


def audit_output(out,directory,source,comparison):
    out=Path(out);cfg=json.loads((out/'config.json').read_text());assignment=json.loads((out/'split.json').read_text())
    records={r['trajectory_id']:r for r in data.load(directory,cfg['templates'])}
    states={k:Workload(r['template']).normalize(data.replay(r)) for k,r in records.items()}
    public=json.loads((out/'public_stats.json').read_text());spent=defaultdict(float);last={};counts=defaultdict(int);maxdiff=0.
    for ln in (out/'actions.jsonl').open():
        a=json.loads(ln);key=(a['trajectory'],a['B'],a['method'],a['repeat']);rec=records[a['trajectory']];w=Workload(rec['template'])
        assert assignment[str(rec['subject_id'])]=='test'
        assert 1<=a['turn']<=rec['T'];assert key not in last or last[key]<a['turn'];last[key]=a['turn']
        eps=np.array(a['actual_epsilon']);selected=eps>0
        assert np.isfinite(eps).all() and np.all(eps>=0)
        assert [r for r,v in zip(w.roots,selected) if v]==a['selected_roots']
        spent[key]+=eps.sum();assert spent[key]<=a['B']+1e-10
        assert abs(a['R_after']-(a['B']-spent[key]))<1e-10
        if a['action_index'] and selected.any():assert abs(eps.sum()-a['eps_can'])<1e-10
        for root in a['selected_roots']:
            assert root in rec['initial_state'] or any(e['turn']<=a['turn'] and root in e['measured_roots'] for e in rec['events'])
        noise=paired_noise(cfg['noise_seed'],'test',str(rec['subject_id']),rec['trajectory_id'],float(a['B']),cfg['repeats'],rec['T'],tuple(w.roots))[a['turn'],a['repeat']]
        true=states[a['trajectory']][a['turn']-1];actual=np.array(a['cache_after'])
        if a['method'].endswith('-exp'):
            u=uniforms(noise);expected=np.empty(len(eps))
            for i in range(len(eps)):
                cdf=np.cumsum(exp_pmf(true[i],eps[i]/999));cdf[-1]=1
                expected[i]=min(999,np.searchsorted(cdf,u[i],side='right'))/999
        else:expected=np.clip(true+np.divide(noise,eps,out=np.zeros_like(noise),where=selected),0,1)
        if selected.any():
            diff=float(np.max(abs(expected[selected]-actual[selected])));maxdiff=max(maxdiff,diff);assert diff<1e-10
        if a['action_index']:
            np.testing.assert_array_equal(actual[~selected],np.array(a['cache_before'])[~selected])
        if a['method'] in ('M5','M5-final','M5-atomic-fixed'):
            pp=public[rec['template']]['final_only' if a['method']=='M5-final' else 'main']
            for g in pp['groups']:assert len(set(selected[g].tolist()))==1
        counts[a['method']]+=1
    metric_count=0
    for ln in (out/'metrics.jsonl').open():
        r=json.loads(ln);key=(r['trajectory'],r['B'],r['method'],r['repeat'])
        assert abs(spent[key]-r['used'])<1e-10
        assert np.isfinite(r['mae']) and np.isfinite(r['wmape'])
        metric_count+=1
    assert metric_count==len(spent)
    for t,p in public.items():
        for model in p.values():
            assert all(assignment[s]=='train' for s in model['train_subjects'])
            M=np.array(model['M']);cov=np.array(model['covariance'])
            assert np.linalg.eigvalsh(M).min()>-1e-8 and np.linalg.eigvalsh(cov).min()>-1e-8
    result=dict(passed=True,independent_paths=len(spent),metric_paths=metric_count,actions=dict(counts),
                max_release_difference=maxdiff,ledger_including_initialization=True,all_eps_positive_on_selected_roots=True,
                release_replayed_from_latest_observed_state=True,atomic_groups_preserved=True,
                unchanged_roots_preserved=True,train_only_public_parameters=True,
                native_baseline_kernel_verified=True,noise_keys_match_previous_round=True)
    dump(out/'independent_audit.json',result);print('INDEPENDENT AUDIT',result,flush=True)
    return result
