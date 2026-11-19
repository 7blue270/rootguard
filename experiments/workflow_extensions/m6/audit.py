"""Independent serialized-transcript checks; never used by selection/policy."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.core import paired_noise
from experiments.workflow_extensions.m4_carry.run import dump
from experiments.workflow_extensions.m4_carry.rootguard_mopt import uniforms
from .run import model_for,options,FROZEN


def audit_output(out,data_dir,source):
    out=Path(out);cfg=json.loads((out/'config.json').read_text());publics=json.loads((out/'public_stats.json').read_text())
    records={r['trajectory_id']:r for r in data.load(data_dir,cfg['templates'])}
    groups=defaultdict(list)
    for ln in (out/'actions.jsonl').open():
        row=json.loads(ln);groups[row['trajectory'],row['B'],row['method'],row['repeat']].append(row)
    maxdiff=0.;conversions=[];nrelease=0;ledger={}
    for (tr,B,method,rep),logs in groups.items():
        rec=records[tr];w=Workload(rec['template']);p=publics[rec['template']];model=model_for(method,p['main'],p['final_only'])
        z=w.normalize(data.replay(rec));noise=paired_noise(cfg['noise_seed'],'test',str(rec['subject_id']),tr,float(B),cfg['repeats'],rec['T'],tuple(w.roots))
        charge=0.;before=None;times=[]
        for a in logs:
            t=a['turn'];times.append(t);eps=np.array(a['actual_epsilon']);selected=eps>0;kernels=np.array(a['kernels'],bool)
            assert np.isfinite(eps).all() and np.all(eps>=0)
            expected=np.array(a['cache_after']) if before is None else before.copy()
            basis=np.array(model['mu']) if before is None else before
            danger=np.array(model['denominator_roots'])&(basis<=np.array(model['danger_threshold']))
            from experiments.workflow_extensions.m5.public import jacobian
            gradient=np.sum(np.array(model['query_weights'])[:,None]*abs(jacobian(w,basis))/np.array(model['node_scales'])[:,None],axis=0)
            danger|=gradient>np.array(model['a_quantile'])+1e-12
            if not options(method)['mixed']:danger[:]=False
            np.testing.assert_array_equal(kernels,danger)
            u=uniforms(noise[t,rep]);raw=z[t-1]+np.divide(noise[t,rep],eps,out=np.zeros_like(eps),where=selected)
            for i in np.where(selected)[0]:
                if kernels[i]:
                    grid=np.linspace(model['work_lo'][i],model['work_hi'][i],model['grid_points'][i])
                    score=-eps[i]*abs(grid-z[t-1,i])/2;cdf=np.cumsum(np.exp(score-logsumexp(score)));cdf[-1]=1
                    expected[i]=grid[min(np.searchsorted(cdf,u[i],side='right'),len(grid)-1)]
                else:expected[i]=np.clip(raw[i],0,1)
            diff=float(np.max(abs(expected-np.array(a['cache_after']))));maxdiff=max(maxdiff,diff);assert diff<1e-12
            if before is not None:
                np.testing.assert_array_equal(a['cache_before'],before)
                np.testing.assert_array_equal(a['candidate_epsilon'],eps)
                assert abs(a['left_before']-(B-charge))<1e-10
                assert abs(eps.sum()-(a['eps_can'] if selected.any() else 0))<1e-10
                if options(method)['groups']:
                    for g in model['groups']:assert np.all(selected[g]==selected[g[0]])
            charge+=float(eps.sum());assert charge<=B+1e-10
            left=a.get('left_after',a.get('R_after'));assert abs(left-(B-charge))<1e-10
            before=np.array(a['cache_after']);nrelease+=int(selected.sum())
        assert len(set(times))==len(times) and times==sorted(times)
        ledger[tr,B,method,rep]=charge
    for ln in (out/'metrics.jsonl').open():
        r=json.loads(ln)
        if r['method'] not in FROZEN:assert abs(r['used']-ledger[r['trajectory'],r['B'],r['method'],r['repeat']])<1e-10
    # Byte-exact previous results, rather than relying on rounded report tables.
    expected=[]
    for ln in (Path(source)/'metrics.jsonl').open():
        r=json.loads(ln)
        if r['method'] in FROZEN and r['B'] in cfg['evaluated_budgets'] and r['repeat']<cfg['repeats']:expected.append(ln)
    got=[]
    for ln in (out/'metrics.jsonl').open():
        if json.loads(ln)['method'] in FROZEN:got.append(ln)
    assert expected==got
    assignment=json.loads((out/'split.json').read_text())
    for t,p in publics.items():
        actual={str(r['subject_id']) for r in data.partition(list(records.values()),assignment,'train',t)}
        assert set(p['main']['train_subjects'])==actual
        for i,root in enumerate(p['main']['roots']):
            width=p['main']['work_hi'][i]-p['main']['work_lo'][i];m=p['main']['grid_points'][i]
            conversions.append(dict(template=t,root=root,m=m,work_width=width,
                                    epsilon_index_per_metric=width/(m-1) if m>1 else None))
    dump(out/'kernel_epsilon_conversion.json',conversions)
    result=dict(independent_paths=len(groups),root_publications=nrelease,max_release_difference=maxdiff,
                frozen_metric_rows=len(got),frozen_exact=True,ledger=True,positive_epsilon=True,
                causal_latest_values=True,kernel_selection_from_public_or_cache=True,atomic_groups=True,
                train_only=True,normalizer_includes_candidate_value=True,
                approximation='terminal-hold public particle risk, not optimal future-control planning')
    dump(out/'independent_audit.json',result)
    (out/'asserts.log').write_text('\n'.join(k+': PASS' for k in result if result[k] is True)+'\n')
    return result
