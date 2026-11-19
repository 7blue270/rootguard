"""Read-only ledger reconciliation and public-objective solver diagnostics.

Reference optimization is a numerical check, NOT a certified optimality gap,
not used for releases, P selection or test tuning. Original outputs preserved.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from .rootguard_mopt import ORIGINAL
from .run import dump


def audit(out):
    out=Path(out);stats=json.loads((out/'mopt_public_stats.json').read_text())
    metrics={}
    for ln in (out/'metrics.jsonl').open():
        r=json.loads(ln);k=(r['trajectory'],r['B'],r['method'],r['repeat']);assert k not in metrics;metrics[k]=r
    groups={};max_excess=0.
    for ln in (out/'actions.jsonl').open():
        r=json.loads(ln);k=(r['trajectory'],r['B'],r['method'],r['repeat']);assert k in metrics
        eps=np.asarray(r['actual_epsilon']);assert np.isfinite(eps).all() and (eps>=0).all()
        assert sum(eps>0)==len(r['selected_roots'])
        if r['method']!='M4-carry':
            assert (eps>0).all()
            np.testing.assert_allclose(eps,999*np.array(r['epsilon_grid']),rtol=0,atol=1e-10)
        g=groups.setdefault(k,dict(spent=0.,last=0,m=0))
        assert r['turn']>g['last'];g['last']=r['turn'];g['spent']+=eps.sum()
        g['m']+=r['action_index']>0 and len(r['selected_roots'])>0
        assert g['spent']<=r['B']+1e-10
        max_excess=max(max_excess,g['spent']-r['B'])
    assert metrics.keys()==groups.keys()
    for k,g in groups.items():
        r=metrics[k];assert abs(g['spent']-r['used'])<1e-10 and g['m']==r['m']
    failures=[]
    for ln in (out/'mopt_allocations.jsonl').open():
        r=json.loads(ln)
        if r['solver']['success']:continue
        info=stats[r['template']];roots=info['roots'];h=np.array([info['sensitivities'][x] for x in roots])
        # Derive original delta from the same frozen domain used by adapter.
        from experiments.workflow_extensions.finite_cache.workload import Workload
        dw=Workload(r['template']).width/999;t=np.full(len(h),499.5);G=r['total_grid'];lower=.001/G
        obj=lambda q:ORIGINAL['objective_absolute'](q*G,h,dw,t,1000)
        jac=lambda q:ORIGINAL['gradient_absolute'](q*G,h,dw,t,1000)*G
        q=np.array(r['epsilon_grid'])/G
        z=np.maximum(q-lower,0);projected=np.full(len(q),lower)+(1-len(q)*lower)*z/z.sum() if z.sum()>0 else np.full(len(q),1/len(q))
        refs=[]
        for start in [projected,np.full(len(q),1/len(q))]:
            ref=minimize(obj,start,jac=jac,bounds=[(lower,1.)]*len(q),constraints={'type':'eq','fun':lambda a:a.sum()-1,'jac':lambda a:np.ones(len(a))},method='SLSQP',options={'ftol':1e-10,'maxiter':1000})
            if ref.success and np.min(ref.x)>=lower-1e-9 and abs(ref.x.sum()-1)<1e-8:refs.append(ref)
        if refs:
            ref=min(refs,key=lambda x:obj(x.x));base=float(obj(ref.x));den=max(abs(base),1e-12)
            item=dict(template=r['template'],total_metric=r['total_metric'],reference_converged=True,
                      original_objective=float(obj(q)),projected_objective=float(obj(projected)),reference_objective=base,
                      relative_returned_difference=(float(obj(q))-base)/den,
                      relative_projected_excess=(float(obj(projected))-base)/den,
                      original_floor_deficit_metric=float(max(0,.001-min(r['epsilon_grid']))*999))
        else:item=dict(template=r['template'],total_metric=r['total_metric'],reference_converged=False)
        failures.append(item)
    result=dict(passed=True,paths=len(groups),max_budget_excess=max_excess,epsilon_unit_reconciled=True,
                solver_non_success_calls=len(failures),reference_unresolved=sum(not r['reference_converged'] for r in failures),
                solver_audit='two feasible share-parameterized SLSQP starts on the SAME original public moment objective; no releases or selections changed; not an optimality certificate',
                max_abs_relative_returned_objective_difference=max([abs(r['relative_returned_difference']) for r in failures if r['reference_converged']],default=0),
                max_relative_projected_excess=max([r['relative_projected_excess'] for r in failures if r['reference_converged']],default=0),
                max_floor_deficit_metric=max([r['original_floor_deficit_metric'] for r in failures if r['reference_converged']],default=0))
    dump(out/'independent_audit.json',result);dump(out/'solver_reference_audit.json',failures)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);audit(p.parse_args().output)
