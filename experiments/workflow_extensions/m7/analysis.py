"""Paired patient-plus-noise summaries with ratio-of-means wMAPE."""
from collections import defaultdict

import numpy as np

from ..m4_carry.core import key_seed
from ..m4_carry.analysis import write_csv

MAPS=(('snapshot_map','informed'),('history_map','informed'),('snapshot_map','uniform'))
UTILITY=('mae','wmape','rce','cmr','intermediate_wmape','final_query_wmape','p95',
         'query_objective','refreshes','roots','spent')


def combine_attack(row, attacks, roots):
    for attack in attacks:
        label=attack['attack']+'_'+attack['prior']
        for i,root in enumerate(roots):
            row[f'{label}:num:{root}']=attack['root_mae'][i]
            row[f'{label}:den:{root}']=attack['root_truth_abs'][i]
        row[f'{label}_normalized_mae']=attack['normalized_mae']
        row[f'{label}_success_005']=100*attack['success_005']
        for scope,numerator,denominator in [
            ('initial','initial_root_mae','initial_root_truth_abs'),
            ('release','root_release_mae','root_release_truth_abs'),
        ]:
            if numerator not in attack or denominator not in attack:
                continue
            for i,root in enumerate(roots):
                row[f'{label}_{scope}:num:{root}']=attack[numerator][i]
                row[f'{label}_{scope}:den:{root}']=attack[denominator][i]


def mean_valid(values,axis):
    count=np.isfinite(values).sum(axis=axis)
    return np.divide(np.nansum(values,axis=axis),count,out=np.full_like(count,np.nan,dtype=float),where=count>0)


def ratio(a,b):
    return np.divide(100*a,b,out=np.full_like(b,np.nan,dtype=float),where=np.isfinite(b)&(b>1e-14))


def derived(fields,ctx):
    result={k:fields[k] for k in UTILITY if k in fields}
    result['wmape']=ratio(fields['mae'],fields['final_truth_abs'])
    nodes=[]
    for node,weight in zip(ctx.nodes,ctx.query_weights):
        a,b='num:'+node,'den:'+node
        if a not in fields:
            continue
        value=ratio(fields[a],fields[b])
        result['node_wmape:'+node]=value
        if node==ctx.work.target:
            result['final_query_wmape']=value
        elif weight>0:
            nodes.append((weight,value))
    if nodes:
        num=sum(w*np.nan_to_num(v,nan=0.) for w,v in nodes)
        den=sum(w*np.isfinite(v) for w,v in nodes)
        result['intermediate_wmape']=np.divide(num,den,out=np.full_like(den,np.nan),where=den>0)
    for attack,prior in MAPS:
        label=attack+'_'+prior
        for metric in ('normalized_mae','success_005'):
            key=f'{label}_{metric}'
            if key in fields:
                result[key]=fields[key]
        for scope in ('','_initial','_release'):
            name=label+scope
            if f'{name}:num:{ctx.work.roots[0]}' in fields:
                root_values=[ratio(fields[f'{name}:num:{root}'],fields[f'{name}:den:{root}'])
                             for root in ctx.work.roots]
                result[f'{name}_root_wmape']=mean_valid(np.stack(root_values),axis=0)
    return result


def quantile(a):
    a=np.asarray(a)
    good=a[np.isfinite(a)]
    return np.quantile(good,[.025,.975]).tolist() if len(good) else [None,None]


def value(a):
    return float(a) if np.isfinite(a) else None


def summarize(outputs,out,contexts,n_boot):
    cells=defaultdict(dict)
    for r in outputs:
        cells[r['template'],r['B']].setdefault(r['method'],{})[r['subject'],r['repeat']]=r
    summaries,comparisons,aggregates=[],[],[]
    curve=defaultdict(list)
    for (t,B),methods in cells.items():
        ctx=contexts[t]
        keys=sorted(methods['M7'])
        subjects=sorted({s for s,_ in keys})
        reps=sorted({r for _,r in keys})
        rng=np.random.default_rng(key_seed('m7-bootstrap',t))
        patient=rng.integers(len(subjects),size=(n_boot,len(subjects),1))
        noise=rng.integers(len(reps),size=(n_boot,len(subjects),len(reps)))
        estimators={}
        for method,rows in methods.items():
            if set(rows)!=set(keys):
                raise ValueError(f'Unmatched prior/reference paths for {t}, {B}, {method}')
            numeric=sorted({k for r in rows.values() for k,v in r.items()
                            if isinstance(v,(int,float)) and k not in ('B','repeat')})
            panels=np.array([[
                [np.nan if rows[s,r].get(k) is None else rows[s,r][k] for k in numeric]
                for r in reps] for s in subjects],float)
            point=dict(zip(numeric,mean_valid(panels,axis=(0,1))))
            draws=dict(zip(numeric,mean_valid(panels[patient,noise],axis=(1,2)).T))
            p,z=derived(point,ctx),derived(draws,ctx)
            estimators[method]=p,z
            for metric,estimate in p.items():
                lower,upper=quantile(z[metric])
                summaries.append(dict(template=t,B=B,method=method,metric=metric,estimate=value(estimate),
                                      lo95=lower,hi95=upper,patients=len(subjects),repeats=len(reps)))
        pairs=[('M7',method) for method in methods if method!='M7']
        pairs += [('M7-S','RG-static-exp'),('M7-D','RG-periodic-exp'),
                  ('M7-D','RG-periodic-cal'),('M7-D','M7-D-final')]
        for method,ref in pairs:
            p,z=estimators[method]
            q,y=estimators[ref]
            for metric in p.keys()&q.keys():
                if not np.isfinite(p[metric]) or not np.isfinite(q[metric]):
                    continue
                delta=p[metric]-q[metric]
                draws=z[metric]-y[metric]
                lower,upper=quantile(draws)
                k=(t,method,ref,metric)
                curve[k].append((delta,draws))
                comparisons.append(dict(template=t,B=B,method=method,reference=ref,metric=metric,
                    estimate=float(p[metric]),reference_estimate=float(q[metric]),delta=float(delta),
                    lo95=lower,hi95=upper,patients=len(subjects),repeats=len(reps)))
    for (t,method,ref,metric),pairs in sorted(curve.items()):
        lower,upper=quantile(np.mean([z for _,z in pairs],axis=0))
        aggregates.append(dict(template=t,method=method,reference=ref,metric=metric,
                       delta=float(np.mean([p for p,_ in pairs])),lo95=lower,hi95=upper,budgets=len(pairs)))
    for name,rows in [('summary',summaries),('paired',comparisons),('across_budgets',aggregates)]:
        write_csv(out/f'{name}.csv',rows)
