"""Patient-equal ratio metrics, paired/nested CI, censored nonmonotone B_min."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from experiments.workflow_extensions.m4_carry.analysis import bootstrap_curves,q95,bmin,write_csv
from experiments.workflow_extensions.m4_carry.core import key_seed
from .run import METHODS


def interval(x):
    a=np.asarray(x);a=a[np.isfinite(a)]
    return q95(a) if len(a) else [None,None]


def ratio(a,b):
    return np.divide(a,b,out=np.full_like(a,np.nan),where=abs(b)>1e-14)


def fmt(x):return 'NA' if x is None or not np.isfinite(x) else f'{x:.3f}'


def tensors(rows,methods,budgets,subjects,R,field):
    g=defaultdict(list)
    for r in rows:
        v=r.get(field)
        if v is not None:g[r['method'],r['B'],r['subject'],r['repeat']].append(v)
    x=np.full((len(methods),len(budgets),len(subjects),R),np.nan)
    for mi,m in enumerate(methods):
        for bi,B in enumerate(budgets):
            for si,s in enumerate(subjects):
                for rep in range(R):
                    a=g.get((m,B,s,rep))
                    if a:x[mi,bi,si,rep]=np.mean(a)
    return x


def analyze(out):
    out=Path(out);cfg=json.loads((out/'config.json').read_text());stats=json.loads((out/'public_stats.json').read_text())
    targets=json.loads((out/'targets.json').read_text());budgets=cfg['evaluated_budgets'];R=cfg['repeats'];D=cfg['bootstrap']
    by=defaultdict(list)
    for ln in (out/'metrics.jsonl').open():
        r=json.loads(ln);by[r['template']].append(r)
    summary=[];paired=[];bmins=[];bmeffects=[]
    contrasts=[('M6',m) for m in METHODS if m!='M6']+[('M6-iso-cross','M6-iso-single-step'),('M6-iso-atomic','M6-iso-cross'),('M6-dynamic-fixed','M6-fixed-no-carry')]
    for t,rows in by.items():
        subjects=sorted({r['subject'] for r in rows});seed=key_seed(cfg['bootstrap_seed'],t)
        fields=['mae','final_truth_abs','query_loss','rce','cmr','p95','projection_rate','m','spend_fraction','MissRate','harmful_refresh','post_stop_mae','frac_veto',
                'danger_rate','danger_mae','frac_root_veto','grid_release_fraction','outside_work_grid_fraction','outside_work_truth_fraction']
        fields += [s+'__'+node for node in stats[t]['main']['nodes'] for s in ['node_num','node_den']]
        raw={};means={};pb={};nb={}
        for f in fields:
            x=tensors(rows,METHODS,budgets,subjects,R,f);raw[f]=x
            if np.isfinite(x).all():
                means[f]=x.mean(axis=(-1,-2));pb[f],nb[f]=bootstrap_curves(x,D,seed)
            else:
                # Entirely undefined RCE or undefined ratio/event denominators.
                count=np.isfinite(x).sum(axis=(-1,-2))
                means[f]=np.divide(np.nansum(x,axis=(-1,-2)),count,out=np.full(x.shape[:2],np.nan),where=count>0)
                # Event-ratio intervals are omitted rather than treating NA as zero.
        for dest in [means,pb,nb]:
            dest['wmape']=100*ratio(dest['mae'],dest['final_truth_abs'])
            for node in stats[t]['main']['nodes']:
                a='node_num__'+node;b='node_den__'+node
                if a in dest and b in dest:dest['node_wmape__'+node]=100*ratio(dest[a],dest[b])
            intermediate=[(j,n) for j,n in enumerate(stats[t]['main']['nodes']) if n!=__import__('experiments.workflow_extensions.finite_cache.workload',fromlist=['Workload']).Workload(t).target]
            q=stats[t]['main']['query_weights'];den=sum(q[j] for j,n in intermediate)
            if den:
                vals=[(q[j],dest['node_wmape__'+n]) for j,n in intermediate if 'node_wmape__'+n in dest]
                num=sum(weight*np.nan_to_num(v,nan=0.) for weight,v in vals)
                valid=sum(weight*np.isfinite(v) for weight,v in vals)
                dest['intermediate_wmape']=ratio(num,valid)
            else:dest['intermediate_wmape']=np.full_like(dest['mae'],np.nan)
        reported=['mae','wmape','rce','cmr','intermediate_wmape','query_loss','p95','projection_rate','m','spend_fraction','MissRate','harmful_refresh','post_stop_mae','frac_veto',
                  'danger_rate','danger_mae','frac_root_veto','grid_release_fraction','outside_work_grid_fraction','outside_work_truth_fraction']+[k for k in means if k.startswith('node_wmape__')]
        for mi,m in enumerate(METHODS):
            for bi,B in enumerate(budgets):
                r=dict(template=t,method=m,B=B,n_patients=len(subjects),repeats=R)
                for f in reported:
                    v=means[f][mi,bi];r[f]=float(v) if np.isfinite(v) else None
                    if f in pb:
                        for name,boot in [('patient',pb),('two_layer',nb)]:
                            lo,hi=interval(boot[f][mi,bi]);r[f+'_'+name+'_lo']=lo;r[f+'_'+name+'_hi']=hi
                summary.append(r)
        for a,b in contrasts:
            ai=METHODS.index(a);bi=METHODS.index(b)
            for f in ['mae','wmape','rce','intermediate_wmape','query_loss','p95','spend_fraction','m']:
                if f not in pb:continue
                for bj,B in enumerate(budgets):
                    diff=means[f][ai,bj]-means[f][bi,bj]
                    if not np.isfinite(diff):continue
                    r=dict(template=t,B=B,metric=f,method=a,reference=b,delta=float(diff))
                    for name,boot in [('patient',pb),('two_layer',nb)]:r[name+'_lo'],r[name+'_hi']=interval(boot[f][ai,bj]-boot[f][bi,bj])
                    paired.append(r)
        for scenario,ts in targets[t].items():
            for target in ['lo','hi']:
                tau=ts['tau_'+target];bm=bmin(means['mae'],budgets,tau)
                bs={name:bmin(boot['mae'].transpose(0,2,1),budgets,tau) for name,boot in [('patient',pb),('two_layer',nb)]}
                for mi,m in enumerate(METHODS):
                    r=dict(template=t,scenario=scenario,target=target,tau=tau,method=m,B_min=float(bm[mi]) if np.isfinite(bm[mi]) else None,reached=bool(np.isfinite(bm[mi])))
                    for name,x in bs.items():
                        lo,hi=q95(x[mi]);r[name+'_lo']=lo if np.isfinite(lo) else None;r[name+'_hi']=hi if np.isfinite(hi) else None
                        r[name+'_reach_rate']=float(np.isfinite(x[mi]).mean())
                    bmins.append(r)
                ai=METHODS.index('M6')
                for base in FOCUS:
                    bi=METHODS.index(base);both=np.isfinite(bm[ai])&np.isfinite(bm[bi]);r=dict(template=t,scenario=scenario,target=target,reference=base,ratio=float(bm[ai]/bm[bi]) if both else None,Saving=float(1-bm[ai]/bm[bi]) if both else None)
                    for name,x in bs.items():
                        ok=np.isfinite(x[ai])&np.isfinite(x[bi]);r[name+'_both_reach_rate']=float(ok.mean())
                        r[name+'_ratio_conditional_lo'],r[name+'_ratio_conditional_hi']=interval(x[ai,ok]/x[bi,ok])
                    bmeffects.append(r)
    write_csv(out/'summary.csv',summary);write_csv(out/'paired_summary.csv',paired);write_csv(out/'bmin.csv',bmins);write_csv(out/'bmin_paired.csv',bmeffects)
    attack_summary,attack_pairs=analyze_attacks(out,cfg,stats)
    endpoint_analysis(out,cfg,stats)
    ucb_analysis(out,summary,targets,budgets)
    from .report import report as write_report
    write_report(out,cfg,stats,summary,paired,bmins,attack_summary,attack_pairs)


FOCUS=['RG-static-exp','RG-periodic-exp','RG-static-lap','RG-periodic-lap','M4-carry','M5']


def analyze_attacks(out,cfg,stats):
    groups=defaultdict(list)
    for ln in (out/'attack_metrics.jsonl').open():
        r=json.loads(ln);groups[r['template'],r['attack'],r['prior']].append(r)
    summary=[];paired=[];budgets=cfg['evaluated_budgets'];methods=cfg['attack_methods'];R=cfg['repeats'];D=cfg['bootstrap']
    for (t,attack,prior),rows in groups.items():
        subjects=sorted({r['subject'] for r in rows});seed=key_seed(cfg['bootstrap_seed'],t)
        for i,root in enumerate(stats[t]['main']['roots']):
            for r in rows:r['num__'+root]=r['root_mae'][i];r['den__'+root]=r['root_truth_abs'][i]
        rootvals={};rootboots={};rootnested={}
        for root in stats[t]['main']['roots']:
            x=tensors(rows,methods,budgets,subjects,R,'num__'+root);y=tensors(rows,methods,budgets,subjects,R,'den__'+root)
            p,n=bootstrap_curves(x,D,seed);pd,nd=bootstrap_curves(y,D,seed)
            rootvals[root]=100*ratio(x.mean(axis=(-1,-2)),y.mean(axis=(-1,-2)))
            rootboots[root]=100*ratio(p,pd);rootnested[root]=100*ratio(n,nd)
        val=np.mean(list(rootvals.values()),axis=0);bp=np.mean(list(rootboots.values()),axis=0);bn=np.mean(list(rootnested.values()),axis=0)
        nx=tensors(rows,methods,budgets,subjects,R,'normalized_mae');succ=tensors(rows,methods,budgets,subjects,R,'success_005')
        for mi,m in enumerate(methods):
            for bi,B in enumerate(budgets):
                r=dict(template=t,attack=attack,prior=prior,B=B,method=m,root_macro_wmape=float(val[mi,bi]),normalized_mae=float(nx[mi,bi].mean()),success_005=float(succ[mi,bi].mean()),n_patients=len(subjects))
                for root,v in rootvals.items():r['wmape__'+root]=float(v[mi,bi])
                for name,boot in [('patient',bp),('two_layer',bn)]:r[name+'_lo'],r[name+'_hi']=interval(boot[mi,bi])
                summary.append(r)
        ai=methods.index('M6')
        for base in FOCUS:
            bi=methods.index(base)
            for bj,B in enumerate(budgets):
                r=dict(template=t,attack=attack,prior=prior,B=B,reference=base,delta_reconstruction_wmape=float(val[ai,bj]-val[bi,bj]))
                for name,boot in [('patient',bp),('two_layer',bn)]:r[name+'_lo'],r[name+'_hi']=interval(boot[ai,bj]-boot[bi,bj])
                paired.append(r)
    write_csv(out/'attack_summary.csv',summary);write_csv(out/'attack_paired.csv',paired)
    return summary,paired


def ucb_analysis(out,summary,targets,budgets):
    rows=[]
    for t,ts in targets.items():
        for scenario,ss in ts.items():
            for target in ['lo','hi']:
                tau=ss['tau_'+target]
                for method in METHODS:
                    group=[r for r in summary if r['template']==t and r['method']==method]
                    for mode in ['patient','two_layer']:
                        passing=[r['B'] for r in group if r.get('mae_'+mode+'_hi',float('inf'))<=tau]
                        rows.append(dict(template=t,scenario=scenario,target=target,tau=tau,method=method,
                                         UCB_type=mode,B_min=min(passing) if passing else None,reached=bool(passing),
                                         convention='pointwise 95% bootstrap upper endpoint; no simultaneous-curve coverage claim'))
    write_csv(out/'bmin_ucb.csv',rows)


def endpoint_analysis(out,cfg,stats):
    groups=defaultdict(list)
    for ln in (out/'frozen_attack_endpoints.jsonl').open():
        r=json.loads(ln)
        if r['attack']=='snapshot_map' and r['prior']=='informed':groups[r['template']].append(r)
    for ln in (out/'attack_metrics.jsonl').open():
        r=json.loads(ln)
        if r['method']=='M6' and r['B']==960 and r['attack']=='snapshot_map' and r['prior']=='informed':groups[r['template']].append(r)
    summary=[];paired=[];methods=cfg['attack_methods'];R=cfg['repeats'];D=cfg['bootstrap']
    for t,rows in groups.items():
        subjects=sorted({r['subject'] for r in rows});seed=key_seed(cfg['bootstrap_seed'],t)
        for endpoint,prefix in [('initial','initial_root'),('release','root_release')]:
            point=[];pbs=[];nbs=[]
            for i,root in enumerate(stats[t]['main']['roots']):
                for r in rows:r['num']=r[prefix+'_mae'][i];r['den']=r[prefix+'_truth_abs'][i]
                x=tensors(rows,methods,[960.],subjects,R,'num');y=tensors(rows,methods,[960.],subjects,R,'den')
                p,n=bootstrap_curves(x,D,seed);pd,nd=bootstrap_curves(y,D,seed)
                point.append(100*ratio(x.mean(axis=(-1,-2)),y.mean(axis=(-1,-2))))
                pbs.append(100*ratio(p,pd));nbs.append(100*ratio(n,nd))
            v=np.mean(point,axis=0)[:,0];p=np.mean(pbs,axis=0)[:,0];n=np.mean(nbs,axis=0)[:,0]
            for mi,m in enumerate(methods):
                row=dict(template=t,method=m,endpoint=endpoint,B=960.,macro_wmape=float(v[mi]))
                row['patient_lo'],row['patient_hi']=interval(p[mi]);row['two_layer_lo'],row['two_layer_hi']=interval(n[mi]);summary.append(row)
            ai=methods.index('M6')
            for mi,m in enumerate(methods):
                if m=='M6':continue
                row=dict(template=t,reference=m,endpoint=endpoint,B=960.,delta=float(v[ai]-v[mi]))
                row['patient_lo'],row['patient_hi']=interval(p[ai]-p[mi]);row['two_layer_lo'],row['two_layer_hi']=interval(n[ai]-n[mi]);paired.append(row)
    write_csv(out/'attack_endpoint_summary.csv',summary);write_csv(out/'attack_endpoint_paired.csv',paired)


