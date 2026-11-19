"""Global patient-bound paired bootstrap; raw MAE is never pooled by template."""
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.workflow_extensions.m4_carry.analysis import write_csv
from experiments.workflow_extensions.m4_carry.run import dump
from .protocol import SELECTORS


def average(x,axis):
    n=np.isfinite(x).sum(axis=axis)
    return np.divide(np.nansum(x,axis=axis),n,out=np.full(np.shape(n),np.nan),where=n>0)


def interval(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return [float(v) for v in np.quantile(x,[.025,.975])] if len(x) else [None,None]


def ratio(x,y):return np.divide(x,y,out=np.full_like(np.asarray(x,float),np.nan),where=abs(y)>1e-12)


def bootstrap(x,patient_draws,repeat_draws):
    # [selector,template,budget,global_subject,repeat]. Missing template/subject
    # cells stay NaN. A subject's multiplicity/repeat draw is bound everywhere.
    D=len(patient_draws);p=np.empty(x.shape[:3]+(D,));n=np.empty_like(p);means=average(x,-1)
    for d,idx in enumerate(patient_draws):
        p[...,d]=average(means[...,idx],-1)
        selected=x[...,idx,:]
        selected=np.take_along_axis(selected,repeat_draws[d][None,None,None,:,:],axis=-1)
        n[...,d]=average(selected,(-1,-2))
    return p,n


def add_ci(row,name,point,p,n):
    row[name]=float(point) if np.isfinite(point) else None
    for label,x in [('patient',p),('two_layer',n)]:row[name+'_'+label+'_lo'],row[name+'_'+label+'_hi']=interval(x)


def analyze(out):
    started=time.perf_counter();out=Path(out);cfg=json.loads((out/'config.json').read_text())
    ts=cfg['templates'];bs=cfg['budgets'];R=cfg['repeats'];D=cfg['bootstrap'];methods=cfg['selectors']
    records=[json.loads(s) for s in (out/'metrics.jsonl').open()];subjects=sorted({r['subject'] for r in records})
    fields=['mae','truth_abs','rce','p95','spent','n_actions','n_root_releases','projection_rate']
    shapes=(len(methods),len(ts),len(bs),len(subjects),R)
    raw={field:np.full(shapes,np.nan) for field in fields};indices=[{s:i for i,s in enumerate(a)} for a in [methods,ts,bs,subjects]]
    sums=defaultdict(list)
    for row in records:
        ix=tuple(mapping[row[key]] for mapping,key in zip(indices,['selector','template','B','subject']))+(row['repeat'],)
        for field in fields:
            if row[field] is not None:sums[field,ix].append(row[field])
    for (field,ix),values in sums.items():raw[field][ix]=np.mean(values)
    rng=np.random.default_rng(cfg['bootstrap_seed']);pd=rng.integers(len(subjects),size=(D,len(subjects)));rd=rng.integers(R,size=(D,len(subjects),R))
    np.savez_compressed(out/'bootstrap_indices.npz',subjects=np.array(subjects),patient=pd,repeat=rd)
    points={f:average(x,(-1,-2)) for f,x in raw.items()};patient={};nested={}
    for field in ['mae','truth_abs','rce','p95','projection_rate']:
        patient[field],nested[field]=bootstrap(raw[field],pd,rd)
    for dest in [points,patient,nested]:dest['wmape']=100*ratio(dest['mae'],dest['truth_abs'])
    S=methods.index('S');Q=methods.index('D');Default=methods.index('Default')
    oracle=np.minimum(points['mae'][S],points['mae'][Q]);oracle_p=np.minimum(patient['mae'][S],patient['mae'][Q]);oracle_n=np.minimum(nested['mae'][S],nested['mae'][Q])
    summaries=[];effects=[];cells=[];aggregate=[];bmins=[];crossings=[]
    frozen=[json.loads(s) for s in (out/'fold_predictions.jsonl').open()]
    public=[json.loads(s) for s in (out/'public_predictions.jsonl').open()]
    main={(r['level'],r['template'],r['B']):r for r in public if r['delta']==cfg['delta'] and r['component']==cfg['component'] and r['model']==cfg['primary_model']}
    for ti,t in enumerate(ts):
        n_patients=sum(any(r['template']==t and r['subject']==s for r in records) for s in subjects)
        # Global patient index arrays are shared; no sample-size fiction from 6
        # budgets, 20 mechanisms, or patients repeated across four templates.
        for bi,B in enumerate(bs):
            gain=points['mae'][S,ti,bi]-points['mae'][Q,ti,bi]
            gp=patient['mae'][S,ti,bi]-patient['mae'][Q,ti,bi];gn=nested['mae'][S,ti,bi]-nested['mae'][Q,ti,bi]
            pci=interval(gp);nci=interval(gn)
            true_side='D' if pci[0]>0 and nci[0]>0 else 'S' if pci[1]<0 and nci[1]<0 else 'undecided'
            folds=[r for r in frozen if r['template']==t and r['B']==B]
            for mi,method in enumerate(methods):
                row=dict(template=t,B=B,selector=method,n_patients=n_patients,repeats=R)
                for field in fields+['wmape']:
                    if field in patient:add_ci(row,field,points[field][mi,ti,bi],patient[field][mi,ti,bi],nested[field][mi,ti,bi])
                    else:row[field]=float(points[field][mi,ti,bi]) if np.isfinite(points[field][mi,ti,bi]) else None
                row['spent_fraction']=row['spent']/B;summaries.append(row)
                if method in ['S','D','B2-val']:continue
                row=dict(template=t,B=B,selector=method,n_patients=n_patients,true_side=true_side,
                         mu_S=float(points['mae'][S,ti,bi]),mu_D=float(points['mae'][Q,ti,bi]),
                         loss_ratio_true=float(points['mae'][S,ti,bi]/points['mae'][Q,ti,bi]),
                         D_fraction=float(np.mean([r['choices'][method]=='D' for r in folds])),
                         known_correct_fraction=None if true_side=='undecided' else float(np.mean([r['choices'][method]==true_side for r in folds])))
                add_ci(row,'G_true',gain,gp,gn)
                levelkey={'L0':'G_L0','L1':'G_L1','L2':'G_L2','Local-L0':'G_local_L0','Local-L1':'G_local_L1','Local-L2':'G_local_L2','TrainLookup':'G_train_lookup'}.get(method)
                row['predicted_gain']=float(np.mean([r[levelkey] for r in folds])) if levelkey else None
                unique=sorted({r['choices'][method] for r in folds});row['predicted_side']=unique[0] if len(unique)==1 else 'mixed_outer_folds'
                # For cross-fitted policies the aggregate excess can be negative
                # (patient-specific fold choices can beat either fixed side).
                # Do not silently clip it or mislabel it nonnegative regret.
                add_ci(row,'cell_oracle_excess',points['mae'][mi,ti,bi]-oracle[ti,bi],patient['mae'][mi,ti,bi]-oracle_p[ti,bi],nested['mae'][mi,ti,bi]-oracle_n[ti,bi])
                row['patient_oracle_regret']=float(average(average(raw['mae'][mi,ti,bi],-1)-np.minimum(average(raw['mae'][S,ti,bi],-1),average(raw['mae'][Q,ti,bi],-1)),-1))
                row['oracle_note']='cell oracle uses post-hoc pooled means; excess may be signed for fold-varying selectors; patient oracle is optimistic and not deployable'
                cells.append(row)
                for base in ['S','D','Default','TrainLookup','B2-val']:
                    j=methods.index(base);rr=dict(template=t,B=B,selector=method,reference=base)
                    x=points['mae'][j,ti,bi]-points['mae'][mi,ti,bi];xp=patient['mae'][j,ti,bi]-patient['mae'][mi,ti,bi];xn=nested['mae'][j,ti,bi]-nested['mae'][mi,ti,bi]
                    add_ci(rr,'utility_gain',x,xp,xn)
                    add_ci(rr,'gain_percent',100*ratio(x,points['mae'][j,ti,bi]),100*ratio(xp,patient['mae'][j,ti,bi]),100*ratio(xn,nested['mae'][j,ti,bi]))
                    effects.append(rr)
        # Equal budget weights, but never average MAE across different templates.
        H=(points['mae'][Default,ti]-oracle[ti]).mean();Hp=(patient['mae'][Default,ti]-oracle_p[ti]).mean(axis=0);Hn=(nested['mae'][Default,ti]-oracle_n[ti]).mean(axis=0)
        for mi,method in enumerate(methods):
            if method in ['S','D','B2-val']:continue
            rr=dict(template=t,selector=method,n_patients=n_patients,n_budgets=len(bs),eta=None,practical_decision='evidence_insufficient_without_eta')
            add_ci(rr,'H',H,Hp,Hn)
            U=(points['mae'][Default,ti]-points['mae'][mi,ti]).mean();Up=(patient['mae'][Default,ti]-patient['mae'][mi,ti]).mean(axis=0);Un=(nested['mae'][Default,ti]-nested['mae'][mi,ti]).mean(axis=0)
            add_ci(rr,'U',U,Up,Un)
            add_ci(rr,'cell_oracle_excess',(points['mae'][mi,ti]-oracle[ti]).mean(),(patient['mae'][mi,ti]-oracle_p[ti]).mean(axis=0),(nested['mae'][mi,ti]-oracle_n[ti]).mean(axis=0))
            Hpositive=H>0 and interval(Hp)[0]>0 and interval(Hn)[0]>0
            rr['capture_ratio']=float(U/H) if Hpositive else None
            rr['capture_ratio_reason']='reported only when both 95% lower endpoints of H exceed zero' if Hpositive else 'H not clearly positive; ratio suppressed'
            for base in ['TrainLookup','B2-val','S','D']:
                j=methods.index(base)
                add_ci(rr,'gain_vs_'+base,(points['mae'][j,ti]-points['mae'][mi,ti]).mean(),(patient['mae'][j,ti]-patient['mae'][mi,ti]).mean(axis=0),(nested['mae'][j,ti]-nested['mae'][mi,ti]).mean(axis=0))
            aggregate.append(rr)
        for selector in ['L0','L1','L2','truth']:
            vals=[];decidable=[]
            for bi,B in enumerate(bs):
                if selector=='truth':v=points['mae'][S,ti,bi]-points['mae'][Q,ti,bi]
                else:v=next(r['predicted_gain'] for r in cells if r['template']==t and r['B']==B and r['selector']==selector)
                vals.append(v);decidable.append(next(r['true_side'] for r in cells if r['template']==t and r['B']==B and r['selector']=='L0')!='undecided')
            for i in range(1,len(bs)):
                if (vals[i]>0)!=(vals[i-1]>0):crossings.append(dict(template=t,selector=selector,B_left=bs[i-1],B_right=bs[i],gain_left=vals[i-1],gain_right=vals[i],both_endpoints_decidable=decidable[i-1] and decidable[i],note='observed adjacent sign reversal, not proof of unique/monotone threshold'))
        for scenario,ss in json.loads((out/'targets.json').read_text())[t].items():
            for goal in ['lo','hi']:
                tau=ss['tau_'+goal]
                for mi,method in enumerate(methods):
                    passing=np.flatnonzero(points['mae'][mi,ti]<=tau);bm=bs[passing[0]] if len(passing) else None
                    rr=dict(template=t,scenario=scenario,target=goal,tau=tau,selector=method,B_min_grid=bm,
                            left_grid_boundary=bm==bs[0],status='test_grid_reached' if bm else 'not_reached_in_test_grid',
                            note='minimum among six evaluated budgets; nonmonotonicity allowed; no continuum threshold or lower bound claimed')
                    for label,boot in [('patient',patient),('two_layer',nested)]:
                        curves=boot['mae'][mi,ti];v=np.min(np.where(curves<=tau,np.array(bs)[:,None],np.inf),axis=0)
                        rr[label+'_reach_rate']=float(np.isfinite(v).mean());finite=v[np.isfinite(v)]
                        rr[label+'_conditional_lo'],rr[label+'_conditional_hi']=interval(finite)
                    bmins.append(rr)
    # Complete sensitivity rows are retained, never used to choose the winner.
    sensitivity=[]
    for row in public:
        ti=ts.index(row['template']);bi=bs.index(row['B']);i=S if row['choice']=='S' else Q
        rr={k:row[k] for k in ['template','B','level','delta','component','model','choice','G_session','R_session','local_choice','mean_empirical_correlation','offline_seconds_per_configuration']}
        rr['actual_selected_MAE']=float(points['mae'][i,ti,bi]);rr['actual_cell_oracle_excess']=float(points['mae'][i,ti,bi]-oracle[ti,bi])
        rr['differs_from_primary']=row['choice']!=main[row['level'],row['template'],row['B']]['choice'];sensitivity.append(rr)
    hit=[]
    for method in [m for m in methods if m not in ['S','D','B2-val']]:
        cc=[r for r in cells if r['selector']==method];valid=[r for r in cc if r['true_side']!='undecided']
        hit.append(dict(selector=method,n_cells=len(cc),decidable_cells=len(valid),coverage=len(valid)/len(cc),
             mean_correct_fraction_on_decidable=float(np.mean([r['known_correct_fraction'] for r in valid])) if valid else None,
             fully_correct_cells=sum(r['known_correct_fraction']==1 for r in valid),
             wrongly_choose_D_fraction_when_S=float(np.mean([r['D_fraction'] for r in valid if r['true_side']=='S'])) if any(r['true_side']=='S' for r in valid) else None,
             wrongly_choose_S_fraction_when_D=float(np.mean([1-r['D_fraction'] for r in valid if r['true_side']=='D'])) if any(r['true_side']=='D' for r in valid) else None,
             note='24 cells are diagnostic summaries, not 24 independent samples; mixed outer-fold rules use patient decision fractions'))
    write_csv(out/'summary.csv',summaries);write_csv(out/'table1_cell_decisions.csv',cells);write_csv(out/'table2_baseline_gains.csv',effects)
    write_csv(out/'table3_H_U_M7.csv',aggregate);write_csv(out/'hit_summary.csv',hit);write_csv(out/'sensitivity_evaluation.csv',sensitivity)
    write_csv(out/'sign_crossings.csv',crossings);write_csv(out/'bmin.csv',bmins)
    costs=[];cal=[json.loads(s) for s in (out/'l2_calibration.jsonl').open()];fitby={r['fold']:r for r in cal}
    for method in methods:
        lookup={(r['fold'],r['template'],r['B']):r['choices'][method] for r in frozen};key=next(iter(lookup))
        start=time.perf_counter()
        for _ in range(10000):_v=lookup[key]
        deploy=(time.perf_counter()-start)/10000
        for t in ts:
            ff=[r for r in frozen if r['template']==t]
            private=method in ['L2','Local-L2','Default','TrainLookup','B2-val','Nonlinearity']
            count=0. if not private else np.mean([len(r['L2_global_training_subjects'] if method in ['L2','Local-L2'] else r['validation_subjects'] if method=='B2-val' else r['training_subjects']) for r in ff])
            off=np.mean([r['L2_prediction']['offline_seconds_per_configuration'] for r in ff]) if method in ['L2','Local-L2'] else np.mean([main['L1' if method in ['L1','Local-L1'] else 'L0',t,B]['offline_seconds_per_configuration'] for B in bs]) if method in ['L0','L1','Local-L0','Local-L1','NoAST-L0','NoBudget-L0'] else None
            costs.append(dict(template=t,selector=method,training_patients_mean=float(count),
                 L2_valid_differences_mean=float(np.mean([fitby[r['fold']]['valid_differences'] for r in ff])) if method in ['L2','Local-L2'] else 0,
                 public_calendars=len(json.loads((out/'public_calendars.json').read_text())[t]),
                 offline_MC_seconds_per_configuration=float(off) if off is not None else None,
                 deployment_dictionary_seconds=deploy,full_session_spend='B for every selected S/D or periodic strategy',
                 note='MC offline cost shared with its local/noAST ablations; training replay totals in timing.json; dictionary microbenchmark excludes model fitting and IO'))
    write_csv(out/'table2_implementation_cost.csv',costs)
    np.savez_compressed(out/'loss_tensor.npz',mae=raw['mae'],methods=np.array(methods),templates=np.array(ts),budgets=np.array(bs),subjects=np.array(subjects))
    bench={r['template']:{key:r[key] for key in ['H','H_patient_lo','H_patient_hi','H_two_layer_lo','H_two_layer_hi','U','U_patient_lo','U_patient_hi','U_two_layer_lo','U_two_layer_hi','capture_ratio','practical_decision']} for r in aggregate if r['selector']=='L0'}
    dump(out/'research_decision.json',dict(primary='L0',by_template=bench,eta=None,
         decision='no unconditional practical go/no-go without eta; report directional benefits and strategy-specific scope',
         scope='simple fixed S/D on known templates and previously examined patients; does not settle all dynamic mechanisms or unseen AST generalization'))
    dump(out/'statistical_audit.json',dict(global_subjects=len(subjects),bootstrap=D,
         bound_patient_indices_across_templates=True,bound_repeat_indices_across_methods_and_budgets=True,
         equal_patient_weight=True,training_uncertainty_included=False,pointwise_CI=True,multiplicity_corrected=False,
         no_raw_MAE_pooling_across_templates=True,oracle_bias_disclosed=True,negative_crossfit_cell_excess_not_clipped=True,
         analysis_seconds=time.perf_counter()-started))
    from .report import report
    report(out,cfg,summaries,cells,effects,aggregate,hit,sensitivity)
    from .figures import plot
    plot(out)
