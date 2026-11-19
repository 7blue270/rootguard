"""Freeze every outer-fold predictor before evaluating held-out noise streams."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.run import dump,line
from experiments.workflow_extensions.m6.public import fit_public as fit_old_public
from .protocol import ROOT,TEMPLATES,REFERENCE,REFERENCE_SOURCES,config,calendars,fold_validation,SELECTORS,prior_bounds
from .engine import real_replay
from .predict import predict_grid
from .calibrate import estimate_delta


def mean_score(cache,records,B,method):
    by={}
    for rec in records:by.setdefault(str(rec['subject_id']),[]).append(float(np.mean(cache[rec['trajectory_id'],B,method])))
    return float(np.mean([np.mean(a) for a in by.values()]))


def run(args):
    started=time.perf_counter();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Output directory must be empty; old results cannot be overwritten')
    cfg=config(args.precheck);dump(out/'config.json',cfg)
    records=data.load(args.data,cfg['templates']);subjects=sorted({str(r['subject_id']) for r in records})
    assert all(len({r['subject_id'] for r in records if r['template']==t})>=10 for t in cfg['templates'])
    assert all(r['T']==40 for r in records)
    dump(out/'input_hashes.json',data.input_hashes(args.data,cfg['templates']))
    dump(out/'versions.json',dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    source_paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'experiments/workflow_extensions/finite_cache/workload.py',
        ROOT/'experiments/workflow_extensions/finite_cache/data.py',ROOT/'experiments/workflow_extensions/m4_carry/core.py',
        ROOT/'experiments/workflow_extensions/m6/public.py',ROOT/'experiments/workflow_extensions/m5/public.py',Path('/home/M6.2_实验方案_Rhat信息预算与分侧验证.md')]
    dump(out/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    source=ROOT/'experiments/workflow_extensions/outputs/m6/real_v2_20261117'
    oldcfg=json.loads((source/'config.json').read_text());dump(out/'targets.json',json.loads((source/'targets.json').read_text()))
    dump(out/'reference_intervals.json',dict(intervals=REFERENCE,sources=REFERENCE_SOURCES,
         normalized={t:dict(zip(['lo','hi'],[a.tolist() for a in prior_bounds(Workload(t),'L1')])) for t in cfg['templates']}))
    public={t:calendars(records,t) for t in cfg['templates']};dump(out/'public_calendars.json',public)
    dump(out/'folds.json',{s:dict(zip(['outer_train','inner_validation'],fold_validation(subjects,s,cfg))) for s in subjects})
    # Input/output examples freeze exact formulas including clamps before labels.
    examples={}
    for t in cfg['templates']:
        w=Workload(t);z=np.full(len(w.roots),.2);raw=w.denormalize(z)
        examples[t]=dict(roots=w.roots,raw=raw,normalized=z,outputs=w.evaluate(raw),target=w.target)
    dump(out/'formula_examples.json',examples)
    main={};raw_public={}
    with (out/'public_predictions.jsonl').open('w') as ph:
        for t in cfg['templates']:
            w=Workload(t)
            for level in ['L0','L1']:
                for model in cfg['models']:
                    for delta in cfg['deltas']:
                        for component in cfg['components']:
                            primary=model==cfg['primary_model'] and delta==cfg['delta'] and component==cfg['component']
                            rows,losses=predict_grid(w,public[t],cfg,level,delta,component,model,keep_losses=primary)
                            for row in rows:
                                line(ph,row)
                                if primary:main[level,t,row['B']]=row
                            if primary:
                                for B,fields in losses.items():
                                    for name,value in fields.items():raw_public[f'{level}__{t}__{B}__{name}']=value
            print('PUBLIC',t,'sensitivity frozen','seconds',round(time.perf_counter()-started),flush=True)
    np.savez_compressed(out/'public_primary_losses.npz',**raw_public)
    public_seconds=time.perf_counter()-started
    # Separate calibration noise avoids reusing held-out evaluation noise for
    # training choices. Every held-out patient's rows are excluded from its fit.
    training={};train_models=['S','D']+['P'+str(P) for P in cfg['periods']]
    training_started=time.perf_counter()
    with (out/'training_metrics.jsonl').open('w') as th:
        for rec in records:
            w=Workload(rec['template'])
            for B in cfg['budgets']:
                for method in train_models:
                    result=real_replay(rec,B,method,w,cfg,'m62_training',cfg['train_repeats'])
                    training[rec['trajectory_id'],B,method]=result['mae']
                    for rep,mae in enumerate(result['mae']):line(th,dict(subject=str(rec['subject_id']),trajectory=rec['trajectory_id'],template=rec['template'],B=B,method=method,repeat=rep,mae=mae))
    training_seconds=time.perf_counter()-training_started
    fold_predictions={};prediction_count=0;fold_started=time.perf_counter()
    with (out/'fold_predictions.jsonl').open('w') as fh,(out/'l2_calibration.jsonl').open('w') as ch:
        for heldout in subjects:
            train_ids,validation_ids=fold_validation(subjects,heldout,cfg)
            training_records=[r for r in records if str(r['subject_id']) in train_ids]
            fit,edges=estimate_delta(training_records,heldout);line(ch,dict(fold=heldout,**fit))
            assert set(fit['subjects'])<=set(train_ids)
            for t in cfg['templates']:
                train=[r for r in training_records if r['template']==t]
                test=[r for r in records if r['template']==t and str(r['subject_id'])==heldout]
                if not test:continue
                validation=[r for r in train if str(r['subject_id']) in validation_ids]
                if not validation:raise ValueError('No inner validation patients; specify a different public split before test')
                w=Workload(t);l2,_=predict_grid(w,public[t],cfg,'L1',fit['delta'],cfg['component'],cfg['primary_model'])
                l2={r['B']:r for r in l2}
                oldpublic=fit_old_public(train,w,oldcfg);nu=oldpublic['nonlinearity']
                defaultS=np.mean([mean_score(training,train,B,'S') for B in cfg['budgets']])
                defaultD=np.mean([mean_score(training,train,B,'D') for B in cfg['budgets']])
                default='D' if defaultS>defaultD else 'S'
                for B in cfg['budgets']:
                    lookupS=mean_score(training,train,B,'S');lookupD=mean_score(training,train,B,'D')
                    period=min(cfg['periods'],key=lambda P:(mean_score(training,validation,B,'P'+str(P)),P))
                    a=main['L0',t,B];b=main['L1',t,B];c=l2[B]
                    choices={'S':'S','D':'D','Default':default,'TrainLookup':'D' if lookupS>lookupD else 'S','B2-val':'P'+str(period),
                             'L0':a['choice'],'L1':b['choice'],'L2':c['choice'],
                             'Local-L0':a['local_choice'],'Local-L1':b['local_choice'],'Local-L2':c['local_choice'],
                             'NoAST-L0':a['no_ast_choice'],'NoBudget-L0':main['L0',t,cfg['fixed_no_budget']]['choice'],
                             'Nonlinearity':'S' if nu>cfg['nonlinearity_threshold'] else 'D'}
                    record=dict(fold=heldout,template=t,B=B,choices=choices,
                        G_L0=a['G_session'],G_L1=b['G_session'],G_L2=c['G_session'],
                        G_local_L0=a['G_local'],G_local_L1=b['G_local'],G_local_L2=c['G_local'],
                        G_train_lookup=lookupS-lookupD,L2_delta=fit['delta'],nonlinearity=nu,
                        L2_prediction={key:c[key] for key in ['loss_S','loss_D','R_session','G_session','local_loss_S','local_loss_D','R_local','G_local','offline_seconds_per_configuration']},
                        training_subjects=sorted({str(r['subject_id']) for r in train}),
                        L2_global_training_subjects=fit['subjects'],L2_valid_differences=fit['valid_differences'],
                        validation_subjects=sorted({str(r['subject_id']) for r in validation}),selected_P=period,
                        train_default_losses=dict(S=defaultS,D=defaultD),
                        sources=dict(L0='AST/domain/public calendar only',L1='independent reference prior',L2='one outer-train scalar',
                                     TrainLookup='outer-train m62_training losses',B2='inner validation m62_training losses'),
                        leakage_check=heldout not in train_ids)
                    digest=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest();record['config_hash']=digest
                    line(fh,record);fold_predictions[heldout,t,B]=record;prediction_count+=1
            print('FOLD',heldout,'delta',round(fit['delta'],6),'predictions frozen','seconds',round(time.perf_counter()-started),flush=True)
    freeze_files=['config.json','input_hashes.json','public_calendars.json','public_predictions.jsonl','fold_predictions.jsonl','reference_intervals.json','folds.json','targets.json']
    freeze={f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in freeze_files}
    dump(out/'PREDICTIONS_FROZEN.json',dict(hashes=freeze,all_fold_predictions_complete=True,test_stream_evaluated=False,
         predictions=prediction_count,training_noise_phase='m62_training',test_noise_phase='m62_evaluation',
         note='training streams use all eligible patients but own held-out rows are excluded in every fold; no held-out evaluation draws generated yet'))
    fold_seconds=time.perf_counter()-fold_started;evaluation_started=time.perf_counter()
    with (out/'metrics.jsonl').open('w') as mh,(out/'actions.jsonl').open('w') as ah:
        for rec in records:
            w=Workload(rec['template']);sid=str(rec['subject_id'])
            for B in cfg['budgets']:
                pred=fold_predictions[sid,rec['template'],B]
                # All selectors directly index this same S/D pair; B2 uses its
                # separately preselected periodic schedule and paired noise.
                results={}
                for method in ['S','D',pred['choices']['B2-val']]:
                    result=real_replay(rec,B,method,w,cfg,'m62_evaluation',cfg['repeats'],detailed=True);results[method]=result
                    ident=dict(subject=sid,trajectory=rec['trajectory_id'],template=rec['template'],B=B,execution=method)
                    for j,h in enumerate(result['histories']):
                        for rep in range(cfg['repeats']):line(ah,{**ident,'repeat':rep,'action_index':j,'turn':h['turn'],
                            'epsilon':h['epsilon'],'cache':h['cache'][rep],'projection':h['projection'][rep]})
                for selector,method in pred['choices'].items():
                    r=results[method]
                    for rep in range(cfg['repeats']):line(mh,dict(subject=sid,trajectory=rec['trajectory_id'],template=rec['template'],B=B,
                        selector=selector,execution=method,repeat=rep,**{f:r[f][rep] for f in ['mae','truth_abs','rce','spent','n_actions','n_root_releases','projection_rate','p95']},
                        prediction_hash=pred['config_hash']))
            print('EVALUATE',rec['template'],sid,'seconds',round(time.perf_counter()-started),flush=True)
    assert all(hashlib.sha256((out/f).read_bytes()).hexdigest()==h for f,h in freeze.items())
    dump(out/'timing.json',dict(simulation_and_prediction_seconds=time.perf_counter()-started,
         public_prediction_and_sensitivity_seconds=public_seconds,training_replays_seconds=training_seconds,
         outer_fold_fit_seconds=fold_seconds,actual_evaluation_seconds=time.perf_counter()-evaluation_started))
    from .audit import audit
    audit(out,args.data)
    from .analysis import analyze
    analyze(out)
    print('DONE',out,'seconds',round(time.perf_counter()-started),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--data',default=str(ROOT/'data_mine/finite_cache'));p.add_argument('--precheck',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
