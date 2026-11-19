"""M6 frozen-cohort diagnosis. Selection on public synthesis precedes ALL test."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.core import Plan
from experiments.workflow_extensions.m4_carry.run import ROOT,dump,line
from experiments.workflow_extensions.m4_carry.planning import public_population
from experiments.workflow_extensions.m5.run import make_plan
from experiments.workflow_extensions.m5.metrics import evaluate
from .public import fit_public
from .core import simulate,Policy
from .attack import evaluate_attack

FROZEN=['RG-static-exp','RG-periodic-exp','RG-static-lap','RG-periodic-lap','M4-carry','M5']
NEW=['M6','M6-lap-only','M6-mean-only','M6-final',
     'M6-iso-single-step','M6-iso-cross','M6-iso-atomic','M6-fixed-no-carry','M6-dynamic-fixed']
METHODS=FROZEN+NEW


def config(source,precheck=False):
    cfg=json.loads((source/'config.json').read_text())
    cfg.update(methods=METHODS,attack_methods=FROZEN+['M6'],
               specification='M6_机制设计与优化逻辑.md (2026-11-17)',precheck=precheck,
               c_threshold=.70,nu_threshold=3.,nu_grid_threshold=1.,gradient_quantile=.90,
               danger_quantile=.50,grid_base=128,grid_max=1000,skeleton_points=4096,skeleton_seed=20261117,
               danger_rule='small AST denominator below train median OR public/cache workload gradient above train-work Q90',
               lambda_harm=.1,eta_tail=.1,scenario_jitter=.05,posterior_variance_floor=1e-8,
               risk_rollout='terminal hold to T; posterior approximation from train synthetic particles',
               risk_harm='probability of larger cumulative scaled-query loss over remaining horizon',
               risk_tail='unweighted particle/turn p95; expectation and harm use posterior weights',
               threshold_source='predeclared constants; no test fitting; c=.70,nu=3 demonstration values retained without claiming separation',
               primary_metric='full-trajectory final-node MAE and fixed-public-target B_min',
               planner_objective='public synthetic query-weighted scaled MAE, same carry executor',
               iso_plan={'K':2,'rho':.5},notes='M6 mechanism diagnosis; same analyzed patients, not fresh confirmation')
    if precheck:
        g=cfg['evaluated_budgets'];cfg.update(repeats=2,bootstrap=100,public_repeats=2,evaluated_budgets=[g[0],g[len(g)//2],g[-1]])
    return cfg


def options(method):
    return dict(mixed=method!='M6-lap-only',
                groups=method not in ('M6-iso-single-step','M6-iso-cross'),
                horizon=method!='M6-iso-single-step',
                force=method.startswith('M6-iso-'),carry=method!='M6-fixed-no-carry')


def model_for(method,public,final):
    p=final if method=='M6-final' else public
    if method=='M6-mean-only':p={**p,'allocation_weights':np.maximum(p['a_mean'],1e-9).tolist()}
    return p


def run(args):
    started=time.monotonic();source=Path(args.source).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Output must be empty; never overwrite earlier experiments')
    cfg=config(source,args.precheck);dump(out/'config.json',cfg)
    assignment=json.loads((source/'split.json').read_text());targets=json.loads((source/'targets.json').read_text())
    oldstats=json.loads((Path(args.original)/'public_stats.json').read_text())
    records=data.load(args.data,cfg['templates']);hashes=data.input_hashes(args.data,cfg['templates'])
    assert hashes==json.loads((source/'input_hashes.json').read_text())
    for name,obj in [('input_hashes',hashes),('split',assignment),('targets',targets)]:dump(out/(name+'.json'),obj)
    paths=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'experiments/workflow_extensions/m5').glob('*.py'))+list((ROOT/'experiments/workflow_extensions/m4_carry').glob('*.py'))+[
        ROOT/'experiments/workflow_extensions/finite_cache/workload.py',ROOT/'experiments/workflow_extensions/finite_cache/data.py',Path('/home/M6_机制设计与优化逻辑.md')]
    dump(out/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    # Reuse the exact original M-Opt optimizer, validation-selected periods,
    # M5 observations, extended metrics and attacks, without refitting them.
    frozen=0
    with (out/'metrics.jsonl').open('w') as mh:
        for ln in (source/'metrics.jsonl').open():
            r=json.loads(ln)
            if r['method'] in FROZEN and r['B'] in cfg['evaluated_budgets'] and r['repeat']<cfg['repeats']:mh.write(ln);frozen+=1
    with (out/'attack_metrics.jsonl').open('w') as ah:
        for ln in (source/'attack_metrics.jsonl').open():
            r=json.loads(ln)
            if r['method'] in FROZEN and r['B'] in cfg['evaluated_budgets'] and r['repeat']<cfg['repeats']:ah.write(ln)
    # The historical all-budget file predates initial/release denominator
    # fields. Reuse its separately audited B960 endpoint extension, not guesses
    # obtained from whole-trajectory errors.
    with (out/'frozen_attack_endpoints.jsonl').open('w') as ah:
        for ln in (source/'attack_checks_960.jsonl').open():
            r=json.loads(ln)
            if r['method'] in FROZEN and r['repeat']<cfg['repeats']:ah.write(ln)
    dump(out/'frozen_provenance.json',dict(source=str(source),metric_rows=frozen,exact_row_copy=True,
         original_optimizer='unchanged RootGuard optimal_budget_allocation_exact(mode=abs)',
         original_grid_points=1000,original_metric_to_grid='epsilon_grid=epsilon_metric/999'))
    dump(out/'frozen_source_manifests.json',{name:json.loads((source/name).read_text()) for name in ['source_hashes.json','input_hashes.json','config.json']})
    contexts={};publics={};plans={}
    # Train-only selection of K/rho on synthetic populations. Validation unused
    # here: B2 validation selection was already frozen in the previous outputs.
    with (out/'public_plan_scores.jsonl').open('w') as sh:
        for t in cfg['templates']:
            w=Workload(t);train=data.partition(records,assignment,'train',t)
            public=fit_public(train,w,cfg);final=fit_public(train,w,cfg,True)
            publics[t]=dict(main=public,final_only=final);contexts[t]=(w,public,final);plans[t]={}
            population=[r for d in cfg['delta_scenarios'].values() for r in public_population(train,w,cfg,d)]
            candidates=[make_plan(K,rho,len(w.roots)) for K in cfg['K_grid'] if K for rho in cfg['rho_grid']]
            policy=Policy(w,public,cfg)
            for B in cfg['evaluated_budgets']:
                if public['regime']=='STATIC':winner=Plan(1.)
                else:
                    scores=[]
                    for plan in candidates:
                        losses=[]
                        for rec in population:
                            res=simulate(rec,B,plan,public,cfg,'m6_public_plan',cfg['public_repeats'],policy=policy)
                            truth=w.evaluate(data.replay(rec));pred=w.from_cache(res['caches'])
                            loss=sum(public['query_weights'][j]*np.mean(abs(pred[node]-truth[node]))/public['node_scales'][j] for j,node in enumerate(w.nodes))
                            losses.append(loss)
                        score=float(np.mean(losses));scores.append((score,plan.K,plan.rho,plan))
                        line(sh,dict(template=t,B=B,plan=plan.to_dict(),score=score))
                    winner=min(scores,key=lambda v:v[:3])[-1]
                plans[t][str(B)]=winner.to_dict()
            print('SELECT',t,public['regime'],'c',round(public['cancellation'],4),'nu',round(public['nonlinearity'],4),'groups',public['groups'],'seconds',round(time.monotonic()-started),flush=True)
    dump(out/'public_stats.json',publics);dump(out/'selections.json',plans)
    dump(out/'phase_a_complete.json',dict(all_selections_complete=True,test_evaluated=False))
    audit=dict(paths=0,ledger_ok=True,counts={},frozen_exact_row_copy=True,train_only_selection=True)
    with (out/'metrics.jsonl').open('a') as mh,(out/'actions.jsonl').open('w') as ah,(out/'attack_metrics.jsonl').open('a') as attackh:
        for t,(w,p,fp) in contexts.items():
            test=data.partition(records,assignment,'test',t);audit['counts'][t]=dict(patients=len(test),trajectories=len(test))
            assert not (set(p['train_subjects']) & {str(r['subject_id']) for r in test})
            policies={m:Policy(w,model_for(m,p,fp),cfg,**{k:v for k,v in options(m).items() if k!='carry'}) for m in NEW}
            for bi,B in enumerate(cfg['evaluated_budgets']):
                pp=plans[t][str(B)];plan=Plan(pp['rho'],tuple(pp['fractions']),tuple(tuple(s) for s in pp['sets']))
                fixed=make_plan(cfg['iso_plan']['K'],cfg['iso_plan']['rho'],len(w.roots))
                for rec in test:
                    for method in NEW:
                        model=model_for(method,p,fp)
                        selected_plan=fixed if method.startswith('M6-iso-') or method in ('M6-fixed-no-carry','M6-dynamic-fixed') else plan
                        result=simulate(rec,B,selected_plan,model,cfg,'test',cfg['repeats'],detailed=True,policy=policies[method],**options(method))
                        metrics=evaluate(rec,result,w,oldstats[t],dict(cfg,_B=B),p)
                        metrics['grid_release_fraction']=result['n_grid']/result['n_pub']
                        metrics['outside_work_grid_fraction']=np.divide(result['n_outside_grid'],result['n_grid'],out=np.full(cfg['repeats'],np.nan),where=result['n_grid']>0)
                        truthz=w.normalize(data.replay(rec));metrics['outside_work_truth_fraction']=np.full(cfg['repeats'],np.mean((truthz<model['work_lo'])|(truthz>model['work_hi'])))
                        ident=dict(template=t,subject=str(rec['subject_id']),trajectory=rec['trajectory_id'],B=B,method=method)
                        for r in range(cfg['repeats']):
                            line(mh,{**ident,'repeat':r,**{k:v[r] for k,v in metrics.items()},'K':result['K'],'rho':selected_plan.rho,'stop_reason':result['stop_reason']})
                            line(ah,{**ident,'repeat':r,'turn':1,'action_index':0,'actual_epsilon':result['anchor_epsilon'][r],
                                     'cache_after':result['anchor_cache'][r],'kernels':result['anchor_kernels'][r],'R_after':B-result['anchor_epsilon'][r].sum()})
                            for h in result['histories']:
                                line(ah,{**ident,'repeat':r,'turn':h['turn'],'action_index':h['action_index'],'planned_roots':h['planned_roots'],
                                    **{key:h[key][r] for key in ['selected','actual_epsilon','candidate_epsilon','eps_can','left_before','left_after','cache_before','cache_after','kernels','risk_skip','risk_selected','predicted_harm','predicted_gain']}})
                        if method=='M6':
                            for row in evaluate_attack(rec,result,w,model,cfg):line(attackh,{**ident,**row})
                        audit['paths']+=cfg['repeats']
                if bi%8==0 or bi==len(cfg['evaluated_budgets'])-1:print('TEST',t,bi+1,'/',len(cfg['evaluated_budgets']),'paths',audit['paths'],'seconds',round(time.monotonic()-started),flush=True)
    audit['simulation_seconds']=time.monotonic()-started;dump(out/'audit.json',audit)
    from .audit import audit_output
    audit_output(out,args.data,source)
    from .analysis import analyze
    analyze(out)
    print('DONE',out,'seconds',round(time.monotonic()-started),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default=str(ROOT/'experiments/workflow_extensions/outputs/m5/real_v2_20261117'))
    ap.add_argument('--original',default=str(ROOT/'experiments/workflow_extensions/outputs/m4_carry/real_v1_20261116'))
    ap.add_argument('--data',default=str(ROOT/'data_mine/finite_cache'));ap.add_argument('--output',required=True);ap.add_argument('--precheck',action='store_true')
    run(ap.parse_args())

if __name__=='__main__':main()
