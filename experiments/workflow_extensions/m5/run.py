"""Frozen-cohort M5 experiment; phase A selection completes before test evaluation."""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.core import Plan,paired_noise
from experiments.workflow_extensions.m4_carry.run import ROOT,dump,line
from .public import fit_public
from .core import simulate
from .metrics import evaluate
from .attack import evaluate_attack

NEW=['M5-alloc-static','M5-local-veto-fixed','M5-trace-fixed','M5-atomic-fixed','M5','M5-final']
FROZEN=['RG-static-exp','RG-periodic-exp','RG-static-lap','RG-periodic-lap','M4-carry','M4-static']
METHODS=FROZEN+NEW


def config(source,precheck):
    cfg=json.loads((source/'config.json').read_text())
    cfg.update(work_quantile=.1,coupling_threshold=.5,nonlinearity_threshold=1.,directional_threshold=.5,
               allocation_weight_floor=1e-9,node_scale_floor=1e-6,gain_tolerance=1e-15,
               query_scaling='train mean absolute node value',validation_objective='query_weighted_scaled_MAE',
               attack_grid_points=1000,attack_methods=['RG-static-exp','RG-periodic-exp','RG-static-lap','RG-periodic-lap','M4-carry','M5'],
               methods=METHODS,precheck=precheck,
               specification='M5_优化方案与逻辑理由.md (2026-11-17)',
               projection_penalty='not specified in document; no additional penalty invented',
               interval_sup='monomial corners exact; MELD corners plus clamp neighborhoods, utility heuristic not privacy sensitivity')
    if precheck:
        grid=cfg['evaluated_budgets'];cfg.update(repeats=2,validation_repeats=2,bootstrap=100,evaluated_budgets=[grid[0],grid[len(grid)//2],grid[-1]])
    return cfg


def make_plan(K,rho,k):
    return Plan(1.) if K==0 else Plan(rho,tuple(j/(K+1) for j in range(1,K+1)),(tuple(range(k)),)*K)


def load_frozen(source,comparison,budgets,R):
    metric={};acts=defaultdict(list)
    for folder,names in [(comparison,set(FROZEN)-{'M4-static'}),(source,{'B0'})]:
        for ln in (folder/'metrics.jsonl').open():
            r=json.loads(ln)
            if r['method'] in names and r['B'] in budgets and r['repeat']<R:
                m='M4-static' if r['method']=='B0' else r['method']
                metric[r['trajectory'],r['B'],m,r['repeat']]=r
        for ln in (folder/'actions.jsonl').open():
            r=json.loads(ln)
            if r['method'] in names and r['B'] in budgets and r['repeat']<R:
                m='M4-static' if r['method']=='B0' else r['method']
                acts[r['trajectory'],r['B'],m,r['repeat']].append(r)
    return metric,acts


def replay_frozen(rec,B,method,R,metric,actions,w):
    rows=[metric[rec['trajectory_id'],B,method,r] for r in range(R)]
    logs=[sorted(actions[rec['trajectory_id'],B,method,r],key=lambda a:a['action_index']) for r in range(R)]
    assert all([a['turn'] for a in logs[0]]==[a['turn'] for a in rr] for rr in logs)
    k=len(w.roots);hist=[];caches=np.empty((R,rec['T'],k));total=np.zeros(R)
    for j in range(len(logs[0])):
        aa=[r[j] for r in logs];t=aa[0]['turn'];cache=np.array([a['cache_after'] for a in aa]);eps=np.array([a['actual_epsilon'] for a in aa])
        mask=eps>0;charge=eps.sum(axis=1);before=total.copy();total+=charge
        end=logs[0][j+1]['turn']-1 if j+1<len(logs[0]) else rec['T'];caches[:,t-1:end]=cache[:,None,:]
        if j==0:anchor=cache.copy();anchor_eps=eps.copy()
        else:
            hist.append(dict(turn=t,action_index=j,planned_roots=tuple(range(k)),selected=mask,actual_epsilon=eps,candidate_epsilon=eps.copy(),
                             eps_can=np.array([a.get('eps_can',float(np.sum(a['actual_epsilon']))) for a in aa]),
                             cache_before=np.array([a['cache_before'] for a in aa]),cache_after=cache,
                             left_before=B-before,left_after=B-total.copy()))
    # Reconcile serialized charges against saved metrics before extended evaluation.
    np.testing.assert_allclose(total,[r['used'] for r in rows],atol=1e-10,rtol=0)
    assert np.all(total<=B+1e-10)
    return dict(caches=caches,spent=total,left=B-total,m=np.array([r['m'] for r in rows]),
                n_pub=np.array([r['n_pub'] for r in rows]),n_proj=np.array([r['n_pub']*r['projection_rate'] for r in rows]),
                empty=np.array([0 if r.get('frac_veto') is None else r['frac_veto']*r['K'] for r in rows]),
                root_veto=np.array([0 if r.get('frac_root_veto') is None else r['frac_root_veto']*len(w.roots)*r['K'] for r in rows]),
                root_count=np.array([len(w.roots)*r['K'] for r in rows]),K=len(hist),stop=int(rows[0]['stop_turn']),histories=hist,
                anchor_cache=anchor,anchor_epsilon=anchor_eps,stop_reason=rows[0].get('stop_reason','frozen'),frozen_rows=rows)


def validation_score(records,B,plan,public,cfg,old_stats):
    bypatient=defaultdict(list);w=Workload(records[0]['template'])
    for rec in records:
        r=simulate(rec,B,plan,public,cfg,'validation',cfg['validation_repeats'],old_stats=old_stats)
        truth=w.evaluate(data.replay(rec));pred=w.from_cache(r['caches']);loss=0.
        for j,node in enumerate(w.nodes):
            ix=[q['turn']-1 for q in rec['queries'] if q['node']==node]
            if ix:loss+=public['query_weights'][j]*np.mean(abs(pred[node][:,ix]-truth[node][ix]))/public['node_scales'][j]
        bypatient[str(rec['subject_id'])].append(float(loss))
    return float(np.mean([np.mean(v) for v in bypatient.values()]))


def write_actions(handle,ident,result,rec,B,method,public=None):
    R=result['caches'].shape[0];w=Workload(rec['template'])
    for r in range(R):
        line(handle,{**ident,'repeat':r,'turn':1,'action_index':0,'selected_roots':w.roots,
                     'actual_epsilon':result['anchor_epsilon'][r],'cache_after':result['anchor_cache'][r],
                     'R_after':B-float(result['anchor_epsilon'][r].sum())})
        for h in result['histories']:
            line(handle,{**ident,'repeat':r,'turn':h['turn'],'action_index':h['action_index'],
                         'selected_roots':[root for root,yes in zip(w.roots,h['selected'][r]) if yes],
                         **{k:h[k][r] for k in ['actual_epsilon','cache_before','cache_after','eps_can']},
                         'R_before':h['left_before'][r],'R_after':h['left_after'][r]})


def run(args):
    started=time.monotonic();source=Path(args.source).resolve();comparison=Path(args.comparison).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Output must be empty; old experiments cannot be overwritten')
    cfg=config(source,args.precheck);dump(out/'config.json',cfg)
    assignment=json.loads((source/'split.json').read_text());oldstats=json.loads((source/'public_stats.json').read_text())
    targets=json.loads((source/'targets.json').read_text());records=data.load(args.data,cfg['templates'])
    hashes=data.input_hashes(args.data,cfg['templates'])
    assert hashes==json.loads((source/'input_hashes.json').read_text())
    dump(out/'input_hashes.json',hashes);dump(out/'split.json',assignment);dump(out/'targets.json',targets)
    source_paths=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'experiments/workflow_extensions/m4_carry').glob('*.py'))+[ROOT/'experiments/workflow_extensions/finite_cache/workload.py',Path('/home/M5_优化方案与逻辑理由.md')]
    dump(out/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    contexts={};publics={};plans={}
    with (out/'validation_scores.jsonl').open('w') as vh:
        for t in cfg['templates']:
            w=Workload(t);train=data.partition(records,assignment,'train',t);validation=data.partition(records,assignment,'validation',t)
            public=fit_public(train,w,cfg);fp=fit_public(train,w,cfg,final_only=True)
            assert not(set(public['train_subjects'])&{str(r['subject_id']) for r in validation})
            publics[t]=dict(main=public,final_only=fp);contexts[t]=(w,public,fp);plans[t]={}
            for bi,B in enumerate(cfg['evaluated_budgets']):
                candidates=[make_plan(0,1,len(w.roots))]+[make_plan(K,rho,len(w.roots)) for K in cfg['K_grid'] if K for rho in cfg['rho_grid']]
                scores=[]
                for plan in candidates:
                    score=validation_score(validation,B,plan,public,cfg,oldstats[t]);scores.append((score,plan.K,plan.rho,plan))
                    line(vh,dict(template=t,B=B,plan=plan.to_dict(),score=score,objective=cfg['validation_objective']))
                plans[t][str(B)]=min(scores,key=lambda a:a[:3])[-1].to_dict()
            print('SELECT',t,'budgets',len(plans[t]),'groups',public['groups'],'exponent',public['exponent'],'elapsed',round(time.monotonic()-started,1),flush=True)
    dump(out/'public_stats.json',publics);dump(out/'selections.json',plans)
    dump(out/'phase_a_complete.json',dict(all_selections_complete=True,test_evaluated=False))
    oldmetrics,oldactions=load_frozen(source,comparison,cfg['evaluated_budgets'],cfg['repeats'])
    audit=dict(paths=0,ledger_ok=True,max_budget_excess=0.,max_frozen_mae_difference=0.,
               selections_before_test=True,train_only_public_statistics=True,prior_excludes_test=True,
               causal_owner_releases=True,atoms_never_split=True,gate_execution_epsilon_equal=True,
               diagnostic_only=True,counts={})
    with (out/'metrics.jsonl').open('w') as mh,(out/'actions.jsonl').open('w') as ah,(out/'attack_metrics.jsonl').open('w') as attackh:
        for t,(w,public,fp) in contexts.items():
            test=data.partition(records,assignment,'test',t);audit['counts'][t]=dict(patients=len({str(r['subject_id']) for r in test}),trajectories=len(test))
            assert not(set(public['train_subjects'])&{str(r['subject_id']) for r in test})
            for bi,B in enumerate(cfg['evaluated_budgets']):
                pp=plans[t][str(B)];main_plan=Plan(pp['rho'],tuple(pp['fractions']),tuple(tuple(s) for s in pp['sets']))
                for rec in test:
                    for method in METHODS:
                        model=fp if method=='M5-final' else public
                        if method in FROZEN:
                            result=replay_frozen(rec,B,method,cfg['repeats'],oldmetrics,oldactions,w)
                            rho=result['frozen_rows'][0]['rho']
                        else:
                            plan=Plan(1.) if method=='M5-alloc-static' else main_plan
                            gate='local' if method=='M5-local-veto-fixed' else 'singletons' if method=='M5-trace-fixed' else 'atomic'
                            carry=method in ('M5','M5-final')
                            result=simulate(rec,B,plan,model,cfg,'test',cfg['repeats'],gate=gate,carry=carry,old_stats=oldstats[t],detailed=True);rho=plan.rho
                        met=evaluate(rec,result,w,oldstats[t],dict(cfg,_B=B),public)
                        if method in FROZEN:
                            diff=np.max(abs(met['mae']-np.array([r['mae'] for r in result['frozen_rows']])))
                            audit['max_frozen_mae_difference']=max(audit['max_frozen_mae_difference'],float(diff));assert diff<1e-9
                            # Preserve every old secondary denominator/NA convention;
                            # extended query/RCE metrics are computed from the cache.
                            for rep,saved in enumerate(result['frozen_rows']):
                                for field in set(met)&set(saved):
                                    met[field][rep]=np.nan if saved[field] is None else saved[field]
                        audit['paths']+=cfg['repeats'];audit['max_budget_excess']=max(audit['max_budget_excess'],float(np.max(result['spent']-B)))
                        assert np.all(result['spent']<=B+1e-10)
                        if method in ('M5','M5-atomic-fixed','M5-final'):
                            for h in result['histories']:
                                np.testing.assert_array_equal(h['candidate_epsilon'],h['actual_epsilon'])
                                for g in model['groups']:
                                    v=h['selected'][:,g];assert np.all(v==v[:,:1])
                        ident=dict(template=t,subject=str(rec['subject_id']),trajectory=rec['trajectory_id'],B=B,method=method)
                        for r in range(cfg['repeats']):
                            line(mh,{**ident,'repeat':r,**{k:v[r] for k,v in met.items()},'K':result['K'],'rho':rho,'stop_reason':result['stop_reason']})
                        write_actions(ah,ident,result,rec,B,method)
                        if method in cfg['attack_methods']:
                            kernel='native_exp' if method.endswith('-exp') else 'laplace'
                            for row in evaluate_attack(rec,result,w,public,kernel,cfg):line(attackh,{**ident,**row})
                if bi%8==0 or bi==len(cfg['evaluated_budgets'])-1:print('TEST',t,bi+1,'/',len(cfg['evaluated_budgets']),'paths',audit['paths'],'elapsed',round(time.monotonic()-started,1),flush=True)
    audit['elapsed_simulation_seconds']=time.monotonic()-started;dump(out/'audit.json',audit)
    from .audit import audit_output
    audit_output(out,args.data,source,comparison)
    from .analysis import analyze
    analyze(out)
    print('DONE',out,'elapsed',round(time.monotonic()-started,1),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=str(ROOT/'experiments/workflow_extensions/outputs/m4_carry/real_v1_20261116'))
    p.add_argument('--comparison',default=str(ROOT/'experiments/workflow_extensions/outputs/m4_carry/rootguard_mopt_20261116'))
    p.add_argument('--data',default=str(ROOT/'data_mine/finite_cache'));p.add_argument('--output',required=True);p.add_argument('--precheck',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
