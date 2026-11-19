"""Repair only the L1-gate ablation objective; preserve every other raw row."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.core import Plan
from experiments.workflow_extensions.m4_carry.run import ROOT,dump,line
from .core import simulate
from .metrics import evaluate
from .run import write_actions


def run(args):
    prior=Path(args.prior).resolve();out=Path(args.output).resolve();source=Path(args.source).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Output must be empty')
    cfg=json.loads((prior/'config.json').read_text());cfg['derived_from']=str(prior)
    cfg['repair']='Only local L1-gate ablation extended to the same frozen multi-node query objective; no tuning'
    for name in ['split.json','targets.json','input_hashes.json','public_stats.json','selections.json','phase_a_complete.json','validation_scores.jsonl','source_hashes.json','attack_metrics.jsonl']:
        shutil.copyfile(prior/name,out/name)
    dump(out/'config.json',cfg)
    before={};after={};method='M5-local-veto-fixed'
    for filename in ['metrics.jsonl','actions.jsonl']:
        a=hashlib.sha256();b=hashlib.sha256()
        with (prior/filename).open() as src,(out/filename).open('w') as dest:
            for text in src:
                if json.loads(text)['method']!=method:
                    a.update(text.encode());dest.write(text);b.update(text.encode())
        before[filename]=a.hexdigest();after[filename]=b.hexdigest()
    assert before==after
    records=data.load(args.data,cfg['templates']);assignment=json.loads((out/'split.json').read_text())
    stats=json.loads((out/'public_stats.json').read_text());oldstats=json.loads((source/'public_stats.json').read_text())
    plans=json.loads((out/'selections.json').read_text());paths=0
    with (out/'metrics.jsonl').open('a') as mh,(out/'actions.jsonl').open('a') as ah:
        for t in cfg['templates']:
            w=Workload(t);public=stats[t]['main'];test=data.partition(records,assignment,'test',t)
            for B in cfg['evaluated_budgets']:
                pp=plans[t][str(B)];plan=Plan(pp['rho'],tuple(pp['fractions']),tuple(tuple(s) for s in pp['sets']))
                for rec in test:
                    result=simulate(rec,B,plan,public,cfg,'test',cfg['repeats'],gate='local',carry=False,old_stats=oldstats[t],detailed=True)
                    met=evaluate(rec,result,w,oldstats[t],dict(cfg,_B=B),public)
                    ident=dict(template=t,subject=str(rec['subject_id']),trajectory=rec['trajectory_id'],B=B,method=method)
                    for r in range(cfg['repeats']):
                        line(mh,{**ident,'repeat':r,**{k:v[r] for k,v in met.items()},'K':result['K'],'rho':plan.rho,'stop_reason':result['stop_reason']})
                    write_actions(ah,ident,result,rec,B,method);paths+=cfg['repeats']
            print('REPAIR',t,'paths',paths,flush=True)
    inherited=json.loads((prior/'audit.json').read_text());inherited['repaired_local_ablation_paths']=paths
    dump(out/'audit.json',inherited)
    dump(out/'repair_audit.json',dict(only_method_recomputed=method,recomputed_paths=paths,unaffected_row_hashes_before=before,
                                   unaffected_row_hashes_after=after,all_other_raw_rows_identical=True,parameters_and_selections_changed=False,
                                   attack_raw_identical=hashlib.sha256((prior/'attack_metrics.jsonl').read_bytes()).hexdigest()==hashlib.sha256((out/'attack_metrics.jsonl').read_bytes()).hexdigest()))
    files=list(Path(__file__).parent.glob('*.py'))
    dump(out/'repair_source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    from .audit import audit_output
    audit_output(out,args.data,source,args.comparison)
    from .analysis import analyze
    analyze(out)
    print('DONE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prior',required=True);p.add_argument('--output',required=True)
    p.add_argument('--source',default=str(ROOT/'experiments/workflow_extensions/outputs/m4_carry/real_v1_20261116'))
    p.add_argument('--comparison',default=str(ROOT/'experiments/workflow_extensions/outputs/m4_carry/rootguard_mopt_20261116'))
    p.add_argument('--data',default=str(ROOT/'data_mine/finite_cache'));run(p.parse_args())
