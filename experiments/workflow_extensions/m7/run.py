"""M7 matched-baseline benchmark and optional historical M5/M6 comparison."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ..finite_cache import data
from ..m4_carry.run import ROOT,dump,line
from .analysis import combine_attack,summarize
from .audit import ledger
from .engine import BASE_STATIC, Context, NATIVE, execute
from .metrics import attack_result,utility
from .routing import (fit,make_route,periodic_controls,select_rule,source_scores)

BUDGETS=(80.,318.99,651.22,960.)
ROOT_OUTPUTS=ROOT/'experiments/workflow_extensions/outputs'
DEFAULT_SPLIT=ROOT_OUTPUTS/'m4_carry/rootguard_mopt_20261116/split.json'
LEGACY=ROOT_OUTPUTS/'m6/real_v2_20261117'
SEED=99361116


def read_json(path):
    with path.open() as f:
        return json.load(f)


def old_rows(legacy, contexts, held, budgets, repeats):
    """Frozen M5/M6 rows on the exact same subject, trajectory, budget and noise phase."""
    paths={}
    valid={(t,str(rec['subject_id']),rec['trajectory_id']) for t,records in held.items() for rec in records}

    def eligible(row):
        return (row['method'] in ('M5','M6') and row['repeat']<repeats and row['B'] in budgets and
                (row['template'],str(row['subject']),row['trajectory']) in valid)

    def key(r):
        return r['template'],r['B'],r['method'],str(r['subject']),r['repeat']

    with (legacy/'metrics.jsonl').open() as f:
        for line_ in f:
            row=json.loads(line_)
            if eligible(row):
                selected={k:row.get(src) for k,src in [
                    ('mae','mae'),('final_truth_abs','final_truth_abs'),('p95','p95'),('rce','rce'),('cmr','cmr'),
                    ('refreshes','m'),('roots','n_pub'),('spent','used')]}
                for node in contexts[row['template']].nodes:
                    selected[f'num:{node}']=row.get(f'node_num__{node}')
                    selected[f'den:{node}']=row.get(f'node_den__{node}')
                paths[key(row)]=dict(selected,template=row['template'],B=row['B'],method=row['method'],
                                     subject=str(row['subject']),repeat=row['repeat'])
    with (legacy/'attack_metrics.jsonl').open() as f:
        for line_ in f:
            row=json.loads(line_)
            if eligible(row):
                combine_attack(paths[key(row)],[row],contexts[row['template']].work.roots)
    return paths


def run(args):
    root=args.data
    templates=tuple(args.templates)
    out=args.output
    records=data.load(root,templates)
    if args.split:
        split=read_json(args.split)
    elif DEFAULT_SPLIT.exists() and root.resolve()==(ROOT/'data_mine/finite_cache').resolve() and set(templates)<=set(NATIVE):
        split=read_json(DEFAULT_SPLIT)
    else:
        split=data.split(records,20261110)
    if not {str(rec['subject_id']) for rec in records} <= set(split):
        raise ValueError('The supplied split omits patients from the selected inputs')
    partitions={t:{p:data.partition(records,split,p,t) for p in ('train','validation','test')} for t in templates}
    for t,parts in partitions.items():
        if len(parts['train'])<4 or len(parts['validation'])<2 or not parts['test']:
            raise ValueError(f'{t}: at least four training, two calibration, and one test patient required')
    hashes=data.input_hashes(root,templates)
    if args.legacy:
        if args.repeats>20:
            raise ValueError('The archived M5/M6 comparison supports at most 20 repeats')
        frozen_hashes=read_json(args.legacy/'input_hashes.json')
        if any(frozen_hashes.get(n)!=h for n,h in hashes.items()) or split!=read_json(args.legacy/'split.json'):
            raise ValueError('Legacy pairing requires exact frozen inputs and patient split')
        oldconfig=read_json(args.legacy/'config.json')
        if oldconfig['noise_seed']!=SEED or not set(args.budgets)<=set(oldconfig['evaluated_budgets']):
            raise ValueError('Legacy pairing requires the original stream and evaluated budgets')
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',dict(templates=templates,budgets=args.budgets,repeats=args.repeats,
                                fit_repeats=args.fit_repeats,bootstrap=args.bootstrap,profile=args.profile,
                                noise_seed=SEED,legacy=str(args.legacy) if args.legacy else None,
                                eval_phase='test',task_gate='half normalized final + half normalized actual queries',
                                calibration='branch profile, same rule on reference, validation and test'))
    dump(out/'input_hashes.json',hashes)
    dump(out/'split.json',split)
    dump(out/'source_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(__file__).parent.glob('*.py'))})
    budgets=tuple(sorted(set(args.budgets)))
    if not budgets or budgets[0]<=0:
        raise ValueError('Supply a positive budget grid')

    def ctx_for(t,ref):
        return Context(t,ref,SEED)

    scores={}
    for objective in ('final','workload'):
        scores[objective]={t:source_scores(t,partitions[t]['train'],partitions[t]['validation'],budgets,
                                           args.profile,objective,args.fit_repeats,ctx_for) for t in templates}
        print('SOURCE',objective,flush=True)
    contexts,prepared,freeze={}, {},{}
    for t,parts in partitions.items():
        train,pilot=parts['train'],parts['validation']
        ctx=contexts[t]=ctx_for(t,train)
        setup={}
        for objective in ('final','workload'):
            features=fit(ctx,train,pilot,budgets,args.profile,objective,args.fit_repeats,'m7-target')
            rule=select_rule(t,scores[objective])
            setup[objective]=make_route(features,rule)
        controls={B:periodic_controls(ctx,train+pilot,B,args.fit_repeats) for B in budgets}
        prepared[t]=setup,controls
        freeze[t]=dict(routes={k:dict(rule=v.rule,choices=v.choices,
                                   branches={B:[asdict(a) for a in pair] for B,pair in v.branches.items()})
                               for k,v in setup.items()},
                       periodic_controls={B:[asdict(a) for a in pair] for B,pair in controls.items()},
                       references=len(train),calibration=len(pilot),held=len(parts['test']))
        print('FREEZE',t,{k:[int(v.choices[B]) for B in budgets] for k,v in setup.items()},flush=True)
        Context.forecasts.cache_clear()
    dump(out/'source_selection_scores.json',scores)
    dump(out/'frozen_routes.json',freeze)

    paths=[]
    stats=dict(ledger_checked_paths=0,attack_rows=0,selected_matches=0)
    with (out/'metrics.jsonl').open('w') as utility_file,(out/'map.jsonl').open('w') as attack_file:
        for t,parts in partitions.items():
            ctx=contexts[t]
            setup,controls=prepared[t]
            final,aware=setup['final'],setup['workload']
            for rec in parts['test']:
                for B in budgets:
                    S,D=aware.branches[B]
                    P,PC=controls[B]
                    methods={'RG-static-exp':BASE_STATIC,'RG-periodic-exp':P,'RG-periodic-cal':PC,
                             'M7-S':S,'M7-D':D,'M7-D-final':final.branches[B][1],
                             'M7-final':final.choose(B),'M7':aware.choose(B)}
                    computed={}
                    for method,branch in methods.items():
                        if branch not in computed:
                            result=execute(rec,B,branch,ctx,'test',args.repeats,True)
                            ledger(result,B,branch)
                            metrics=utility(ctx,rec,result)
                            attacks=attack_result(ctx,rec,result,branch)
                            computed[branch]=metrics,attacks
                            stats['ledger_checked_paths']+=args.repeats
                        metrics,attacks=computed[branch]
                        by_rep={r:[] for r in range(args.repeats)}
                        for row in attacks:
                            by_rep[row['repeat']].append(row)
                        for rep in range(args.repeats):
                            ident=dict(template=t,B=B,method=method,subject=str(rec['subject_id']),repeat=rep,
                                       branch='S' if branch.period is None else 'D',
                                       period=branch.period,trajectory=rec['trajectory_id'])
                            row={**ident,**{k:float(v[rep]) for k,v in metrics.items()}}
                            line(utility_file,row)
                            for item in by_rep[rep]:
                                line(attack_file,{**ident,**item})
                                stats['attack_rows']+=1
                            combine_attack(row,by_rep[rep],ctx.work.roots)
                            paths.append(row)
                            if method=='M7':
                                assert branch==aware.branches[B][int(aware.choices[B])]
                                stats['selected_matches']+=1
            print('EVALUATED',t,flush=True)
            Context.forecasts.cache_clear()
    if args.legacy:
        old=old_rows(args.legacy,contexts,{t:p['test'] for t,p in partitions.items()},budgets,args.repeats)
        with (out/'legacy_paired.jsonl').open('w') as f:
            for row in old.values():
                line(f,row)
        paths.extend(old.values())
        stats['legacy_rows']=len(old)
    summarize(paths,out,contexts,args.bootstrap)
    dump(out/'audit.json',stats)
    print(f'COMPLETED {out}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data_mine/finite_cache')
    p.add_argument('--split',type=Path)
    p.add_argument('--templates',nargs='+',default=NATIVE)
    p.add_argument('--budgets',nargs='+',type=float,default=BUDGETS)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--fit-repeats',type=int,default=10)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--profile',choices=('portable','historical'),default='portable')
    p.add_argument('--legacy',type=Path,help=f'Optional paired M5/M6 archive, for example {LEGACY}')
    run(p.parse_args())
