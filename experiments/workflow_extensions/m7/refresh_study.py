"""AutoResearch refresh attribution and original iso-utility targets."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path

import numpy as np

from ..finite_cache import data
from ..m4_carry.analysis import write_csv
from ..m4_carry.core import key_seed
from ..m4_carry.run import ROOT,dump,line
from .analysis import combine_attack,derived,mean_valid,quantile
from .audit import ledger
from .engine import BASE_PERIODIC,BASE_STATIC,Branch,Context,NATIVE,execute
from .metrics import attack_result,utility
from .routing import BASELINE_PERIODS
from .run import BUDGETS,DEFAULT_SPLIT,LEGACY,SEED

HISTORY=LEGACY
REF=ROOT/'experiments/workflow_extensions/outputs/m4_carry/real_v1_20261116'
ALPHA=0.85
REPEATS=20
DEV_REPEATS=8
BOOT=1000
COMPARATORS=('S-cal','P-cal','P-fusion','D-no-fusion','D-forced-matched')


def read(path):
    return json.loads(path.read_text())


def setup(datadir):
    records=data.load(datadir,NATIVE)
    split=read(DEFAULT_SPLIT)
    hashes=data.input_hashes(datadir,NATIVE)
    assert hashes==read(REF/'input_hashes.json')
    assert split==read(REF/'split.json')
    cfg=read(REF/'config.json')
    parts={t:{role:data.partition(records,split,role,t) for role in ('train','validation','test')}
           for t in NATIVE}
    return parts,hashes,tuple(cfg['evaluated_budgets'])


def profiles(period):
    p=replace(BASE_PERIODIC,period=period,calibration='marginal',shrink=ALPHA)
    gate=replace(p,gate='workload',packages='singles',carry=True)
    return {
        'P-cal':p,
        'P-fusion':replace(p,fusion=True),
        'D-latest':gate,
        'D-fusion':replace(gate,fusion=True),
        'D-fusion-65':replace(gate,fusion=True,anchor=.65),
        'D-latest-65':replace(gate,anchor=.65),
        'D-fusion-80':replace(gate,fusion=True,anchor=.8),
        'P-latest-35':replace(p,anchor=.35),
        'P-latest-65':replace(p,anchor=.65),
        'P-fusion-fast':replace(p,fusion=True,drift_multiplier=4.),
        'D-latest-35':replace(gate,anchor=.35),
        'D-fusion-fast-35':replace(gate,fusion=True,anchor=.35,drift_multiplier=4.),
    }


def development(out,datadir):
    out.mkdir(parents=True,exist_ok=False)
    parts,hashes,grid=setup(datadir)
    static=replace(BASE_STATIC,calibration='marginal',shrink=ALPHA)
    logs,selection=[],{}
    for template,partition in parts.items():
        pilot=partition['validation']
        ctx=Context(template,partition['train'],SEED)
        period_scores={}
        validation={}
        for B in BUDGETS:
            periods={}
            for period in BASELINE_PERIODS:
                p=profiles(period)['P-cal']
                periods[period]=trial(ctx,pilot,B,p)
            best=min(periods,key=lambda p:periods[p]['mae'].mean())
            period_scores[B]=dict(period=best,scores={p:r['mae'].mean() for p,r in periods.items()})
            methods={'S-cal':trial(ctx,pilot,B,static),'P-cal':periods[best]}
            for name,branch in profiles(best).items():
                if name in ('P-fusion','D-latest','D-fusion','D-fusion-65','D-latest-65','D-fusion-80'):
                    methods[name]=trial(ctx,pilot,B,branch)
            validation[B]=methods
        ranks={}
        for name in validation[BUDGETS[0]]:
            if name in ('S-cal','P-cal'):
                continue
            score=[]
            refresh=[]
            for B,methods in validation.items():
                ref=methods['P-cal']
                current=methods[name]
                relative=current['mae'].mean()/max(ref['mae'].mean(),1e-12)
                query=current['query_objective'].mean()/max(ref['query_objective'].mean(),1e-12)
                score.append(.75*relative+.25*query)
                refresh.append(current['did_refresh'].mean())
            ranks[name]=(float(np.mean(score)),float(np.mean(refresh)))
        eligible=[name for name,(_,activity) in ranks.items() if activity>=.25]
        winner=min(eligible,key=lambda name:ranks[name][0])
        previous='P-cal'
        incumbent=1.
        for name,(score,activity) in ranks.items():
            keep=activity>=.25 and score<incumbent
            logs.append(dict(template=template,candidate=name,metric=score,refresh_fraction=activity,
                             decision='keep' if keep else 'discard',incumbent=previous))
            if keep:
                previous,incumbent=name,score
        selection[template]=dict(winner=winner,score=ranks[winner][0],validation=ranks,anchor_periods=period_scores,
                    control_score={B:{name:{k:float(v[k].mean()) for k in ('mae','query_objective','did_refresh')}
                                      for name,v in methods.items()} for B,methods in validation.items()})
        print(template,winner,ranks[winner],flush=True)
        Context.forecasts.cache_clear()
        Context.marginal_grid.cache_clear()
    dump(out/'selection.json',selection)
    dump(out/'config.json',dict(input_hashes=hashes,grid=grid,selection_budgets=BUDGETS,validation_repeats=DEV_REPEATS,
                                selection_metric='.75 paired normalized final MAE + .25 paired normalized query objective',
                                eligible='dynamic actually refreshes >=25% of validation replay paths'))
    with (out/'iterations.tsv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(logs[0]),delimiter='\t')
        writer.writeheader()
        writer.writerows(logs)
    print(f'{np.mean([x["score"] for x in selection.values()]):.8f}',flush=True)


def refine_bun(out,datadir,previous):
    """Spend a second development round only on the template that lost."""
    out.mkdir(parents=True,exist_ok=False)
    parts,hashes,grid=setup(datadir)
    config=read(previous/'config.json')
    assert config['input_hashes']==hashes and tuple(config['grid'])==grid
    chosen=read(previous/'selection.json')
    t='BUN_CR'
    plan=chosen[t]
    ctx=Context(t,parts[t]['train'],SEED)
    pilot=parts[t]['validation']
    candidates=('P-latest-35','P-latest-65','P-fusion-fast','D-latest-35','D-fusion-fast-35')
    logs=[]
    for name in candidates:
        scores,refreshes=[],[]
        for B in BUDGETS:
            ref=plan['control_score'][str(B)]['P-cal']
            P=plan['anchor_periods'][str(B)]['period']
            r=trial(ctx,pilot,B,profiles(P)[name])
            score=.75*r['mae'].mean()/ref['mae']+.25*r['query_objective'].mean()/ref['query_objective']
            scores.append(score)
            refreshes.append(r['did_refresh'].mean())
            plan['control_score'][str(B)][name]={k:float(v.mean()) for k,v in r.items()}
        val,frac=float(np.mean(scores)),float(np.mean(refreshes))
        keep=frac>=.25 and val<plan['score']
        logs.append(dict(template=t,candidate=name,metric=val,refresh_fraction=frac,
                         decision='keep' if keep else 'discard',incumbent=plan['winner']))
        plan['validation'][name]=[val,frac]
        if keep:
            plan['winner'],plan['score']=name,val
    dump(out/'config.json',{**config,'initial_development':str(previous),
                            'followup_reason':'all first-round BUN/Cr candidates lost on validation'})
    dump(out/'selection.json',chosen)
    with (out/'iterations.tsv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(logs[0]),delimiter='\t')
        writer.writeheader()
        writer.writerows(logs)
    print(t,plan['winner'],plan['score'],flush=True)


def trial(ctx, records, B, branch):
    rows=[]
    for record in records:
        val=execute(record,B,branch,ctx,'m7-refresh-validation',DEV_REPEATS)
        metrics=utility(ctx,record,val)
        rows.append(dict(mae=metrics['mae'].mean(),query_objective=metrics['query_objective'].mean(),
                         did_refresh=(metrics['refreshes']>0).mean()))
    return {name:np.array([r[name] for r in rows]) for name in rows[0]}


def route_choice(a,b):
    """Use validation patients only, requiring a positive margin for switching."""
    gain=a['mae']-b['mae']
    return bool(gain.mean()>0.5*gain.std(ddof=1)/np.sqrt(len(gain)))


def choices_for_grid(part,ctx,grid,selection):
    plans={}
    for B in grid:
        ref=part['validation']
        s=replace(BASE_STATIC,calibration='marginal',shrink=ALPHA)
        controls=[(p,trial(ctx,ref,B,profiles(p)['P-cal'])) for p in BASELINE_PERIODS]
        P,pc=min(controls,key=lambda r:(r[1]['mae'].mean(),r[0]))
        choices=profiles(P)
        winner=choices[selection['winner']]
        latest=replace(winner,fusion=False)
        forced=replace(winner,gate='off',packages='all',carry=False)
        sv=trial(ctx,ref,B,s)
        wv=trial(ctx,ref,B,winner)
        choose_d=route_choice(sv,wv)
        choose_p=route_choice(sv,pc)
        plans[B]=dict(branches={'S-cal':s,'P-cal':choices['P-cal'],
                      'P-fusion':choices['P-fusion'],'D-new':winner,'D-no-fusion':latest,
                      'D-forced-matched':forced,'R-new':winner if choose_d else s,
                      'R-control':choices['P-cal'] if choose_p else s},
                      route_D=choose_d,control_D=choose_p,period=P,
                      validation_gain=float((sv['mae']-wv['mae']).mean()),
                      validation_refresh=float(wv['did_refresh'].mean()))
    return plans


def harm(ctx,record,result):
    raw=ctx.work.evaluate(data.replay(record))[ctx.work.target]
    previous=result['anchor_service']
    count=np.zeros(len(previous),int)
    for i,h in enumerate(result['histories']):
        start=h['turn']-1
        end=result['histories'][i+1]['turn']-1 if i+1<len(result['histories']) else len(raw)
        old=ctx.work.from_cache(previous)[ctx.work.target]
        new=ctx.work.from_cache(h['service'])[ctx.work.target]
        worse=np.mean(abs(new[:,None]-raw[start:end]),axis=1)>np.mean(abs(old[:,None]-raw[start:end]),axis=1)+1e-12
        count+=worse&h['selected'].any(axis=1)
        previous=h['service']
    return count


def confirm(out,datadir,dev,grid_limit=None):
    out.mkdir(parents=True,exist_ok=False)
    parts,hashes,grid=setup(datadir)
    config=read(dev/'config.json')
    assert config['input_hashes']==hashes and tuple(config['grid'])==grid
    chosen=read(dev/'selection.json')
    if grid_limit:
        # Optional resource-bounded replay selects evenly spaced grid points,
        # retaining the four already declared anchor budgets.
        grid=tuple(sorted({grid[i] for i in np.rint(np.linspace(0,len(grid)-1,grid_limit)).astype(int)}|set(BUDGETS)))
    dump(out/'config.json',dict(budgets=grid,repeats=REPEATS,validation_repeats=DEV_REPEATS,bootstrap=BOOT,
                                input_hashes=hashes,dev=str(dev),primary_targets=str(REF/'targets.json')))
    contexts,plans={},{}
    for template,part in parts.items():
        ctx=contexts[template]=Context(template,part['train'],SEED)
        plans[template]=choices_for_grid(part,ctx,grid,chosen[template])
        Context.forecasts.cache_clear()
        Context.marginal_grid.cache_clear()
        print('FREEZE',template,sum(p['route_D'] for p in plans[template].values()),'/',len(grid),flush=True)
    dump(out/'frozen.json',{t:{B:{**p,'branches':{m:asdict(b) for m,b in p['branches'].items()}}
                             for B,p in v.items()} for t,v in plans.items()})
    rows=[]
    audit=dict(paths=0,map_rows=0)
    with (out/'paths.jsonl').open('w') as fh,(out/'map.jsonl').open('w') as ah:
        for template,part in parts.items():
            ctx=contexts[template]
            for B in grid:
                plan=plans[template][B]
                for record in part['test']:
                    cache={}
                    for name,branch in plan['branches'].items():
                        if name=='D-no-fusion' and B not in BUDGETS:
                            continue
                        if name=='D-forced-matched' and B not in BUDGETS:
                            continue
                        if branch not in cache:
                            result=execute(record,B,branch,ctx,'test',REPEATS,True)
                            ledger(result,B,branch)
                            u=utility(ctx,record,result)
                            u['did_refresh']=(u['refreshes']>0).astype(float)
                            u['harmful_refreshes']=harm(ctx,record,result)
                            attacks=attack_result(ctx,record,result,branch) if B in BUDGETS else []
                            by_rep=defaultdict(list)
                            for item in attacks:
                                by_rep[item['repeat']].append(item)
                            cache[branch]=u,by_rep
                            audit['paths']+=REPEATS
                        u,by_rep=cache[branch]
                        for rep in range(REPEATS):
                            meta=dict(template=template,B=B,method=name,subject=str(record['subject_id']),repeat=rep)
                            row={**meta,**{key:float(value[rep]) for key,value in u.items()}}
                            line(fh,row)
                            for item in by_rep[rep]:
                                line(ah,{**meta,**item})
                                audit['map_rows']+=1
                            combine_attack(row,by_rep[rep],ctx.work.roots)
                            rows.append(row)
                if B in BUDGETS:
                    print('EVAL',template,B,'paths',audit['paths'],flush=True)
            Context.forecasts.cache_clear()
            Context.marginal_grid.cache_clear()
    archive=historical(parts,grid)
    rows.extend(archive)
    audit['archived_rows']=len(archive)
    summarize(rows,out,contexts,grid,read(REF/'targets.json'))
    dump(out/'audit.json',audit)


def historical(parts,grid):
    assert read(HISTORY/'input_hashes.json')==read(REF/'input_hashes.json')
    assert read(HISTORY/'split.json')==read(DEFAULT_SPLIT)
    assert read(HISTORY/'config.json')['noise_seed']==SEED
    identifiers={(t,str(r['subject_id']),r['trajectory_id']) for t,p in parts.items() for r in p['test']}
    wanted=('RG-static-exp','RG-periodic-exp','M5','M6')
    selected=[]
    with (HISTORY/'metrics.jsonl').open() as f:
        for text in f:
            row=json.loads(text)
            if row['method'] not in wanted or row['B'] not in grid or row['repeat']>=REPEATS:
                continue
            if (row['template'],str(row['subject']),row['trajectory']) not in identifiers:
                continue
            selected.append({k:row[k] for k in ('template','B','subject','repeat','method','mae','final_truth_abs','p95')})
    return selected


def estimate(table,keys,patient,noise,ctx):
    fields=sorted({k for row in table.values() for k in row if k not in ('B','repeat')
                    and isinstance(row[k],(int,float))})
    a=np.array([[[table[s,r].get(field,np.nan) for field in fields] for r in keys[1]] for s in keys[0]],float)
    center=dict(zip(fields,mean_valid(a,(0,1))))
    draws=dict(zip(fields,mean_valid(a[patient,noise],(1,2)).T))
    for group in (center,draws):
        if 'harmful_refreshes' in group:
            den=group['refreshes']
            group['harm_fraction']=np.divide(group['harmful_refreshes'],den,
                                            out=np.full_like(den,np.nan,dtype=float),where=den>0)
    return derived(center,ctx)|{k:center[k] for k in ('did_refresh','harm_fraction') if k in center},            derived(draws,ctx)|{k:draws[k] for k in ('did_refresh','harm_fraction') if k in draws}


def bmin(curve,grid,tau):
    return np.min(np.where(curve<=tau,np.asarray(grid),np.inf),axis=-1)


def finite(x):
    return float(x) if np.isfinite(x) else None


def summarize(rows,out,contexts,grid,thresholds):
    panels=defaultdict(dict)
    for row in rows:
        key=row['template'],row['B'],row['method']
        ident=str(row['subject']),row['repeat']
        assert ident not in panels[key]
        panels[key][ident]=row
    summary,pairs,targets,iso_pairs,macro=[],[],[],[],[]
    for t,ctx in contexts.items():
        example=panels[t,grid[0],'R-new']
        ss=sorted({s for s,_ in example})
        rr=sorted({r for _,r in example})
        rng=np.random.default_rng(key_seed('m7-refresh-bootstrap',t))
        patient=rng.integers(len(ss),size=(BOOT,len(ss),1))
        noise=rng.integers(len(rr),size=(BOOT,len(ss),len(rr)))
        curves=defaultdict(list)
        effects=defaultdict(list)
        for B in grid:
            scores={}
            for tt,bb,method in panels:
                if (tt,bb)!=(t,B):
                    continue
                table=panels[t,B,method]
                assert set(table)==set(example),(t,B,method)
                if B in BUDGETS:
                    p,z=estimate(table,(ss,rr),patient,noise,ctx)
                else:
                    a=np.array([[table[s,r]['mae'] for r in rr] for s in ss])
                    p={'mae':a.mean()}
                    z={'mae':a[patient,noise].mean(axis=(1,2))}
                scores[method]=p,z
                curves[method].append((B,p['mae'],z['mae']))
                for k,v in p.items():
                    lo,hi=quantile(z[k])
                    summary.append(dict(template=t,B=B,method=method,metric=k,estimate=finite(v),
                                        lo95=lo,hi95=hi,patients=len(ss),repeats=len(rr)))
            contrasts=[('D-new',m) for m in COMPARATORS if m in scores]
            contrasts += [('P-fusion','P-cal'),('R-new','R-control'),('R-new','S-cal'),('R-new','M6')]
            for a,b in contrasts:
                p,z=scores[a]
                q,w=scores[b]
                for k in sorted(p.keys()&q.keys()):
                    if not np.isfinite(p[k]) or not np.isfinite(q[k]):
                        continue
                    d=p[k]-q[k]
                    sample=z[k]-w[k]
                    lo,hi=quantile(sample)
                    pairs.append(dict(template=t,B=B,method=a,reference=b,metric=k,delta=float(d),
                                      lo95=lo,hi95=hi,candidate=float(p[k]),baseline=float(q[k])))
                    if B in BUDGETS:
                        effects[a,b,k].append((d,sample))
        for (a,b,k),r in effects.items():
            lo,hi=quantile(np.mean([v for _,v in r],axis=0))
            macro.append(dict(template=t,method=a,reference=b,metric=k,delta=float(np.mean([v for v,_ in r])),
                              lo95=lo,hi95=hi,anchor_budgets=len(r)))
        for scenario,fixed in thresholds[t].items():
            for bound in ('lo','hi'):
                tau=fixed['tau_'+bound]
                reached={}
                for method,curve in curves.items():
                    if [r[0] for r in curve]!=list(grid):
                        continue
                    _,mean,draws=zip(*curve)
                    m=bmin(np.array(mean),grid,tau)
                    samples=np.array(draws).T
                    boot=bmin(samples,grid,tau)
                    conservative=bmin(np.quantile(samples,.95,axis=0),grid,tau)
                    reached[method]=(m,boot)
                    conditional=quantile(boot)
                    targets.append(dict(template=t,scenario=scenario,target=bound,tau=tau,method=method,
                                        B_min=finite(m),B_min_UCB95=finite(conservative),reached=bool(np.isfinite(m)),
                                        bootstrap_reached=float(np.isfinite(boot).mean()),
                                        B_min_lo_reached=conditional[0],B_min_hi_reached=conditional[1]))
                for a,b in [('D-new','P-cal'),('D-new','S-cal'),('P-fusion','P-cal'),
                            ('R-new','R-control'),('R-new','S-cal'),('R-new','M6'),
                            ('R-new','RG-static-exp'),('R-new','RG-periodic-exp')]:
                    ma,ba=reached[a]
                    mb,bb=reached[b]
                    paired=np.isfinite(ba)&np.isfinite(bb)
                    lo,hi=quantile(1-ba[paired]/bb[paired])
                    iso_pairs.append(dict(template=t,scenario=scenario,target=bound,tau=tau,method=a,reference=b,
                         candidate_B_min=finite(ma),reference_B_min=finite(mb),
                         saving=float(1-ma/mb) if np.isfinite(ma) and np.isfinite(mb) else None,
                         both_bootstrap_reached=float(paired.mean()),saving_lo_reached=lo,saving_hi_reached=hi))
    for name,output in [('summary',summary),('paired',pairs),('anchor_average',macro),
                        ('bmin',targets),('bmin_paired',iso_pairs)]:
        write_csv(out/f'{name}.csv',output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('development','refine-bun','confirm'))
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--dev',type=Path)
    p.add_argument('--data',type=Path,default=ROOT/'data_mine/finite_cache')
    p.add_argument('--grid-limit',type=int,help='Optional coarse replay, always includes the four anchor budgets')
    args=p.parse_args()
    if args.stage=='development':
        development(args.output,args.data)
    elif args.dev is None:
        p.error('--dev is required for confirm')
    elif args.stage=='refine-bun':
        refine_bun(args.output,args.data,args.dev)
    else:
        confirm(args.output,args.data,args.dev,args.grid_limit)
