"""B=960 MAP diagnostics at common initial roots and actual publication times.

Replays saved transcripts; never changes mechanisms, plans or noise. Verifies
that all previously reported MAP results are reproduced before adding endpoints.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.core import key_seed
from experiments.workflow_extensions.m4_carry.analysis import bootstrap_curves,write_csv
from experiments.workflow_extensions.m4_carry.run import ROOT,dump,line
from .run import replay_frozen
from .attack import evaluate_attack
from .analysis import tensors,interval,ratio,fmt


def run(out,directory=None):
    out=Path(out);directory=directory or ROOT/'data_mine/finite_cache'
    cfg=json.loads((out/'config.json').read_text());public=json.loads((out/'public_stats.json').read_text());assignment=json.loads((out/'split.json').read_text())
    assert data.input_hashes(directory,cfg['templates'])==json.loads((out/'input_hashes.json').read_text())
    methods=cfg['attack_methods'];metrics={};actions=defaultdict(list);previous={}
    for filename,dest in [('metrics.jsonl',metrics),('actions.jsonl',actions)]:
        for ln in (out/filename).open():
            r=json.loads(ln)
            if r['B']!=960 or r['method'] not in methods:continue
            key=(r['trajectory'],r['B'],r['method'],r['repeat'])
            if filename=='metrics.jsonl':dest[key]=r
            else:dest[key].append(r)
    for ln in (out/'attack_metrics.jsonl').open():
        r=json.loads(ln)
        if r['B']==960:previous[r['trajectory'],r['method'],r['repeat'],r['attack'],r['prior']]=r
    rows=[];maxdiff=0.
    with (out/'attack_checks_960.jsonl').open('w') as handle:
        for rec in data.load(directory,cfg['templates']):
            if assignment[str(rec['subject_id'])]!='test':continue
            w=Workload(rec['template']);pp=public[w.template]['main']
            for method in methods:
                result=replay_frozen(rec,960.,method,cfg['repeats'],metrics,actions,w)
                kernel='native_exp' if method.endswith('-exp') else 'laplace'
                for r in evaluate_attack(rec,result,w,pp,kernel,cfg):
                    old=previous[rec['trajectory_id'],method,r['repeat'],r['attack'],r['prior']]
                    diff=np.max(abs(np.array(r['root_mae'])-old['root_mae']));maxdiff=max(maxdiff,float(diff));assert diff<1e-10
                    rr=dict(template=w.template,subject=str(rec['subject_id']),trajectory=rec['trajectory_id'],B=960.,method=method,**r)
                    line(handle,rr);rows.append(rr)
    summary=[];paired=[]
    for t in cfg['templates']:
        for attack,prior in [('snapshot_map','uniform'),('snapshot_map','informed'),('history_map','informed')]:
            ss=[r for r in rows if r['template']==t and r['attack']==attack and r['prior']==prior]
            subjects=sorted({r['subject'] for r in ss});seed=key_seed(cfg['bootstrap_seed'],t)
            for endpoint,numerator,denominator in [('initial','initial_root_mae','initial_root_truth_abs'),('release','root_release_mae','root_release_truth_abs')]:
                vals=[];pboots=[];nboots=[];roots=public[t]['main']['roots']
                for i,root in enumerate(roots):
                    for r in ss:r['num']=r[numerator][i];r['den']=r[denominator][i]
                    x=tensors(ss,methods,[960.],subjects,cfg['repeats'],'num');y=tensors(ss,methods,[960.],subjects,cfg['repeats'],'den')
                    p,n=bootstrap_curves(x,cfg['bootstrap'],seed);pd,nd=bootstrap_curves(y,cfg['bootstrap'],seed)
                    vals.append(100*ratio(x.mean(axis=(-1,-2)),y.mean(axis=(-1,-2))))
                    pboots.append(100*ratio(p,pd));nboots.append(100*ratio(n,nd))
                val=np.mean(vals,axis=0);pb=np.mean(pboots,axis=0);nb=np.mean(nboots,axis=0)
                for mi,m in enumerate(methods):
                    r=dict(template=t,B=960,endpoint=endpoint,attack=attack,prior=prior,method=m,root_macro_wmape=float(val[mi,0]))
                    for i,root in enumerate(roots):r['wmape__'+root]=float(vals[i][mi,0])
                    for name,boot in [('patient',pb),('two_layer',nb)]:r[name+'_lo'],r[name+'_hi']=interval(boot[mi,0])
                    summary.append(r)
                ai=methods.index('M5')
                for reference in [m for m in methods if m!='M5']:
                    bi=methods.index(reference);r=dict(template=t,endpoint=endpoint,attack=attack,prior=prior,reference=reference,delta=float(val[ai,0]-val[bi,0]))
                    for name,boot in [('patient',pb),('two_layer',nb)]:r[name+'_lo'],r[name+'_hi']=interval(boot[ai,0]-boot[bi,0])
                    paired.append(r)
    write_csv(out/'attack_endpoint_summary.csv',summary);write_csv(out/'attack_endpoint_paired.csv',paired)
    dump(out/'attack_checks_audit.json',dict(passed=True,paths=len(metrics),attack_rows=len(rows),max_previous_map_difference=maxdiff,
                                          frozen_transcripts=True,no_new_private_releases=True,no_parameter_or_prior_selection=True,
                                          initial_target_identical_across_methods=True,
                                          analysis_source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('attack.py')]}))
    text=['## 9. MAP端点复核：避免把陈旧误差当作隐私提升','',
          '历史Viterbi在若干模板上比单版本MAP更差。这是所选轨迹先验/路径众数与误差指标错配的实测结果，不能把历史模型当成更强对手，也不能以其较大误差宣称隐私更好。原版风格snapshot MAP保留为主攻击；history作为模型敏感性诊断。','',
          '以下预指定B=960，使用相同实际转录重构，不产生新发布。共同初始根值在所有方法相同，避免不同刷新时间/根状态导致目标不一致；实际发布时刻指标单独列出，避免缓存回答期间的陈旧误差，但其评价时刻随方法变化。两者均为逐根wMAPE后宏平均，越低表示攻击越成功。','',
          '|模板|方法|初始根 snapshot informed %|发布时刻 snapshot informed %|','|---|---|---:|---:|']
    for t in cfg['templates']:
        for m in methods:
            rr={r['endpoint']:r for r in summary if r['template']==t and r['method']==m and r['attack']=='snapshot_map' and r['prior']=='informed'}
            text.append('|'+t+'|'+m+'|'+fmt(rr['initial']['root_macro_wmape'])+'|'+fmt(rr['release']['root_macro_wmape'])+'|')
    text+=['','M5相对M4-carry的共同初始根重构差值；正数表示M5在该攻击下更难重构，负数表示更易重构。不是DP强弱证明。','',
           '|模板|差值（百分点）|两层95% CI|','|---|---:|---|']
    for r in paired:
        if r['endpoint']=='initial' and r['attack']=='snapshot_map' and r['prior']=='informed' and r['reference']=='M4-carry':
            text.append(f'|{r["template"]}|{r["delta"]:.3f}|[{fmt(r["two_layer_lo"])},{fmt(r["two_layer_hi"])}]|')
    text+=['',f'端点复核重放与既有全轨迹MAP最大差={maxdiff:.3g}。所有先验仍为原train固定先验，未按test挑攻击。原始端点统计及区间见attack_checks_960.jsonl、attack_endpoint_summary.csv、attack_endpoint_paired.csv。','']
    report=out/'REPORT-zh.md';old=report.read_text().split('## 9. MAP端点复核')[0].rstrip();report.write_text(old+'\n\n'+'\n'.join(text))
    print('MAP ENDPOINT AUDIT',len(metrics),'paths, max difference',maxdiff,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--data')
    args=p.parse_args();run(args.output,args.data)
