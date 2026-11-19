"""Frozen B2-val schedule + M4-carry online veto/rollover; no re-selection.

This aligns opportunity times, rho=.5 and all-root candidate sets to B2-val.
Actual releases/selected roots/spend can differ due to veto. It does not claim
fixed-action pure-timing matching or isolate timing from every offline choice.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from .core import Plan, Policy, simulate, evaluate, key_seed, realize
from .run import ROOT, dump, line
from .analysis import write_csv, bootstrap_curves, q95, bmin

METHODS=['B2-val','M4-carry','M4-carry-B2schedule']


def load_plan(value):
    return Plan(value['rho'],tuple(value['fractions']),tuple(tuple(s) for s in value['sets']),value['period'])


def aligned_pair(rec,B,b2plan,policy,cfg,phase,repeats):
    baseline=simulate(rec,B,b2plan,'periodic',policy,cfg,phase,repeats,True)
    carry=simulate(rec,B,b2plan,'carry',policy,cfg,phase,repeats,True)
    assert baseline['K']==carry['K']
    assert [h['turn'] for h in baseline['histories']]==[h['turn'] for h in carry['histories']]
    assert [h['planned_roots'] for h in baseline['histories']]==[h['planned_roots'] for h in carry['histories']]
    np.testing.assert_array_equal(baseline['anchor_epsilon'],carry['anchor_epsilon'])
    np.testing.assert_array_equal(baseline['anchor_cache'],carry['anchor_cache'])
    assert np.all(carry['spent']<=B+1e-10)
    return baseline,carry


def analyze(out,cfg,rows,targets):
    summary,paired,bmins=[] ,[],[]
    for template in cfg['templates']:
        rr=[r for r in rows if r['template']==template]
        subjects=sorted({r['subject'] for r in rr})
        budgets=cfg['evaluated_budgets'];R=cfg['repeats']
        groups=defaultdict(list)
        for r in rr:groups[r['method'],r['B'],r['subject'],r['repeat']].append(r)
        x=np.empty((3,len(budgets),len(subjects),R))
        for mi,m in enumerate(METHODS):
            for bi,B in enumerate(budgets):
                for ni,s in enumerate(subjects):
                    for rep in range(R):x[mi,bi,ni,rep]=np.mean([r['mae'] for r in groups[m,B,s,rep]])
                cell=[r for r in rr if r['method']==m and r['B']==B]
                d=dict(template=template,B=B,method=m,patients=len(subjects),repeats=R)
                for metric in ['mae','p95','m','spend_fraction','projection_rate','frac_veto','harmful_refresh','MissRate','K']:
                    pv=[]
                    for s in subjects:
                        v=[r.get(metric) for r in cell if r['subject']==s and r.get(metric) is not None]
                        if v:pv.append(float(np.mean(v)))
                    d[metric]=float(np.mean(pv)) if pv else None
                summary.append(d)
        p,n=bootstrap_curves(x,cfg['bootstrap'],key_seed(cfg['bootstrap_seed'],template))
        mean=x.mean(axis=(-1,-2))
        for ai,bi in [(2,0),(2,1)]:
            for bidx,B in enumerate(budgets):
                d=dict(template=template,B=B,method=METHODS[ai],reference=METHODS[bi],delta_mae=float(mean[ai,bidx]-mean[bi,bidx]),ratio=float(mean[ai,bidx]/mean[bi,bidx]))
                for name,boot in [('patient',p),('two_layer',n)]:
                    d[name+'_lo'],d[name+'_hi']=q95(boot[ai,bidx]-boot[bi,bidx])
                paired.append(d)
        for scenario,ts in targets[template].items():
            for target in ('lo','hi'):
                tau=ts['tau_'+target]
                bm=bmin(mean,budgets,tau)
                for mi,m in enumerate(METHODS):
                    d=dict(template=template,scenario=scenario,target=target,tau=tau,method=m,B_min=float(bm[mi]) if np.isfinite(bm[mi]) else None)
                    for name,boot in [('patient',p),('two_layer',n)]:
                        bb=bmin(boot[mi].T,budgets,tau)
                        a,b=q95(bb)
                        d[name+'_reach_rate']=float(np.isfinite(bb).mean())
                        d[name+'_lo']=a if np.isfinite(a) else None
                        d[name+'_hi']=b if np.isfinite(b) else None
                    bmins.append(d)
    write_csv(out/'summary.csv',summary)
    write_csv(out/'paired_summary.csv',paired)
    write_csv(out/'bmin.csv',bmins)
    txt=['# 同周期骨架下的 M4-carry 补测','',
         '复用原实验数据/划分/统计/54预算/已选P及原M4计划；没有重新规划或调参。新方法与B2-val共享ρ=0.5、全部候选根和精确相同候选时刻，仅在线veto与结转不同。实际发布集合、次数和支出可能不同；不是固定动作包的纯时间实验。', '',
         f'R={cfg["repeats"]}，患者配对及患者＋机制两层bootstrap={cfg["bootstrap"]}。原B2和M4逐路径复现断言通过；所有非空发布ε>0、Σε≤B，初始化相同断言通过。', '',
         '|模板|B2-val MAE @960|原M4-carry|同周期carry|同周期−B2两层95% CI|同周期m / 支出%|', '|---|---:|---:|---:|---|---|']
    for t in cfg['templates']:
        z={r['method']:r for r in summary if r['template']==t and r['B']==960}
        d=next(r for r in paired if r['template']==t and r['B']==960 and r['reference']=='B2-val')
        ci=f'[{d["two_layer_lo"]:.3g}, {d["two_layer_hi"]:.3g}]'
        txt.append(f'|{t}|{z["B2-val"]["mae"]:.4f}|{z["M4-carry"]["mae"]:.4f}|{z[METHODS[2]]["mae"]:.4f}|{ci}|{z[METHODS[2]]["m"]:.3f} / {100*z[METHODS[2]]["spend_fraction"]:.2f}|')
    meld_change=next(r for r in paired if r['template']=='MELD_NA' and r['B']==960 and r['reference']=='M4-carry')
    anemia_change=next(r for r in paired if r['template']=='ANEMIA' and r['B']==960 and r['reference']=='B2-val')
    anemia_endpoint={r['method']:r['mae'] for r in summary if r['template']=='ANEMIA' and r['B']==960}
    anemia_old_percent=100*(anemia_endpoint['M4-carry']/anemia_endpoint['B2-val']-1)
    endpoint=[r for r in summary if r['B']==960 and r['method']==METHODS[2]]
    full=all(abs(r['spend_fraction']-1)<1e-10 and r['frac_veto']==0 for r in endpoint)
    txt += ['', 'B2-val与原M4都采用本轮统一的√ĥ规则，未直接调用仓库RQ1/RQ4的原版M-Opt求解器。ĥ是最终任务在公开归一化域的解析/保守导数界；原版M-Opt使用公开均值处的局部敏感度，按原机制的矩目标数值优化。两者继承敏感度引导分配思想，但不是同一实现；本轮不能称为复现原版RootGuard M-Opt。', '',
            '原B2在每B由validation选择P，候选{2,3,4,5,6,8,10,12}，周期单位为测量机会，初始化固定B/2，剩下B/2均分实际计划刷新次数，逐根按√ĥ；每次刷新全部根覆盖缓存，不平均历史输出。M4的ρ、K、时刻和候选根来自公开计划，可以K=0、ρ=1；两者相同的是会话预算上限，原来不固定实际支出/次数/逐根ε。', '',
            f'注意同周期与原M4比较：当前冻结源中，MELD-Na在B=960两者K=2、ρ=0.5且两次全根候选，替换时刻后MAE差={meld_change["delta_mae"]:.4f}，两层配对区间[{meld_change["two_layer_lo"]:.4f},{meld_change["two_layer_hi"]:.4f}]。ANEMIA/BUN_CR的机会数、FIB4的K=0退化状态也随周期骨架改变，不能把它们的新旧差异全归因时刻。噪声沿用实际turn配对键，未固定动作包/单位噪声重放。', '',
            f'ANEMIA与B2在B=960的差={anemia_change["delta_mae"]:.3g}，最终任务不依赖rbc，veto释放该根极小预算后重新分配；应区分实际可忽略差异与机制收益。该预算四模板是否都花满B且没有整机会否决：{full}。条件成立时，差异来自根级veto/重分配，未体现预算结转增益。', '',
            '|ANEMIA中档目标|B2-val B_min|原M4-carry B_min|同周期carry B_min|', '|---|---:|---:|---:|']
    for target in ('lo','hi'):
        vals={r['method']:r['B_min'] for r in bmins if r['template']=='ANEMIA' and r['scenario']=='medium' and r['target']==target}
        txt.append('|'+target+'|'+ '|'.join(str(vals[m]) for m in METHODS)+'|')
    if (out/'anemia_diagnostics.json').exists():
        diag=json.loads((out/'anemia_diagnostics.json').read_text())
        a=diag['patient_mean']
        txt += ['', f'ANEMIA五位test患者：Hb/Hct相对初值的患者均权平均绝对变化分别{100*a["hb_relative_drift"]:.2f}%/{100*a["hct_relative_drift"]:.2f}%，最终MCHC为{100*a["mchc_relative_drift"]:.2f}%；同批相对增量相关系数{diag["relative_increment_correlation"]:.3f}（{diag["simultaneously_measured_events"]}个双根测量事件）。冻结真实初始MCHC的全轨迹MAE={a["frozen_true_initial_mae"]:.3f}，是离线描述统计，不是下界证明。', '',
                f'Hb与Hct共同变化在比值中大量抵消。原正式实验的ANEMIA中档B_min改善发生在K=0、全B初始化；B2把同一B拆给初始化与后续刷新且覆盖缓存，引入的新噪声在这些轨迹上比降低的陈旧误差更大。B2只调P且不允许静态选择；这两处收益不能归因carry创新。在B=960原M4相对B2的MAE变化为{anemia_old_percent:+.1f}%；B_min优势不是全预算MAE优势。']
    txt += ['', '全网格MAE/配对区间见summary.csv与paired_summary.csv；原公开τ的网格达标及删失见bmin.csv。原报告及输出保持不变。', '',
            '若在同一候选时刻还强制每次全根发布，则没有veto，也没有可结转的跳过金额；在同初始化/逐根ε/同噪声下会逐路径等于B2-val。这里保留veto，所以相同的是候选周期，而不保证实际发布时间完全一致。', '',
            'ANEMIA原因诊断见anemia_diagnostics.json：只评价MCHC=100·Hb/Hct，不能推广为全部贫血相关节点。在稳定比值任务上，多次覆盖发布引入的噪声可能大于减少陈旧误差的收益。原显著B_min改善发生在K=0、ρ=1，等同B0。', '',
            '复现：','', '```bash','cd /home/rootguard-research',
            '/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m4_carry.periodic_skeleton --output experiments/workflow_extensions/outputs/m4_carry/periodic_skeleton_reproduce','```','']
    (out/'REPORT-zh.md').write_text('\n'.join(txt))


def anemia_diagnostics(records,assignment):
    w=Workload('ANEMIA');per=[];dx=[];dy=[]
    for rec in records:
        if rec['template']!='ANEMIA' or assignment[str(rec['subject_id'])]!='test':continue
        x=data.replay(rec);y=w.evaluate(x)[w.target]
        rel=abs(x[:,:2]/x[0,:2]-1)
        per.append(dict(subject=str(rec['subject_id']),frozen_true_initial_mae=float(np.mean(abs(y-y[0]))),
                        hb_relative_drift=float(rel[:,0].mean()),hct_relative_drift=float(rel[:,1].mean()),
                        mchc_relative_drift=float(np.mean(abs(y/y[0]-1)))))
        for e in rec['events']:
            if {'hb','hct'}<=set(e['measured_roots']):
                t=e['turn']-1;d=x[t,:2]/x[t-1,:2]-1;dx.append(d[0]);dy.append(d[1])
    return dict(definition='Offline descriptive statistics only; not used for planning/selection or a lower-bound proof',
                per_patient=per,patient_mean={k:float(np.mean([r[k] for r in per])) for k in per[0] if k!='subject'},
                simultaneously_measured_events=len(dx),relative_increment_correlation=float(np.corrcoef(dx,dy)[0,1]))


def run(args):
    source=Path(args.source).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use an empty output directory')
    cfg=json.loads((source/'config.json').read_text())
    if args.precheck:
        cfg['repeats']=2;cfg['bootstrap']=100
        grid=cfg['evaluated_budgets']
        cfg['evaluated_budgets']=[grid[0],grid[len(grid)//2],grid[-1]]
    cfg['supplement']='exact B2 candidate schedule/rho/all-root skeleton; veto/carry preserved'
    dump(out/'config.json',cfg)
    assignment=json.loads((source/'split.json').read_text())
    stats=json.loads((source/'public_stats.json').read_text())
    selections=json.loads((source/'selections.json').read_text())
    targets=json.loads((source/'targets.json').read_text())
    records=data.load(args.data,cfg['templates'])
    assert data.input_hashes(args.data,cfg['templates'])==json.loads((source/'input_hashes.json').read_text())
    # Reconcile with saved original paths, including the first two reps in precheck.
    original={}
    for ln in (source/'metrics.jsonl').open():
        r=json.loads(ln)
        if r['method'] in METHODS[:2] and r['B'] in cfg['evaluated_budgets'] and r['repeat']<cfg['repeats']:
            original[r['trajectory'],r['B'],r['method'],r['repeat']]=r
    rows=[];reconciled=0;maxdiff=0.;paths=0
    with (out/'metrics.jsonl').open('w') as mh,(out/'schedules.jsonl').open('w') as sh:
        for template in cfg['templates']:
            policy=Policy(Workload(template),stats[template],cfg)
            test=[r for r in records if r['template']==template and assignment[str(r['subject_id'])]=='test']
            for bi,B in enumerate(cfg['evaluated_budgets']):
                sel=selections[template][str(B)]
                b2plan=load_plan(sel['B2-val']);oldplan=load_plan(sel['M4-carry'])
                for rec in test:
                    baseline,new=aligned_pair(rec,B,b2plan,policy,cfg,'test',cfg['repeats'])
                    old=simulate(rec,B,oldplan,'carry',policy,cfg,'test',cfg['repeats'],True)
                    for method,result in zip(METHODS,[baseline,old,new]):
                        met=evaluate(rec,result,policy.w,stats[template],{**cfg,'_B':B},True)
                        for r in range(cfg['repeats']):
                            ident=dict(template=template,trajectory=rec['trajectory_id'],subject=str(rec['subject_id']),B=B,method=method,repeat=r)
                            row={**ident,**{k:v[r] for k,v in met.items()},'K':result['K']}
                            if method in METHODS[:2]:
                                saved=original[rec['trajectory_id'],B,method,r]
                                for k in ['mae','p95','used','m','projection_rate']:
                                    diff=abs(float(row[k])-float(saved[k]));maxdiff=max(maxdiff,diff)
                                    assert diff<1e-10,(template,B,method,k,diff)
                                reconciled+=1
                            line(mh,row);rows.append(json.loads(json.dumps(row,default=lambda v:v.item() if isinstance(v,np.generic) else v)))
                            line(sh,{**ident,'anchor_epsilon':result['anchor_epsilon'][r],
                                     'opportunities':[dict(turn=h['turn'],roots=h['planned_roots'],selected_roots=np.where(h['selected'][r])[0],
                                                           actual_epsilon=h['actual_epsilon'][r],remaining=h['left_after'][r]) for h in result['histories']]})
                            paths+=1
                if bi%10==0 or bi==len(cfg['evaluated_budgets'])-1:print(template,bi+1,'/',len(cfg['evaluated_budgets']),flush=True)
    # JSON null for undefined metrics, consistent with previous logs.
    from .run import clean
    rows=clean(rows)
    dump(out/'anemia_diagnostics.json',anemia_diagnostics(records,assignment))
    analyze(out,cfg,rows,targets)
    dump(out/'audit.json',dict(passed=True,paths=paths,reconciled_original_paths=reconciled,max_original_difference=maxdiff,
                              same_candidate_times=True,same_anchor=True,no_reselection=True,precheck=args.precheck))
    dump(out/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('core.py')]})
    print('DONE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=str(ROOT/'experiments/workflow_extensions/outputs/m4_carry/real_v1_20261116'))
    p.add_argument('--data',default=str(ROOT/'data_mine/finite_cache'))
    p.add_argument('--output',required=True)
    p.add_argument('--precheck',action='store_true')
    run(p.parse_args())
