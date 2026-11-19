"""Component interpretation using already frozen raw results, no new tuning."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .analysis import bootstrap_curves,q95,write_csv
from .core import allocate,key_seed
from .run import ROOT

NAMES=['sqrt_h-static-lap','sqrt_h-periodic-lap','M4-carry','MOpt-static-lap']
MARK='## 9. 现有配对数据对组件的解释'


def diagnose(source,out):
    source=Path(source);out=Path(out);cfg=json.loads((out/'config.json').read_text())
    stats=json.loads((source/'public_stats.json').read_text())
    old={}
    name_map={'B0':NAMES[0],'B2-val':NAMES[1],'M4-carry':NAMES[2]}
    for ln in (source/'metrics.jsonl').open():
        r=json.loads(ln)
        if r['method'] in name_map and r['B'] in cfg['evaluated_budgets'] and r['repeat']<cfg['repeats']:
            k=(r['template'],name_map[r['method']],r['B'],r['subject'],r['repeat'])
            old.setdefault(k,[]).append(r['mae'])
    for ln in (out/'metrics.jsonl').open():
        r=json.loads(ln)
        if r['method']=='RG-static-lap':
            k=(r['template'],NAMES[3],r['B'],r['subject'],r['repeat'])
            old.setdefault(k,[]).append(r['mae'])
    effects=[];endpoint=[]
    for t in cfg['templates']:
        subs=sorted({k[3] for k in old if k[0]==t});budgets=cfg['evaluated_budgets'];R=cfg['repeats']
        x=np.empty((4,len(budgets),len(subs),R))
        for mi,m in enumerate(NAMES):
            for bi,B in enumerate(budgets):
                for ni,s in enumerate(subs):
                    for rep in range(R):x[mi,bi,ni,rep]=np.mean(old[t,m,B,s,rep])
        p,n=bootstrap_curves(x,cfg['bootstrap'],key_seed(cfg['bootstrap_seed'],t))
        mean=x.mean(axis=(-1,-2))
        for a,b,label in [(0,3,'static_allocation'),(2,0,'dynamic_over_same_static_allocation'),(2,1,'dynamic_over_same_periodic_allocation')]:
            for bi,B in enumerate(budgets):
                r=dict(template=t,B=B,comparison=label,method=NAMES[a],reference=NAMES[b],
                       mae=float(mean[a,bi]),reference_mae=float(mean[b,bi]),delta_mae=float(mean[a,bi]-mean[b,bi]))
                for kind,boot in [('patient',p),('two_layer',n)]:r[kind+'_lo'],r[kind+'_hi']=q95(boot[a,bi]-boot[b,bi])
                effects.append(r)
    logs=[json.loads(l) for l in (out/'mopt_allocations.jsonl').open()]
    weights=[]
    for t in cfg['templates']:
        info=next(l for l in logs if l['template']==t and l['total_metric']==960)
        h=np.array(stats[t]['h']);new=allocate(np.array(960.),np.ones(len(h),bool),h)
        from experiments.workflow_extensions.finite_cache.workload import Workload
        roots=Workload(t).roots
        for i,root in enumerate(roots):weights.append(dict(template=t,root=root,MOpt_epsilon_metric=info['epsilon_metric'][i],sqrt_h_epsilon_metric=float(new[i]),sqrt_h_share=float(new[i]/960)))
    write_csv(out/'component_effects.csv',effects);write_csv(out/'allocation_at_960.csv',weights)
    text=[MARK,'',
          '以下只复用已有原始结果，不新增test选择。静态分配对照具有相同B、真值初始化、连续Laplace发布核、相同逐根单位噪声和固定缓存，区别为分配规则整体（局部/全域敏感度、精确目标/闭式及预算下限）。它能评价分配替换，不等于仅改变敏感度估计的单因素消融。动态对照还含日程/ρ/根级veto等差别。','',
          '|模板|对照（B=960）|MAE差|两层95% CI|','|---|---|---:|---|']
    for r in effects:
        if r['B']==960:text.append(f'|{r["template"]}|{r["method"]} − {r["reference"]}|{r["delta_mae"]:.4f}|[{r["two_layer_lo"]:.4f}, {r["two_layer_hi"]:.4f}]|')
    meld=[r for r in effects if r['template']=='MELD_NA' and r['B']==960]
    if meld:
        a=next(r for r in meld if r['comparison']=='static_allocation')
        b=next(r for r in meld if r['comparison']=='dynamic_over_same_static_allocation')
        c=next(r for r in meld if r['comparison']=='dynamic_over_same_periodic_allocation')
        text+=['',f'MELD-Na在同Laplace核下，原M-Opt静态MAE={a["reference_mae"]:.3f}，只换成全域界√ĥ静态分配后={a["mae"]:.3f}，再采用M4动态系统={b["mae"]:.3f}。这是一条有顺序的对照路径，不是Shapley分解；不能据此把全部收益归给结转。', '',
               f'已有√ĥ周期基线MAE={c["reference_mae"]:.3f}，与M4的差值区间[{c["two_layer_lo"]:.4f},{c["two_layer_hi"]:.4f}]。因此即使M4优于原M-Opt，仍需核查是否主要是分配改进，而不是复杂在线机制。']
    na=next((r for r in weights if r['template']=='MELD_NA' and r['root']=='na'),None)
    if na:text+=['',f'本批train均值Na=137.62，落在冻结公式Na截断段，原局部梯度为0，原M-Opt在B=960只分到ε_metric≈{na["MOpt_epsilon_metric"]:.3f}；全域界静态分配约{na["sqrt_h_epsilon_metric"]:.3f}。这与“局部平坦导致演化后预算不足”的解释一致；本轮没有单独做Na分配消融，不能认定它是唯一原因。']
    from collections import Counter
    root_sets=Counter()
    for ln in (out/'actions.jsonl').open():
        r=json.loads(ln)
        if r['method']=='M4-carry' and r['B']==960 and r['action_index']>0:
            root_sets[r['template'],tuple(r['selected_roots'])]+=1
    write_csv(out/'released_root_sets_at_960.csv',[dict(template=k[0],roots=','.join(k[1]),count=v) for k,v in root_sets.items()])
    total_meld=sum(v for (t,s),v in root_sets.items() if t=='MELD_NA')
    partial_meld=sum(v for (t,s),v in root_sets.items() if t=='MELD_NA' and len(s)<4)
    if total_meld:
        text+=['',f'B=960的MELD-Na在{total_meld}个后续实际动作中，有{partial_meld}个只发布部分根（{partial_meld/total_meld:.1%}）；该模块确实触发，但独立收益尚未由这个系统比较识别。具体根集合次数见released_root_sets_at_960.csv。']
    text+=['','既有同计划M4/carry对照在全部24个目标上的B_min相同，B=960四模板的MAE也相同；本轮新增RootGuard对照不改变这一结转归因事实。部分根更新的独立收益仍需固定日程/动作总预算的选根消融。','',
           '### 数值审计','']
    auditfile=out/'independent_audit.json'
    if auditfile.exists():
        a=json.loads(auditfile.read_text())
        text += [f'原版{a["solver_non_success_calls"]}个非成功状态调用均完成同一公开目标的独立参考重求，未解决数={a["reference_unresolved"]}；原返回目标值与参考解最大相对差约{100*a["max_abs_relative_returned_objective_difference"]:.4f}%。原最终归一化使预算下限最多低于设定值{a["max_floor_deficit_metric"]:.5f}归一化ε；ε>0与会话总预算仍合法。保留原输出未替换；参考重求只用于数值诊断，不是全局最优性证明。']
    path=out/'REPORT-zh.md';base=path.read_text().split(MARK)[0].rstrip();path.write_text(base+'\n\n'+'\n'.join(text)+'\n')
    print('component effects written',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default=str(ROOT/'experiments/workflow_extensions/outputs/m4_carry/real_v1_20261116'));p.add_argument('--output',required=True)
    a=p.parse_args();diagnose(a.source,a.output)
