"""Post-hoc mechanism explanations; never changes selection or test execution."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.run import dump
from experiments.workflow_extensions.m4_carry.analysis import write_csv
from experiments.workflow_extensions.m5.public import allocate


def rows(path):return list(csv.DictReader(path.open()))
def number(r,key):
    v=r.get(key)
    return float(v) if v not in (None,'','NA') else np.nan
def fmt(v):return 'NA' if not np.isfinite(v) else f'{v:.3f}'


def run(out,data_dir):
    out=Path(out);cfg=json.loads((out/'config.json').read_text());stats=json.loads((out/'public_stats.json').read_text())
    summaries=rows(out/'summary.csv');pairs=rows(out/'paired_summary.csv');endpoint=rows(out/'attack_endpoint_paired.csv')
    bm=rows(out/'bmin_paired.csv');audit=json.loads((out/'independent_audit.json').read_text())
    records={r['trajectory_id']:r for r in data.load(data_dir,cfg['templates'])}
    tables={t:{r['method']:r for r in summaries if r['template']==t and number(r,'B')==960} for t in cfg['templates']}
    details=[];factors=[];extra=defaultdict(list)
    for ln in (out/'metrics.jsonl').open():
        r=json.loads(ln)
        if r['B']==960 and r['method']=='M6':extra[r['template']].append(r)
    roots=defaultdict(lambda:defaultdict(list))
    # Value-only action diagnostics. Counterfactual cumulative harms below use
    # offline truth ONLY after all test execution/selection has finished.
    for ln in (out/'actions.jsonl').open():
        a=json.loads(ln)
        if a['B']!=960 or a['method']!='M6':continue
        p=stats[a['template']]['main'];w=Workload(a['template']);eps=np.array(a['actual_epsilon']);selected=eps>0
        for i,root in enumerate(w.roots):
            roots[a['template'],root]['epsilon'].append(eps[i])
            if selected[i]:
                roots[a['template'],root]['pub_epsilon'].append(eps[i]);roots[a['template'],root]['at_lower'].append(a['cache_after'][i]<=1e-14)
                roots[a['template'],root]['grid'].append(a['kernels'][i])
                if a['action_index']==0:roots[a['template'],root]['anchor_epsilon'].append(eps[i])
        if a['action_index']==0 or not selected.any():continue
        truth=w.evaluate(data.replay(records[a['trajectory']]))
        before=w.from_cache(np.array(a['cache_before']));after=w.from_cache(np.array(a['cache_after']));start=a['turn']-1
        lo=sum(p['query_weights'][j]*np.sum(abs(before[node]-truth[node][start:]))/p['node_scales'][j] for j,node in enumerate(w.nodes))
        ln_=sum(p['query_weights'][j]*np.sum(abs(after[node]-truth[node][start:]))/p['node_scales'][j] for j,node in enumerate(w.nodes))
        details.append(dict(template=a['template'],subject=a['subject'],trajectory=a['trajectory'],repeat=a['repeat'],turn=a['turn'],
                            predicted_harm=a['predicted_harm'],predicted_gain=a['predicted_gain'],
                            actual_cumulative_harm=ln_>lo+1e-12,actual_cumulative_drop=lo-ln_,
                            counterfactual='hold before/after cache to T, matching policy terminal-hold target'))
    for (t,root),d in roots.items():
        p=stats[t]['main'];i=p['roots'].index(root)
        factors.append(dict(template=t,root=root,a_mean=p['a_mean'][i],a_quantile=p['a_quantile'][i],
                            allocation_mode=p['allocation_mode'],anchor_epsilon_mean=np.mean(d['anchor_epsilon']),
                            root_epsilon_per_trajectory=np.sum(d['epsilon'])/len(extra[t]),
                            root_publications_per_trajectory=len(d['pub_epsilon'])/len(extra[t]),
                            lower_domain_output_fraction=np.mean(d['at_lower']),grid_fraction=np.mean(d['grid'])))
    write_csv(out/'root_diagnostics_960.csv',factors);write_csv(out/'risk_calibration_960.csv',details)
    decision={}
    for t,ss in tables.items():
        contrasts=[r for r in pairs if r['template']==t and number(r,'B')==960 and r['metric']=='mae' and r['method']=='M6']
        byref={r['reference']:r for r in contrasts}
        def better(ref):return number(byref[ref],'patient_hi')<0 and number(byref[ref],'two_layer_hi')<0
        targets=[r for r in bm if r['template']==t and r['reference']=='RG-periodic-exp']
        privacy=next(r for r in endpoint if r['template']==t and r['reference']=='M5' and r['endpoint']=='initial')
        p95=next(r for r in pairs if r['template']==t and number(r,'B')==960 and r['method']=='M6' and r['reference']=='RG-periodic-exp' and r['metric']=='p95')
        decision[t]=dict(actual_regime=stats[t]['main']['regime'],
             beats_M5_at_960=better('M5'),beats_both_static_at_960=better('RG-static-exp') and better('RG-static-lap'),
             public_targets_saving_ge_10pct=sum(number(r,'Saving')>=.1 for r in targets),public_targets_total=len(targets),
             bmin_not_both_reached=sum(not np.isfinite(number(r,'ratio')) for r in targets),
             p95_noninferiority_supported=number(p95,'patient_hi')<=0 and number(p95,'two_layer_hi')<=0,
             p95_point_change=number(ss['M6'],'p95')-number(ss['RG-periodic-exp'],'p95'),
             harmful_refresh_point_change=number(ss['M6'],'harmful_refresh')-number(ss['RG-periodic-exp'],'harmful_refresh'),
             initial_MAP_more_successful_than_M5=number(privacy,'patient_hi')<0 and number(privacy,'two_layer_hi')<0)
    static_all=all(decision[t]['beats_both_static_at_960'] for t in ['ANEMIA','FIB4'])
    dynamic_all=all(decision[t]['public_targets_saving_ge_10pct']==decision[t]['public_targets_total'] and decision[t]['p95_noninferiority_supported'] and decision[t]['harmful_refresh_point_change']<0 for t in ['MELD_NA','BUN_CR'])
    privacy_all=not any(d['initial_MAP_more_successful_than_M5'] for d in decision.values())
    verdict=dict(by_template=decision,static_2of2=static_all,dynamic_all_public_targets=dynamic_all,
                 no_systematic_initial_MAP_worsening_vs_M5=privacy_all,ledger_audit_passed=audit['ledger'],
                 overall_go=static_all and dynamic_all and privacy_all,
                 limitations='Bmin targets may be censored; p95 noninferiority requires both pointwise paired upper endpoints <=0; no multiplicity correction')
    dump(out/'go_no_go.json',verdict)
    lines=['## 补充诊断与go/no-go','',
        '以下解释只在冻结运行结束后计算，不反馈到参数、分组或test选择。','',
        '|模板|平均实际刷新|整机会否决率|网格发布比例|EM真值超工作区间比例|全轨迹真值超工作区间比例|有害刷新率|','|---|---:|---:|---:|---:|---:|---:|']
    for t,ss in tables.items():
        r=ss['M6'];lines.append('|'+t+'|'+'|'.join(fmt(number(r,k)) for k in ['m','frac_veto','grid_release_fraction','outside_work_grid_fraction','outside_work_truth_fraction','harmful_refresh'])+'|')
    lines+=['','根预算与边界输出（同一模板内比较，域单位不同不可横向混加）：','',
       '|模板|根|a_mean|a_Q90|初始化平均ε|全程逐根累计ε|输出完整域下界比例|网格发布比例|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in factors:lines.append('|'+r['template']+'|'+r['root']+'|'+'|'.join(fmt(r[k]) for k in ['a_mean','a_quantile','anchor_epsilon_mean','root_epsilon_per_trajectory','lower_domain_output_fraction','grid_fraction'])+'|')
    lines+=['','机制解释：','',
       '- MELD的人群μ可能处于Na裁剪平台，使a_mean,Na=0；正数floor只保证ε>0，不能保证足够精度。对尚未发布的其他患者，Na仍可能落入敏感区。局部均值梯度零不等于这个根在完整域中不重要，这是低ν分支的明确弱点。',
       '- FIB4在本轮train统计下仍走动态分支，初始化未触发EM，且预算按√a_Q90而不是M5的工作区间权重^(2/3)分配。动态锚点只得到rho·B，分母ALT/PLT的尾部噪声仍可经比值放大；核/分配/计划同时变化，单一原因须看mean-only、lap-only及固定计划消融。',
       '- ANEMIA若M6与M5完全相同，是两者都静态、分配相同、初始化未触发换核所致，不能当作新增符号组或跨期判据的收益。',
       '- 预测跨期有害概率与实际同一terminal-hold反事实是否变坏在risk_calibration_960.csv单列；它与全轨迹实际刷新后的未来策略、有害刷新（当轮最终节点）是三个不同口径。','',
       '|模板|发布动作数|平均预测跨期变坏概率|实际跨期变坏比例|平均跨期误差下降|','|---|---:|---:|---:|---:|']
    for t in cfg['templates']:
        aa=[r for r in details if r['template']==t]
        vals=[np.mean([r[k] for r in aa]) if aa else np.nan for k in ['predicted_harm','actual_cumulative_harm','actual_cumulative_drop']]
        lines.append('|'+t+'|'+str(len(aa))+'|'+'|'.join(fmt(v) for v in vals)+'|')
    lines+=['','预指定整体目标判断：','',
       f'- 静态ANEMIA/FIB4同时胜过两个静态原核/同核对照：{static_all}。',
       f'- 动态MELD/BUN在全部冻结公开目标达到≥10%预算节省，且p95双区间不劣、有害率下降：{dynamic_all}。网格删失不算成功。',
       f'- 相对M5初始化MAP没有显著系统性变得更易攻击：{privacy_all}。这是经验目标，不是DP合法性条件。',
       f'- 整体：{"GO" if verdict["overall_go"] else "NO-GO，当前不支持宣称M6达到整套优化目标"}。',
       '', '这里NO-GO指整体性能主张未过关，不能理解为预算违规；应先处理局部零梯度、区制/危险核未触发及风险模型失配，而不是据test即时调阈值。',
       '', '补充诊断复现（在主运行完成后执行）：','', '```bash',
       '/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m6.diagnostics --output experiments/workflow_extensions/outputs/m6/real_reproduce',
       '```']
    text='\n'.join(lines)+'\n';(out/'DIAGNOSTICS-zh.md').write_text(text)
    path=out/'REPORT-zh.md';original=path.read_text();marker='## 补充诊断与go/no-go'
    if marker in original:original=original[:original.index(marker)]
    path.write_text(original+'\n'+text)
    dump(out/'diagnostic_source_hashes.json',{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps(verdict,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);ap.add_argument('--data',default='data_mine/finite_cache')
    a=ap.parse_args();run(a.output,a.data)
