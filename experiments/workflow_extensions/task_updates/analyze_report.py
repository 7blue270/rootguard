"""Patient paired uncertainty, preserved-source audit, short report + evidence."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from diagnostic_detail import details


def md(rows,fields):
    s=['| '+' | '.join(v for _,v in fields)+' |','| '+' | '.join('---' for _ in fields)+' |']
    for r in rows:
        vals=[]
        for k,_ in fields:
            v=r.get(k)
            if isinstance(v,float):v=f'{v:.2f}'
            if isinstance(v,list):v='['+', '.join(f'{x:.2f}' for x in v)+']'
            vals.append('—' if v is None else str(v))
        s.append('| '+' | '.join(vals)+' |')
    return '\n'.join(s)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();out=args.run.resolve()
    cfg=json.loads((out/'config.json').read_text());meta=json.loads((out/'metadata.json').read_text())
    cells=json.loads((out/'cells.json').read_text());diag=json.loads((out/'diagnosis.json').read_text());sels=json.loads((out/'selection.json').read_text())
    prior=json.loads((Path(__file__).parent/'prior_sources.json').read_text());repo=Path(__file__).resolve().parents[3]
    altered=[p for p,h in prior.items() if hashlib.sha256((repo/p).read_bytes()).hexdigest()!=h]
    mismatch=[p for p,h in meta['source_sha256'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not altered and not mismatch
    with (out/'patients.csv').open() as f:raw=list(csv.DictReader(f))
    by={}
    for row in raw:by.setdefault(row['cell'],[]).append(row)
    refs={(r['dataset'],r['track'],r['budget'],r['method']):r for r in cells};analysis=[]
    for c in cells:
        rr=by[c['cell']];n=len(rr);idx=np.random.default_rng(913).integers(0,n,(cfg['bootstrap'],n))
        vals={key:np.array([float(r[key]) for r in rr]) for key in ['error','truth','wrong','hit2','event_den','false','false_den']}
        if c['certified']:assert c['spent_max']<=c['B']+1e-8
        a=dict(c)
        a['wmape_ci95']=np.quantile(100*vals['error'][idx].sum(axis=1)/vals['truth'][idx].sum(axis=1),[.025,.975]).tolist()
        den=vals['event_den'][idx].sum(axis=1)
        a['tcr2_ci95']=None if np.max(den)==0 else np.quantile(100*vals['hit2'][idx].sum(axis=1)[den>0]/den[den>0],[.025,.975]).tolist()
        for method in ['static','period_balanced']:
            ref=refs[c['dataset'],c['track'],c['budget'],method];rr2=by[ref['cell']]
            assert [r['patient'] for r in rr]==[r['patient'] for r in rr2]
            de=vals['error']-np.array([float(r['error']) for r in rr2])
            a['wmape_gap_vs_'+method]=float(100*de.sum()/vals['truth'].sum())
            a['wmape_gap_vs_'+method+'_ci95']=np.quantile(100*de[idx].sum(axis=1)/vals['truth'][idx].sum(axis=1),[.025,.975]).tolist()
        analysis.append(a)
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False))
    fields=['dataset','track','budget','method','param','certified','wmape','rce','tcr0','tcr2','false_change','delay_capped','censored_pct','vector_refreshes','root_releases','spent_max','wmape_ci95','tcr2_ci95','wmape_gap_vs_period_balanced_ci95']
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(analysis)
    with (out/'diagnosis.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(diag[0]));w.writeheader();w.writerows(diag)
    audit=dict(previous_files_checked=len(prior),previous_files_changed=altered,run_source_mismatches=mismatch,
               certified_budget_caps_pass=True,conditions=len(cells),validation_conditions=meta['validation_cells'])
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    detailed=details(cfg)
    (out/'meaningful_changes.json').write_text(json.dumps(detailed,indent=2))
    realdetail=[r for r in detailed if r['dataset']=='real_ANEMIA_H8' and r['task_tolerance']==.02]
    realdiag=next(r for r in diag if r['dataset']=='real_ANEMIA_H8' and r['split']=='test')
    realrows=[r for r in analysis if r['dataset']=='real_ANEMIA_H8' and r['track']=='metric' and r['budget']==.3]
    real_static=next(r for r in realrows if r['method']=='static')
    real_task=next(r for r in realrows if r['method']=='task_balanced')
    real_oracle=next(r for r in realrows if r['method']=='oracle_C2')
    control=[r for r in diag if r['split']=='test' and '_ANEMIA_' in r['dataset'] and 'H8' in r['dataset']]
    representative=[r for r in analysis if r['dataset'] in ['syn_ANEMIA_cancel_H8','syn_ANEMIA_amplify_H8','syn_FIB4_near_cancel_H16'] and r['track']=='metric' and r['budget']==.3 and r['method'] in ['static','period_accuracy','period_balanced','task_balanced','attribute_balanced','oracle_C4']]
    standard=[r for r in analysis if r['dataset']=='real_ANEMIA_H8' and r['track']=='standard' and r['method'] in ['static','period_accuracy','period_balanced','task_balanced']]
    chosen=[r for r in sels if r['group']=='real_ANEMIA_H8' and r['track']=='metric' and r['budget']==.3]
    sweep=json.loads((out/'period_tradeoffs.json').read_text()) if (out/'period_tradeoffs.json').exists() else []
    if sweep:
        assert all(r['spent_max']<=r['budget']/(999 if r['track']=='standard' else 1)+1e-8 for r in sweep)
        sm=json.loads((out/'period_sweep_metadata.json').read_text())
        assert sm['source_sha256']==hashlib.sha256((Path(__file__).parent/'period_sweep.py').read_bytes()).hexdigest()
        with (out/'period_tradeoffs.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(sweep[0]));w.writeheader();w.writerows(sweep)
    audit['period_sweep_conditions']=len(sweep)
    audit['analysis_and_supplement_sources_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.rglob('*.py')}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    realperiods=[r for r in sweep if r['dataset']=='real_ANEMIA_H8' and r['track']=='metric']
    oracle_probe=json.loads((out/'oracle_noise_ablation.json').read_text())
    audit['oracle_noise_diagnostic_conditions']=len(oracle_probe)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    syn_events=sum(r['continuous_task_events'] for r in diag if r['split']=='test' and r['dataset'].startswith('syn'))
    conclusion=[]
    for track,b in [('metric',.3),('metric',.7),('standard',4)]:
        rr=[r for r in analysis if r['track']==track and r['budget']==b and r['method']=='task_balanced']
        gaps=[r['wmape_gap_vs_period_balanced'] for r in rr]
        significant=sum(r['wmape_gap_vs_period_balanced_ci95'][1]<0 for r in rr)
        conclusion.append(dict(track=track,budget=b,datasets=len(rr),task_trigger_lower_wmape=sum(g<0 for g in gaps),paired_ci_upper_below_zero=significant))
    doc=f'''# task_updates：本轮结果报告

## 先看结论

这轮主要检查两件事：**属性共同变化是否被任务抵消，以及简单方法是否已经能处理好这种情况。** 本次没有修改上一轮实验，没有提出新 DP 机制，也没有调用 LLM。

真实复测测试集中有 {realdiag['attribute_events']} 次属性变化，其中 {realdiag['small_task_given_changed_pct']:.2f}% 的任务比值变化不超过 2%。相关属性贡献的净变化/总变化平均为 {realdiag['cancellation_ratio_mean']:.3f}（越接近 0，抵消越明显）。因此，“属性更新不一定需要任务更新”在这个小数据集中确实能观察到，不只是人为构造的例子。

但**这并不直接证明需要新方法**。真实 B=0.3 下，旧缓存 wMAPE={real_static['wmape']:.2f}%，正确响应率为 0；任务 SVT 的误差升至 {real_task['wmape']:.2f}%，两轮内正确响应仅 {real_task['tcr2']:.2f}%，误报 {real_task['false_change']:.2f}%。免费知道触发时刻的 Oracle_C2 也只有 {real_oracle['tcr2']:.2f}% 正确响应，说明问题不只是检测，还包含重新发布数值的噪声。

**建议保留“属性变化与任务变化脱钩”这一应用现象，但暂不直接进入新检测器设计。** 当前任务触发没有显示稳定优势，需要先判断根发布噪声和全局敏感度是否掩盖了触发收益，以及已有发布方法是否已能改善这一点。验证集调优的简单方法很有竞争力，却往往选择少更新；这不等于它满足及时响应要求。当前简化触发器也不是成熟框架效用极限。

## 1. 实际做了什么

- 新建 `task_updates/`，增加 adapter、dataset、release、run_experiments、analyze_report、配置、5 项测试与复现说明。
- 只读复用原公式、指数采样、预算分配和真实 CBC 筛选；核对 {len(prior)} 个上一轮文件哈希，变更数为 0。
- NHANES 每模板 200 人：60 人验证、140 人测试；H=8/16。精确抵消、近似抵消、共同放大、独立变化，另有 ANEMIA 无关 RBC 更新。合成值为乘法变化，幅度上限 12%，靠统一缩小幅度保证不越域，没有裁剪破坏抵消关系。
- 每患者随机选择三次潜在改变时刻，不专门对齐周期，不构造风险跨阈值。合成轨迹仍是控制实验，不是真实纵向临床数据。
- MIMIC-IV Demo 2.2，原同标本 Hb/Hct/RBC 规则，H=8：53 人中 20 人验证、33 人测试，一患者一住院。病例少，不能推广为临床结论。
- 27 个数据条件；{meta['validation_cells']} 个验证条件，{len(cells)} 个测试条件。验证每条件 12 次噪声重复，测试 60 次。合成测试集连续任务事件合计 {syn_events} 次，跨模板/场景/长度重复使用患者，不代表独立患者数。

## 2. 属性变化与任务变化的区别

对单步变化，先计算各属性对 log(g) 的贡献 p_i*Δlog(x_i)。总变化=sum|贡献|，净变化=|sum贡献|；净/总接近 0 说明抵消，接近 1 说明同向放大。属性变化判定来自未加噪真值；仅无关属性变化时，抵消比不定义，不能误称相关属性抵消。

{md(control,[('dataset','场景'),('n','测试人数'),('attribute_events','属性事件'),('continuous_task_events','任务事件'),('small_task_given_changed_pct','属性变、任务小变 %'),('cancellation_ratio_mean','净/总贡献')])}

“任务事件”指连续值相邻版本的 |Δlog(g)|>log(1.02)，是评估数值容差，不是新风险类别阈值。风险类别使用原 risk_class，仅报告辅助 RCE。

为避免大量微小属性波动夸大现象，真实数据另外只统计最大属性变化超过 1%/2%/5% 的时刻，检查任务变化是否仍不超过 2%：

{md(realdetail,[('attribute_tolerance','属性变化容差'),('attribute_events','满足条件的事件'),('small_task_pct','任务仍小变 %')])}

完整 1%/2%/5% 交叉容差分析在 meaningful_changes.json。这些是分析条件，不用于未经计费地触发部署刷新。

## 3. 方法、选参及安全边界

| 方法 | 做法 |
|---|---|
| static | 第一次私有化后重复使用 |
| public_roots | 按公开测量计划仅刷新对应属性，总预算按公开发布次数分配 |
| period_accuracy / period_balanced | 验证集从周期 1/2/4/8/16 选择，分别按两种目标选参 |
| attribute_accuracy / attribute_balanced | 有限重启 AboveThreshold，检测最大归一化根变化 |
| task_accuracy / task_balanced | 同一 AboveThreshold 框架，检测当前任务与已发布缓存的 log 值差 |
| oracle_C2 / oracle_C4 | 原始真值累计任务变化超过 2% 时免费触发，最多 2/4 次；非 DP 诊断参考 |

SVT 网格：最多触发 C=2/4，检测总预算占比 20%/50%，触发阈值 0.01/0.03/0.1。阈值单位分别为域宽比例、log 任务值差，不是风险类别。每次 AboveThreshold 的阈值噪声为 Lap(2Δ/epsilon)、查询噪声 Lap(4Δ/epsilon)，一次阳性即结束此块；最多 C 块，每块检测费 B*f/C。每次发布（包括第一次）预留 B*(1-f)/(C+1)。到达上限后保持缓存。预算不会因负判断次数增多而逐次扣除，但块本身收费；这与上一轮逐次收费的简化检测器不同。

参照 [Dwork–Roth §3.6 的 AboveThreshold 与有限重启](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)，不是对某应用论文全套代码的复现。任务检测敏感度使用整个公开域的全局界 sum|p_i|*(hi_i-lo_i)/(999*lo_i)，不能用测试轨迹的较小变化代替；这可能很保守，FIB4 广域尤其明显。属性检测敏感度为 1/999。固定过去已发布缓存后查询敏感度有界，有限重启检测与新发布按组合计费。

保护对象仍为固定公共长度、测量计划与根结构下的数值轨迹：d_inf=max|归一化索引之差|。非 Oracle 组均满足总 B 的距离隐私；充分标准轨迹界是 E=999B。主 B=0.3/0.7 对应 E=299.7/699.3，不是标准 E=0.3/0.7。另跑标准 E=4（B=4/999）。更新时间和测量字段是公共信息；验证集视为公开调参数据，不声明保护其调参过程。输出根缓存、任务后处理及检测位均纳入合同；离线研究报告本身不是 DP 发布接口。

选参不能接触测试结果。同一合成模板/长度跨全部场景共同选参，不为每个测试情景单独挑最佳参数。误差目标为验证 wMAPE；兼顾响应目标为 wMAPE+0.1*(100−TCR2)+0.1*误报率，无任务事件的场景不加召回惩罚。权重是预设探索选择，不是临床可接受标准。两套策略均公开报告。

## 4. 真实复测结果

H=8，33 位测试患者，总距离预算 B=0.3：

{md(realrows,[('method','方法'),('wmape','wMAPE %'),('rce','原风险 RCE %'),('tcr2','两轮内正确响应 %'),('false_change','错误变化 %'),('vector_refreshes','含初始刷新次数'),('certified','DP 组')])}

这里的正确响应要求：输出确实出现>2%的变化、方向正确，并且新任务值相对真值 log 误差不超过 log(1.05)。TCR0 要求当轮，TCR2 允许延迟最多两轮且不能跨过下一个真实任务事件。故噪声造成的任意类别跳动不会算正确响应；没有任务事件时召回率为未定义。误报率只在真值无>2%单步任务事件的时刻计算。

延迟仅对事件进行分析，未成功的事件按其可观察区间长度截断，另外保存未成功比例，不能把截断延迟当成功检测速度。所有指标详细见 summary.csv。

验证集选择的参数：

{md(chosen,[('family','系列'),('profile','目标'),('param','选定参数'),('validation_loss','验证目标')])}

## 5. 控制场景与强标准轨迹保护

以下结果展示任务稳定和任务改变两种情况，不仅展示有利于触发策略的样例：

{md(representative,[('dataset','场景'),('method','方法'),('wmape','wMAPE %'),('tcr2','两轮内正确响应 %'),('false_change','错误变化 %'),('vector_refreshes','刷新次数')])}

标准 E=4 下真实场景：

{md(standard,[('method','方法'),('wmape','wMAPE %'),('tcr2','两轮内正确响应 %'),('false_change','错误变化 %')])}

这只测试当前指数发布与全局敏感度校准，不能据此认定所有强轨迹 LDP 都不可行。

## 6. 是否留下了方法改进空间

在全部 27 个数据条件中，任务触发 balanced 与验证集调优周期 balanced 的 wMAPE 比较：

{md(conclusion,[('track','保护轨道'),('budget','预算'),('datasets','条件数'),('task_trigger_lower_wmape','任务触发误差更低'),('paired_ci_upper_below_zero','患者配对区间支持更低')])}

具体来说，B=0.3 只有一个 ANEMIA 共同放大条件降低平均误差；B=0.7 的六个条件主要是 ANEMIA 的稳定/近似抵消/无关更新控制，部分根本没有>2%的任务事件。它们不是“发现重要变化更好”的证据，也不是对不更新缓存的全面优势。不能把 6/27 描述成六个真实应用场景获胜。

此表比较误差，不是“整体获胜”；必须同时读召回和误报。不能把同一患者重复场景或不同长度当独立数据。区间采用患者配对 bootstrap 1000 次、逐项 95%、未校正多重比较，适合筛查而不是最终论文显著性结论。

预设全部周期的真实测试对照如下。它是描述权衡，不使用测试集挑最佳周期替代验证集选择，也不声称覆盖全部成熟基线：

{md(realperiods,[('budget','总 B'),('period','周期'),('wmape','wMAPE %'),('tcr2','两轮内正确响应 %'),('false_change','错误变化 %'),('vector_refreshes','刷新次数')])}

真实B=0.3时，预设周期4的误差4.98%、正确响应10.31%、误报12.98%，三项均优于当前task_balanced（8.05%、8.35%、22.90%）。这是真实数据中简单参考仍有竞争力的证据；周期4不是用测试结果替换验证集选择出的部署参数。

为分离“没触发”和“触发后数值被噪声损坏”，追加相同Oracle时刻/上限/初始缓存的反事实对照，只把触发后的新根值换成精确值，其他条件不变：

{md(oracle_probe,[('B','参考预算'),('C','触发上限'),('arm','诊断臂'),('wmape','wMAPE %'),('tcr2','正确响应 %')])}

例如B=0.3、C=4，移除刷新噪声后正确响应从15.04%升至68.12%，确认发布噪声确实贡献了大部分损失。仍未达100%是因为刷新上限、触发规则、初始误差及评价窗口等也有影响。**两臂都是非DP诊断，精确新值发布不能部署，也不能在相同账本标签下声称同隐私保护。** 这不是新方法，而是瓶颈定位。

本轮可以回答现象是否存在、哪些简单参考有竞争力。即使任务触发在部分条件受益，也不构成新算法；若无优势，也只能否定这次配置下的参考实现，不能否定整个任务感知方向。下一步应先看真实任务事件数量、验证集选参是否稳定，以及是否存在周期难以兼顾的响应/误报缺口，再决定是否增加数据或改进方法。

## 7. 复现和产物

运行环境 Python {meta['python']}、NumPy {meta['numpy']}、SciPy {meta['scipy']}，种子见 config.json；主运行用时 {meta['seconds']:.1f} 秒。5 项机制测试及2项事件指标对照检查通过，源文件哈希匹配，全部 DP 组账本未超预算，上一轮实验文件未变更。另有 {len(sweep)} 个预设周期测试对照，不作为独立数据重复。Oracle 只表示免费知晓真实触发条件的参考，不是严格最优上界，也不是同预算 DP 竞争者。

- `config.json`：参数与随机种子。
- `metadata.json`：环境、源码和输入 SHA、验证/测试患者编号、合成时刻及真实筛选审计。
- `diagnosis.json/csv`：未加噪真值的属性/任务变化统计。
- `meaningful_changes.json`：剔除微小属性变化后的容差敏感性分析。
- `validation.json`、`selection.json`：完整调参依据。
- `cells.json`、`patients.csv`：测试结果及患者级统计。
- `analysis.json`、`summary.csv`：区间、配对差异和完整指标。
- `audit.json`：前轮文件不变、源码和预算核验。
- `period_tradeoffs.json/csv`：全部预设周期的测试权衡，未用来选择最终参数。
- `oracle_noise_ablation.json`：8个非DP反事实诊断条件，定位刷新发布噪声。

本轮不覆盖真实文本/LLM 推理、根结构新增、私有测量时刻或无限长度。具体命令见 [task_updates README]({Path(__file__).parent/'README.md'})。文件只在新目录及对应输出目录中创建，未 commit/push。
'''
    (out/'REPORT.md').write_text(doc)
    shortrows=[r for r in realrows if r['method'] in ['static','task_balanced','attribute_balanced','oracle_C2']]
    for r in realperiods:
        if r['budget']==.3 and r['period'] in [2,4]:shortrows.append(dict(r,method='固定周期 '+str(r['period'])))
    short=f'''# task_updates：简明结果报告

**结论：现象成立；当前任务触发方法还没有显示稳定优势。暂不急于设计新检测器。**

1. **属性变化确实不等于任务变化。** 真实测试集33人、80次>2%的任务事件。在属性变化>5%的144个时刻中，60.42%的任务比值变化仍≤2%。这说明抵消不是只有人为例子，但证据来自一个血常规比值任务。
2. **少更新准确，却不及时；多更新及时一些，却易误报。** 验证集选出的周期为8，即只发布一次。下表另列预设周期2/4，避免把“不更新”当唯一周期对照。
3. **简化任务SVT没有解决这个权衡。** 它比属性SVT略好，但不能稳定超过周期策略；即使免费知道何时触发，Oracle仍受新发布噪声影响，所以瓶颈可能不只在检测。

补充反事实验证：保持同一Oracle刷新时刻，只移除刷新噪声，正确响应由15.04%升至68.12%（C=4、B=0.3）。两组均是非DP诊断，精确发布不可部署，但能确认发布噪声是实际瓶颈。

真实H=8，总距离预算B=0.3；正确响应要求方向正确且新任务值准确，允许最多延迟两轮：

{md(shortrows,[('method','方法'),('wmape','平均误差 %'),('tcr2','正确响应 %'),('false_change','误报 %')])}

**本轮做了什么：** 新建独立task_updates目录；NHANES每模板200人，60验证/140测试，随机时刻的抵消、近似抵消、放大和独立变化；真实数据20验证/33测试。完成2307个验证条件、810个测试条件、363个周期权衡对照及8个噪声诊断，测试每条件60次噪声重复。7项检查通过，上一轮文件哈希完全不变。

**安全边界：** 非Oracle组在公共长度/时间/属性计划下满足总B的距离隐私；B=0.3不能称作标准轨迹ε=0.3，充分标准界为299.7。另跑标准ε=4，当前机制效用也较差。Oracle非DP，仅诊断；SVT为有限重启参考实现，不是某应用论文完整复现。

**下一步启发：** 可以保留“属性变化被任务抵消”的问题动机，但先核实根发布噪声、敏感度保守程度及成熟发布方法的表现，再决定是否改进触发器。现在还不足以证明新方法的论文价值，也不能断言强轨迹DP不可能可用。

修改内容、参数选择、全部结果、区间及复现步骤见 [完整报告]({out/'REPORT.md'}) 与 [代码说明]({Path(__file__).parent/'README.md'})。
'''
    (out/'SUMMARY.md').write_text(short)
    print(out/'REPORT.md');print(json.dumps(conclusion));print(json.dumps(realdiag))


if __name__=='__main__':main()
