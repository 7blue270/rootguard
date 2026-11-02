import json,csv
from pathlib import Path
import numpy as np
from common import HERE


def table(rows,fields):
    lines=['| '+' | '.join(v for _,v in fields)+' |','| '+' | '.join('---' for _ in fields)+' |']
    for r in rows:
        vals=[]
        for k,_ in fields:
            v=r.get(k)
            if isinstance(v,float):v=f'{v:.2f}'
            if isinstance(v,list):v='['+', '.join(f'{x:.2f}' for x in v)+']'
            vals.append('—' if v is None else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def build(out,rows,macro,meta,cfg):
    from generation_audit import build as audit_generation
    gen=audit_generation(out,cfg)
    selections=json.loads((out/'selection.json').read_text())
    audit=json.loads((out/'audit.json').read_text())
    source_counts={s:sum(d['source']==s for d in meta['datasets'].values()) for s in ['legacy_synthetic','stress','observed']}
    real=[dict(dataset=did,n=len(d['test_ids']),events=d['test_events'],event_patients=d['test_event_patients'],validation_n=len(d['validation_ids'])) for did,d in meta['datasets'].items() if d['source']=='observed']
    coremacro=[r for r in macro if r['K']==2]
    by={(r['dataset'],r['B'],r['comparison'],r['method']):r for r in rows}
    B_rows=[r for r in rows if r['comparison']=='B' and r['B']==.7 and r['param']['K']==2]
    impact=[]
    for did in cfg['B_cases']:
        for B in cfg['budgets']:
            for K in cfg['B_K']:
                get=lambda ie,re:by[(did,B,'B',f'initial_{ie}_refresh_{re}_K{K}')]
                ee=get('exact','exact');en=get('exact','noisy');ne=get('noisy','exact');nn=get('noisy','noisy')
                impact.append(dict(dataset=did,B=B,K=K,ideal=ee['recall1'],all_noisy=nn['recall1'],
                    conditional_initial_loss=en['recall1']-nn['recall1'],conditional_refresh_loss=ne['recall1']-nn['recall1'],
                    total_publication_gap=ee['recall1']-nn['recall1'],
                    interaction=ee['recall1']-en['recall1']-ne['recall1']+nn['recall1']))
    write=lambda name,data:(out/name).write_text(json.dumps(data,indent=2,allow_nan=False))
    write('B_quadrant_effects.json',impact)
    fee=[];schedules=[]
    for r in rows:
        if r['comparison']=='B_fee' and r['method'].endswith('_reserved'):
            family=r['method'].split('_')[0];ref=by[r['dataset'],r['B'],'B_fee',family+'_full']
            fee.append(dict(dataset=r['dataset'],B=r['B'],family=family,K=r['param']['K'],fraction=r['param']['fraction'],
                full_recall1=ref['recall1'],reserved_recall1=r['recall1'],gap_reserved_minus_full=r['recall1']-ref['recall1']))
        if r['comparison']=='B_schedule' and r['method'].endswith('oracle_count_matched'):
            family=r['method'].split('_')[0];ref=by[r['dataset'],r['B'],'B_schedule',family+'_svt_frozen']
            assert abs(r['refreshes']-ref['refreshes'])<1e-10
            schedules.append(dict(dataset=r['dataset'],B=r['B'],family=family,refreshes=r['refreshes'],
                frozen_svt_recall1=ref['recall1'],matched_oracle_recall1=r['recall1'],gap=r['recall1']-ref['recall1']))
    write('detection_fee_effects.json',fee);write('frozen_schedule_effects.json',schedules)
    feas=[]
    for B in cfg['budgets']:
        for family in ['period','attribute','task']:
            for floor in cfg['recall_constraints']:
                rr=[s for s in selections if s['B']==B and s['family']==family and s['constraint']==floor]
                feas.append(dict(B=B,family=family,recall_floor=floor,groups=len(rr),feasible_groups=sum(s['validation_feasible'] for s in rr)))
    a2_real=[r for r in rows if r['source']=='observed' and r['B']==.7 and r['comparison']=='A2' and r['method'] in ['static','public_roots','period_target20','attribute_target20','task_target20']]
    a1_real=[r for r in rows if r['source']=='observed' and r['comparison']=='A1' and r['method']=='oracle_K2']
    # Descriptive 5pp grouping is not a clinical threshold or confirmatory test.
    stressmac=[r for r in macro if r['source']=='stress']
    mean_loss=float(np.mean([r['total_publication_gap'] for r in impact]))
    timing=float(np.mean([r['mean_gap_recall1'] for r in stressmac]))
    timing_bounds=[r for r in stressmac if r['mean_gap_recall1']>=5 and r['gap_recall1_ci95'][0]>0]
    stress_only=[r for r in coremacro if r['source']=='stress']
    b_template=[]
    for name in cfg['templates']:
        dids={did for did,d in meta['datasets'].items() if d['template']==name}
        rr=[r for r in impact if r['dataset'] in dids]
        b_template.append(dict(template=name,combinations=len(rr),ideal_mean=float(np.mean([r['ideal'] for r in rr])),
            noisy_mean=float(np.mean([r['all_noisy'] for r in rr])),gap_mean=float(np.mean([r['total_publication_gap'] for r in rr]))))
    mean_fee=float(np.mean([r['gap_reserved_minus_full'] for r in fee]))
    mean_schedule=float(np.mean([r['gap'] for r in schedules]))
    partial_stable=[r for r in gen if 'stress_HOMA_stable' in r['dataset']]
    decision='B：优先检查已有发布机制，暂不设计新机制' if mean_loss>=10 else 'C/D：证据不足，或潜在空间较小；暂缓机制设计'
    direction=dict(decision=decision,mean_B_publication_gap_pp=mean_loss,mean_stress_A1_gap_pp=timing,
                   mean_detection_budget_reservation_effect_pp=mean_fee,mean_exact_frozen_schedule_gap_pp=mean_schedule,
                   stress_macro_groups=len(stressmac),stress_groups_gap_atleast5pp_CI_above0=len(timing_bounds),
                   note='描述性归纳，非临床标准/理论不可能性证明；真实、合成分开见表')
    write('research_decision.json',direction)
    short=f'''# Stage1 A/B：简明研究价值报告

**判断：{decision}。**

本轮不是新算法评测。原27条件完整保留，新增30个类别事件压力条件和真实H16，共58个条件；B=0.3/0.7，正式测试每条件60次重复。完成{len(rows)}个测试对照、{meta['validation_cells']}个验证条件。9项协议测试和5条件预检查通过。

**1. 同资源下，更新时间的价值有多大？** 下表为额外实际刷新K=2时，Oracle相对验证集选相位的周期发布，严格ResponseRecall@1的变化，单位百分点。括号为患者配对95%探索区间；多场景/两个长度共用患者抽样，不当作独立患者。

{table(coremacro,[('source','数据类型'),('template','任务'),('B','总B'),('mean_gap_recall1','Oracle增益'),('gap_recall1_ci95','95%区间')])}

**2. 发布噪声的损失：** 预定8个案例、两个预算和两个K，完全精确与全部加噪之间的响应差平均为{mean_loss:.2f}个百分点。四组完整结果见详细报告，不能把这些平均值解释为真实临床损失或严格独立误差贡献。完全精确组仍受刷新额度限制。

**3. 可部署方法是否合格？** 真实数据B=0.7下：

{table(a2_real,[('dataset','真实队列'),('method','方法'),('recall1','响应@1 %'),('rce','RCE %'),('cmr','CMR %'),('false_change','稳定期误报 %'),('validation_feasible','验证约束可行')])}

“不可行”时仍展示诊断候选，不能称其满足响应要求。是否满足测试约束须看实测响应和误报，不由验证可行标签保证。

**下一步建议：** 不建议此刻停止动态更新方向；A1存在同资源的时机收益，但优先转向核查已有发布机制，暂缓设计新触发器或残差方案。四组噪声损失在三个函数中均存在，仍不能证明新方法有论文价值。真实事件有限，不能由受控跨阈值轨迹替代临床证据。若成熟方法已解决缺口，或更丰富真实任务无法复现有意义空间，应停止机制创新投入。

检测预算预留在同Oracle日程下使响应平均变化{mean_fee:.2f}个百分点；冻结SVT日程、精确输出与匹配实际次数的Oracle日程之间平均相差{mean_schedule:.2f}个百分点。这两项是不同反事实，不能相加成总体检测损失。

所有可部署组采用有限T8/16、公开计划下的距离轨迹预算B；不等于标准轨迹ε=B。全部Oracle、精确值、冻结日程诊断均非DP。未调用LLM、未改动既有实验、未提交Git。

[完整流程、RQ1–RQ5分析与限制]({out/'REPORT.md'}) · [全条件指标和区间]({out/'summary.csv'})
'''
    (out/'SUMMARY.md').write_text(short)
    doc=f'''# RootGuard Stage1 A/B研究价值判定：完整报告

日期2026-10-13。代码位于task_updates/stage1_testab，运行HEAD `{meta['git_head']}`。预检查v2通过后正式运行；本轮没有新DP机制、残差发布或LLM调用。

## 1. 数据与规则

原26合成条件加真实H8的27条件保留；新增压力30条件，另加真实H16。类型统计 `{source_counts}`。每模板NHANES200基样本，前60验证/后140测试；压力轨迹复用同患者身份。原旧合成生成函数未改动。多个场景、两个长度和60次噪声不能变成额外独立患者。

压力模式：单次持续变化、间隔长的3次变化、连续两轮变化、每轮阈值跨越、属性变化但类别不变。H8/16；半数围绕第一风险边界，半数第二边界。普通变化两侧目标为b±min(0.1b,0.4*(b2−b1))，反复边界模式为b±0.003b。用原monomial公式在根公开log域内求解等式，保持FIB4年龄和无关根；类别稳定模式沿任务不变的log零空间变化。没有后处理裁剪破坏事件。域可行与所需事件数逐患者断言验证。这是数学受控压力，未证明临床生理合理性。

真实CBC规则完全复用阶段0，H8/H16分开，不合并患者；H16按H8已有身份划分，避免跨长度验证/测试混入。H8全部资源49事件/20有事件患者、H16全部资源51事件/16患者，资源门槛未达；本轮用户明确授权继续。测试子集：

{table(real,[('dataset','队列'),('validation_n','验证人数'),('n','测试人数'),('events','测试事件'),('event_patients','有事件患者')])}

原H8测试33事件/13有事件患者保持一致。真实事件少，只作现实检查；其他模板没有真实纵向数据，不能把压力事件当实证临床事件。

生成审计显示ANEMIA/FIB4类别稳定压力轨迹全部200人确有属性变化；HOMA仅140/200人有变化，另60人受投影域边界限制没有变化。验证/测试具体人数如下，零变化轨迹不用于证明脱钩现象；相关方法结果按有/无实际属性变化另存HOMA_stable_subgroups.json。主实验保留原生成规则和全体患者，没有根据方法效果删除患者。

{table(partial_stable,[('dataset','稳定条件'),('split','划分'),('patients','人数'),('moving_patients','实际属性变化人数')])}

## 2. 事件与指标

每轮顺序测量→判断刷新→更新缓存→回答，事件是原真值类别跨越。严格响应需事件前一轮输出正确旧类别，在0/1/2轮窗口出现旧→正确新类别输出；下一个事件发生前截止。初始或前态错误不剔除，严格失败；另报告恢复新状态。一个输出变化最多配一个事件，重复/错误变化为假事件。无真事件Recall/F1未定义，无输出变化Precision未定义。

EventPrecision@2=正确匹配/全部输出类别变化；Recall@2=正确匹配/全部真事件；F1由事件计数计算。稳定期误报为真类别未变的时刻出现未匹配输出变化的比例，排除已匹配合法迟到响应。原始稳定期输出变化率另保留，不能把合法迟到计成误报。假事件数还包括错向、错新类别、重复和旧态错误的未匹配变化。

延迟只对成功匹配求均值，同时报告失败率，避免将失败变成虚假快速响应。邻接事件、端点距阈值≤2%的事件、类别至少保持3个观测版本的事件另分组，分组可重叠。wMAPE辅助；AoII为错误持续年龄×归一化类别距离的时间平均。

RCE遵从论文公式(4)：100*mean(|pred−true|/(C−1))，当前三模板C=3；CMR=100*mean(pred!=true)。两者分别报告，不解释为真实临床决策改变比例。2%连续数值规则不再用于主事件。

## 3. A1固定实际资源：纯时机诊断

初始+额外K=1/2/4次，共K+1实际发布，单次向量预算B/(K+1)，同原指数机制、公开均值M-Opt权重。初始缓存一致、第j次发布使用相同uniforms，预算和次数严格相等。逆CDF对不同真值产生不同噪声分布，配对随机数不是强行使用相同加性误差。

公共等间隔计划相位−1/0/+1在验证集选；Oracle先取前K个真类别事件，不足则按固定公共候选补足，依赖真值时间且非DP，不是全局最优上界。SVT没有强塞入固定K主比较。

Oracle−周期的K=2宏平均效应：

{table(coremacro,[('source','类型'),('template','模板'),('B','B'),('conditions','有事件条件'),('unique_patients','独立身份数'),('mean_gap_recall1','ΔRecall@1百分点'),('gap_recall1_ci95','95%区间'),('min_condition_gap','最小条件效应'),('max_condition_gap','最大条件效应')])}

所有K及逐条件效应/区间见A1_macro_effects.json、analysis.json和summary.csv。真实逐窗口A1：

{table(a1_real,[('dataset','真实队列'),('B','B'),('recall1','Oracle@1 %'),('gap_recall1_vs_period','相对周期差'),('gap_recall1_ci95','95%区间'),('gap_rce_vs_period','RCE差')])}

## 4. A2总预算部署比较与选参

周期、公开计划、属性SVT和任务SVT全部总B。周期无需检测，可以将整个B用于发布。SVT沿用前轮有限重启AboveThreshold，不设计变体：C2/4，检测比例20%/50%，阈值0.01/0.03/0.1；全局敏感度不因观测变化小而降低。周期网格1/2/4/8/16。静态为负对照。

验证约束为平均有事件条件Recall@1≥20/50/80%，平均稳定期误报≤20%，只是探索规则。可行候选按RCE、CMR排序。不可行时选验证响应最高候选供诊断，明确不可行，不用静态缓存冒充满足要求。合成按模板/长度跨条件共同选参，真实窗口单独验证。该平均约束不保证每个模式均达标。

{table(feas,[('B','B'),('family','方法系列'),('recall_floor','响应下限 %'),('groups','验证组数'),('feasible_groups','可行组数')])}

真实部署结果B=0.7（目标20候选）：

{table(a2_real,[('dataset','队列'),('method','方法'),('recall0','响应@0'),('recall1','响应@1'),('recall2','响应@2'),('rce','RCE'),('cmr','CMR'),('false_change','稳定期误报'),('refreshes','实际发布数'),('validation_feasible','验证可行')])}

完整预设周期权衡、其他目标与预算均在summary.csv。不根据测试结果替换选参。验证可行不等于测试或临床可行。

## 5. 四组B诊断

8个案例在formal.json中预先固定，3函数/4事件模式/2真实窗口；K2/4。同Oracle日程、额度、评价窗口、随机数，初始精确/加噪×刷新精确/加噪四组。全部非DP；完全精确包含精确初始状态，仍受日程和次数限制。

以下B=0.7、K=2的全部四组结果：

{table(B_rows,[('dataset','案例'),('method','组合'),('recall1','响应@1'),('recovery1','恢复新状态@1'),('rce','RCE'),('cmr','CMR'),('false_change','稳定期误报')])}

全部加噪和完全精确的响应差，32个案例/预算/K组合平均{mean_loss:.2f}个百分点。它是这组选定诊断的描述性平均，不是独立样本统计。各条件初始噪声条件差、刷新噪声条件差及差分交互量保存在B_quadrant_effects.json；不可机械相减为独立贡献，初始正确状态是严格响应的重要前提。

按任务分开的诊断均值（ANEMIA包含真实及压力数据，另两任务仅压力）：

{table(b_template,[('template','任务'),('combinations','诊断组合'),('ideal_mean','完全精确响应均值'),('noisy_mean','全部加噪响应均值'),('gap_mean','差值百分点')])}

## 6. 检测费用与冻结日程的额外诊断

为了避免把较小发布预算的差异全归于检测错误，额外保持Oracle时刻/K/随机数不变，只把根发布份额从全B改为所选SVT剩余份额。这是预留检测预算的条件影响，结果在detection_fee_effects.json。不是让Oracle变成可部署DP。

再冻结所选SVT的真实日程，使用精确初始/刷新输出；每患者每随机重复给Oracle完全相同的实际刷新次数，比较精确输出日程质量。结果在frozen_schedule_effects.json；没有重新运行一个精确缓存下的新检测器。这只是固定实现日程的反事实，不与A1预先固定K混用，也不与四组差值相加成总因果分解。

同Oracle条件发布份额预留的平均效应为{mean_fee:.2f}百分点，精确输出下冻结日程与匹配次数Oracle的平均差为{mean_schedule:.2f}百分点。它们不是在相同体系中独立的检测误差与费用，不能相加；每案例明细分别保存，数值既包括预算对初始缓存的影响，也可能反映检测受到原缓存噪声的影响。

## 7. RQ1–RQ5与研究方向决策

RQ1：脱钩现象在三种原任务的受控属性依赖中存在，精确抵消和无关变化条件无类别事件，近似抵消可有少量边界事件。压力数据保证跨越提供广覆盖，但不能证明现实任务普遍频繁变化；真实证据只有ANEMIA。

RQ2：同实际资源的Oracle价值用A1判断。压力条件宏效应在所有模板/预算/K上的均值为{timing:.2f}个百分点；{len(timing_bounds)}/{len(stressmac)}个宏组效应≥5百分点且探索区间下界>0。5百分点仅为描述性分档，非临床标准，不仅凭获胜数量决定价值；逐条件范围和区间需同时看。

RQ3：四组B定位发布噪声和初始错误，完全精确参考受刷新上限限制；发布差异不能证明所有DP发布都不可用。当前指数根发布的噪声结构可能压缩时机收益，但未经成熟发布方法对照不能预定残差有效。

RQ4：部署结果以响应约束可行性及完整误差/误报权衡判定，不按平均wMAPE选择不更新。简单周期可能具有竞争力，SVT参考未调遍所有成熟策略，失败不否定成熟方法整体。

RQ5：依赖抵消、无关根和类别边界确实影响任务是否需要更新，独立根加噪可能损害派生类别。但这不是普通多变量时间序列DP未处理该现象的新颖性证明；RootGuard特有优势仍需另行验证。

**当前决策：{decision}；暂不停止动态方向，A的时机潜力也存在，但不优先设计新触发器。** 三任务压力K=2、B=0.7时，Oracle响应增益分别为ANEMIA约37.98、FIB4约12.36、HOMA约9.93百分点，说明有限机制并非完全抹掉时机价值；但当前发布噪声损失更大、部署约束可行性有限，先检查成熟发布方法更有信息价值。真实证据需要补充，不能由受控结果自动进入论文设计。若成熟方法已形成足够权衡，或更多真实任务无法复现有意义缺口，建议停止机制创新投入。当前不能因有限机制失败给出“所有DP不可能”的结论。

## 8. 隐私、统计与复现审计

保护T8/16、公共计划/根结构下整段数值轨迹的d_inf距离隐私。B0.3/0.7对应充分标准轨迹界299.7/699.3，不称标准ε0.3/0.7。部署根机制与有界AboveThreshold沿用前轮，检测/新值/可观察决定组合计费。权重与参数来自公共/验证资料，不用测试原始值免费决定部署刷新。原始离线报告、队列筛选未声称DP；Oracle和精确/冻结诊断均非DP，spent字段不显示有限隐私保证，nominal_calibration_cost只是参考尺度。

数学合同与前提详见PROTOCOL.md；有限精度和完整系统没有机器形式化证明。账本通过只核验实现费用，不代替证明。参考 [Dwork–Roth §3.6](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)。

60次噪声是离线独立模拟，预算B针对每个条件的一次完整轨迹发布历史；若真实系统同时公开全部重复，必须另行组合，不能仍声称总B。不同条件和窗口也是各自的评估合同，不能自动重置预算用于无限历史。

患者级配对bootstrap{cfg['bootstrap']}次，逐项95%区间，无确证性多重检验；跨条件宏平均共享患者抽样权重，避免场景/长度重复伪造样本量。噪声先按患者平均。真实样本及验证组小，存在选参波动，压力分布是人为设计，结论保持探索性。

环境Python{meta['python']}、NumPy{meta['numpy']}、SciPy{meta['scipy']}，主运行{meta['seconds']:.1f}秒。{len(rows)}测试对照、{meta['validation_cells']}验证对照；既有源码{audit['previous_files_checked']}文件变更0。来源SHA、完整新增源码快照、配置/种子、患者清单、患者统计、验证和选择都保存。9项协议测试与v2预检查通过；所有部署预算≤B，所有A1实际次数为K+1。初次正式输出在报告前复核时发现合法迟到响应被稳定期变化率计入，修正统计并补充回归用例后完整重跑选参与测试，最终使用stage1_formal_v2结果。初次输出保留但不作结论依据。旧结果未覆盖，无自动commit/push。

产物：metadata/config/source_snapshot、validation/selection、cells/patients、analysis/summary.csv、A1_macro_effects、B_quadrant_effects、detection_fee_effects、frozen_schedule_effects、generation_audit、HOMA_stable_subgroups、research_decision/audit和测试日志。主要代码：data_cases构造、event_metrics匹配、publishing复用和排程、run执行、analyze统计。本轮新增目录内变更，旧源码哈希检查通过。
'''
    (out/'REPORT.md').write_text(doc)
