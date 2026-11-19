"""Concise Chinese internal report, including untriggered switches and limits."""
import csv
import json
import numpy as np


def fmt(x):return 'NA' if x is None or not np.isfinite(float(x)) else f'{float(x):.3f}'


def report(out,cfg,stats,summary,paired,bmins,attacks,attack_pairs):
    def ci(r):return '['+fmt(r['two_layer_lo'])+', '+fmt(r['two_layer_hi'])+']'
    def verdict(r):
        if r['patient_hi']<0 and r['two_layer_hi']<0:return '改善'
        if r['patient_lo']>0 and r['two_layer_lo']>0:return '退化'
        return '证据不足'
    endpoint=list(csv.DictReader((out/'attack_endpoint_summary.csv').open()))
    endpoint_pairs=list(csv.DictReader((out/'attack_endpoint_paired.csv').open()))
    # endpoint CSV is written immediately after base analysis; report is called
    # only after that step, so no implicit rerun or test-based choice occurs.
    lines=['# M6 实现、机制审核与真实轨迹对照实验','',
       f'内部机制诊断；R={cfg["repeats"]}，患者级/患者＋机制两层配对 bootstrap={cfg["bootstrap"]}，预算{len(cfg["evaluated_budgets"])}点（80–960）。沿用M5患者划分、完整公开域、任务、公开Δ与τ；M-Opt对照及M5原始结果逐行原样复制。test此前已用于机制诊断，本轮不是新患者泛化证据。','',
       '## 主要结果：B=960','',
       'MAE、wMAPE、RCE和中间查询wMAPE越低越好。所有模板单独判断，不能用另一模板的收益抵消退化。','',
       '|模板|M5 MAE|M6 MAE|相对M5变化|两层配对95% CI（M6−M5）|判断|','|---|---:|---:|---:|---|---|']
    tables={t:{r['method']:r for r in summary if r['template']==t and r['B']==960} for t in cfg['templates']}
    for t,s in tables.items():
        r=next(r for r in paired if r['template']==t and r['B']==960 and r['metric']=='mae' and r['method']=='M6' and r['reference']=='M5')
        lines.append(f'|{t}|{s["M5"]["mae"]:.3f}|{s["M6"]["mae"]:.3f}|{100*(s["M6"]["mae"]/s["M5"]["mae"]-1):+.1f}%|{ci(r)}|{verdict(r)}|')
    lines+=['','## 对照结果与指标口径','',
      'wMAPE=100×患者均权绝对误差/患者均权真实绝对值，bootstrap重新计算分子分母。RCE沿用RootGuard RQ1三类别序数误差100×mean(|预测类别−真类别|/2)，ANEMIA/FIB4可用，其他模板NA。中间查询wMAPE在真实查询轮逐节点求比值，再按固定train查询频率宏平均；BUN_CR没有中间节点，NA。p95先逐患者/机制路径计算再均权。','',
      '|模板|方法|MAE|wMAPE %|RCE %|中间查询wMAPE %|p95|刷新m|支出比例|投影率|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    main=['RG-static-exp','RG-periodic-exp','RG-static-lap','RG-periodic-lap','M5','M6']
    for t,s in tables.items():
        for m in main:
            r=s[m];lines.append('|'+t+'|'+m+'|'+'|'.join(fmt(r.get(f)) for f in ['mae','wmape','rce','intermediate_wmape','p95','m','spend_fraction','projection_rate'])+'|')
    lines+=['','原核exp的预算分配直接复用原RootGuard exact M-Opt、1000格完整域指数核；lap为相同分配替换连续Laplace核的机制对照。周期基线在validation选P，初始化B/2，余下B/2等分到全根刷新；M6共享会话预算B上限，但实际支出可能因否决少于B，计划也不同。比较不能解释为同刷新次数或同逐根ε实验。','',
      '|模板|参考|MAE差 M6−参考|患者95% CI|两层95% CI|判断|','|---|---|---:|---|---|---|']
    for r in paired:
        if r['B']==960 and r['metric']=='mae' and r['method']=='M6' and r['reference'] in main[:-1]:
            lines.append(f'|{r["template"]}|{r["reference"]}|{r["delta"]:.3f}|[{fmt(r["patient_lo"])}, {fmt(r["patient_hi"])}]|{ci(r)}|{verdict(r)}|')
    lines+=['','## 公开骨架、实际触发与M5遗留问题','',
       '阈值在读取test前写入config：c*=0.70、ν*=3、ν_grid*=1；Q10–Q90工作区间，梯度Q90，4096个固定公开随机工作点。危险判据有两个分支：AST负幂分母根低于train中位数，或该根在当前公开基点的工作量梯度高于固定工作区间梯度Q90。静态初始化用公开μ判断，动态刷新只用已发布缓存。文档没有给定这些参数的唯一数值，本轮是明确记录的实例化，不声称阈值已获最优或鲁棒性验证。','',
       '|模板|train c|train ν|模式|分配|原子组|B960 K/rho|初始化网格根|','|---|---:|---:|---|---|---|---|---|']
    selections=json.loads((out/'selections.json').read_text())
    for t,v in stats.items():
        p=v['main'];s=selections[t]['960.0'];roots=p['roots']
        atoms=';'.join('{'+','.join(roots[i] for i in g)+'}' for g in p['groups'])
        from .kernels import danger
        flags=danger(np.array([p['mu']]),p)[0]
        grid=[r for r,d in zip(roots,flags) if d]
        lines.append(f'|{t}|{p["cancellation"]:.3f}|{p["nonlinearity"]:.3f}|{p["regime"]}|{p["allocation_mode"]}|{atoms}|{s["K"]}/{s["rho"]}|{",".join(grid) or "无"}|')
    lines+=['','FIB4是否被判静态、是否实际换核必须看上表，不能套用合成sanity的4/4分离说法。K≥1只保证提供刷新机会，仍可全部否决。ANEMIA的三个比值共享根，经符号组传递闭包后为一个三根包；FIB4保留AST派生节点边界，分成age/ast与alt/plt两组。','',
       '跨期风险使用train轨迹加固定扰动构成的公开粒子，按缓存、上一ε和公开批次龄计算近似后验权重；当前候选刷新读取粒子的当前隐状态，再按同一实际核加噪。期望累计尺度化查询误差＋0.1×跨期变坏概率＋0.1×尾部p95。预测窗口到T，候选缓存在窗口内保持；这是terminal-hold前瞻近似，不是求解未来全部刷新策略的贝尔曼最优值。规划器对公开Δ合成轨迹调用与test完全相同的结转执行器，在动态K∈{1,2}、rho∈{0.5,0.7,0.85}中选；不调用validation重选M6。','',
       '与M5相比，修复了隐式K=0塌缩（动态分支）、由权重阈值拆散符号组、分配指数混杂及混合核攻击似然问题；是否带来效用改善须由上面的配对结果判断。公开先验不能实时看见尚未发布的漂移，这个信息限制仍存在。','',
       '## 消融','',
       'iso三方法强制每个固定机会实际发布，统一K=2、rho=0.5、等间隔对齐时间、每动作总ε和初始化；用于单轮→跨期→符号分组。根集合和逐根分配随选择变化，故不是纯时间实验。fixed-no-carry与dynamic-fixed允许否决、共享同一公开计划，仅切换结转；否决会改变实际次数和后续支出，不能同时宣称实际次数固定。主M6与这些固定计划消融的差异还包含模式/计划选择。','',
       '|模板|方法−参考|MAE差|两层95% CI|','|---|---|---:|---|']
    ab=[('M6-iso-cross','M6-iso-single-step'),('M6-iso-atomic','M6-iso-cross'),('M6-dynamic-fixed','M6-fixed-no-carry')]
    for r in paired:
        if r['B']==960 and r['metric']=='mae' and ((r['method'],r['reference']) in ab or r['method']=='M6' and r['reference'] in ('M6-lap-only','M6-mean-only','M6-final')):
            lines.append(f'|{r["template"]}|{r["method"]}−{r["reference"]}|{r["delta"]:.3f}|{ci(r)}|')
    lines+=['','中间节点目标消融（同一M6计划，改变公开目标会同时改变分配/风险，并非单个矩阵元素消融）：','',
      '|模板|M6−M6-final 中间wMAPE差|两层95% CI|','|---|---:|---|']
    for r in paired:
        if r['B']==960 and r['method']=='M6' and r['reference']=='M6-final' and r['metric']=='intermediate_wmape':
            lines.append(f'|{r["template"]}|{r["delta"]:.3f}|{ci(r)}|')
    lines+=['','## MAP重构攻击与端点','',
       '候选输入仍为完整公开域1000格，绝不把攻击候选限定在输出工作区间。混合核逐根逐动作记录，指数核似然包含随候选输入变化的精确归一化常数；Laplace包括裁剪边界质量。snapshot只用最近实际发布；history在train最大跳变的独立有界随机游走假设下做网格Viterbi，是模型敏感性对照，不保证比snapshot更强。共享配对种子不提供给攻击器；实际部署须秘密随机性。','',
       '攻击wMAPE越低代表重构越容易。全轨迹攻击混有陈旧误差，故单独报告初始化根与实际发布时点。发布时点因方法不同也不同，初始化才是共同时间端点。','',
       '|模板|方法|全轨迹snapshot-informed wMAPE %|history-informed %|初始化 %|发布端点 %|','|---|---|---:|---:|---:|---:|']
    for t in cfg['templates']:
        for m in main:
            a=next(r for r in attacks if r['template']==t and r['method']==m and r['B']==960 and r['attack']=='snapshot_map' and r['prior']=='informed')
            h=next(r for r in attacks if r['template']==t and r['method']==m and r['B']==960 and r['attack']=='history_map' and r['prior']=='informed')
            e={r['endpoint']:r for r in endpoint if r['template']==t and r['method']==m}
            lines.append(f'|{t}|{m}|{a["root_macro_wmape"]:.3f}|{h["root_macro_wmape"]:.3f}|{fmt(e["initial"]["macro_wmape"])}|{fmt(e["release"]["macro_wmape"])}|')
    lines+=['','M6−M5初始化重构wMAPE差（负值为攻击更成功）：','',
       '|模板|差值|两层95% CI|','|---|---:|---|']
    for r in endpoint_pairs:
        if r['reference']=='M5' and r['endpoint']=='initial':lines.append(f'|{r["template"]}|{fmt(r["delta"])}|{ci(r)}|')
    lines+=['','## 固定公开τ的B_min','',
       '扫描全部预算点，不做单调包络或二分；未达标为右删失，仅说明网格内未达到。bmin.csv含两层区间和达标率，bmin_paired.csv的Saving区间以双方达标为条件并列双方达标比例。bmin_ucb.csv另列点态95%上端点达标预算，不声称整条曲线同时覆盖。','',
       '|模板|公开Δ/目标|τ|RG静态exp|RG周期exp|M5|M6|','|---|---|---:|---:|---:|---:|---:|']
    targets=json.loads((out/'targets.json').read_text())
    for t,tt in targets.items():
        for sc in tt:
            for target in ['lo','hi']:
                rr={r['method']:r for r in bmins if r['template']==t and r['scenario']==sc and r['target']==target}
                vals=[rr[m]['B_min'] for m in ['RG-static-exp','RG-periodic-exp','M5','M6']]
                lines.append('|'+t+'|'+sc+'/'+target+'|'+fmt(rr['M6']['tau'])+'|'+'|'.join('网格内未达标' if v is None else f'{v:.2f}' for v in vals)+'|')
    lines+=['','## 大问题与可解释边界','',
       '1. 区制门是公开经验规则，不是“证明静态最优”。train c与ν会随目标缩放、工作点采样变化；演示阈值不保证真实FIB4进入静态。当前统计量/危险阈值可能使预期模块根本不触发，这是方法设计有效性问题，不能靠代码账本通过掩盖。',
       '2. 动态决策的收益依赖公开粒子模型是否拟合真实回摆/漂移。terminal-hold风险没有联合优化未来机会，不构成最优调度证明；本轮不能把观察到的提升归结为普遍跨期最优。',
       '3. EM输出限Q10–Q90，超工作区间真值不可表示，会有系统性偏差。投影率只统计Laplace裁剪前出界；outside_work_grid_fraction单列EM发布真值超工作区间比例，不能用投影率为0宣称没有偏差。',
       '4. 公开train统计条件化不保护这些train个体；保证针对当前test会话、给定公开域/日程/项目和骨架。所有根预算顺序相加，任何停机前缀都不超B。效用提升不意味着更强DP或更难MAP。',
       '5. test每模板仅4–5患者，CI以患者为单位、机制重复不是新患者。点态多重比较未校正，目标/阈值选择不确定性未包含；后续正面结果需新患者或外层验证。','',
       '## 审计与复现','']
    audit=json.loads((out/'independent_audit.json').read_text())
    lines+=[f'序列化转录独立审计：{audit["independent_paths"]}路径、{audit["root_publications"]}个根发布；最大发布重放差{audit["max_release_difference"]:.3g}；冻结旧结果逐行完全一致。完整断言见asserts.log，机制公式修正见源码MECHANISM-REVIEW-zh.md。',
       '','```bash','cd /home/rootguard-research',
       '/root/miniconda3/envs/rootguard/bin/python -m unittest experiments.workflow_extensions.m6.test_m6 -v',
       '/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m6.run --precheck --output experiments/workflow_extensions/outputs/m6/precheck_reproduce',
       '/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m6.run --output experiments/workflow_extensions/outputs/m6/real_reproduce','```','',
       '输出目录须为空。完整原始动作/预算阶梯见actions.jsonl；完整曲线见summary.csv；两层配对差值见paired_summary.csv；MAP均匀/人群先验与历史结果见attack_summary.csv。']
    (out/'REPORT-zh.md').write_text('\n'.join(lines)+'\n')
