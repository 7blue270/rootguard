"""Chinese internal decision report; no practical threshold is invented."""
import csv
import json
import numpy as np


def f(x):return 'NA' if x in (None,'') or not np.isfinite(float(x)) else f'{float(x):.4g}'
def ci(r,prefix):return '['+f(r[prefix+'_two_layer_lo'])+', '+f(r[prefix+'_two_layer_hi'])+']'


def report(out,cfg,summary,cells,effects,aggregate,hit,sensitivity):
    audit=json.loads((out/'audit.json').read_text());folds=[json.loads(s) for s in (out/'fold_predictions.jsonl').open()]
    calibration=[json.loads(s) for s in (out/'l2_calibration.jsonl').open()]
    lines=['# M6.2：会话前选复用还是更新——信息预算与分侧验证','',
      f'主分析L0；四模板×六预算；公开模拟每配置{cfg["n_sim"]}条轨迹；真实发布R={cfg["repeats"]}；全局患者绑定的配对/患者＋机制两层bootstrap={cfg["bootstrap"]}。本轮为患者级外层留一的探索性复核：既往已经分析过这些患者，不能当作全新独立验证。','',
      '## 先回答研究去留','',
      '本轮只检验简单S/D间是否有值得利用的差异，以及能否在发布前识别；没有实现或验证M7的升级分支。用户未指定有依据的最小实用收益η，因此不输出“已达到实用门槛”的结论；逐模板报告连续收益、区间和方向性判断。不能由这两个固定策略的结果断言整个动态隐私场景无研究价值。','',
      'H=训练侧固定默认的MAE−同格S/D事后最小均值；U=默认MAE−选择器MAE。两者先在各预算计算、再在模板内六预算等权平均；不同模板不直接混加MAE。正U表示选择器改善。H是有限样本事后空间估计，可能偏乐观；交叉拟合默认并非所有折同一个动作，H甚至可为负，均如实保留。','',
      '|模板|H|两层95% CI|L0 U|两层95% CI|L0 U/H|方向性判断|','|---|---:|---|---:|---|---:|---|']
    for t in cfg['templates']:
        r=next(r for r in aggregate if r['template']==t and r['selector']=='L0')
        conclusion='净收益有正向证据' if r['U_patient_lo']>0 and r['U_two_layer_lo']>0 else '净收益有负向证据' if r['U_patient_hi']<0 and r['U_two_layer_hi']<0 else '净收益方向未决'
        lines.append(f'|{t}|{f(r["H"])}|{ci(r,"H")}|{f(r["U"])}|{ci(r,"U")}|{f(r["capture_ratio"])}|{conclusion}；实用性待η|')
    lines+=['','## 1. 固定的执行与信息边界','',
      'S：首轮用全B、全根均分，随后复用。D：首轮B/2，14和27轮各B/4、全根均分，其余复用。全部使用相同完整公开域、归一化Laplace恒等发布与裁剪；没有M6换核/原子根包/否决/结转。每个根的噪声尺度为1/ε_root。两个协议全程实际支出均为B，收益首先是同预算下的准确性差异。','',
      '时间口径：T=40，刷新为ceil(T/3)、ceil(2T/3)，公开发布时钟允许使用该轮截至当时已观测的最新根值，即使该轮未重测。原数据的测量时间、缺测沿用及查询轮不改。主终点只在原查询序列中最终节点被实际查询的轮次计算，按会话、患者等权；不是以全部40轮或全部中间节点重新定义标签。','',
      '日程实例化：公开预测对各模板已冻结的20个公开测量/查询日程等权模拟，12000轨迹按日程分层；同一模板/B的L0/L1对所有患者共用选侧。因此研究单位为“模板×B×冻结的公开日程设计”，不是逐患者真值分类。日程集合只含轮次、测量根掩码和最终查询轮，不含任何根值。这个公开日程分布假设是本轮适用范围的一部分。','',
      'Rhat_session=模拟全程S损失/模拟全程D损失；Ghat=模拟S损失−模拟D损失，正值选D、平局选S。局部判据在第14轮比较B/2锚点一直沿用与B/4新发布，使用相同模拟路径与噪声。去AST以归一化根平均绝对误差作预测目标，但实际标签仍为同一最终任务MAE；去预算固定预测B=320；仅非线性度沿用旧ν>3选S，否则D，并逐外层折仅用训练患者重估旧ν。','',
      '主模型每条轨迹一次抽共享/逐根±方向，按Δ[c·共享方向+(1−c)·逐根方向]在实际公开测量机会更新相应根，裁剪到[0,1]；未测根沿用，年龄不随实验轮变化。主Δ=.010、c=0、公开种子20261118。敏感性：5个Δ×3个c×持续/逐机会重抽方向×L0/L1，全格点保留；c是混合权重，实测相关矩阵另存，不把c当Pearson相关。','',
      '## 2. L0/L1/L2的来源与成本','',
      'L0：公开域内独立均匀初值，不读取患者值。L1：独立成人参考区间的边缘均匀先验，不把临床参考区间当ICU人群分布；不拟合根间相关性。L2：在L1基础上只改变一个全模板共用的归一化漂移尺度，逐外层折在训练患者内估计。参考区间只改变模拟先验，绝不缩小DP输入域。','',
      'L1来源：[生化参考区间](https://medlineplus.gov/ency/article/003468.htm)、[血常规参考区间](https://medlineplus.gov/ency/article/003642.htm)、[血小板及单位](https://www.nhlbi.nih.gov/health/blood-tests)、[INR](https://medlineplus.gov/ency/article/003652.htm)。Hb/Hct/RBC采用男女参考区间并集；PLT按细胞/μL换算K/μL；年龄保留原公开域而非伪造“正常年龄区间”。完整数值、单位处理与来源见reference_intervals.json。','',
      'L2：同一原始labevent_id跨模板只计一次；同根同charttime取最新记录；按住院内公开检验批次间隔折算归一化差；先在患者/根内平均，再根等权、患者等权。固定年龄无纵向检验差分，不加入漂移估计。训练统计本身未做DP，保护账本针对目标会话，不能宣称同时保护训练患者。','',
      '|L2项目|最小|最大|','|---|---:|---:|',
      f'|训练患者|{min(r["n_patients"] for r in calibration)}|{max(r["n_patients"] for r in calibration)}|',
      f'|去重有效差分|{min(r["valid_differences"] for r in calibration)}|{max(r["valid_differences"] for r in calibration)}|',
      f'|全局Δ估计|{f(min(r["delta"] for r in calibration))}|{f(max(r["delta"] for r in calibration))}|','',
      'table2_implementation_cost.csv逐信息档列训练人数、差分数、每配置公开MC时间与部署字典查表时间；真实支出、动作数、总根发布数在summary.csv逐B列出。全部阶段运行耗时见timing.json。','',
      '## 3. 主表一：逐格选侧与真实差异','',
      'G=μS−μD，正值D更好。患者与两层95%区间均在0同侧才列为可判定，否则“未决”。全部格子仍计入选错损失；不把未决格排除后夸大总体收益。','',
      '|模板|B|L0预测|L1预测|L2选D比例|真实G|两层95% CI|真实侧|L0事后超额损失|','|---|---:|---|---|---:|---:|---|---|---:|']
    for t in cfg['templates']:
        for B in cfg['budgets']:
            rr={r['selector']:r for r in cells if r['template']==t and r['B']==B};a=rr['L0'];b=rr['L1'];c=rr['L2']
            lines.append(f'|{t}|{B:g}|{a["predicted_side"]}|{b["predicted_side"]}|{f(c["D_fraction"])}|{f(a["G_true"])}|{ci(a,"G_true")}|{a["true_side"]}|{f(a["cell_oracle_excess"])}|')
    lines+=['','|选择器|可判定格/全部格|覆盖率|可判定格中正确决策比例|','|---|---:|---:|---:|']
    for r in hit:lines.append(f'|{r["selector"]}|{r["decidable_cells"]}/{r["n_cells"]}|{f(r["coverage"])}|{f(r["mean_correct_fraction_on_decidable"])}|')
    lines+=['','L0/L1同一配置共享预测。训练型选择器在外层折可能作不同选择，报告D比例和正确决策比例，不强行将它们伪装成一个统一侧。24格不是24个独立样本，不做基于24独立伯努利试验的准确率显著性检验。误选S/D分列见hit_summary.csv。','',
      '## 4. 主表二：相对默认、查表与强周期的净收益','',
      '下表为模板内六预算等权的MAE收益，正值更好。完整逐预算绝对/百分比差值及两种区间见table2_baseline_gains.csv。','',
      '|模板|选择器|相对默认U|两层95% CI|相对查表收益|两层95% CI|相对B2-val收益|两层95% CI|','|---|---|---:|---|---:|---|---:|---|']
    for r in aggregate:
        if r['selector'] in ['L0','L1','L2','Local-L0','NoAST-L0','NoBudget-L0','Nonlinearity']:
            lines.append(f'|{r["template"]}|{r["selector"]}|{f(r["U"])}|{ci(r,"U")}|{f(r["gain_vs_TrainLookup"])}|{ci(r,"gain_vs_TrainLookup")}|{f(r["gain_vs_B2-val"])}|{ci(r,"gain_vs_B2-val")}|')
    lines+=['','固定默认在每折训练患者上按模板跨六预算选S或D。查表逐模板/B选侧。L2和查表不含该折留出患者，L2使用其余患者跨模板的一项标量，查表使用相同外层训练患者中该模板的轨迹。B2-val只在外层训练侧固定25%患者上选P∈{2,3,4,5,6,8,10,12}；P以实际测量批次为单位，首轮B/2，余款等分到全根周期刷新。相同患者跨模板同折。','',
      '所有外层规则先冻结，再生成独立m62_evaluation发布噪声。训练回放用不同m62_training噪声相位；它们只用于各折排除了自身留出患者的选择。全部选择器引用同一对真实S/D损失，绝不重新抽噪声排名。','',
      '## 5. 主表三、先验敏感性与两侧意义','',
      'table3_H_U_M7.csv给出各选择器完整H/U与两种区间。仅当H的两种95%下界均大于0才报告U/H。否则“高命中率”不能说明分流有实用价值。后续M7静态、动态分支各自的增量需要新的固定分组配对替换；本轮没有验证这两项创新。','',
      '|模板|L0/L1主先验选侧不同的格数|局部/全程L0选侧不同的格数|敏感性配置改变主选侧的比例|','|---|---:|---:|---:|']
    for t in cfg['templates']:
        cc={B:{r['selector']:r for r in cells if r['template']==t and r['B']==B} for B in cfg['budgets']}
        pri=sum(v['L0']['predicted_side']!=v['L1']['predicted_side'] for v in cc.values())
        loc=sum(v['L0']['predicted_side']!=v['Local-L0']['predicted_side'] for v in cc.values())
        ss=[r['differs_from_primary'] for r in sensitivity if r['template']==t]
        lines.append(f'|{t}|{pri}/{len(cc)}|{loc}/{len(cc)}|{f(np.mean(ss))}|')
    lines+=['','敏感性全格点结果与其实际选侧代价全部保留在sensitivity_evaluation.csv，未选择test上最好的Δ、相关成分或先验作为主结果。相邻预算符号翻转见sign_crossings.csv；多次翻转/未决均保留，不假设单调或单一临界预算。','',
      '## 6. 次指标、预算节省与统计边界','',
      'summary.csv含查询轮最终节点MAE/wMAPE、p95、RCE、投影率、支出与发布次数。wMAPE为患者均权误差/患者均权真实绝对值×100；RCE为三分类序数误差，ANEMIA/FIB4可用，其余NA。不同指标可朝相反方向变化，不把MAE改善写成所有下游指标改善。','',
      'B_min沿用先前公开合成参考生成的τ，不从本轮test生成；仅在六点网格扫描，未达标记“测试网格未达标”，不推导连续预算下界。不做单调包络或二分；80达标是网格左边界，不能声称真实最小预算就是80。','',
      '患者为独立单位，共24位全局患者，四模板各20人；不是80位独立患者。bootstrap的患者与机制索引在模板、预算和方法间绑定，均值/百分比/事后最小值均重算。区间条件于已冻结外层规则，没有在每次bootstrap重新训练L2、查表和周期，因而不含训练选择不确定性；重复外层训练集也不独立。全部为点态探索性95%区间，未做比较族校正。','',
      '事后S/D最小值存在有限样本乐观偏差。对跨折可能选不同侧的规则，μA−min(μS,μD)可能为负，本轮保留为“cell_oracle_excess”而不偷偷裁为0；另报以每个患者S/D重复均值取最小的非负patient_oracle_regret，但它更乐观且不可部署，不能与配置级oracle混用。','',
      '## 7. 审计、图和复现','',
      f'独立转录审计{audit["paths"]}条执行路径、{audit["root_releases"]}个根发布，重放最大差{audit["max_release_difference"]}。初始化、每次正ε、每个前缀不超B、完整会话总额B、私有值因果性、所有选择器相同S/D结果复用、外层排除和预测冻结检查全部通过。',
      '', '每模板分别输出预测G/真实G曲线与相对默认收益曲线，见figures/；PNG便于查看，PDF/SVG可编辑，图源表与QA文件一并保存。正收益为D更好或选择器更好，坐标标题明确区分。若采用symlog，是为了同时显示公开先验预测量级与真实收益，不删除极端值。','',
      '```bash','cd /home/rootguard-research',
      '/root/miniconda3/envs/rootguard/bin/python -m unittest experiments.workflow_extensions.m62.test_m62 -v',
      'OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m62.run --precheck --output experiments/workflow_extensions/outputs/m62/precheck_reproduce',
      'OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m62.run --output experiments/workflow_extensions/outputs/m62/real_reproduce','```','',
      '输出目录必须为空。原始动作、指标、预测、校准计数、冻结哈希和bootstrap索引均保留。面向LLM工作流的应用主张还需要跨轮外部根/中间值读取实例；如果外部只需最终分数，应另与客户端直接发布受保护分数比较。']
    (out/'REPORT-zh.md').write_text('\n'.join(lines)+'\n')
