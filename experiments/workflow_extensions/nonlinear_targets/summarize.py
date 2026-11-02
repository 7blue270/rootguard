import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent
linear={'non_hdl','ldl','pp','map','mbp','nlr_sum'}
nodes={'ANEMIA':['mcv','mch','mchc'],'AIP':['non_hdl','ldl','aip'],'CONICITY':['bmi','wthr','conicity'],'VASCULAR':['pp','map','mbp','ppi'],'FIB4':['fib4_prod','fib4_denom','fib4'],'TYG':['tyg_prod','tyg_half','tyg'],'HOMA':['homa_prod','homa'],'NLR':['nlr_sum','nlr_diff','nlr']}
with (P/'formula_inventory.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['corpus','unit','type','basis'])
 for t,ns in nodes.items():
  for n in ns:w.writerow(['RootGuard_derived_numeric_nodes',t+':'+n,'affine' if n in linear else 'nonlinear','classify after composing to raw root attributes; exclude categorical nodes'])
 for n in ['apsiii','lods','oasis','sapsii','sirs','sofa','meld']:w.writerow(['official_selected_score_programs',n,'nonlinear','piecewise thresholds/rounding/logs as function of underlying measured inputs; one program is one unit'])
D=json.loads((P/'results/results.json').read_text());A=json.loads((P/'results/allocations.json').read_text())
lines=['# M-Opt复杂目标函数探索实验','',
'## 结论','',
'普通平滑非线性不是本轮发现的主要缺陷。M-Opt在原有四个平滑模板、P/F比值及线性血气对照中总体保持较好表现。更值得关注的是参考均值处零梯度引起的预算饥饿，以及分段/舍入目标使M-Opt退化为均匀分配。',
'', '最严重退化出现在去掉舍入的MELD-Na诊断变体；保留官方舍入的同一公式，在三次划分中全部退回均匀分配，没有发生相同的劣化。不能将诊断变体的退化冒充原始官方MELD评分的失败。',
'', '## 1. 线性/非线性的可统计范围','',
'MIMIC-IV是记录数据库，没有“全部计算因子”的穷尽目录，也没有每种公式在真实对话中调用频次。不能给出全数据库或真实工作流的线性/非线性总体比例。以下是明确限定语料的逐项分类，不是随机抽样推断。',
'', '|统计范围|单位数|仿射/线性|非线性|解释|','|---|---:|---:|---:|---|',
'|RootGuard原有数值派生节点|24|6（25%）|18（75%）|表达式全部展开到原始根值；不计8个分类节点|',
'|RootGuard最终连续目标|8|0|8（100%）|MCHC/AIP/Conicity/PPI/FIB4/TyG/HOMA/NLR|',
'|官方score目录6个程序＋organfailure/meld.sql|7|0|7（100%）|完整评分程序相对原始测量是非线性；不代表MIMIC全部公式|',
'',
'6个根值仿射节点是non_hdl、ldl、pp、map、mbp、nlr_sum。tyg_half对中间乘积是线性的，但对TG与Glucose是双线性的；homa同理。nlr_diff含绝对值，全域为分段线性，归入非线性。分类阈值若另算节点也属于全域非线性，但未混入24个数值节点的分母。',
'',
'官方7个程序：APSIII、LODS、OASIS、SAPSII、SIRS、SOFA、MELD/MELD-Na。前6个包含阈值分段，MELD包含截断、对数、舍入和钠修正交互。score目录本来就是评分程序集合，存在明确选择偏向，不能由100%外推临床所有计算均是非线性。血气A–a梯度固定气压条件下为7.13*FiO2-PaCO2/0.8-PaO2，是实际仿射反例；P/F=100*PaO2/FiO2是非线性。',
'',
'官方版本固定为303d26c623dcc9c49cc0f204468d4acc2f063797；SQL保存在sources/，逐项清单见formula_inventory.csv。',
'', '## 2. 实验实现与数据','',
'- 新增run.py、verify.py、summarize.py，新增sources/与各次结果目录；未修改原机制、原实验或data_mine中的四份精简数据。',
'- 使用MIMIC-IV Demo 2.2原始检验和患者年龄表。每任务每患者取最早完整同刻测量，避免把重复测量当独立患者。血气限定明确ART标本，FiO2按官方SQL规范化百分比。',
'- 同一100位Demo患者按固定哈希划为50位参考、50位测试，各任务按可用性取子集。没有阈值/预算选参；三个划分种子为20261023/24/25。相同患者会跨任务和划分重复，不把这些当独立复制样本。',
'- 预算B=0.03/0.1/0.3/0.7/1/3；每预算60次随机重复，m=1000；每个样本重新初始化一次发布，不做多轮累计发布。',
'- M-Roots每根B/k。M-Opt直接加载utils/utils.py原optimal_budget_allocation_exact及其函数依赖，epsilon_min=.001，原abs目标，默认网格中心矩。新目标导数由参考均值处中心差分提供，并用10倍不同步长检查一致性。原代码get_local_partials是预定义节点的硬编码，因此新公式需要补导数；本次是同一分配方法的目标扩展，不是原代码开箱支持。',
'- 噪声采用preempt/sanitizer.py原连续索引离散指数概率分布，向量化逆CDF采样；与原函数300个配对抽样逐个核对一致。没有使用RQ4中先round索引的另一实现。配对随机数用于降低比较方差，不保证逐样本哪种方法更好。',
'- 数值域为每任务已公开Demo观测的min/max，包括公开测试值；均值只使用参考患者。因此它是公开有限基准域实验，不能包装为完全独立于测试数据的部署域估计。未剪裁真实测试值。域宽/参考均值可影响结论。',
'- B为原索引距离机制的组合预算，不能等同同数值的标准整用户ε-DP；本次只评估单次机制效用，不推导新的隐私保证。',
'- 指标为MAE、wMAPE、95分位绝对误差。原4个分类任务另报告原论文RCE与普通CMR；其他评分未人为添加风险类别，所以不计算其RCE/CMR。患者级配对bootstrap 2000次，先对患者内60次噪声求平均；大量单元比较属于探索性，不作多重检验后的确证性主张。',
'', '## 3. 测试任务与适用边界','',
'ANEMIA、FIB4、NLR、TYG为原模板对照；AADO2为仿射对照；PF_ratio为官方比值；MELD_smooth去掉基础MELD舍入；MELDNA_smooth在去舍入基础上保留钠截断与交互；MELDNA_exact保留官方舍入和条件。',
'',
'所有MELD变体均把透析状态固定为无，使用同刻化验而不是官方ICU首日最大/最小值；所以是该条件下的化验评分函数实验，未复现完整临床队列、透析识别或首日聚合流程。SOFA_lab3仅是血小板、胆红素、肌酐肾脏子项之和，不是完整SOFA，也未包含尿量。TYG未确认空腹条件，FIB4年龄仅为年级估计。',
'', '## 4. 首次划分主要结果','',
'下表wMAPE单位为%，越小越好；完整六预算、MAE、尾部误差、RCE、CMR及配对区间在results/results.json。','',
'|任务|参考/测试患者|B=.7 M-Roots|B=.7 M-Opt|B=3 M-Roots|B=3 M-Opt|','|---|---:|---:|---:|---:|---:|']
for name in A:
 r=next(r for r in D if r['task']==name and r['B']==.7);s=next(r for r in D if r['task']==name and r['B']==3)
 lines.append(f"|{name}|{A[name]['reference_patients']}/{A[name]['test_patients']}|{r['methods']['M-Roots']['wMAPE']:.3f}|{r['methods']['M-Opt']['wMAPE']:.3f}|{s['methods']['M-Roots']['wMAPE']:.3f}|{s['methods']['M-Opt']['wMAPE']:.3f}|")
lines+=['','### 关键失效诊断','',
'首次划分MELDNA_smooth参考钠均值138.897，高于137截断点，因此|∂g/∂Na(μ)|=0，Na分到.001。B=.7均匀Na预算是.175，B=3均匀Na预算是.75，而M-Opt仍给.001。它只在参考均值处局部不敏感，不能解释为所有测试患者都不依赖Na。',
'',
'B=.7时M-Opt/均匀MAE比为1.834，MAE差的患者级95%区间[.256,.804]；B=3时比为5.894，差区间[.439,1.110]。后者wMAPE从1.178%升至6.945%，95分位绝对误差从.500升至5.456。这是数值任务退化，不是临床伤害比例。',
'',
'保留官方舍入的MELDNA_exact在均值处所有偏导均为零，原分配器明确回退均匀，因此它与M-Roots完全一致；SOFA_lab3同理。这证明本方法在这些任务上无法发挥敏感度分配优势，没有证明官方离散评分产生巨大预算优化差错。',
'', '## 5. 更换患者划分','',
'|划分种子|MELDNA_smooth，B=.7的MAE比|B=3的MAE比|保留舍入MELDNA_exact|','|---|---:|---:|---|']
for seed,folder in [(20261023,P/'results'),(20261024,P/'results_seed_20261024'),(20261025,P/'results_seed_20261025')]:
 data=json.loads((folder/'results.json').read_text());a=next(x for x in data if x['task']=='MELDNA_smooth' and x['B']==.7);b=next(x for x in data if x['task']=='MELDNA_smooth' and x['B']==3)
 lines.append(f"|{seed}|{a['mae_ratio_opt_over_roots']:.3f}|{b['mae_ratio_opt_over_roots']:.3f}|均匀回退，完全相同|")
lines+=['',
'第三划分参考胆红素均值.848<1，胆红素也在截断平坦区，Na与胆红素均被分到最低预算，导致更严重退化。这表明结果依赖参考均值所在分支，不是非线性次数越多就越差。其余种子全部任务结果保留在各自目录，不仅保留有利于问题假设的目标。三个划分使用重叠人群，不能作为三个独立临床复现实验。',
'', '## 6. 研究价值判断','',
'1. 不支持“RootGuard未考虑非线性”这一论述：原8个最终目标本来就是非线性，多个平滑非线性任务中M-Opt仍表现较好。',
'2. 支持把问题收窄为：参考均值处的局部梯度，能否代表跨分支、截断区与测试人群的任务重要性？诊断变体中零梯度根的最低预算导致稳定退化；量级随参考群体显著变化。',
'3. 对官方原始离散评分，目前证据是“优化退化为均匀”，而不是“出现巨大差错”。不能用去舍入变体替代原公式来制造失败结论。',
'4. 当前证据值得做一次范围明确的后续确认，但不足以直接进入新机制设计：应先寻找独立、实际使用的平滑截断/交互目标，增加患者规模，并检查不同公开域宽及参考人群下的稳定性。此轮没有实现修补算法。',
'', '## 复现','',
'```bash','cd /home/rootguard-research','/root/miniconda3/envs/rootguard/bin/python experiments/workflow_extensions/nonlinear_targets/run.py','/root/miniconda3/envs/rootguard/bin/python experiments/workflow_extensions/nonlinear_targets/run.py --seed 20261024','/root/miniconda3/envs/rootguard/bin/python experiments/workflow_extensions/nonlinear_targets/run.py --seed 20261025','/root/miniconda3/envs/rootguard/bin/python experiments/workflow_extensions/nonlinear_targets/verify.py','python experiments/workflow_extensions/nonlinear_targets/summarize.py','```','',
'结果60单元/划分，共180单元；配对重复60次/单元。verify.py验证原采样一致性、公式边界、预算守恒、患者唯一及零梯度回退。正式运行只保存数据，无远程LLM调用。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('wrote REPORT.md and formula_inventory.csv')
