# M5：工作区间分配、耦合迹门控、原子根包

独立目录，复用finite_cache公式/数据与M4的根发布器；不改变既有实验源码或结果。

- `public.py`：train-only工作区间、全部AST节点解析导数、查询权重、协方差及原子组。
- `core.py`：同预算候选包评估、L2迹门控、因果结转与固定包消融。
- `metrics.py`：最终MAE/wMAPE、RQ1 ordinal RCE、CMR、实际中间查询逐节点误差。
- `attack.py`：1000网格MAP（均匀/人群Gaussian先验）及历史Viterbi扩展。
- `attack_checks.py`：B=960共同初始根/实际发布时刻重构，排除把缓存陈旧误认为隐私改善。
- `run.py`：冻结输入，validation选计划，完成全部选择后评估test。
- `audit.py`：独立序列化账本和发布噪声/因果值重放。
- `analysis.py`：患者配对及患者＋机制两层CI、非单调B_min和报告。
- `MECHANISM-REVIEW-zh.md`：规格解释、机制审查与已完成测试。

```bash
cd /home/rootguard-research
/root/miniconda3/envs/rootguard/bin/python -m unittest experiments.workflow_extensions.m5.test_m5 -v
/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m5.run --precheck --output experiments/workflow_extensions/outputs/m5/precheck_reproduce
/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m5.run --output experiments/workflow_extensions/outputs/m5/real_reproduce
```

输出目录必须不存在或为空。默认复用`outputs/m4_carry/real_v1_20261116`的数据划分、配置与τ，以及`outputs/m4_carry/rootguard_mopt_20261116`的原版M-Opt基线。缺少旧输出时需先复现旧实验，再运行本比较，不悄悄重选基线。

主比较为M5、原M4-carry、原M-Opt静态/周期（原生指数核及相同Laplace核）。同时包含局部分配静态、逐根旧门控、迹门控、原子组、结转以及最终任务权重的诊断对照。不同单位中间节点先独立计算wMAPE，再按train查询频率宏平均；BUN_CR没有中间节点，其指标为NA。

实验是已分析过的demo患者上的机制诊断；公开统计只从train计算，test不用于选择参数。置信区间不包括再次选择参数的不确定性。MAP重构误差越低表示攻击越成功，不能当作隐私提升。

最终正式结果：`outputs/m5/real_v2_20261117/REPORT-zh.md`。初版正式目录保留；仅一个L1门控消融的查询目标修正后，在新目录重算该方法19440条路径，其他原始行逐字节一致，见`repair_audit.json`。`repair_local_ablation.py`用于记录这次有限修正；正常从头复现直接运行上面的主命令。
