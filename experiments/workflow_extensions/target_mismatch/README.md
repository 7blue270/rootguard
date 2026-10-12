# 目标函数未知或不准确：机制级探索实验

研究问题：固定公开信息下，近似目标函数造成的敏感度失配，是否使 M-Opt 的真实任务效用下降，甚至劣于 M-Roots？不涉及目标切换、NER 错误或根数据更新。

## 原实现复用与实验隔离

- 输入：`data/nhanes_benchmark_200.json`、`data/holdout_population_means.json`。
- 根属性及域：`preempt/sanitizer.py:TEMPLATE_DOMAINS`，采用原版 RQ1 的逐模板域。
- 优化：`utils/utils.py:optimal_budget_allocation_exact`，`mode=abs`，默认网格中点噪声矩；敏感度在固定公共均值处计算。
- 噪声：原版 `Sanitizer.M_exponential_discrete` 的未取整指数概率；批量逆 CDF 采样，测试与原方法逐次相同。
- 分类：`utils/risk_class.py:get_risk_class`。ANEMIA/HOMA 的类别定义遵循这个 RQ 评价接口，不使用 DAG 中二分类表达式；RCE 是分类不一致比例，不是归一化类别距离。
- 原源码不修改。`core.py` 从原文件 AST 加载原优化器函数及静态域，避免导入与本实验无关的 NER、模型和姓名数据库。所需函数缺失即报错；运行记录源文件哈希。
- 每个患者/噪声重复只产生一份根缓存，后续结果为真实任务的确定性后处理；本实验是数值机制实验，不运行完整 LLM 会话。

只改变分配所用敏感度；加噪后始终使用真实函数评价。公共均值、域、网格、总预算等在同一比较内固定。总预算 B 直接取 0.1、0.3、0.7，不等同于论文每次调用的 epsilon。

## 三组与失配定义

M-Roots：每个根 B/k；真实 M-Opt：真实函数公共均值敏感度；近似 M-Opt：失配敏感度。无噪声数据核对作为真值检查。

真实函数均为幂乘形式。记公开均值 mu、真实指数 p，近似函数：

`ghat(x) = g(mu) * product((x_i / mu_i) ** q_i)`。

一次改变一个根。非零真实指数 `q_j=p_j*(1+level)`，其余不变；ANEMIA 的 RBC 真指数为零，设置 `q_RBC=max(level,0)`，负 level 不扫描。

- level=0：完整恢复真实函数。
- level=-0.1/-0.25：所选指数及均值敏感度低估 10%/25%。
- level=-1：所选属性被近似函数完全遗漏。
- 确认实验增加 -0.9/-0.95/-0.99：仍保留非零敏感度的严重低估。
- 对 RBC，level 表示引入的指数，不表示相对于零的百分比。

`weights` 家族直接扰动敏感度，用于诊断而非可部署近似函数；零敏感度根加入的域尺度重要性等于 level 乘以最大真实域尺度重要性。两个家族在非零根上可能给出完全相同分配，这些单元不构成独立证据。

## 文件职责

| 文件 | 职责 |
|---|---|
| core.py | 原源码适配、真函数、近似函数、分配与指数采样 |
| run_pilot.py | 数据核对、扫描、保存逐患者/逐重复误差及运行来源 |
| analyze.py | 患者配对 bootstrap、固定患者的噪声 Monte Carlo 区间 |
| configs/*.json | 探索扫描和两个独立种子的确认配置 |
| tests/test_invariants.py | 真值、零失配、梯度、预算、原采样和缓存隔离检查 |

## 运行

从仓库根目录执行。环境需要 NumPy 和 SciPy；无需 API key、LLM 或 GPU。本次使用已有 `/root/miniconda3/envs/rootguard/bin/python`，未修改环境。

```bash
PYTHON=/root/miniconda3/envs/rootguard/bin/python
$PYTHON -m unittest discover -s experiments/workflow_extensions/target_mismatch/tests -v

$PYTHON experiments/workflow_extensions/target_mismatch/run_pilot.py \
  --config experiments/workflow_extensions/target_mismatch/configs/pilot.json \
  --out experiments/workflow_extensions/outputs/target_mismatch/<新的运行标识>

$PYTHON experiments/workflow_extensions/target_mismatch/analyze.py \
  experiments/workflow_extensions/outputs/target_mismatch/<运行标识>
```

确认实验更换配置为 `confirmation_a.json`、`confirmation_b.json`，各自使用新的输出目录。运行器拒绝覆盖已有目录。

## 输出与统计解释

`config.json`、`metadata.json`、`cells.json`、`patients.csv`、`draws.npz` 为运行输入/来源和结果；分析产生 `analysis.json`、`summary.csv`、`checks.json`、`SUMMARY.md`。

- wMAPE = 100 * sum(各患者平均绝对误差) / sum(真实目标绝对值)。
- RCE = 100 * 平均分类错误比例。组间差值单位为百分点，越低越好。
- importance_l1 是真实/近似的归一化 `|gradient|*domain_width` 向量间 L1 距离，范围 0–2。
- 先平均同一患者的噪声重复，再做 2000 次患者配对 bootstrap。另用每次完整患者队列的配对误差差值计算噪声 Monte Carlo 标准误及 t 区间；两个区间分别报告，不能互换。
- 同一模板各组使用共享独立均匀随机数作配对采样。不同噪声重复、患者、根之间仍独立；种子固定且可重现。
- 区间为探索性逐单元 95% 区间，未校正多重比较；两次确认使用同一患者的独立噪声，证明噪声稳定性，不是新队列验证。
- 不据此声称隐私失效、现实失配发生率、全局最优或论文创新性已确立。

全部生成结果写入已忽略的 `../outputs/target_mismatch/`，代码和配置保持可跟踪。
