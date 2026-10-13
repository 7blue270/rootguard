# 动态根属性：探索实验

2026-10-12 已实现并运行。目标是筛查固定缓存的时效性、跨版本预算与任务效用的矛盾；当前没有提出新的优化算法。没有改动 RootGuard 核心实现或上一次目标函数失配实验。

## 隐私合同

有限长度 H、根属性结构、测量时刻和更新哪些根属性均为公共信息。秘密是数值轨迹。将根属性归一化为 `z=(m-1)*(x-lo)/(hi-lo)`，使用 `d_inf=max_{i,t}|z_it-z'_it|`。发布单根的指数机制满足 epsilon 距离隐私；公开计划或私有测试决定的发布费用相加不超过 B，从而完整 transcript 满足 B 距离隐私。同一版本的重复查询复用同一缓存，只产生后处理。

每根的指数概率是 `exp(-epsilon*|grid-z|/2)/Z(z)`；三角不等式分别约束分子和归一化因子各 epsilon/2。私有测试分数 `max_i |z_i-pred_i|` 在固定过去 transcript 下是 1-Lipschitz，添加尺度 `1/epsilon_test` 的 Laplace 噪声；测试位和条件刷新均计费。

因为 `d_inf <= m-1`，设置 `B=E/(m-1)` 足以获得标准纯 E-轨迹 LDP。这是充分校准，未声称最紧。主实验 B=0.1/0.3/0.7、m=1000，对应标准界 99.9/299.7/699.3，不能称作小 epsilon 的强标准 LDP。另跑 E=1/4/8 的标准轨迹实验。更新事件、时间戳、文本、根集合新增、未知长度和人口筛选均不在当前隐私合同内。

## 文件与方法

| 文件 | 内容 |
|---|---|
| `data.py` | NHANES 合成轨迹；MIMIC-IV Demo 2.2 同标本 CBC 复测筛选 |
| `mechanisms.py` | 缓存、公开刷新计划、跨版本账本、私有测试参考实现、指标 |
| `run_experiments.py` | 主实验与标准 LDP 实验，输入 SHA256、版本、患者结果、部分轨迹 |
| `confirm.py` | 同一 24 位患者/同一住院的截断序列；独立种子、200 次重复 |
| `attack_probe.py` | 公共已知偏移下的标量 MAP 重构诊断 |
| `analyze.py` | 患者配对 bootstrap 与固定患者噪声重复置信区间 |
| `build_report.py` | 结果审计、证据表和完整中文报告 |
| `configs/*.json` | 三组实验的固定参数 |
| `tests/test_dynamic.py` | 预算、缓存、指数概率界、测试敏感度、数据筛选等验证 |

`static` 首次私有化后一直复用；`full_uniform/full_opt` 每版本刷新全部根；`root_uniform/root_opt` 仅按公共更新计划刷新对应根；`period2/period4` 每 2/4 个版本刷新全部根。Opt 使用原始绝对误差代理目标，公开发布次数 n_i 下每次 epsilon_i=beta_i/n_i，优化纵向根预算 beta_i，sum beta_i=B。数值最小预算设为总预算比例 `1e-6/k`，不是原静态实现固定 0.001 下限，因此这些是预算受控的动态参考实现，不能说直接复跑原静态 M-Opt。

`predictive30/predictive100` 借鉴 Predictive DP for Mobility Traces 的“私有测试 + 发布/复用”思想，预测器是上一缓存，阈值为 30/100 个归一化网格点，每步 B/H，测试占 15%。这不是原论文完整复现，未进行阈值调优；未使用未经计费的真实变化检测器。

`fixed_per_version` 每版本用 B，总界 H*B，只作成本参照。`unsafe_shared_noise` 使用第一次误差平移后续真实值，公开差分等于真实差分，没有有效轨迹隐私保证。它可能越域/产生 NaN，仅作为泄漏反例，不能进入公平效用比较。

## 数据、指标与复现

合成轨迹来源是仓库 NHANES 静态数据，绝非观测纵向临床轨迹。三个模板 FIB4/HOMA/ANEMIA，8 类控制情形：无更新、同值新测量、渐变、突变、振荡、稀疏根更新、仅无关根变化、保持比值的相关变化（最后两类仅 ANEMIA）。变化量是公开域宽的 1%/5%，不代表真实医学可行变化。H=4 的振荡在采样点发生混叠，主要用 H=8/16 解读振荡。越域合成值裁剪且记录数量。

真实复测使用 [MIMIC-IV Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/) 开放数据，Hb/Hct/RBC 同标本、同 charttime、单位匹配，排除越出原 NHANES 域的整组数据。一患者选一个符合长度的住院；按 charttime 排序，不模拟 storetime 到达过程。默认数据路径在 `data.MIMIC_DIR`，迁移机器时在 JSON 加 `mimic_dir` 指向解压后含 `labevents.csv.gz` 的目录。均值沿用仓库 holdout 人口均值。

主指标是当前任务 wMAPE 与 RCE（沿用原 `risk_class.py` 阈值）；另记录根误差、类别变化漏报、稳定期误报、以版本数计的截断延迟及账本。漏报定义为真实类别改变但预测没有改变，不等于新类别识别成功率。bootstrap 以患者为单位，噪声重复和版本不当作独立患者；区间为逐项 95%，未校正多重比较。

在仓库根目录运行（`python` 需 NumPy/SciPy；本次用 `/root/miniconda3/envs/rootguard/bin/python`）：

```bash
python -m unittest discover -s experiments/workflow_extensions/root_updates/tests -v
python experiments/workflow_extensions/root_updates/run_experiments.py --config experiments/workflow_extensions/root_updates/configs/exploration.json --out experiments/workflow_extensions/outputs/root_updates/exploration_20261012
python experiments/workflow_extensions/root_updates/run_experiments.py --config experiments/workflow_extensions/root_updates/configs/standard_ldp.json --out experiments/workflow_extensions/outputs/root_updates/standard_ldp_20261012
python experiments/workflow_extensions/root_updates/confirm.py --config experiments/workflow_extensions/root_updates/configs/confirmation.json --out experiments/workflow_extensions/outputs/root_updates/confirmation_20261012
python experiments/workflow_extensions/root_updates/attack_probe.py --out experiments/workflow_extensions/outputs/root_updates/attack_20261012
python experiments/workflow_extensions/root_updates/analyze.py experiments/workflow_extensions/outputs/root_updates/exploration_20261012
python experiments/workflow_extensions/root_updates/analyze.py experiments/workflow_extensions/outputs/root_updates/standard_ldp_20261012
python experiments/workflow_extensions/root_updates/analyze.py experiments/workflow_extensions/outputs/root_updates/confirmation_20261012
python experiments/workflow_extensions/root_updates/build_report.py
```

运行目录必须不存在。重复实验请使用新的目录，并相应调整报告脚本的运行目录参数。生成产物均在忽略目录 `outputs/root_updates/`；代码、参数和 README 受 Git 管理。每个正式运行保留 `config.json`、`metadata.json`、`cells.json`、`patients.csv`、`analysis.json`、`summary.csv`。只有选定案例保存完整缓存轨迹，不声称保存所有运行的全部随机轨迹。
