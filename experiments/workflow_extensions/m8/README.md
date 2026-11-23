# M8 有限轨迹隐私发布

默认策略为上一轮冻结的 PhysioNet BUN/Cr 版本。本目录包含原算法拟合与分析入口、冻结计划、训练引用清单、精确采样部署入口，以及本轮高预算漏刷实验。原研究的四档平均与本轮按预算的比较分别见[原始实验](original_results.md)和[本轮实验及局限](RESULTS.md)。

## 当前定版与阅读入口

以 `load_original()` 加载的 `original_bun.json` 中的 `route` 为原 M8。该配置对应 PhysioNet BUN/Cr 第二轮冻结的自动选侧策略。实验表中的 `dynamic` 为每次会话都采用动态分支的对照。后续研究脚本的 `m8` 列指风险加权候选，`m8_original` 列才对应原冻结路由。

| 有限会话预算锚点 | 公开测量批次 < 8 | 公开测量批次 ≥ 8 |
|---|---|---|
| 80、318.99 | 全额首发，复用缓存 | 全额首发，复用缓存 |
| 651.22 | 全额首发，复用缓存 | 首轮 40%，余下 60% 在中点附近购买一次双根刷新，与旧缓存融合 |
| 960 | 全额首发，复用缓存 | 首轮 50%，余下 50% 在中点附近购买一次双根刷新，采用最新校准值 |

非锚点预算按对数距离取最近配置，实际花费按传入的 B 计算。首发使用 M-Opt 根预算；两条启用的刷新计划均在刷新时混合 M-Opt 和训练梯度权重。会话内选侧只用预算和公开批次数。某一患者事后的高波动标签只用于评估。

- [六组消融与收益归因](ATTRIBUTION.md) 给出校准、刷新计划、融合与自动选侧的配对贡献。
- [最终审核记录](FROZEN_AUDIT.md) 给出配置指纹、回放与部署的一致性及验证边界。
- [原 M8 实验结果](original_results.md) 给出数据划分、对照、四档平均、消融、隐私评估和最低预算。
- [后续优化与迁移实验](RESULTS.md) 给出漏刷修补、旧模板和新增三批次患者的结果。

## 运行机制

1. 会话开始前获得公开的预算 B、会话长度、测量机会及查询时钟。离线训练按公开机会数是否达到八次分组。每组在四个预算锚点学习刷新次数、首轮预算比例、根包、包内分配与缓存融合。静态使用全预算首发。动态将余款分摊到实际预约时点，最近机会重合时合并。决策不接收会话当前真值。
2. 初始和各预约时刻只由可信读取器提供获准根的最新测量。每根在公开域归一化、加噪、边际校准后写入缓存。可选的时序融合按公开参考轨迹的漂移和释放精度对旧缓存及新测量加权。发布之间继续从缓存回答。归一化根缓存可传给现有 Workload.from_cache 生成查询值。
3. 原 BUN/Cr 版本在两个较高预算且测量密度高时刷新一次，其余组合静态。补充实验的风险加权路由和最高预算强制刷新均保留为研究候选。默认冻结配置可在 [original_bun.json](original_bun.json) 审阅。

```python
from experiments.workflow_extensions.m8 import load_original, run

ctx, plans = load_original("data_mine/finite_cache_physionet2019")
# public 包含 T、events 与 queries；read_roots(turn, names) 只返回指定根的当时值。
result = run(ctx, plans, 960, public, read_roots)
task_values = ctx.work.from_cache(result["cache"])
```

从仓库根目录用 rootguard 环境执行。study 写出各模板开发与配对回放结果。transfer 使用本地 training.zip 构建新留出队列，读取 study 的冻结计划。analysis 汇总这两个阶段。iso 核对旧队列原预算网格。audit 单独运行时需先建立研究结果目录。

```bash
python -m experiments.workflow_extensions.m8 study
python -m experiments.workflow_extensions.m8 transfer
python -m experiments.workflow_extensions.m8 analysis
python -m experiments.workflow_extensions.m8 iso
python -m experiments.workflow_extensions.m8 audit
```

定版复现实验默认启用第二轮融合候选集；显式设置 `RG_PHYSIONET_FUSION=0` 可重现第一轮消融。原版方案搜索和汇总可分别用 `RG_PHYSIONET_FUSION=1 python -m experiments.workflow_extensions.m8.original_study` 和同环境变量下的 `python -m experiments.workflow_extensions.m8.original_analysis` 重放。复用了 M7 的训练先验、分配器、任务公式、缓存校准、研究回放及重构评估。正式部署入口调用本目录的精确核；实验回放使用同一理论核的浮点实现以保持共同噪声配对。运行研究命令会更新同名实验输出。

## 隐私保证的对象与条件

固定公开信息 P，含长度、时钟、机会、查询、根域、训练参考、计划及 B。两个单用户轨迹的距离为所有轮、所有根归一化差值的最大值。对手见动作、时刻、花费和所有服务缓存。训练参考个体及测量时钟本身不在这一条件保证内。只保护本次会话，同一人的多次会话需要另行组合。

每个选择的根使用有界 1000 格指数核，概率正比于 exp(−ε|x−j/999|/2)。两输入未归一化对数权重差至多 ε|x−x′|/2，对数归一化常数差也至多该值，因此单核似然比至多 exp(ε|x−x′|)。同用户同时更新多根的 ε 相加。单次会话任一动作前缀的精确费用之和不超过 B，所以全程满足纯 B-metric DP，δ=0。缓存校准、时间融合、重复查询、模板函数是已发布值的后处理。

旧部署路径将 52 位均匀数映射至浮点 CDF。在 ε=160，输出格点零在 x=1 的理论概率约为 1.39×10⁻³⁶，低于实现能产生的最小均匀数 2⁻⁵³；同格点在 x=0 处概率约 0.077。两种输入具有不同可达输出，因此旧路径的机器实现不满足该纯 DP 声明。只核查预算不能发现这个问题。

M8 的 [policy.py](policy.py) 用 Fraction 表示输入、域和经归一化的逐根费用。均匀提出格点，用私有随机整数实现概率 exp(−x) 的精确伯努利接受。对 0≤x≤1，连续第 k 次成功的条件概率为 x/k，因此前 k 次均成功的概率为 xᵏ/k!；成功次数为偶数的概率由交错级数得到 exp(−x)。更大的 x 分块独立相乘。拒绝后重新提出，得到目标离散分布，全部格点有正概率。付费采样结束后的浮点处理仅接触释放值与公开信息。精确归一化针对传入的数值及给定公开域。任意有限合法预算在理论上几乎必然终止，运行时间随参数变化。

这项保证假定可信客户端的私有随机位、有效输入和对手仅观察上述 transcript。拒绝采样循环次数、运行延迟、内存访问、异常及操作系统侧信道均超出证明，若对手能精确观测响应耗时，当前实现没有端到端时序隐私保证。系统随机源在理想随机位模型下分析。本轮验证了读回调、精确预算、机会时刻、两个参数点的采样均值；有限抽样用于回归核查，数学保证来自核、组合和观察边界。
