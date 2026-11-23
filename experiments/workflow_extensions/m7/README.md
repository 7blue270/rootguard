# RootGuard M7｜会话前选侧与固定周期发布

```bash
cd /home/rootguard-research
PYTHON=/root/miniconda3/envs/rootguard/bin/python
OPENBLAS_NUM_THREADS=1 "$PYTHON" -m experiments.workflow_extensions.m7.audit
OPENBLAS_NUM_THREADS=1 "$PYTHON" -m experiments.workflow_extensions.m7 \
  --output experiments/workflow_extensions/outputs/m7/reproduce \
  --legacy experiments/workflow_extensions/outputs/m6/real_v2_20261117
OPENBLAS_NUM_THREADS=1 "$PYTHON" -m experiments.workflow_extensions.m7.matched_gate \
  --source experiments/workflow_extensions/outputs/m7/reproduce \
  --output experiments/workflow_extensions/outputs/m7/matched_reproduce
```

`m7` 默认用四个冻结模板、预算 80/318.99/651.22/960、20 次评估噪声和 1,000 次双层配对重采样。默认便携配置对两侧用相同的原生指数核和边际缓存校准。可选 `--profile historical` 复用此前选出的 ANEMIA 梯度混合及联合校准、BUN/Cr 的初始预算和联合校准。跨模板拟合时采用统一配置。历史 M5/M6 比较要求使用匹配的输入哈希、患者划分、预算和噪声流。

## 方法入口

- `engine.Context` 用目标模板历史参考患者拟合任务公式的经验先验。原版 M-Opt 用原生 1000 格指数发布核。校准用该核的完整似然，以当前已发布缓存与估计值做凸组合。主配置收缩系数为 0.85。
- `routing.fit` 在参考轨迹上选动态周期，用另一组目标模板历史校准轨迹配对回放两侧，返回逐预算的预测特征。`routing.source_scores` 在独立的历史划分中拟合规则。`routing.select_rule` 按其他模板的损失选规则，`routing.make_route` 冻结两侧配置及选择。
- `engine.run_live` 接收已冻结的 `Route`、公开机会表与公开查询计划。只运行会话前选出的分支。调用方提供可信的 `read_roots(turn, names)`，返回该轮可用的授权根值。发布噪声来自系统随机源，实验配对种子只用于离线回放。`Route.from_frozen` 可从输出的冻结规则加载选侧结果。
- `M7` 动态门控用半数权重衡量按历史量级标准化的全窗口最终节点误差，半数给该窗口实际查询的标准化节点误差。没有查询时使用最终节点目标。它用校准后的缓存评估复用和模拟发布，预测只读训练轨迹与既有带噪发布。
- `M7-final` 用上一轮的最终节点、原始缓存门控与选侧目标，保留相同的校准输出。`M7-D-final` 强制走该动态分支。相同周期的消融额外排除两套目标选择不同周期的影响。

## 对照与统计

- `RG-static-exp` 用全 B 按原版 M-Opt 一次加噪后固定缓存。
- `RG-periodic-exp` 首轮 B/2，后续按验证搜索选出的固定周期全根更新。每个机会以原版 M-Opt 分配该次预算。`RG-periodic-cal` 用相同缓存校准，周期同样独立择优。
- `M7-S`、`M7-D` 是两条分支单用对照，`M7` 是会话前选侧。M5、M6 可从已冻结的同患者同随机流历史行加入。
- 全程最终节点 MAE 是全 T 轮等权误差。最终 wMAPE 用平均 MAE 除以平均绝对真值。RCE 用原版有序类别距离归一化，MELD-Na 和 BUN/Cr 没有该类别标签，显示为空。中间节点只取实际查询该节点的轮次，先每节点做比值，再用训练分区查询频率加权汇总。BUN/Cr 没有中间节点。
- MAP 攻击沿用 RootGuard 的分根网格模型，含均匀快照、基于训练先验的快照和带历史的 MAP。缓存重复应答没有额外独立观测。主配置的边际校准用公开参数从观测的校准值反求原发布格点，实验核对逐根反演。历史配置的联合校准采用可见原始发布的更强对手模型，标签单列。对手模型中的根值重构误差越低、成功率越高，意味着更容易重构。
- 输入、分流、逐路径效用和 MAP、摘要、配对置信区间及预算审计分别保存。区间条件于已冻结机制和历史患者划分，采用患者与患者内噪声配对重采样，未做多重比较校正。

## 边界

保护对象是固定公开测量与查询日程条件下的单用户整段归一化轨迹。未知或敏感日程、训练样本本身、真实对话跨用户与跨公式转移需要另行验证。离线门控用经验训练场景，不是当前会话的真值或完整最优控制。实际发布应用还需给出有限精度随机采样和服务接口的独立验证。现有缓存校准可以降低任务误差，按转录后处理没有额外隐私预算，边际校准也不增加该观察模型下的重构难度。

## AutoResearch 动态归因与原 iso-utility 目标

```bash
BASE=experiments/workflow_extensions/outputs/m7
OPENBLAS_NUM_THREADS=1 "$PYTHON" -m experiments.workflow_extensions.m7.refresh_study \
  --stage development --output "$BASE/refresh_dev_reproduce"
OPENBLAS_NUM_THREADS=1 "$PYTHON" -m experiments.workflow_extensions.m7.refresh_study \
  --stage refine-bun --dev "$BASE/refresh_dev_reproduce" --output "$BASE/refresh_bun_reproduce"
OPENBLAS_NUM_THREADS=1 "$PYTHON" -m experiments.workflow_extensions.m7.refresh_study \
  --stage confirm --dev "$BASE/refresh_bun_reproduce" --output "$BASE/refresh_iso_reproduce"
```

历史融合只使用已发布的根。新释放完成现有边际校准后，将新估计和历史服务估计按公开的相对不确定度做凸组合。新释放的噪声方差代理为 `min(0.25, 8/epsilon^2)`，旧估计方差增加在训练轨迹上估计的相应时间间隔根二阶变化。它是效用近似，误差可能有相关性。周期门控用同一融合公式评估候选，跳过动作不读取当前根。融合系数由公开训练统计、已见动作和预算决定，只要可见数值精度足够，MAP 诊断先反求最新校准值，再反演该原生格点。预算结论与攻击诊断分开解释。

开发轮在原四预算锚点比较动态家族，主评分为 75% 相对校准强周期的最终 MAE 比和 25% 的查询目标比。候选需实际在至少四分之一的开发路径刷新。原第一次开发的 BUN/Cr 候选全部落后，追加少数较快遗忘或调整初始份额的候选，只在开发数据上再选一次。宽网格回放的候选来自冻结选择，不在测试数据上重调。

主对照包括相同校准的静态、强周期、只增加融合的强周期、候选去除融合、候选改为强制全根更新，以及同样用开发患者选侧的静态/周期对照。等预算主曲线沿用原 54 点网格和旧的 24 个公开阈值。B_min 取第一个实际达到误差阈值的网格点，容许曲线非单调。未达到报告删失，并给出点态上界及配对重采样的达标概率。部分格在最低预算即已达标，不能进一步分辨谁需要更少预算。MAP 与中间查询等次指标在四个预算锚点计算。
