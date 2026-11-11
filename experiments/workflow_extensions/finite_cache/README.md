# 有限轨迹 DP 缓存实验

依据 `/home/方法整体描述.md`（2026-11-10 版本）和本地实现计划。主线为标准归一化 Laplace 重锚、固定会话总预算、经验损失门控；不是增量发布，也不启用历史反馈、私有检测器或 zCDP 账本。

## 运行

从仓库根运行。已有 Python 3.12 环境足够，不需要启动 NER、LLM 或 API。

```bash
FC_PY=/root/miniconda3/envs/rootguard/bin/python

# 输出到独立且已被忽略的 data_mine/finite_cache/，不覆盖四份精简 JSON。
$FC_PY -B -m data_mine.build_datasets --format finite
$FC_PY -B -m data_mine.validate_datasets --format finite \
  --input data_mine/finite_cache --report data_mine/finite_cache/validation.json

$FC_PY -B -m unittest discover -s experiments/workflow_extensions/finite_cache/tests -v

# 工程预检查：R=2、三档预算，不能据此形成效用结论。
$FC_PY -B -m experiments.workflow_extensions.finite_cache.run \
  --config experiments/workflow_extensions/finite_cache/configs/precheck.json \
  --output experiments/workflow_extensions/outputs/finite_cache/precheck_new

# 方案主线 pilot：R=20、预算比值1.1、验证集阈值区间细化到1.05。
$FC_PY -B -m experiments.workflow_extensions.finite_cache.run \
  --config experiments/workflow_extensions/finite_cache/configs/pilot.json \
  --output experiments/workflow_extensions/outputs/finite_cache/pilot_new
```

已存在的运行目录不会被覆盖。数据重建需显式 `--overwrite`；原精简对话文件仍保留。仅补建 MELD 的入口为 `python -m data_mine.build_meld_na --output <新目录>`。

## 模块与流程

- `data.py`：完整轨迹、按患者整体划分、公开当前批次对象；只有离线评估器能回放全部真值。
- `workload.py`：ANEMIA/FIB4/NLR/TYG/MELD-Na 连续公式、公开域、导数界、名义 Jacobian；不对类别函数求导。
- `public_stats.py`：只使用 train 的均值、查询频率、测量频率、漂移分位数及 M。
- `mechanisms.py`：公共分配器、纯 DP 预算组合、覆盖式根缓存；`decide(PublicBatch)` 不接收真值或未来事件；`publish` 单独接收当前根。
- `run.py`：验证集扫描 P/q → 按目标 tau 选参 → 冻结选择 → 相同测试集与预算网格；没有测试集选参。
- `analyze.py`：患者均权聚合、两层 bootstrap、均值/UCB 前沿、验证集选择稳定性与图表。

首轮方法为 B0、B2、B2p（文档 B2′）、M1-a、M1-g。B2p 与 M1 共用配额加权分配及因果 h；B2 保持纯 sqrt(h) 分配，所以 B2p−B2 是预算/分配联合差异，不能称纯账本收益。B2、B2p 的周期分别在 validation 选择；M1-g 复用 M1-a 的 q，以隔离年龄门控。

## 数值口径与明确的实现选择

1. `d_X=max_t max_i |z_i-z'_i|`，所有根和所有发布时间的 epsilon 相加，包括初始化。只保护固定公开到达/测量集合/工作负载条件下的根轨迹；到达信息本身不受保护。
2. `N_max=T-1`：公开有限对话长度、每轮至多一批；候选预算不读取未来实际批次数。`eps_can=min(c*remaining,eps_max)`；B0 全预算初始化，动态默认半预算初始化。
3. `C=0` 对应 `G=∞`；空 S 跳过；h=0 根使用公开固定值，不发布；耗尽的软配额仍通过 q_min 保留正权重。只有全局预算是硬约束。
4. 新真实纵向数据不使用原 NHANES 的窄域。`workload.DOMAINS` 是明确的固定基准域，不是根据测试集计算的经验上下界；可在运行前通过配置 `domains` 覆盖。输入截断数量及任务差异单列。
5. 原始 Laplace 发布保留在归一化根坐标；所有任务对缓存作同一 `[0,1]` 投影再反归一化，保证比值/平方根/对数可计算。代理噪声项仍为 `1/eps_last`，不累积历史噪声。投影对效用的影响不能忽略。
6. 中间节点单位不同，默认先除以该节点公开域上的数值范围，再按 train 查询频率加权。这等价于选择公开尺度后的 T_j；`public_stats.json` 保存尺度、权重和 Abar。可通过配置 `workload_scales` 在运行前固定其他尺度。
7. 单项式的坐标导数界精确求上界；MELD 的复合 clamp 使用保守解析链式上界。M 只用于归一化误差的名义二次评价，不进入触发器。
8. MELD 数值函数固定为旧 pilot 的连续四根版本：三项 log 加 6.43，再作钠修正和最终 `[6,40]` clamp。**不是临床完整 MELD 协议**：未加透析、四舍五入或其他临床条件；不捏造 MELD 类别阈值。公式版本写入 metadata。
9. 真实值用于效用评估，主 MAE 是最终节点在全 T 轮对原始最新已知状态的误差，含输入轮和冻结后轮；另报截断后输入真值误差及实际查询节点的尺度化误差。C 与 L_keep/L_refresh 都是经验代理量，没有高概率效用保证。
10. 当前固定目标 tau 仅为数值基准，不是临床可接受阈值。配置 `targets={}` 时，可从验证集 B2 曲线生成目标（参考预算由配置预先指定），在 test 前保存。未达标只报网格内未达标，不赋真正无穷大。

## 输出与不确定性

所有数据、结果和日志保留本地，写入已被 `.gitignore` 覆盖的目录。

| 文件 | 内容 |
|---|---|
| `splits.json` | 所有模板共用的患者划分 |
| `public_stats.json` | train-only 统计、尺度、M、C、根域 |
| `config.json`、`metadata.json` | 冻结配置、输入/源码/方案哈希、隐私单位与公式版本 |
| `metrics.jsonl` | 每患者、方法、预算、机制重复的全轨迹指标 |
| `ledger.jsonl` | 当前批次决定、处理前/后状态、逐轮快照与实际 DP 发布；本地评估日志不属于公开 transcript |
| `selections.json`、`targets.json` | test 执行前固定的选择与目标 |
| `selection_stability.json` | 仅 validation 的配对 bootstrap P/q 选择频率 |
| `summary.json` | 患者 CI、机制 CI、两层 CI、方差分量、配对 bootstrap 均值 |
| `frontiers.csv` | 均值/UCB 各自的 B_min 和 Saving Ratio；均值口径另报 bootstrap 前沿的不确定性 |

UCB 使用单侧 95% 两层 bootstrap 分位数。均值口径前沿区间以“该 bootstrap 重采样达标”为条件，**必须同时看达标比例**；两方法节省区间同样以双方达标为条件。未达标重采样不会伪装成有限预算。UCB 本身的前沿不另套未经实现的双重 bootstrap。

`aoi_mean` 使用对话轮距离（当前轮减去每根上次发布轮），门控的 `ages` 则使用全局批次数；两者分开。`p95_abs_error` 是每条机制轨迹的绝对误差95分位数，summary 对患者均权聚合该量，不将其误称为全队列合并后的95分位数。

配对模拟流按 `(seed,subject,trajectory,B,repeat,turn,root)` 对齐，方法间不共享动作。记录种子是为了可信本地复现；它不是向对手公布噪声种子的部署协议。部署时需独立保密的随机性。

`frac_age` 把 both 计入年龄相关刷新，但只是行为统计。因果归因以 M1-a/M1-g 的对照差异为主。MissRate 的显著事件为活动根在单批次归一化变化超过配置 `significant_delta`；无显著事件的分母返回 null。

独立重算与可选普通科研图：

```bash
$FC_PY -B -m experiments.workflow_extensions.finite_cache.analyze \
  experiments/workflow_extensions/outputs/finite_cache/pilot_new --bootstrap 500 --plots
```

数据为真实检验值、合成查询日程；根状态是最新已知测量，不宣称同一时刻完整临床评分。NLR/TYG 当前仅作辅助，其小样本不足时 runner 会拒绝独立三划分主实验。此实现只用于 pilot，不宣称已证明门控优于周期或已经达到论文投稿要求。

## MELD 首轮固定协议

`configs/round1_meld.json` 单独固定：MELD_NA、五个主方法、B=80–960（比值1.1）、R=20、显著阈值严格>0.05。三个目标由validation B2在最接近320/640/960格点的曲线生成；本轮不细化预算。

```bash
$FC_PY -B -m experiments.workflow_extensions.finite_cache.run \
  --config experiments/workflow_extensions/finite_cache/configs/round1_meld.json \
  --output experiments/workflow_extensions/outputs/finite_cache/round1_new
$FC_PY -B -m experiments.workflow_extensions.finite_cache.report \
  experiments/workflow_extensions/outputs/finite_cache/round1_new
```

批量重复仅适用于当前不依赖噪声的公开调度，已同逐次参考实现核对全部指标与发布值。全部20次重复保留指标，逐轮完整账本默认保存repeat=0；公开调度/收费在所有重复中相同，噪声流保持独立。

严格显著阈值比较采用1e-12容差排除十进制相等值的浮点误计。首轮历史结果的边界修订仅重算事件/Miss计数，原始指标与修订记录保留在运行目录；未重采噪声或重新选参。当前源码从头运行即得到修正语义，无需再次调用`rescore_significance`。

## 更新时间配对诊断

`matched_timing` 只读已有首轮的划分、公开统计、公式、预算与 validation 选择，不重拟合。
M1 每对先运行；离线 `matched_periodic` 在合法机会中寻找与 K 个等间隔对话轮目标最接近的 K 个不同时间，
复用 M1 的初始化和按序实际 epsilon 总额。共用 `Cache.allocate/publish`，只发布当时测量的活动根。
逐根预算及根集合差异全部记录，不保证逐根费用相等。
噪声按 `(seed,subject,trajectory,B,repeat,action_index,root)` 配对，不含 turn。

```bash
$FC_PY -B -m experiments.workflow_extensions.finite_cache.matched_timing \
  --source experiments/workflow_extensions/outputs/finite_cache/round1_meld_20261110 \
  --output experiments/workflow_extensions/outputs/finite_cache/matched_timing_precheck_new \
  --repeats 2
$FC_PY -B -m experiments.workflow_extensions.finite_cache.matched_timing \
  --source experiments/workflow_extensions/outputs/finite_cache/round1_meld_20261110 \
  --output experiments/workflow_extensions/outputs/finite_cache/matched_timing_formal_new \
  --precheck experiments/workflow_extensions/outputs/finite_cache/matched_timing_precheck_new \
  --repeats 50
```

R=2 会先运行全部单元测试；正式运行要求预检查成功且源码、冻结来源、预指定分析计划一致。
不允许覆盖输出目录。`configs/matched_timing.json` 预先固定六个前沿附近预算和2000次配对 bootstrap，
主检验为六预算患者内均值。患者均权，另保留患者/机制两层区间、机制单独区间及方差分量。
全网格结果也保存，逐预算 CI 是点态诊断。

输出包含 `matched_metrics.jsonl`、`matched_schedules.jsonl`、`paired_summary.csv`、`REPORT-zh.md`、
逐患者差值、刷新时间表、根覆盖差异、预算停止时间及审计。该对照使用 M1 最终 K/费用与完整公开机会表，
属于离线诊断；复用的 test 不能代替新划分确认。单任务 MELD 的中间节点不能充当多个独立任务。
