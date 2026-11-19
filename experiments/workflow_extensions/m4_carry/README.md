# M4-carry：真实有限轨迹机制诊断

独立模块；复用 `finite_cache` 数据 schema、公式、公开归一化域和患者划分函数，不修改旧机制及旧实验输出。六方法：B0、B2-val、B2-pub、M4、M4-carry、M4-carry-replan；没有 M2/M5 附加实验。

## 运行

在仓库根目录，现有 `rootguard` Python 环境具备 numpy / matplotlib：

```bash
cd /home/rootguard-research
PYTHON=/root/miniconda3/envs/rootguard/bin/python
$PYTHON -m unittest experiments.workflow_extensions.m4_carry.test_core -v
$PYTHON -m experiments.workflow_extensions.m4_carry.run --precheck \
  --output experiments/workflow_extensions/outputs/m4_carry/precheck_reproduce
$PYTHON -m experiments.workflow_extensions.m4_carry.run \
  --output experiments/workflow_extensions/outputs/m4_carry/real_v1_reproduce
```

`--config` 默认 `configs/real_v1.json`；`--data` 默认仓库的 `data_mine/finite_cache`。输出目录必须为空；复跑请换新目录。`--precheck` 仅降低随机重复数、bootstrap数，使用80/320/960三个预算，不更改门控与候选常量。

重做独立账本审计、从已完成日志重生成统计/报告/图：

```bash
$PYTHON -m experiments.workflow_extensions.m4_carry.audit --output <已有输出目录>
$PYTHON -m experiments.workflow_extensions.m4_carry.analysis --output <已有输出目录>
```

## 文件职责

- `core.py`：只接收缓存/账本/公开年龄的 `Policy.decide()`；唯一读取待发布根的 `PrivateSource.release()`；统一覆盖缓存/结转账本；隔离的离线评价器。
- `planning.py`：train-only 联合变化方向及公开合成人群、有限候选、代理短名单、fixed/carry各自的MC规划。规划优化的是混合均值风险，不是minimax。
- `run.py`：先完成所有公开τ、计划与validation选择，再处理test误差；保存全路径指标和动作。
- `analysis.py`：患者均权、患者配对及患者＋机制两层 bootstrap、非单调网格扫描和删失、报告/结果图。
- `audit.py`：独立重算已落盘动作的ε支出、根掩码、刷新次数和机会递推，并量化Bmax时M4/carry逐路径差异。
- `plotting.py`：从summary.csv单独重绘带置信带的报告图。
- `test_core.py`：门控、结转、真值隔离、公式、配对噪声及统计口径测试。

## 预先固定的范围

- 四模板为 MELD_NA / ANEMIA / FIB4 / BUN_CR，跨模板同一患者始终属于同一分区。相对全体患者做50/25/25划分，各模板人数无需恰好同一比例。
- 最终节点为优化目标，全40轮平均误差包含初始化及停止后。原始真实测量作为效用真值；输入先按既有公开域裁剪，裁剪率另报。
- ĥ用最终任务导数的公开解析界，保留原型 h_floor=1e-9，所有方法按√ĥ分配；根的公开可见性统一按“截至本轮已知最新值”。
- Δ=[0.005,0.02,0.06]、C_OP=0.5、Cbar90%分位、MC=32等均为config常量。Δ用于等权公开混合规划与分档τ，不用于test分组/选档。
- validation仅选B2-val周期及replan的K/ρ；replan时刻/根集合由carry公开模拟预选。M4/carry始终使用同一fixed公开计划。
- K∈{0,1,2}，ρ∈{0.5,0.7,0.85}；9个公开相对时刻枚举、全根/高h半数根/单根动作；代理每(K,ρ,全根/子集)留1个，MC重排。候选规模有限，未声称全空间最优。
- B2周期以公开测量机会计，不截断刷新；初始化ρ=0.5，余下预算均分实际周期动作数。
- 80..960、1.1粗网格，事先在每个区间加入1.05倍点，避免test驱动细化；所有方法同一网格。
- τ由各公开Δ合成人群B0在960的有限预算参考MAE乘0.85/1.15给出；不是下界。网格未达标记录删失，不删样本、不填960。
- 发布配对键含phase/患者/轨迹/B/重复/实际轮次/根，不含方法；公开校准、公开规划、validation、test的随机流分离。部署噪声不得采用对服务端已公开的确定性流。
- bootstrap固定已选参数及τ，以患者为单位配对重采整条预算曲线；两层在所抽患者内重采机制重复。单预算UCB非全网格同时保证。

## 输出

`config.json`、`split.json`、公开统计/人群/τ、选择及候选评分、输入/代码哈希、`metrics.jsonl`（每患者/轨迹/B/方法/重复）、`actions.jsonl`（含初始化、候选、全否和预算阶梯）、`audit.json`、`independent_audit.json`、`summary.csv`、`paired_mae.csv`、`bmin.csv`、`bmin_paired.csv`、`REPORT-zh.md`及`figures/`。

`bmin.csv`空的B_min表示扫描未达标；bootstrap区间空上界表示包含删失；`bmin_paired.csv`明确双方达标概率，条件比值区间不能脱离该概率解读。

结果仅用于机制诊断：同批demo患者已用于旧轮次分析，且test每模板仅4–5位患者；如出现正面结果仍须新患者/外层验证确认。方法比B2好但等同B0时，应解释为静态选择；carry−M4才是纯结转对照；replan收益含额外validation选择。

原始执行代码哈希见source_hashes.json（启动时记录）；报告/绘图后处理版本见postprocessing_source_hashes.json。报告文字更新不改变已完成的选择或原始机制数据。可用analysis的`--report-only`仅根据已生成CSV更新报告，不重算bootstrap。

## 同周期补测

`periodic_skeleton.py`复用原输出的P、数据划分与统计，不重新选择参数。新carry采用B2-val的相同ρ=0.5、全根候选及精确候选周期，只增加veto/结转；实际发布次数、根集合和支出允许因veto而不同。它不是固定动作包的纯时间实验。

```bash
$PYTHON -m unittest experiments.workflow_extensions.m4_carry.test_periodic_skeleton -v
$PYTHON -m experiments.workflow_extensions.m4_carry.periodic_skeleton --precheck --output experiments/workflow_extensions/outputs/m4_carry/periodic_skeleton_precheck_reproduce
$PYTHON -m experiments.workflow_extensions.m4_carry.periodic_skeleton --output experiments/workflow_extensions/outputs/m4_carry/periodic_skeleton_reproduce
```

`selections.json`中周期计划的K为未展开占位信息；实际候选次数需按P和患者公开机会表实现，使用metrics/schedules中的K。

## 原版M-Opt主对照

`compare_rootguard.py`直接复用`utils/utils.py`原精确分配器，比较原生离散指数核下的静态缓存/周期全根更新，并给出与M4相同Laplace核的分配移植对照。保护域与数据、既有M4配置固定；两种核的周期都只在validation重新选择。所有方法按归一化轨迹metric ε计费，1000网格时`ε_metric=999 ε_grid`。不能把原网格参数与B直接相等。

```bash
$PYTHON -m unittest experiments.workflow_extensions.m4_carry.test_rootguard_mopt -v
$PYTHON -m experiments.workflow_extensions.m4_carry.compare_rootguard --precheck --output experiments/workflow_extensions/outputs/m4_carry/rootguard_mopt_precheck_reproduce
$PYTHON -m experiments.workflow_extensions.m4_carry.compare_rootguard --output experiments/workflow_extensions/outputs/m4_carry/rootguard_mopt_reproduce
```

原分配函数通过AST逐字加载，避免无关的NER/LLM依赖；原生PMF经过与原函数的逐点一致性测试。ANEMIA/FIB4复用原解析DAG梯度；MELD/BUN比值为公式适配。每次原求解器收敛状态记录在`mopt_allocations.jsonl`；独立账本和公开目标参考重求自动运行，不替换原分配输出。

输出包括`REPORT-zh.md`、`METRICS-zh.md`、五方法全路径指标及动作、周期选择/敏感性、`bmin.csv`及`bmin_paired.csv`、分配与数值审计，以及利用既有B0/B2/M4原始结果得到的`component_effects.csv`。组件诊断不是新一轮test调参，也不是各组件的完整Shapley归因。
