# RootGuard 之后：有限流下的「根节点更新保护」阅读地图与创新点草案

> 基准论文：*Dependency-Aware Privacy for Multi-Turn LLM Agents*（RootGuard），arXiv:2605.03188v3，UW-Madison，2026-10-12。
> 代码：`refs/rootguard/`（官方实现，217 个文件，含 RQ1–RQ4 与预生成表格/图）。
> 本文档目的：把「更新/复用」这件事形式化，并给出 4 个可做的创新点 + 实验蓝图 + 审稿人问题清单。

---

## 0. 一句话

RootGuard 的全部保证建立在 **"根一旦被加噪，就永远有效"** 之上；你要做的，是把这句话改成 **"根是一个分段常值（piecewise-constant）的随机过程，隐私成本由状态迁移次数 m 与迁移幅度驱动，而不是由对话轮数 t 驱动"**——于是"何时算更新、更新时花多少预算、刷新时刻本身会不会泄密"成为三个真正的科学问题。

---

## 1. RootGuard 到底做了什么（以及它靠什么成立）

### 1.1 设定

- 用户侧 agent 持有 k 个私有数值根 `X = (x1..xk)`（NHANES 体检指标：身高、体重、血压、血糖、血脂、AST/ALT/血小板…），定义域有界 `D_i`。
- 服务方（可能恶意/被注入/被调取）计算目标 `T = h∘g`，例如 FIB-4、HOMA-IR、BMI、conicity。
- 对手目标：从 t 轮交互中重建 `X`；对手可选择任意派生函数 `f` 反复查询。

### 1.2 机制（三件事）

1. **一次性加噪**：`x̂_i = M_{ε_i}(x_i)`，机制可选 Laplace / bounded-Laplace / staircase / 离散指数机制。
2. **缓存 + 确定性后处理**：所有后续发布都是 `f(x̂)`；重复查询返回同一缓存值。由后处理定理，隐私只取决于初始加噪。
3. **预算分配**：总预算 `B = t·ε`（对手逼出 t 轮，用户"获得" t·ε 的总预算），按 `ε_i ∝ sqrt(|h_i|·D_i)` 分配到根上，其中 `h_i` 是目标对根 i 的敏感度。

### 1.3 四个隐含假设（你的切入点）

| 编号 | 假设 | 一旦打破会怎样 |
|---|---|---|
| **A1 静态性** | 根在一次会话内不变 | 更新后所有派生值仍是旧值 → 效用崩塌（陈旧性） |
| **A2 一次性** | 只需对"初始值"付一次预算 | 更新 = 新的私有输入，需要新预算 → 退化为 `m·ε` 组合 |
| **A3 时间独立** | 相邻状态之间没有可利用的相关性 | 对手可用状态空间先验（Kalman / 变点先验）把两次独立加噪"对齐"，攻击强于论文的 iid MAP |
| **A4 身份完美** | 系统知道"哪个值是哪根""是不是同一个值" | 误判会同时造成**预算浪费**和**可链接性泄漏**（见 F4） |

> 论文的"双重不对称"（轮数越多，RootGuard 越好、独立加噪越差）完全建立在 A1 上。**更新一旦引入，这个不对称会反转**：轮数多往往意味着状态迁移多，而每次迁移对"重新加噪"是成本、对"不刷新"是陈旧。这是你论文里最漂亮的一句 motivation。

---

## 2. 把「更新」写进形式化

### 2.1 状态过程

对每根 i，定义状态序列 `s_i(t)`，`t = 1..T`（**有限流**：T 有界，总预算 B 有界）：

```
s_i(t) = s_i(τ_i(j))     for t ∈ [τ_i(j), τ_i(j+1))
```

- `τ_i(j)`：第 j 个变点（change point）；
- 迁移幅度 `Δ_i(j) = s_i(τ_i(j)) − s_i(τ_i(j−1))`；
- 三个可调的结构参数：**迁移率** `λ = m/T`、**幅度分布**（小漂移 vs 跳变）、**每根是否独立迁移**。

### 2.2 两种隐私单位（这是必须选的第一件事）

| 单位 | 定义 | 代价 | 适用 |
|---|---|---|---|
| **值级（RootGuard 现状）** | 对手学到 `s_i(t)` 的分辨率 ≤ σ | 每次刷新 `ε`，共 `m·ε` | 只要"值"保密 |
| **轨迹级（建议）** | 两条轨迹若总变差为 `d`，则 `e^{ε·d}`-不可区分 | `ε·Σ|Δ|/D`，小漂移几乎免费 | 只要"变化"不可分辨 |

轨迹级的关键洞察：**小幅变化本来就难与"没变"区分，所以它天然便宜**。RootGuard 完全没利用这件事——它对"体重变了 0.2kg"和"体重变了 20kg"收一样的钱。

### 2.3 有限流的三个硬约束

1. **T 有界但未知**：刷新次数 m 是数据依赖的随机变量 → **不能用朴素组合定理**，要用 pay-as-you-go 组合 / privacy filter（Rogers et al., NeurIPS 2016）。
2. **B 固定**：不能像 RootGuard 那样让预算随轮数增长（否则刷新永远免费）。
3. **在线决策**：必须在看到 `u_t` 的当下决定刷不刷，不能回看。

---

## 3. 先想攻击，再想机制：五个失效模式

### F1 陈旧性（benign，但直接毁效用）
更新后仍用旧 `x̂` → FIB-4/HOMA 判级错。度量：`staleness = |s_i(t) − x̂_i|`，直接进 wMAPE / RCE。

### F2 假更新平均（false-refresh averaging）
若检测器把"同一值的再次提及"误判为更新并重新加噪，对手得到同一真值的两份独立噪声 → 方差减半；q 次误判 → `1/√q`。
→ **检测器的假阳性率 p 是一个隐私参数**，不是工程指标。

### F3 刷新时刻本身就是信道（★ 最锐利）
策略 `d_t = 1[|Δ_acc| > θ]` 是数据的函数。服务方即使只看到"派生值跳了一下"或"请求模式变了"，也能推断 `|x_t − x_{t−1}| > θ`。
给定先验 `x_{t−1} ~ N(μ, σ²)`，观测到"刷新"会把后验收窄到壳层 `|x−μ| > θ`，方差下降可解析计算。
→ **刷新策略必须本身是 DP 的**（data-independent schedule，或 noisy/randomized schedule，或 decoy refresh）。

### F4 拆分身份（split-identity）
同一个根被误判为两个不同根 → 两份独立噪声，既浪费预算，又给出"这两个提及指向同一实体"的链接证据（这在 RootGuard 的缓存语义下本来是零成本消灭的）。
→ 需要一个 **DP-aware 的去重/同一性判定**，并给出误拆分概率 p 下的隐私损失上界。

### F5 时间相关先验（attack upgrade）
论文的 MAP 对手用 iid 先验。更新场景下对手应该用**状态空间先验**：`x_t = x_{t−1} + noise`，观测是 `x̂_t`。
后果：对手不需要"平均 q 次"，只需一个 Kalman 平滑，就能把独立加噪的多次发布压到接近单次噪声水平。**这是你相对论文必须要做的攻击升级**，否则审稿人会说你打的是稻草人。

---

## 4. 四个创新点草案（按推荐度排序）

### I1 信任半径刷新（Trust-Radius Refresh, TRR）：让预算随"变化幅度"缩放 ★★★★☆

**一句话**：客户端维护一个"锚点 + 明文累计漂移"，只有当累计漂移超过锚点噪声的一半（`θ = α·σ`）时才重新加噪；刷新次数 `m ≈ TV/θ`，隐私成本 `ε_0 + m·ε_r`，而**绝对误差永远不超过初始 σ**。

**新在哪**：RootGuard 的 ε 与"变没变、变了多少"无关；TRR 把 metric-DP 的距离语义用到"状态迁移"上——小幅变化便宜、大跳变贵。

**要证的命题**：
- (P1) 机制 `M(x_0, Δ_1..Δ_m) = (x̂_0, Δ̂_1..Δ̂_m)` 在轨迹度量 `d = Σ|Δ_j|/D` 下满足 `(ε_0 + ε_r·d)`-DP；
- (P2) 重建误差上界 `σ_0 + O(θ)`，且与 m 无关（这是与"每次重加噪"最本质的区别）；
- (P3) 给定 TV 预算与 T，最优 `θ*` 的闭式解（Laplace 下可解），以及 `θ*` 随 `λ` 的单调性。

**技术难点（要提前想清楚）**：`Δ̂_j` 的噪声尺度若取 `D/(ε|Δ_j|)`，小 `Δ` 会得到巨大噪声，累计误差爆炸。候选解法：噪声上限截断到 `σ_0`（即小于分辨率的更新直接"看不见"），或者用"锚点 + 有界增量"的两层结构。**这个难点本身就是论文的一节**。

**怎么测**：θ 扫描；与"每次重加噪""永不刷新"对比 wMAPE vs m 曲线；看交叉点。

**风险**：审稿人会说"这就是 metric DP 的平凡应用"。→ 必须靠 I2/I3 补足"非平凡性"。

---

### I2 调度隐私（Schedule-DP）：把"何时更新"当成机制的一部分 ★★★★★

**一句话**：证明在 F3 下，**即使值被完美加噪，刷新时刻的可见性仍会泄漏 `|Δ| > θ` 这一谓词**；给出"值-调度"联合 DP 定义，并设计把调度泄漏压到 `ε_s` 的机制（周期化 cooldown / 噪声化阈值 / decoy refresh）。

**新在哪**：这是文献里没人正面处理的一块。w-event DP 假设**事件流是给定的**，从不讨论"事件本身要靠观测来判定"；streaming DP 假设发布时刻固定。你把"检测"抬进威胁模型。

**要证的命题**：
- (P1) 负结果：任何确定性阈值调度 `d_t = 1[|Δ|>θ]` 的调度通道，其泄漏量 `≥ log(1/Pr[|Δ|>θ])` 量级（给出与 θ/σ 的具体表达式）；
- (P2) 修复：`w`-cooldown + 随机化触发（指数机制选刷新时刻）使联合机制满足 `(ε_v, ε_s)`-DP，且额外预算 ≤ `c·ε_v`；
- (P3) decoy refresh（同时刷新若干未变的根）可以把"哪个根变了"的泄漏压到 `log k` 之外的常数。

**怎么测**：构造"只观察刷新模式"的对手，量化它对 `Δ` 的重建误差；对比 deterministic / cooldown / randomized / decoy 四种策略的 privacy-utility 曲线。

**风险**：需要明确威胁模型（服务方能否看到请求时间/派生值跳变）。写清楚就是加分项，写不清就是漏洞。

---

### I3 检测器吃噪声：隐私-检测-效用三难 ★★★☆☆

**一句话**：检测器若读明文，检测准但调度泄漏（I2）；若只读已加噪的 cache，则检测错误率随 ε 上升 → 误刷新（F2）或漏刷新（F1）。三者不可同时最优，给出 Pareto 前沿。

**新在哪**：把"detect update or reuse"从启发式变成框架内的一等公民，并把 F2/F4 的泄漏量化。

**要证的命题**：
- 误刷新率 p 下的隐私损失 `≈ ε·(1 + p·q)`（q 为重复提及次数），以及重建误差的 `√(1+p)` 收缩；
- 给定 (ε, 漂移分布)，存在检测阈值使 staleness + budget 的加权和最小；三者的 Pareto 前沿形状（很可能是"L"形）。

**怎么测**：在合成轨迹上注入受控的"重复提及/改写/单位换算/四舍五入"，测检测 P/R 与最终 wMAPE 的耦合。

---

### I4 有限流最优停止与 (m/t, B) 相图 ★★★★☆（做"图"和"定理"都好吃）

**一句话**：把刷新决策写成有限时域 POMDP（状态 = 锚误差 + 剩余预算 + 距离上次刷新的轮数），在 Laplace + 线性目标下求闭式/DP 解，画出相图：**缓存区 / 刷新区 / 放弃缓存改用树机制区**。

**新在哪**：RootGuard 的结论是"轮数越多越好"；相图把它推广成"轮数、迁移率、预算"的三维权衡，并给出**何时 RootGuard 范式本身失效**的边界——这是最有引用潜力的一张图。

**要证的命题**：
- 最优策略是阈值型（threshold policy），阈值随剩余预算递减；
- 相界 `m*(t, B)` 的解析近似；
- 在 `m/t → 1`（每轮都变）时，最优退化为 w-event / 树机制，RootGuard 严格劣——**这是对原论文的一个"边界条件"结论**，很适合作为 related work 的对话点。

---

### 推荐主线

> **I2 为主（问题新）+ I1 为机制（可落地）+ I4 为图（好讲）+ I3 为消融（防御性）**。
> 一句话故事：*"RootGuard 假设根不变；真实对话里根会变。我们证明'何时刷新'本身是一条隐私信道，并给出幅度感知的刷新机制与有限预算下的最优策略。"*

---

## 5. 实验蓝图（直接映射到 `refs/rootguard/`）

| 组件 | 复用 | 要改 |
|---|---|---|
| 数据 | `data/nhanes_benchmark_200.json`、8 个模板 | 合成纵向轨迹：随机游走（体重 ±0.3kg/周、血糖 ±5）+ 临床事件跳变（+30）+ 缺失 |
| 机制 | `utils.sanitize_dep_aware_v8`、`optimal_budget_allocation_exact`、`blap_*`、`staircase_*` | 把 sensitivity 换成"轨迹敏感度"；加 refresh 循环 |
| 攻击 | `rq2_reconstruction/`（MAP 对手） | 先验从 iid 换成状态空间（Kalman / 变点）+ 调度观察通道 |
| 基线 | — | ①RootGuard-static ②每轮重加噪 ③oracle-change 重加噪 ④w-event/树机制 ⑤独立加噪 |
| 指标 | `wMAPE`、`RCE` | 加：重建误差 vs m、staleness、预算消耗曲线、调度通道对手的误差 |
| 消融 | — | θ、cooldown w、decoy 比例、检测假阳率 p、迁移率 λ |

> 真实纵向数据建议：MIMIC-IV / eICU（重复测量）；NHANES 本身是横截面，只能造轨迹——这点要主动在 limitation 里说。

---

## 6. 审稿人会问的 8 个问题（先写答案）

1. **和 w-event DP 的本质区别？** w-event 保护"事件流"，假设事件已给定；我们保护"事件判定"和"判定时刻"。
2. **为什么 t 越大预算越大？** RootGuard 的建模争议点。你要么沿用（会话级 B），要么直接改成固定 B——固定 B 更干净，也更能凸显"更新要花钱"。
3. **调度可见性假设合理吗？** 需要 threat model 明确列出三条可观察信道：派生值跳变、请求时间、请求内容。
4. **检测器读明文是否违反威胁模型？** 客户端可信（与 RootGuard 相同假设），但要说明"检测结果如何影响发布"。
5. **合成轨迹真实吗？** 用 MIMIC/eICU 兜底 + 敏感性分析。
6. **自适应对手覆盖了吗？** post-processing + pay-as-you-go（privacy filter），注意 m 是随机变量。
7. **只对目标函数有效？** 加"任意派生函数"的鲁棒性实验（沿用 RQ1 的对抗查询设定）。
8. **和"增量/流式 DP 发布"的差异？** 我们是"根级缓存 + 稀疏刷新"，不是"逐时刻发布统计量"；成本模型不同（按幅度而非按时刻）。

---

## 7. 阅读清单（按用途分组）

**必读·流式 DP 的工具箱**
1. Dwork, Naor, Pitassi, Rothblum. *Differential Privacy Under Continual Observation.* STOC 2010.（树机制，`O(log T)`）
2. Chan, Shi, Song. *Private and Continual Release of Statistics.* ICALP 2010 / TISSEC 2011.
3. Kellaris, Papadopoulos, Xiao, Papadias. *Differentially Private Event Sequences over Infinite Streams.* VLDB 2014.（w-event，最直接的对照）
4. Dwork, Naor, Pitassi, Rothblum, Yekhanin. *Pan-Private Streaming Algorithms.* ICS 2010.（入侵模型）
5. Li, Miklau, Hay, McGregor, Rastogi. *The Matrix Mechanism.* VLDB 2015.（跨时刻 workload 最优噪声）

**必读·数据依赖 / 自适应组合**
6. Rogers, Roth, Ullman, Vadhan. *Privacy Odometers and Filters: Pay-as-you-Go Composition.* NeurIPS 2016.（m 随机时的正确组合工具）
7. Whitehouse, Ramdas, Rogers, Wu. *Fully Adaptive Composition.* 2023.（更紧的 filter）
8. Kifer, Machanavajjhala. *Pufferfish.* PODS 2014.（相关数据的隐私框架）
9. 相关数据 DP / dependent DP 的近期工作（检索关键词：`dependent differential privacy correlated data`）——**需核对具体文献**。

**必读·度量 DP 与文本**
10. Chatzikokolakis, Andrés, Bordenabe, Palamidessi. *Broadening the Scope of DP Using Metrics.* PETS 2013.（RootGuard 的 ε-mDP 来源）
11. Feyisetan, Balle, Drake, Diethe. *Calibrated Multivariate Perturbations for Text.* WSDM 2020.

**对照·多轮对话 DP**
12. DP-Rewrite（Igamberdiev et al., 2024）、PrivLM-BLOOM（Utpala et al., 2023）、DP-Prompt（Tang et al., 2024）、CAPE（RootGuard 引作 [9]）、InferDPT（2024）、AdaDPS（2025）。
    → 这些是"逐次加噪"的代表，正是你论文 introduction 里要打的靶子。

**基准**
13. RootGuard, arXiv:2605.03188（+ `refs/rootguard/` 代码）。

---

## 8. 已下载资源

```
default-workspace/
├── refs/rootguard/          # 官方实现（217 文件）：preempt/ 消毒器、utils/ 预算分配、rq1–rq4、data/、预生成 tables+plots
├── skills/paper-lookup/     # K-Dense 科学技能库（54k★）文献检索：24 文件
├── skills/literature-review/# 同上：系统综述工作流（PRISMA、筛选、引用核验）：12 文件
└── skills/dp-research-radar/# 我按你的场景写的定制技能（文献三角 + 创新点五问 + 攻击优先检查表）
```

- `paper-lookup` / `literature-review` 是 Anthropic Agent Skills 格式（`SKILL.md` + frontmatter），可直接被支持该标准的 agent 读取；DSH 若未自动索引，把目录放到其 skills 路径或重启会话即可。
- 官方代码可直接 `pip install -e .`（Python ≤3.12），RQ1–RQ3 纯 CPU 约 10–30 分钟，RQ4 需要 `OPENAI_API_KEY`（约 $2–5）。

---

## 9. 待你确认（会直接改变上面的排序）

1. **"有限流"** 具体指：T 有界？还是预算 B 有界？还是"每次只允许发布有限个值"？
2. 更新是**用户主动改口**（"其实我体重是 72"），还是**时间演化**（复诊新测量），还是两者都有？
3. 你已经有的"检测更新或复用"是**规则/NER**，还是**模型判定**？有没有初步结果？
4. 目标是**会议论文**（NeurIPS/PETS/USENIX）还是**期刊/毕业大论文**？这决定 I2 的负结果要不要做成主贡献。
5. 预算模型：沿用 RootGuard 的 `B = t·ε`，还是改成会话级固定 `B`？
