---
name: dp-research-radar
description: For differential-privacy / privacy-preserving dialogue research. Turns a vague scenario into a formal privacy statement, runs a three-track literature sweep (foundation, adjacent, application), attacks the design before designing the mechanism, and stress-tests each candidate novelty claim with five questions. Use when the user asks for related work, novelty checks, or "is this idea new enough".
license: MIT
metadata:
  version: "0.1"
  domain: differential-privacy, privacy-preserving-dialogue
---

# DP Research Radar

A repeatable protocol for "我有个场景，帮我找文献 + 想创新点" in differential privacy.

## When to use

- 用户给出一个隐私场景，要求 related work、novelty check、或"这个点够不够新"。
- 用户想在已有机制（如 RootGuard、DP-Rewrite、CAPE、PrivLM-BLOOM）上做扩展。
- 需要把一个模糊的"保护 X"变成可证明的 `(ε, δ)` 陈述。

## Step 1 — 形式化（先写这六行，写不出来就说明场景没想清）

```
隐私单元 U = ?            # 一条记录 / 一次发布 / 一个用户的整条轨迹 / 一个状态迁移
邻接关系 ~ = ?            # 两条数据集差什么（一条记录？一次更新？一个时刻？）
距离 d(·,·) = ?           # 离散度量 / L1 / 轨迹总变差 / 加权度量
机制 M = ?                # 加噪对象、噪声族、预算分配
对手视图 V = ?            # 对手能看到什么（值 / 时刻 / 请求模式 / 派生值跳变）
保证 = ?                  # (ε,δ)-DP / ε-mDP / w-event / pan-privacy / 调度隐私
```

**红旗**：如果"隐私单元"是"一次发布"，而发布次数是数据依赖的（比如"检测到变化才发布"），普通组合定理就**不成立**，必须用 pay-as-you-go 组合 / privacy filter（Rogers-Roth-Ullman-Vadhan 2016）。

## Step 2 — 三线检索（每条线都要有，缺一条就会被审稿人打）

| 线 | 目的 | 查询模板 |
|---|---|---|
| **Foundations** | 找到可复用的定理工具 | `"differential privacy" AND ("continual observation" OR "streaming" OR "w-event" OR "pan-privacy" OR "matrix mechanism")` |
| **Adjacent** | 找到"换了个壳的同一问题" | `"privacy" AND ("state change" OR "update" OR "drift" OR "temporal correlation" OR "adaptive composition")` |
| **Application** | 找到同场景的 baseline | `("multi-turn" OR "dialogue" OR "agent") AND ("differential privacy" OR "sanitization")` |

执行方式（任选其一）：
- `skills/paper-lookup/` — 多库检索（arXiv / PubMed / Semantic Scholar / OpenAlex）。
- `skills/literature-review/` — 需要 PRISMA 式可复现综述时用。
- arXiv API：`http://export.arxiv.org/api/query?search_query=all:%22continual+observation%22&sortBy=submittedDate&max_results=20`
- Semantic Scholar：`https://api.semanticscholar.org/graph/v1/paper/search?query=...&fields=title,year,abstract,citationCount`

**判定"新"的三条硬标准**（缺一不可）：
1. 邻接关系或隐私单元与已有工作**不同**（不是换个数据集）；
2. 存在**已有机制做不到**的负面结果或下界；
3. 新机制在**同一威胁模型**下严格优于至少一个强 baseline。

## Step 3 — 先攻击，再设计（固定顺序，不许跳）

对候选设计逐条问：

- [ ] 对手能否通过**发布时刻 / 发布模式 / 跳变**反推私有量？（调度信道）
- [ ] 两次发布是否**相关**（同一真值、相邻状态）？能否被平均 / 平滑 / 对齐？
- [ ] 检测/分流步骤读的是**明文还是加噪值**？读明文则调度泄密，读噪声则检测退化——二选一必须写清。
- [ ] 发布次数是否**数据依赖**？组合定理是否仍然有效？
- [ ] 误判（假阳/假阴）的隐私代价是多少？（把它写成一个参数 p，而不是当成工程噪声）
- [ ] 自适应对手（根据历史发布选下一次查询）覆盖了吗？

## Step 4 — 创新点五问（每个候选点都要答）

1. **一句话**：这个点用一句不含术语的话怎么讲？
2. **新在哪**：和最近的 3 篇相比，隐私单元/邻接关系/威胁模型/机制哪一项变了？
3. **要证的命题**：至少一条定理或一条下界（"我们观察到…"不算）。
4. **怎么测**：哪个 baseline、哪个指标、哪张图能一眼看出差异？
5. **怎么死**：最可能被审稿人一句话否掉的点是什么？提前写 rebuttal。

## Step 5 — 输出（固定格式，便于后续迭代）

```markdown
## 场景形式化（六行）
## 文献地图（三线 × 每线 3–5 篇 × 一句话差异）
## 攻击面（按严重度排序）
## 创新点（每个：一句话 / 新在哪 / 要证的命题 / 怎么测 / 风险）
## 推荐主线（主贡献 + 支撑贡献 + 图）
## 待确认问题（不超过 5 个，且每个都会改变结论）
```

## 反向检查（交付前）

- 每条引用都能点开、标题与作者对得上（不确定的标"需核对"，不要编）。
- 每个"新"都有对应的**已有工作**被明确点名对比。
- 每个机制都有对应的**攻击**在同一威胁模型下评估。
- 图表能在没有正文的情况下讲清故事。
