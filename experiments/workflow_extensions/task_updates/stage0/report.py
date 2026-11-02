"""Short gate decision and detailed evidence; no experiment execution."""
import json
from pathlib import Path


def table(rows,fields):
    lines=['| '+' | '.join(v for _,v in fields)+' |','| '+' | '.join('---' for _ in fields)+' |']
    for r in rows:
        vals=[]
        for k,_ in fields:
            v=r.get(k)
            if isinstance(v,float):v=f'{v:.2f}'
            vals.append('—' if v is None else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def build(out,summaries,thresholds,provenance):
    out=out.resolve();real=[r for r in summaries if r['source']=='observed']
    primary=next(r for r in real if r['cohort']=='real_H8_existing_test')
    near=[r for r in thresholds if r['cohort']==primary['cohort']]
    fields=[('cohort','队列'),('patients','人数'),('events','类别事件'),('event_patients','有事件患者'),('pass_event_gate','达标')]
    concise=f'''# 阶段0：真实风险类别事件审计

**结论：未达到“至少50次类别变化、至少20位有事件患者”的探索性门槛。停止在阶段0，等待用户确认；未运行A/B。**

{table(real,fields)}

现有测试集的33次变化来自 **{primary['event_patients']}位独立患者**，不是33位；前三位患者贡献 {primary['top3_event_share_pct']:.2f}% 的事件。方向为 {primary['directions']}（类别编号上升/下降，不等于临床风险恶化/改善）。

相邻事件的轮次间隔：{primary['event_round_gaps']}。其中连续两轮发生事件的相邻对数为 {primary['adjacent_event_pairs']}。真实发生时刻未知，记录的是首次测量到新类别的时刻。

阈值附近统计（ANEMIA原指标100*Hb/Hct，边界32和36；相对容差为分析条件）：

{table(near,[('scope','范围'),('relative_radius','阈值相对半径'),('near_count','邻近数'),('total','总数'),('near_pct','邻近比例 %')])}

**指标纠正：** 原论文RCE为平均归一化类别距离，CMR为普通类别错误率。此前本方向结果中的“RCE”数值实际是CMR，不能冒充论文RCE。公式校验通过，旧结果文件未改写。

**下一步判断：** 当前数据可用于指标和代码正确性检查，尚不适合启动完整研究价值比较。需要更多独立患者的观测类别事件，不能通过合并H8/H16、重复场景或噪声重复凑门槛。FIB4/HOMA当前只有合成纵向轨迹，不能替代真实事件证据。

详细事件表、每患者计数、阈值距离、时间间隔与后续执行规则见 [完整审计报告]({out/'REPORT.md'})。
'''
    (out/'SUMMARY.md').write_text(concise)
    groups=[]
    for r in real:
        groups.append(f"### {r['cohort']}\n\n方向：`{r['directions']}`；发生轮次：`{r['round_distribution']}`。\n\n有事件患者计数分布：`{r['active_event_counts']}`。\n\n同患者相邻事件的轮次间隔：`{r['event_round_gaps']}`；时间间隔（小时）：`{r['event_hour_gaps']}`。\n\n事件前后测量间隔（小时）：`{r['observation_gap_hours']}`。连续轮事件对：{r['adjacent_event_pairs']}；前三位患者事件占比：{r['top3_event_share_pct']:.2f}%。")
    synthetic=[r for r in summaries if r['source']=='synthetic' and r['cohort'].endswith('_test')]
    doc=f'''# task_updates 阶段0：事件与指标审计完整报告

日期：2026-10-13。仅执行真值事件统计、公式校验与报告，不运行检测、发布、Oracle或A/B；不设计新DP机制。执行时HEAD：`{provenance['git_head']}`。

## 1. 启动门槛与决策

门槛为同一个真实评估队列至少50次风险类别变化，来自至少20位有事件独立患者。它只是探索性资源门槛，不代表统计充分性或临床要求。验证集和测试集分别审计；可用全集仅说明数据资源，不把验证数据并入原测试结论。

{table(real,fields)}

全部真实队列均未达门槛，因此**本轮停在阶段0，不启动完整A/B**。H8全集49次、20位患者虽接近门槛，仍不擅自降标准；H16虽然51次，却仅16位患者，不能只看事件总数。

H8和H16有 {provenance['H8_H16_patient_overlap']} 位重叠患者；其中 {provenance['H8_H16_same_patient_different_initial_times']} 位的选定序列初始时间不同，可能来自不同住院/序列选择。因此不能直接拼接或把两个窗口当独立患者。患者重叠子队列仅作审计，不是额外独立证据。

## 2. 数据和事件定义

只读复用现有MIMIC-IV Demo2.2 CBC筛选：Hb/Hct/RBC同标本同charttime、单位匹配、唯一值、一个患者一个符合长度的住院，排除越出原公开NHANES根域的整组面板。按charttime排序，每时间取唯一标本；日期去标识化，间隔按同患者日期差计算。不是按storetime模拟到达。

主事件为原风险函数 `c(g(X_t)) != c(g(X_(t-1)))`。当前观测模板为ANEMIA，g=100*Hb/Hct，类别0为<32，类别1为32至36（含边界），类别2为>36。这里按类别编号描述方向，不能把类别编号升高等同临床风险恶化。原始连续2%变化保留为辅助计数，不代替类别事件。

事件时间记为首次测量到新类别的第t轮，采用人可读轮次1..T，第一轮没有前态变化事件。实际类别跨越时间只知处于两次测量之间，不能声称整个临床过程只发生这些事件。筛选掉的面板和窗口之外可能存在更多变化，当前数量只针对选定观测轨迹。

FIB4/HOMA及其余原模板没有在本轮构建可用的真实纵向序列，详见template_availability.csv。NHANES原数据为静态样本；其合成轨迹仅作检查，不充当真实事件。不另行搜集新数据或改变域来凑门槛。

## 3. 每队列的事件方向、集中程度与时间

{chr(10).join(groups)}

每位患者的事件数及发生轮次见patients.csv；每个事件的旧/新类别、指标值、跨越阈值、前后测量时间见events.csv。包含无事件患者，不只展示事件患者。轮次间隔与小时间隔不同：相邻轮可能相隔许多小时。

## 4. 阈值附近分布

测试队列全部测量值以最近阈值的相对距离统计；事件前/后值以实际跨越的阈值计算距离。若跨两级，记录两个阈值且取最近距离。1%/2%/5%半径为审计容差，不是新增风险类别或临床噪声模型。

{table(near,[('scope','范围'),('relative_radius','相对半径'),('near_count','数'),('total','总数'),('near_pct','比例 %')])}

panels.csv保留每测量的绝对和相对阈值距离，threshold_proximity.csv包含全部队列。对真实CBC另用十进制比较100*Hb与32/36*Hct核查浮点边界；各队列与原函数不一致数为 `{[(r['cohort'],r['decimal_class_mismatches']) for r in real]}`。未对指标值静默取整或改阈值。

## 5. RCE与CMR公式校验

原论文§5.2公式(4)：

`RCE = 100 * mean(abs(pred_class - true_class)/(C-1))`

普通类别错误率：

`CMR = 100 * mean(pred_class != true_class)`

C=3时，错一级对单样本RCE贡献50%，错两级贡献100%；两者对CMR都贡献100%。例如真值[0,1,2]、预测[1,1,0]，RCE=50%，CMR=66.67%。二分类时两者一致，但多分类不能混用。此处不是临床损害的直接实测指标。

原RQ4 aggregate.py的同名rce_with_se实现为1-accuracy，与论文公式不同。本轮以论文为准，单独模块stage0/metrics.py及手算用例进行核验。此前task_updates/root_updates报告中普通类别不一致率被命名RCE，需解释为CMR；旧报告保持原样，本报告记录纠正，不重跑旧实验。

## 6. 后续事件响应的执行顺序和边界

后续若获准运行A/B，每一轮严格采用：**获得当轮测量 → 判断是否刷新 → 如刷新则更新缓存 → 基于该轮缓存生成回答**。当轮回答首次正确反映当轮事件，延迟为0；下一轮才反映，延迟为1。审计事件轮次不会因缓存索引而平移。

严格事件响应需在事件前正确展示旧类别，并在规定窗口内由旧类别转到真实新类别；初始/事件前错误不剔除，严格响应计失败，另报恢复新状态比例。主窗口一轮，另报告0/2轮。窗口在下个真实事件发生前截止，一个输出变化最多匹配一个事件；错误方向、错误新类别、重复或未匹配报警另列。无真实事件时召回未定义。

未成功事件报告失败/删失率，不只报告成功事件的延迟；不会把失败自动补成一轮，也不会让相邻事件共用一次输出变化。本轮仅记录规则，不运行响应实验。

## 7. 后续A/B公平性约束（记录，未执行）

实验A严格区分两种问题：①固定实际发布次数、初始设置、每次预算的时机诊断；②固定总预算和刷新上限的完整策略比较。实际次数不同的差异不能全归因于时机。配对随机数或事后按次数分组只能辅助，不能独立消除此混杂。

实验B将采用初始缓存（精确/加噪）×后续刷新（精确/加噪）四组，保持相同Oracle时刻、窗口和上限。四组全部是非DP诊断，不称可部署方法。交互作用可能存在，差异不机械相减成相互独立的误差贡献；完全无噪组包括精确初始状态。

部署比较固定总B，并计算检测及所有可观察发布；诊断比较尽量控制发布能力。约束不满足时报告不可行，不用静态策略假装满足任务响应。T=8/16公共计划下的整段数值距离隐私保持不变，既非滚动窗口也非无限期标准轨迹ε-DP。

## 8. 代码变动、检查与产物

只新增task_updates/stage0内的audit.py、metrics.py、report.py、tests及README，不修改既有发布实验。核对 {provenance['existing_files_checked']} 个既有实验源码/说明/配置文件，变更数为0。provenance.json记录数据、旧队列清单、源码及论文提取文本SHA。没有任何A/B或私有发布运行。

输出包括：summary.json、patients.csv、events.csv、event_gaps.csv、panels.csv、threshold_proximity.csv、template_availability.csv、metric_formula_checks.json、real_loader_audit.json、provenance.json、测试日志与简明报告。JSON保留分布与分位数；CSV可逐患者逐事件核查。

合成测试队列事件数仅作辅助检查：

{table(synthetic,[('cohort','合成队列'),('events','类别事件'),('event_patients','有事件患者'),('pass_event_gate','数值门槛')])}

表中的合成门槛通过不能解锁真实实验。相同患者的不同控制场景/窗口也不独立，不累加凑事件数。

**研究价值判断：当前真实观测事件不足以按已认可门槛启动完整A/B。先补真实事件资源，或仅继续代码正确性检查；没有证据支持此刻进入机制设计。事件不足也不是证明方向本身无价值。等待用户确认下一步。**
'''
    (out/'REPORT.md').write_text(doc)
    print(out/'SUMMARY.md')
