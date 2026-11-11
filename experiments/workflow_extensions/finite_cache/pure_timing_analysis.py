"""Paired inference and Chinese report for fixed-action timing replays."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .analyze import csv_rows, write_json
from .matched_analysis import summarize_cell
from .pure_timing import M1_TIME, UNIFORM_TIME


def analyze(rows, output, protocol, coverage):
    output = Path(output)
    names = {M1_TIME: 'M1-a', UNIFORM_TIME: 'matched_periodic'}
    adapted = [dict(r, method=names[r['method']]) for r in rows]
    excluded = {'subject_id', 'trajectory_id', 'template', 'B', 'repeat', 'method', 'parameter'}
    metrics = [k for k in rows[0] if k not in excluded] if rows else []
    groups, by_trajectory = defaultdict(list), defaultdict(set)
    for r in adapted:
        groups[(r['parameter'], r['B'])].append(r)
        by_trajectory[(r['trajectory_id'], r['parameter'])].add(r['B'])
    complete = {key for key, bs in by_trajectory.items() if bs == set(protocol['primary_budgets'])}
    primary = [r for r in adapted if (r['trajectory_id'], r['parameter']) in complete]
    result = []
    for parameter in sorted({r['parameter'] for r in primary}):
        rr = [r for r in primary if r['parameter'] == parameter]
        result += summarize_cell(rr, metrics, 'primary_six_budget_complete_trajectory', None,
                                 protocol['bootstrap'], protocol['seed'])
    for (parameter, B), rr in sorted(groups.items()):
        result += summarize_cell(rr, metrics, 'strict_matched_budget', B, protocol['bootstrap'], protocol['seed'])
    for r in result:
        r['uniform_mean'] = r.pop('periodic_mean')
        if r.get('conclusion') == 'supports_better_M1_timing_single_task':
            r['conclusion'] = 'supports_pure_timing_advantage'
    if result:
        csv_rows(output / 'paired_summary.csv', result)
    else:
        (output / 'paired_summary.csv').write_text('scope,B,metric,n_patients,m1_mean,uniform_mean,delta,conclusion\n')
    write_json(output / 'paired_primary.json', [r for r in result if r['scope'].startswith('primary')])
    per_patient = []
    for parameter in sorted({r['parameter'] for r in primary}):
        for sid in sorted({r['subject_id'] for r in primary if r['parameter'] == parameter}):
            rr = [r for r in primary if r['subject_id'] == sid and r['parameter'] == parameter]
            # First trajectory/repeat mean; equal R and the same six budgets here.
            a = float(np.mean([r['mae'] for r in rr if r['method'] == 'M1-a']))
            b = float(np.mean([r['mae'] for r in rr if r['method'] == 'matched_periodic']))
            per_patient.append(dict(subject_id=sid, parameter=parameter, m1_mean=a, uniform_mean=b, delta=a - b))
    csv_rows(output / 'patient_deltas.csv', per_patient)
    write_report(output, result, per_patient, protocol, coverage, len(complete))


def write_report(output, summary, patients, protocol, coverage, complete_count):
    f = lambda v: 'NA' if v is None else f'{v:.4f}'
    pct = lambda v: 'NA' if v is None else f'{v:.2%}'
    ci = lambda r, name: f"[{f(r.get(name + '_ci_low'))}, {f(r.get(name + '_ci_high'))}]"
    labels = {'supports_pure_timing_advantage': '支持M1具有纯更新时间优势',
              'uniform_timing_better': '均匀时间更好，M1当前门控时间未显示优势',
              'insufficient_evidence': '证据不足，未确认纯更新时间优势'}
    primary = [r for r in summary if r['scope'].startswith('primary')]
    main = [r for r in primary if r['metric'] == 'mae']
    text = ['# 纯更新时间配对实验报告', '',
            '本轮为机制诊断，复用已分析过的test。未修改门控、重新估计公开统计或利用test调参；正面结果仍需新划分或外层交叉验证确认。', '',
            '## 结论', '']
    if not main:
        text += ['没有同时完成六个预算严格匹配的轨迹，主检验不成立；不将失败样本填补为成功或零误差。', '']
    for r in main:
        text += [f"**{labels[r['conclusion']]}。** 六预算患者均权 MAE：M1时间={f(r['m1_mean'])}，"
                 f"均匀时间={f(r['uniform_mean'])}；delta={f(r['delta'])}。患者配对95% CI {ci(r, 'patient')}，"
                 f"患者＋机制两层95% CI {ci(r, 'hierarchical')}。", '']
        if r['conclusion'] == 'insufficient_evidence':
            text += ['当前没有确认纯时间收益。上一轮优势主要可能来自根选择或预算集中，但跨0本身不能证明这两项就是原因；本轮没有完成各因素的加性因果分解。', '']
        elif r['conclusion'] == 'uniform_timing_better':
            text += ['在本次固定动作包与合法时间定义下，均匀重放误差更小，当前门控时间应继续优化；不将这一结果外推为所有任务和轨迹中的普遍无效。', '']
    c = coverage['overall']
    text += ['## 严格匹配覆盖率与排除', '',
             '|单位|尝试|成功|覆盖率|', '|---|---:|---:|---:|',
             f"|患者（至少一个配对成功）|{c['attempted_patients']}|{c['successful_patients_any']}|{pct(c['patient_coverage_any'])}|",
             f"|轨迹（至少一个配对成功）|{c['attempted_trajectories']}|{c['successful_trajectories_any']}|{pct(c['trajectory_coverage_any'])}|",
             f"|患者×轨迹×预算×重复|{c['attempted_pairs']}|{c['successful_pairs']}|{pct(c['pair_coverage'])}|",
             f"|后续动作包（含重复）|{c['attempted_actions_including_repeats']}|{c['matched_actions_including_repeats']}|{pct(c['action_coverage'])}|",
             f"|后续动作包（去除R重复）|{c['unique_case_actions']}|{c['unique_case_matched_actions']}|{pct(c['action_coverage'])}|", '',
             f"全部预指定预算和重复都成功的患者数：{c['successful_patients_complete']}；轨迹数：{c['successful_trajectories_complete']}。"
             f"主分析包含 {complete_count} 个轨迹×参数完整单元；按患者均权。", '',
             '|预算|尝试配对|成功配对|动作覆盖率|', '|---:|---:|---:|---:|']
    for b, v in coverage['by_budget'].items():
        text.append(f"|{float(b):.2f}|{v['attempted_pairs']}|{v['successful_pairs']}|{pct(v['action_coverage'])}|")
    text += ['', f"排除配对数：{c['failed_pairs']}。`matching_failures.jsonl` 逐条记录原因和每个动作的合法机会；没有失败时该文件为空。"
             '失败整对排除，不保留部分动作，不改根、不改epsilon、不改噪声。逐预算结果只用该预算严格成功的配对；六预算主检验只用六点均成功的轨迹，避免预算间更换患者样本。', '',
             '本数据在初始化时已观测全部根；且原M1日程本身就是有序合法匹配的见证。因此在本轮选定的“最新已知值”定义下，全局匹配通常不应失败。'
             '覆盖率高不等于时间移动大，移动幅度另见下表及完整时间表。', '',
             '## 冻结协议及合法时间', '',
             f"- 原始冻结来源：`{protocol['source']}`；上一轮配对来源：`{protocol['prior']}`。",
             f"- 精确预算值复用原网格：{', '.join(str(b) for b in protocol['primary_budgets'])}；仅显示时四舍五入。validation选择不变。",
             f"- R={protocol['repeats']}，配对bootstrap={protocol['bootstrap']}次；主检验为六预算患者内平均，再患者均权。逐预算区间为点态诊断。",
             '- 合法机会沿用公开测量批次。按用户确认，动作包内每个根只需截至该轮已观测；本批次未重测时用历史最新值，绝不使用未来观测。当前批次观测先入状态，再发布，再评估该轮查询。',
             '- 均匀目标为1+k(T−1)/(K+1)。确定性有序最小代价匹配最小化平方距离；平局取字典序更早日程；不读取数值。动作序号不变，时间严格递增。',
             '- 初始化、K、第k次有序根集合、逐根epsilon和单位Laplace噪声逐项固定。两方法使用同一个重放函数与原Cache发布、域投影和覆盖更新实现；不调用预算分配器重新分配。',
             '- 噪声按(seed,subject,trajectory,B,repeat,action_index,root)生成，初始化action=0；不按turn重采样。M1原时重放同时与真实运行M1逐项核对发布值和MAE。',
             '- 均匀对照知道最终动作包及完整公开机会表，是离线解释工具，不宣称在线可部署。M1仍只看当前公开批次和历史账本。',
             '- 长期变化：仅删除某一次刷新，后续所有动作、真值输入与噪声保持不变，比较[刷新轮,T]误差；后续同根覆盖自动消除该次影响。正值表示该次刷新降低误差。不同动作效应可重叠，不能相加解释总体收益。',
             '- 不将机制重复当作独立患者。保留患者级、患者＋机制两层、固定患者机制区间与两个方差分量。', '',
             '## 六个固定预算的主指标', '',
             '|B|M1时间MAE|均匀时间MAE|delta|患者配对95% CI|两层95% CI|判断|',
             '|---:|---:|---:|---:|---|---|---|']
    for r in summary:
        if r['scope'] == 'strict_matched_budget' and r['metric'] == 'mae':
            text.append(f"|{r['B']:.2f}|{f(r['m1_mean'])}|{f(r['uniform_mean'])}|{f(r['delta'])}|{ci(r, 'patient')}|{ci(r, 'hierarchical')}|{labels[r['conclusion']]}|")
    text += ['', '## 次指标：六预算聚合', '', '|指标|M1时间|均匀时间|delta|', '|---|---:|---:|---:|']
    names = {'p95_abs_error': '轨迹内p95误差（再均权）', 'query_loss': '尺度化查询误差',
             'workload_loss': '尺度化加权节点误差', 'refresh_count': '后续刷新次数',
             'spent': '含初始化实际epsilon', 'initial_spent': '初始化epsilon', 'refresh_spent': '后续epsilon',
             'miss_rate': 'MissRate（变化当轮未覆盖全部变化根）', 'aoi_mean': '根平均AoI/对话轮',
             'publication_projection_fraction': '发布根投影率', 'cache_projection_fraction': '缓存根×轮投影率',
             'wait_any_observed_mean': '变化至任意刷新等待/仅观察到者',
             'wait_any_restricted_mean': '变化至任意刷新等待/末尾截尾',
             'wait_any_censored_fraction': '任意刷新等待截尾比例',
             'wait_roots_observed_mean': '变化根全部被覆盖等待/仅观察到者',
             'wait_roots_restricted_mean': '变化根全部覆盖等待/末尾截尾',
             'wait_roots_censored_fraction': '变化根全部覆盖等待截尾比例',
             'immediate_error_drop_per_refresh': '每次刷新即时误差下降',
             'immediate_error_drop_per_epsilon': '每epsilon即时误差下降',
             'immediate_scaled_drop_per_refresh': '每次刷新即时尺度化误差下降',
             'long_term_error_drop_per_refresh': '每次刷新后续轨迹平均误差下降',
             'long_term_cumulative_drop_per_refresh': '每次刷新后续累计误差下降',
             'long_term_cumulative_drop_per_epsilon': '每epsilon后续累计误差下降', 'mean_refresh_turn': '平均刷新轮次',
             'paired_time_changed_fraction': '配对动作时间改变比例（配对统计）',
             'paired_absolute_time_shift': '配对动作平均绝对移动轮数（配对统计）'}
    for r in primary:
        if r['metric'] in names:
            text.append(f"|{names[r['metric']]}|{f(r['m1_mean'])}|{f(r['uniform_mean'])}|{f(r['delta'])}|")
    text += ['', '显著阈值沿用严格>0.05（1e-12浮点容差）。等待无后续发布时在T+1右截尾，同时报告观察均值和截尾比例。'
             '投影率指发布值或缓存值超出标准化根域的比例；两方法均只在任务计算时作相同域投影。', '',
             '## 各节点和患者结果', '',
             '|节点|M1时间MAE|均匀时间MAE|delta|患者配对95% CI|', '|---|---:|---:|---:|---|']
    for r in primary:
        if r['metric'].startswith('node_mae__'):
            text.append(f"|{r['metric'].split('__')[1]}|{f(r['m1_mean'])}|{f(r['uniform_mean'])}|{f(r['delta'])}|{ci(r, 'patient')}|")
    text += ['', '|患者|M1时间MAE|均匀时间MAE|delta|', '|---|---:|---:|---:|']
    for r in patients:
        text.append(f"|{r['subject_id']}|{f(r['m1_mean'])}|{f(r['uniform_mean'])}|{f(r['delta'])}|")
    text += ['', '## 审计与文件', '',
             '- `pure_timing_metrics.jsonl`：严格匹配成功样本的所有重复指标；失败样本不混入。',
             '- `pure_timing_schedules.jsonl`：成功/失败状态、M1原日程、固定动作包、逐动作真实输入/来源轮次、发布值与即时/长期误差变化。',
             '- `matching_failures.jsonl` / `coverage.json`：排除原因及患者、轨迹、配对、动作覆盖率。',
             '- `paired_summary.csv` / `paired_primary.json`：全部主次指标的配对差值与三种区间；`patient_deltas.csv`：患者差异。',
             '- `audit.json`：固定包逐项相等、总支出误差<1e-10、不超预算、来源轮次<=发布时间、无重复时间、M1重放一致和冻结来源校验。',
             '- `unit_tests.txt`：正式运行前通过的完整单元测试；`protocol.json`：执行前冻结的口径。', '',
             '结论严格遵守：患者配对和两层95%区间上界均<0支持纯时间优势；下界均>0支持均匀更好；跨0、触0或两种区间判断不一致时为证据不足。'
             '小样本、复用test及离线匹配限制仍存在；六预算均值是预指定局部结论，不能外推为全部预算和任务中的普遍最优。', '']
    (output / 'REPORT-zh.md').write_text('\n'.join(text))
