"""Patient-equal paired inference for the frozen timing diagnostic."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .analyze import csv_rows, write_json


def paired_interval(patient_repeats, draws=2000, seed=20261110):
    """Bootstrap paired differences, never independently resample the methods.

    Each patient is a vector of repeat means (equal trajectories within repeat).
    Patient-only, nested patient+mechanism, and fixed-patient mechanism-only
    intervals are retained. Resampling keys are the same across metric/B cells.
    """
    values = np.asarray(patient_repeats, float)
    n, R = values.shape
    rng = np.random.default_rng(seed)
    patients = rng.integers(n, size=(draws, n))
    repeats = rng.integers(R, size=(draws, n, R))
    patient = values.mean(axis=1)[patients].mean(axis=1)
    hierarchical = values[patients[:, :, None], repeats].mean(axis=(1, 2))
    mechanism = values[np.arange(n)[None, :, None], repeats].mean(axis=(1, 2))
    interval = lambda a: [float(v) for v in np.quantile(a, [.025, .975])]
    within = np.var(values, axis=1, ddof=1) / R if R > 1 else np.zeros(n)
    between = float(np.var(values.mean(axis=1), ddof=1)) if n > 1 else 0.
    return {'patient_ci': interval(patient), 'hierarchical_ci': interval(hierarchical),
            'mechanism_ci': interval(mechanism),
            'variance_mechanism_mean': float(within.sum() / n ** 2),
            'variance_between_mean': max(0., between - float(within.mean())) / n}


def summarize_cell(rows, metrics, scope, B, draws, seed):
    by_case = defaultdict(dict)
    for row in rows:
        key = row['subject_id'], row['trajectory_id'], row['B'], row['repeat']
        assert row['method'] not in by_case[key]
        by_case[key][row['method']] = row
    assert all(set(v) == {'M1-a', 'matched_periodic'} for v in by_case.values())
    output = []
    for metric in metrics:
        # First equal trajectories and budgets per patient/repeat; then equal repeats
        # per patient, then equal patients. This prevents noisy patients dominating.
        cells = defaultdict(list)
        for (sid, trajectory, budget, repeat), pair in by_case.items():
            a, b = pair['M1-a'].get(metric), pair['matched_periodic'].get(metric)
            if a is not None and b is not None:
                cells[(sid, repeat)].append([a, b])
        if not cells:
            continue
        patients = defaultdict(list)
        for (sid, repeat), v in sorted(cells.items()):
            patients[sid].append(np.mean(v, axis=0))
        lens = {len(v) for v in patients.values()}
        if len(lens) != 1:
            # Observed-only waiting/drop statistics can have unequal support.
            # Keep point estimates but do not invent complete bootstrap panels.
            point = np.mean([np.mean(v, axis=0) for v in patients.values()], axis=0)
            intervals = None
        else:
            values = np.array([patients[sid] for sid in sorted(patients)])
            point = values.mean(axis=(0, 1))
            intervals = paired_interval(values[:, :, 0] - values[:, :, 1], draws, seed)
        row = dict(template=rows[0]['template'], parameter=rows[0]['parameter'],
                   scope=scope, B=B, metric=metric, n_patients=len(patients),
                   n_pairs=sum(len(v) for v in cells.values()),
                   m1_mean=float(point[0]), periodic_mean=float(point[1]), delta=float(point[0] - point[1]))
        if intervals:
            for label in ['patient', 'hierarchical', 'mechanism']:
                row[label + '_ci_low'], row[label + '_ci_high'] = intervals[label + '_ci']
            row.update({k: intervals[k] for k in ['variance_mechanism_mean', 'variance_between_mean']})
        if metric == 'mae' and intervals:
            pl, ph = intervals['patient_ci']
            hl, hh = intervals['hierarchical_ci']
            row['conclusion'] = ('supports_better_M1_timing_single_task' if ph < 0 and hh < 0 else
                                 'uniform_timing_better' if pl > 0 and hl > 0 else 'insufficient_evidence')
        output.append(row)
    return output


def analyze(rows, output, protocol, seed):
    output = Path(output)
    excluded = {'subject_id', 'trajectory_id', 'template', 'method', 'parameter', 'B', 'repeat',
                'stop_turn', 'budget_exhaustion_turn', 'last_refresh_turn'}
    metrics = [k for k in rows[0] if k not in excluded]
    groups = defaultdict(list)
    primary = defaultdict(list)
    for r in rows:
        groups[(r['template'], r['parameter'], r['B'])].append(r)
        if r['B'] in protocol['primary_budgets']:
            primary[(r['template'], r['parameter'])].append(r)
    summaries = []
    for (_, _), rr in sorted(primary.items()):
        summaries += summarize_cell(rr, metrics, 'primary_six_budget_mean', None, protocol['bootstrap'], seed)
    for (_, _, B), rr in sorted(groups.items()):
        scope = 'prespecified_frontier_budget' if B in protocol['primary_budgets'] else 'exploratory_grid'
        summaries += summarize_cell(rr, metrics, scope, B, protocol['bootstrap'], seed)
    csv_rows(output / 'paired_summary.csv', summaries)
    # Patient-specific means expose heterogeneity without treating repeats as people.
    patient_rows = []
    for (_, _), rr in sorted(primary.items()):
        for sid in sorted({r['subject_id'] for r in rr}):
            pp = [r for r in rr if r['subject_id'] == sid]
            row = {'subject_id': sid, 'template': pp[0]['template'], 'parameter': pp[0]['parameter']}
            for metric in ['mae', 'miss_rate', 'aoi_mean', 'query_loss', 'publication_projection_fraction']:
                values = {m: float(np.mean([r[metric] for r in pp if r['method'] == m and r[metric] is not None]))
                          for m in ['M1-a', 'matched_periodic']}
                row.update({metric + '_M1': values['M1-a'], metric + '_periodic': values['matched_periodic'],
                            metric + '_delta': values['M1-a'] - values['matched_periodic']})
            patient_rows.append(row)
    csv_rows(output / 'patient_deltas.csv', patient_rows)
    # Calendar counts / coverage collapse repeats: schedules are noise-independent.
    turns, rootdiffs, exhaustion = [], [], []
    schedule_file = output / 'matched_schedules.jsonl'
    with schedule_file.open() as f:
        for line in f:
            r = json.loads(line)
            if r['repeat'] != 0:
                continue
            base = {k: r[k] for k in ['subject_id', 'trajectory_id', 'template', 'B', 'parameter']}
            for method in ['M1-a', 'matched_periodic']:
                s = r[method]
                exhaustion.append(dict(base, method=method, stop_turn=s['stop_turn'], spent=s['spent'],
                                       remaining=r['B'] - s['spent'],
                                       last_refresh_turn=s['transcript'][-1]['turn']))
                for release in s['transcript'][1:]:
                    turns.append(dict(base, method=method, turn=release['turn'], action_index=release['action_index'],
                                      roots=','.join(release['roots']), epsilon=release['epsilon_total']))
            for comparison in r['comparisons'][1:]:
                rootdiffs.append({**base, **comparison, 'common_roots': ','.join(comparison['common_roots'])})
    csv_rows(output / 'refresh_times.csv', turns)
    csv_rows(output / 'root_coverage_differences.csv', rootdiffs)
    csv_rows(output / 'budget_stop.csv', exhaustion)
    write_report(output, summaries, patient_rows, protocol)
    write_json(output / 'paired_primary.json', [r for r in summaries if r['scope'] == 'primary_six_budget_mean'])


def write_report(output, summaries, patient_rows, protocol):
    def num(x):
        return 'NA' if x is None else f'{x:.4f}'
    def interval(row, name):
        return f"[{num(row.get(name + '_ci_low'))}, {num(row.get(name + '_ci_high'))}]"
    primary = [r for r in summaries if r['scope'] == 'primary_six_budget_mean']
    main = [r for r in primary if r['metric'] == 'mae']
    text = ['# 更新时间配对实验报告', '',
            '本实验复用已经分析过的 test，仅用于机制诊断。正面结果仍需新划分或外层交叉验证确认。', '',
            '## 主要判断', '']
    for r in main:
        if r['conclusion'] == 'supports_better_M1_timing_single_task':
            statement = '在本任务、预先指定预算范围内，结果支持 M1 选择了更好的更新时间。'
        elif r['conclusion'] == 'uniform_timing_better':
            statement = '均匀周期时间更好，M1 门控仍需优化。'
        else:
            statement = '当前证据不足，不能判定 M1 选择了更好的更新时间。'
        text += [f"**{r['template']}：{statement}** M1-a MAE={num(r['m1_mean'])}，均匀对照={num(r['periodic_mean'])}，"
                 f"delta={num(r['delta'])}；患者级配对 95% CI {interval(r, 'patient')}，两层 CI {interval(r, 'hierarchical')}。", '']
    text += ['## 冻结协议与比较口径', '',
             f"- 冻结来源：`{protocol['source']}`；只读原数据划分、公式、公开统计量、参数、完整预算网格及 validation 选择。未运行拟合或选择。",
             f"- R={protocol['repeats']}；每个预算、患者、轨迹、重复先实际运行 M1，再生成等 K、等逐次总 epsilon 的对照；总支出包含初始化。",
             '- 主指标：患者均权、患者内轨迹均权的全轨迹最终节点 MAE；delta=M1−均匀，负值有利于 M1。',
             f"- 预先固定重点预算：{', '.join(f'{b:.2f}' for b in protocol['primary_budgets'])}。主检验为这六个预算的患者内均值；全网格和逐预算 CI 为点态诊断，未选最有利预算。",
             f"- 配对 bootstrap {protocol['bootstrap']} 次：患者级重采样；另保留患者＋机制两层区间、固定患者的机制区间及两个方差分量。方法间始终共同重采样。",
             '- 所有患者重复均有完整账本。噪声键为 `(seed,subject,trajectory,B,repeat,action_index,root)`；初始化 action=0；对应刷新中的共同根共享单位 Laplace。',
             '- 均匀时间表用完整公开机会表，确定性最小化与 K 个等间隔对话目标的平方距离，冲突时选更早时间；不读取真值。M1 只读取当前 PublicBatch、公开统计及自身历史。',
             '- **离线对照**知道 M1 的最终 K 和支出序列，不能直接当作可在线部署的周期算法。',
             '- 若时刻变化导致根集合变化，两者只保证同次数、同逐次总 epsilon、同含初始化总 epsilon；逐根份额可变。根选择仍是当前 measured active roots，分配、发布、投影和覆盖更新共用实现。',
             '- 本轮只有 MELD-Na 一个任务；五个计算节点是相关的中间/最终节点，不能据此宣称“多数独立任务一致”。', '',
             '## 预先指定预算的最终节点结果', '',
             '|B|M1 MAE|均匀 MAE|delta|患者配对95% CI|两层95% CI|判断|',
             '|---:|---:|---:|---:|---|---|---|']
    labels = {'supports_better_M1_timing_single_task': '支持M1（单任务）',
              'uniform_timing_better': '均匀更好', 'insufficient_evidence': '证据不足'}
    for r in summaries:
        if r['metric'] == 'mae' and r['scope'] == 'prespecified_frontier_budget':
            text.append(f"|{r['B']:.2f}|{num(r['m1_mean'])}|{num(r['periodic_mean'])}|{num(r['delta'])}|"
                        f"{interval(r, 'patient')}|{interval(r, 'hierarchical')}|{labels[r['conclusion']]}|")
    text += ['', '## 代价与时间解释：六预算均值', '',
             '|指标|M1|均匀|delta|', '|---|---:|---:|---:|']
    names = {'spent': '含初始化实际epsilon', 'initial_spent': '初始化epsilon', 'refresh_count': '后续刷新次数',
             'p95_abs_error': '轨迹内p95绝对误差（再均权）', 'query_loss': '尺度化查询误差',
             'workload_loss': '尺度化加权节点误差', 'miss_rate': 'MissRate（患者均权）',
             'aoi_mean': '根平均AoI/对话轮', 'publication_projection_fraction': '发布根投影率',
             'cache_projection_fraction': '缓存根×轮投影率',
             'refresh_near_change_fraction': '刷新落在显著变化±1轮比例',
             'refresh_after_change_fraction': '刷新落在显著变化后0–1轮比例',
             'refresh_before_change_fraction': '刷新落在显著变化前0–1轮比例',
             'wait_any_observed_mean': '变化至下次任意刷新等待/仅观察到者',
             'wait_any_restricted_mean': '变化至下次任意刷新等待/末尾截尾',
             'wait_any_censored_fraction': '任意刷新等待截尾比例',
             'wait_roots_observed_mean': '变化根全部被后续发布覆盖等待/仅观察到者',
             'wait_roots_restricted_mean': '变化根全部覆盖等待/末尾截尾',
             'wait_roots_censored_fraction': '变化根全部覆盖等待截尾比例',
             'immediate_error_drop_per_refresh': '每次刷新即时绝对误差下降',
             'immediate_error_drop_per_epsilon': '每单位后续epsilon即时绝对误差下降',
             'immediate_scaled_drop_per_refresh': '每次刷新即时尺度化误差下降',
             'immediate_scaled_drop_per_epsilon': '每单位后续epsilon即时尺度化误差下降',
             'mean_refresh_turn': '平均刷新轮次',
             'paired_root_mismatch_fraction': '对应刷新根集合不同的比例'}
    for r in primary:
        if r['metric'] in names:
            text.append(f"|{names[r['metric']]}|{num(r['m1_mean'])}|{num(r['periodic_mean'])}|{num(r['delta'])}|")
    text += ['', '等待按对话轮计；没有后续刷新时于 T+1 右截尾，表中同时给观察均值、截尾等待和截尾比例。'
             '±1轮统计只用于离线解释，不进入门控。即时误差下降以同一当前真值比较刷新前后，可为负；不等同长期因果收益。'
             '投影率是标准化发布根超出[0,1]的比例，任务计算共用域投影。', '',
             '## 节点方向与患者差异', '',
             '|节点|M1 MAE|均匀 MAE|delta|患者配对95% CI|', '|---|---:|---:|---:|---|']
    for r in primary:
        if r['metric'].startswith('node_mae__'):
            text.append(f"|{r['metric'].split('__')[1]}|{num(r['m1_mean'])}|{num(r['periodic_mean'])}|{num(r['delta'])}|{interval(r, 'patient')}|")
    text += ['', '|患者|M1 MAE|均匀 MAE|delta|', '|---|---:|---:|---:|']
    for r in patient_rows:
        text.append(f"|{r['subject_id']}|{num(r['mae_M1'])}|{num(r['mae_periodic'])}|{num(r['mae_delta'])}|")
    exhausted = [r for r in rows_csv(output / 'budget_stop.csv') if r['stop_turn']]
    text += ['', '## 审计与可复核文件', '',
             f"- 预算阈值停止的唯一患者×预算×方法时间表数：{len(exhausted)}；具体时刻见 `budget_stop.csv`。无停止记为 null/空值，不记作第0轮。",
             '- `matched_metrics.jsonl`：所有重复的完整主次指标、各节点结果、逐根epsilon累计与AoI。',
             '- `matched_schedules.jsonl`：全部重复的初始化/后续发布时间、根、逐根epsilon、单位噪声、发布值、门控决策、显著变化及等待。',
             '- `paired_summary.csv`：全预算、各指标配对差值和三种区间；`paired_primary.json` 为预指定六预算聚合。',
             '- `refresh_times.csv` / `root_coverage_differences.csv`：去重后的时间分布及逐次根差异；`patient_deltas.csv` 为患者差异。',
             '- `audit.json`：刷新次数相等、每次及总epsilon误差<1e-10、预算不超支、合法根、噪声配对、均匀时间表确定无重复、M1与旧账本一致及冻结来源哈希。', '',
             '## 结论判定与局限', '',
             '患者配对与两层区间上界都<0时，支持本任务中 M1 更好；下界都>0时，均匀更好；跨0或两种区间判断不一致时记为证据不足。'
             '跨任务结论还需多数任务同方向；本轮单任务无法检验这一条件。'
             '同总支出排除了刷新数量/总隐私花费的解释，但根可用性及由此导致的逐根份额变化仍可能贡献差异。',
             'test 患者只有5人；增加R只减少机制噪声不确定性，不能替代增加患者。'
             '本次噪声按刷新序号生成，不能直接与上一轮按turn生成的MAE数值逐项比较；M1公开决策、时间、根和epsilon与上一轮账本逐项验证。', '']
    (output / 'REPORT-zh.md').write_text('\n'.join(text))


def rows_csv(path):
    import csv
    with Path(path).open() as f:
        return list(csv.DictReader(f))
