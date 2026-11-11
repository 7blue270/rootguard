"""Patient-paired random/oracle timing comparisons; random schedules are nested."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .analyze import csv_rows, write_json
from .matched_analysis import summarize_cell, paired_interval
from .pure_timing import M1_TIME, UNIFORM_TIME
from .timing_supplement import RANDOM, ORACLE


def aggregate_by_case(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['subject_id'], row['trajectory_id'], row['B'], row['repeat'], row['method'])].append(row)
    result = []
    keys = ['subject_id', 'trajectory_id', 'template', 'B', 'repeat', 'parameter', 'method']
    for rr in grouped.values():
        metric_names = [k for k in rr[0] if k not in keys + ['sample_index']]
        r = {k: rr[0][k] for k in keys}
        for name in metric_names:
            values = [v[name] for v in rr if v[name] is not None]
            r[name] = float(np.mean(values)) if values else None
        result.append(r)
    return result


def panel(rows, fields, methods):
    """Patient x repeat x (fields,methods), averaging equal trajectories/budgets."""
    ids = sorted({r['subject_id'] for r in rows})
    reps = sorted({r['repeat'] for r in rows})
    lookup = defaultdict(list)
    for r in rows:
        for f in fields:
            if r.get(f) is not None:
                lookup[(r['subject_id'], r['repeat'], f, r['method'])].append(r[f])
    return np.array([[[float(np.mean(lookup[(sid, repeat, f, method)]))
                       for f in fields for method in methods] for repeat in reps] for sid in ids])


def ratio_summary(rows, scope, B, draws, seed):
    methods = [M1_TIME, UNIFORM_TIME, RANDOM, ORACLE]
    values = panel(rows, ['mae'], methods)
    n, R, _ = values.shape
    point = values.mean(axis=(0, 1))
    rng = np.random.default_rng(seed)
    pi = rng.integers(n, size=(draws, n))
    ri = rng.integers(R, size=(draws, n, R))
    samples = values[pi[:, :, None], ri].mean(axis=(1, 2))
    result = {'scope': scope, 'B': B, 'm1_mae': float(point[0]), 'uniform_mae': float(point[1]),
              'random_mean_mae': float(point[2]), 'oracle_mae': float(point[3])}
    for ref, index in [('random', 2), ('uniform', 1)]:
        den = point[index] - point[3]
        num = point[index] - point[0]
        ratio = num / den if den > 1e-12 else None
        denom_samples = samples[:, index] - samples[:, 3]
        valid = denom_samples > 1e-12
        bootstrap_ratio = (samples[valid, index] - samples[valid, 0]) / denom_samples[valid]
        result.update({ref + '_total_achievable_gain': float(den),
                       ref + '_m1_realized_gain': float(num),
                       ref + '_m1_realized_gain_fraction': float(ratio) if ratio is not None else None,
                       ref + '_gain_fraction_ci_low': float(np.quantile(bootstrap_ratio, .025)) if len(bootstrap_ratio) else None,
                       ref + '_gain_fraction_ci_high': float(np.quantile(bootstrap_ratio, .975)) if len(bootstrap_ratio) else None,
                       ref + '_defined_bootstrap_fraction': float(valid.mean())})
    return result


def schedule_nested_ci(rows, budgets, draws, seed):
    """Adds resampling of 50 schedule draws inside each fixed noise repeat."""
    ids = sorted({r['subject_id'] for r in rows})
    reps = sorted({r['repeat'] for r in rows})
    random = defaultdict(list)
    m1 = {}
    for r in rows:
        key = r['subject_id'], r['repeat'], r['B']
        if r['method'] == RANDOM:
            random[key].append(r['mae'])
        elif r['method'] == M1_TIME:
            m1[key] = r['mae']
    # The current frozen cohort has one trajectory per subject, explicitly checked.
    assert all(len(v) == 50 for v in random.values())
    values = np.array([[[random[(sid, repeat, b)] for b in budgets] for repeat in reps] for sid in ids])
    baselines = np.array([[[m1[(sid, repeat, b)] for b in budgets] for repeat in reps] for sid in ids])
    rng = np.random.default_rng(seed)
    schedule_rng = np.random.default_rng(seed + 17)
    n, R, nb, S = values.shape
    pi = rng.integers(n, size=(draws, n))
    ri = rng.integers(R, size=(draws, n, R))
    samples = []
    for draw in range(draws):
        rr = values[pi[draw, :, None], ri[draw]]
        bb = baselines[pi[draw, :, None], ri[draw]]
        indices = schedule_rng.integers(S, size=rr.shape)
        random_mean = np.take_along_axis(rr, indices, axis=-1).mean()
        samples.append(float(bb.mean() - random_mean))
    return {'ci_low': float(np.quantile(samples, .025)), 'ci_high': float(np.quantile(samples, .975)),
            'bootstrap': draws, 'levels': ['patients', 'noise_repeats', 'random_calendars']}


def analyze(rows, distributions, output, protocol):
    output = Path(output)
    collapsed = aggregate_by_case(rows)
    key_fields = ['subject_id', 'trajectory_id', 'template', 'B', 'repeat', 'parameter', 'method']
    metrics = [k for k in collapsed[0] if k not in key_fields]
    scopes = [('primary_six_budget_mean', None, collapsed)]
    scopes += [('prespecified_budget', b, [r for r in collapsed if r['B'] == b]) for b in protocol['executed_budgets']]
    results, gains, diagnostics = [], [], []
    for scope, B, rr in scopes:
        for reference in [RANDOM, ORACLE, UNIFORM_TIME]:
            adapted = [dict(r, method='M1-a' if r['method'] == M1_TIME else 'matched_periodic')
                       for r in rr if r['method'] in [M1_TIME, reference]]
            ss = summarize_cell(adapted, metrics, scope, B, protocol['bootstrap'], protocol['seed'])
            for s in ss:
                s['comparison'] = f'{M1_TIME}-minus-{reference}'
                s['reference_mean'] = s.pop('periodic_mean')
                if s['metric'] == 'mae':
                    lo = s['patient_ci_low']; hi = s['patient_ci_high']
                    hlo = s['hierarchical_ci_low']; hhi = s['hierarchical_ci_high']
                    s['conclusion'] = ('M1_significantly_better' if hi < 0 and hhi < 0 else
                                       'reference_significantly_better' if lo > 0 and hlo > 0 else 'insufficient_evidence')
            results.extend(ss)
        gains.append(ratio_summary(rr, scope, B, protocol['bootstrap'], protocol['seed']))
        dd = [r for r in distributions if B is None or r['B'] == B]
        for metric in ['m1_win_rate', 'm1_percentile', 'tie_rate', 'random_mae_mean', 'random_mae_median',
                       'random_mae_q025', 'random_mae_q975']:
            ids = sorted({r['subject_id'] for r in dd})
            reps = sorted({r['repeat'] for r in dd})
            values = np.array([[np.mean([r[metric] for r in dd if r['subject_id'] == sid and r['repeat'] == repeat])
                                for repeat in reps] for sid in ids])
            ci = paired_interval(values, protocol['bootstrap'], protocol['seed'])
            diagnostics.append({'scope': scope, 'B': B, 'metric': metric, 'mean': float(values.mean()),
                                **{f'{label}_ci_{end}': ci[label + '_ci'][i]
                                   for label in ['patient', 'hierarchical', 'mechanism']
                                   for i, end in enumerate(['low', 'high'])}})
    csv_rows(output / 'paired_summary.csv', results)
    csv_rows(output / 'gain_summary.csv', gains)
    csv_rows(output / 'random_diagnostics.csv', diagnostics)
    # Four-method tables aggregate random calendars WITHIN each case first.
    tables = []
    for scope, B, rr in scopes:
        for method in [UNIFORM_TIME, RANDOM, M1_TIME, ORACLE]:
            mm = [r for r in rr if r['method'] == method]
            row = {'scope': scope, 'B': B, 'method': method}
            for metric in metrics:
                vals = []
                for sid in sorted({r['subject_id'] for r in mm}):
                    available = [r[metric] for r in mm if r['subject_id'] == sid and r.get(metric) is not None]
                    if available:
                        vals.append(float(np.mean(available)))
                row[metric] = float(np.mean(vals)) if vals else None
            tables.append(row)
    csv_rows(output / 'method_summary.csv', tables)
    nested = schedule_nested_ci(rows, protocol['executed_budgets'], protocol['bootstrap'], protocol['seed'])
    write_json(output / 'schedule_nested_ci.json', nested)
    all_random = np.array([r['mae'] for r in rows if r['method'] == RANDOM])
    pooled = {'mean': float(all_random.mean()), 'median': float(np.median(all_random)),
              'q025': float(np.quantile(all_random, .025)), 'q975': float(np.quantile(all_random, .975)),
              'meaning': 'case/noise/schedule mixture central95, NOT confidence interval for mean',
              'patient_weighting': 'balanced frozen cohort, one trajectory each, same budgets/repeats/schedules'}
    write_json(output / 'random_mixture_distribution.json', pooled)
    write_report(output, results, gains, diagnostics, tables, protocol, nested, pooled)


def write_report(output, summary, gains, diagnostic, tables, protocol, nested, pooled):
    fmt = lambda v: 'NA' if v is None else f'{v:.4f}'
    interval = lambda r, n: f"[{fmt(r.get(n + '_ci_low'))}, {fmt(r.get(n + '_ci_high'))}]"
    primary = [r for r in summary if r['scope'] == 'primary_six_budget_mean' and r['metric'] == 'mae']
    rand = next(r for r in primary if r['comparison'].endswith(RANDOM))
    oracle = next(r for r in primary if r['comparison'].endswith(ORACLE))
    ds = {r['metric']: r for r in diagnostic if r['scope'] == 'primary_six_budget_mean'}
    g = gains[0]
    if rand['conclusion'] == 'M1_significantly_better':
        finding = ('M1显著优于随机，时间选择包含有效信号；但明显落后精确Oracle，仍有优化空间。'
                   if oracle['conclusion'] == 'reference_significantly_better' else
                   'M1显著优于随机，时间选择包含有效信号；与Oracle接近程度需结合差距与收益比例，不以不显著代替等效。')
    else:
        finding = 'M1未显著优于随机；当前不支持继续将纯更新时间作为核心机制。'
    text = ['# 随机时间与精确Oracle补充实验', '', '## 主要判断', '', f'**{finding}**', '',
            f"六预算患者均权：MAE(M1)−MAE(random_mean)={fmt(rand['delta'])}，患者配对95% CI {interval(rand, 'patient')}，"
            f"患者＋机制两层CI {interval(rand, 'hierarchical')}。进一步嵌套随机日程抽样的三层CI为"
            f"[{fmt(nested['ci_low'])}, {fmt(nested['ci_high'])}]。", '',
            f"M1严格优于随机日程的平均比例={ds['m1_win_rate']['mean']:.2%}；M1在随机误差分布中的平均百分位="
            f"{ds['m1_percentile']['mean']:.2f}（越低越好；平局取半权）。"
            '胜率和百分位先在每个固定动作/噪声包的50个随机日程内计算，再患者均权。', '',
            f"MAE(M1)−MAE(oracle)={fmt(oracle['delta'])}；患者CI {interval(oracle, 'patient')}，两层CI {interval(oracle, 'hierarchical')}。", '',
            '## 冻结内容、随机采样与精确性', '',
            f"- 来源：`{protocol['source']}`。复用其初始化、有序动作包、逐根epsilon、单位Laplace噪声、数据划分、公开统计、公式、六预算及R={protocol['repeats']}；bootstrap={protocol['bootstrap']}。",
            f"- 日程种子={protocol['schedule_seed']}；每个(subject,trajectory,B,repeat)抽{protocol['random_schedules_per_pair']}个日程。随机性与发布噪声分开。",
            '- 精确后缀计数得到全部可行有序日程数，均匀抽取整数编号，再确定性反解为日程；独立有放回抽样，重复编号和较差结果都保留。编号、派生种子与全部50个结果均输出。',
            '- 合法机会与上一轮相同：只在原测量批次；每个包内的根截至当轮已观测，允许使用历史最新值；不使用未来根值。',
            '- Oracle完整枚举同一合法时间空间，以全轨迹最终MELD-Na MAE求最小；不是最大变化轮，也不是50个随机样本中最好者。完整枚举给出零离散搜索最优性缺口，浮点评分核对容差1e-10。',
            '- Oracle可见完整真值和固定动作包的已实现噪声，属于离线乐观性能上界（误差下界）；不参与调参、选包或部署结论。',
            '- 随机/Oracle数值重放与原Cache发布实现核对；M1及均匀结果沿用并逐项校验上一轮保存值。根、逐根epsilon、单位噪声、初始化和总支出固定。',
            '- 随机日程不是新患者或新的隐私发布重复。先在固定噪声包内平均50个日程，再以患者/机制重复做配对区间；另报告三层区间反映50次日程抽样的不确定性。',
            '- 主检验仍为六个预指定预算的患者内均值。逐预算区间是点态诊断，没有根据test选预算或重新调参。本轮复用test，仍属机制诊断。', '',
            '## 四种日程的六预算结果', '',
            '|方法|MAE|p95误差|MissRate|AoI|变化至任意刷新等待/截尾|变化根全部覆盖等待/截尾|',
            '|---|---:|---:|---:|---:|---:|---:|']
    labels = {M1_TIME: 'M1时间', UNIFORM_TIME: '均匀时间', RANDOM: '随机时间（先平均50次）', ORACLE: '精确Oracle时间'}
    for r in tables:
        if r['scope'] == 'primary_six_budget_mean':
            text.append(f"|{labels[r['method']]}|{fmt(r['mae'])}|{fmt(r['p95_abs_error'])}|{fmt(r['miss_rate'])}|"
                        f"{fmt(r['aoi_mean'])}|{fmt(r['wait_any_restricted_mean'])}|{fmt(r['wait_roots_restricted_mean'])}|")
    text += ['', 'p95为每条完整轨迹误差的95分位，再按日程/重复/患者聚合。等待以T+1作右截尾；截尾比例另见`method_summary.csv`。', '',
             '## 随机MAE的分布与M1位置', '',
             f"所有病例×预算×机制×随机日程的混合分布：均值={fmt(pooled['mean'])}，中位数={fmt(pooled['median'])}，"
             f"中央95%范围=[{fmt(pooled['q025'])}, {fmt(pooled['q975'])}]。这不是均值置信区间，也不是只由时间引起的离散程度。", '',
             f"逐固定动作/噪声包的条件分布，先求中位数和2.5%/97.5%分位后再均权：中位数均值={fmt(ds['random_mae_median']['mean'])}，"
             f"条件范围端点均值=[{fmt(ds['random_mae_q025']['mean'])}, {fmt(ds['random_mae_q975']['mean'])}]。"
             '每个包的完整条件统计见`random_distribution.jsonl`。', '',
             f"M1胜率患者95% CI {interval(ds['m1_win_rate'], 'patient')}，两层CI {interval(ds['m1_win_rate'], 'hierarchical')}；"
             f"百分位患者CI {interval(ds['m1_percentile'], 'patient')}，两层CI {interval(ds['m1_percentile'], 'hierarchical')}。", '',
             '## 时间选择可实现收益', '',
             f"以随机均值为参照：总可实现MAE下降=random_mean−oracle={fmt(g['random_total_achievable_gain'])}；"
             f"M1已实现=random_mean−M1={fmt(g['random_m1_realized_gain'])}；已实现比例={fmt(g['random_m1_realized_gain_fraction'])}，"
             f"两层bootstrap区间=[{fmt(g['random_gain_fraction_ci_low'])}, {fmt(g['random_gain_fraction_ci_high'])}]。", '',
             f"以均匀时间为参照：总可实现下降={fmt(g['uniform_total_achievable_gain'])}，M1已实现下降={fmt(g['uniform_m1_realized_gain'])}，"
             f"已实现比例={fmt(g['uniform_m1_realized_gain_fraction'])}。", '',
             '比例使用患者均权平均收益之比，不平均小分母的逐病例比例；不截断负收益或负比例。分母为0时记NA，另报bootstrap可定义比例。'
             '没有预设“接近Oracle”的等效界，因此不以差异不显著宣称接近或最优。', '',
             '## 六个预指定预算的主要比较', '',
             '|B|M1−随机均值|患者95% CI|两层95% CI|M1−Oracle|随机参照已实现比例|',
             '|---:|---:|---|---|---:|---:|']
    for b in protocol['executed_budgets']:
        rr = next(r for r in summary if r['B'] == b and r['metric'] == 'mae' and r['comparison'].endswith(RANDOM))
        oo = next(r for r in summary if r['B'] == b and r['metric'] == 'mae' and r['comparison'].endswith(ORACLE))
        gg = next(r for r in gains if r['B'] == b)
        text.append(f"|{b:.2f}|{fmt(rr['delta'])}|{interval(rr, 'patient')}|{interval(rr, 'hierarchical')}|{fmt(oo['delta'])}|{fmt(gg['random_m1_realized_gain_fraction'])}|")
    text += ['', '## 文件与审计', '',
             '- `random_schedules.jsonl`：全部随机日程及每次发布时间、根、逐根epsilon、单位噪声、实际发布值、观测来源轮次和均匀抽样编号。',
             '- `oracle_schedules.jsonl`：精确最优日程、完整搜索规模、目标值与零搜索缺口。',
             '- `supplement_metrics.jsonl`：M1、均匀、全部随机样本及Oracle逐病例/重复指标。',
             '- `paired_summary.csv`：各主次指标的患者级/两层/机制区间；`gain_summary.csv`和`random_diagnostics.csv`：收益比例与胜率/百分位。',
             '- `method_summary.csv`、`random_distribution.jsonl`、`schedule_nested_ci.json`：四方法汇总、条件分布及三层区间。',
             '- `audit.json`：冻结来源、完整搜索计数、基线一致、固定包与预算、合法时间/最新观测检查。', '',
             '只有5位患者；R=50和每包50个日程降低机制与时间抽样误差，不替代新增患者。Oracle使用不可部署的未来信息；其差距仅反映固定动作包下时间可优化空间。', '']
    (output / 'REPORT-zh.md').write_text('\n'.join(text))
