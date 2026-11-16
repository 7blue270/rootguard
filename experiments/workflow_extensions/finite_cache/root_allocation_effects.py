"""Paired factorial contrasts and endpoint-correct Shapley attribution."""
from collections import defaultdict
from itertools import permutations, product
import math
from pathlib import Path
import numpy as np

from .analyze import csv_rows, write_json
from .matched_analysis import paired_interval
from .root_allocation_factorial import CELLS, TIMES, ROOTS, ALLOCS, label


def bit_cell(bits):
    return ('M1_time' if bits[0] else 'uniform_time',
            'M1_roots' if bits[1] else 'uniform_roots', 'quota' if bits[2] else 'sqrt_h')


def effect_definitions():
    result = []
    def add(name, family, terms, **extra):
        coeff = defaultdict(float)
        for cell, value in terms:
            coeff[label(cell)] += value
        result.append(dict(effect=name, family=family, coefficients=dict(coeff), **extra))
    for t, a in product(TIMES, ALLOCS):
        add(f'S|T={t}|A={a}', 'root_conditional', [((t, ROOTS[0], a), 1), ((t, ROOTS[1], a), -1)])
    for t, s in product(TIMES, ROOTS):
        add(f'A|T={t}|S={s}', 'allocation_conditional', [((t, s, ALLOCS[0]), 1), ((t, s, ALLOCS[1]), -1)])
    for s, a in product(ROOTS, ALLOCS):
        add(f'T|S={s}|A={a}', 'time_conditional', [((TIMES[0], s, a), 1), ((TIMES[1], s, a), -1)])
    add('S_main', 'root_main', [((t, s, a), (.25 if s == ROOTS[0] else -.25)) for t, s, a in CELLS])
    add('A_main', 'allocation_main', [((t, s, a), (.25 if a == ALLOCS[0] else -.25)) for t, s, a in CELLS])
    add('T_main', 'time_main', [((t, s, a), (.25 if t == TIMES[0] else -.25)) for t, s, a in CELLS])
    for a in ALLOCS:
        add('S_mean_at_' + a, 'root_allocation_stratum',
            [((t, s, a), (.5 if s == ROOTS[0] else -.5)) for t, s in product(TIMES, ROOTS)])
    for t in TIMES:
        add('S_A_interaction|T=' + t, 'root_allocation_interaction_conditional',
            [((t, s, a), (1 if (s == ROOTS[0]) == (a == ALLOCS[0]) else -1))
             for s, a in product(ROOTS, ALLOCS)])
    add('S_A_interaction_mean', 'root_allocation_interaction',
        [((t, s, a), (.5 if (s == ROOTS[0]) == (a == ALLOCS[0]) else -.5)) for t, s, a in CELLS])
    add('T_S_A_interaction', 'three_way_interaction',
        [((t, s, a), (1 if (t == TIMES[0]) == ((s == ROOTS[0]) == (a == ALLOCS[0])) else -1))
         for t, s, a in CELLS])
    add('original_M1_minus_matched_uniform', 'original_total_gap',
        [(('M1_time', 'M1_roots', 'quota'), 1), (('uniform_time', 'uniform_roots', 'quota'), -1)])
    add('full_cube_M1Q_minus_uniformUH', 'full_cube_total_gap', [(bit_cell((1, 1, 1)), 1), (bit_cell((0, 0, 0)), -1)])
    # Exact uniform average over ALL factor orderings, not one chosen path.
    phi = [defaultdict(float) for _ in range(3)]
    for order in permutations(range(3)):
        state = [0, 0, 0]
        for factor in order:
            previous = bit_cell(state)
            state[factor] = 1
            phi[factor][bit_cell(state)] += 1 / 6
            phi[factor][previous] -= 1 / 6
    for i, factor in enumerate(['T', 'S', 'A']):
        add('full_shapley_' + factor, 'three_factor_shapley_full_cube', list(phi[i].items()))
        terms = list(phi[i].items())
        if i == 2:
            terms += [(bit_cell((0, 0, 1)), -1), (bit_cell((0, 0, 0)), 1)]
        add('contrast_shapley_' + factor, 'common_reference_shapley_difference', terms)
    direct = [defaultdict(float) for _ in range(2)]
    for order in permutations(range(2)):
        state = [0, 0, 1]
        for factor in order:
            previous = bit_cell(state)
            state[factor] = 1
            direct[factor][bit_cell(state)] += .5
            direct[factor][previous] -= .5
    for factor, coef in zip(['T', 'S'], direct):
        add('direct_quota_shapley_' + factor, 'fixed_quota_direct_attribution', list(coef.items()))
    add('direct_quota_shapley_A', 'fixed_quota_direct_attribution', [])
    return result


def weight_vector(definition):
    return np.array([definition['coefficients'].get(label(c), 0.) for c in CELLS])


def make_panel(rows, metric):
    lookup = defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:
            lookup[(r['subject_id'], r['repeat'], (r['T'], r['S'], r['A']))].append(r[metric])
    ids = sorted({r['subject_id'] for r in rows})
    repeats = sorted({r['repeat'] for r in rows})
    if not all(lookup[(sid, repeat, cell)] for sid in ids for repeat in repeats for cell in CELLS):
        return None
    return np.array([[[np.mean(lookup[(sid, repeat, cell)]) for cell in CELLS] for repeat in repeats] for sid in ids])


def analyze(rows, output, protocol):
    output = Path(output)
    bycase = defaultdict(set)
    for row in rows:
        bycase[(row['trajectory_id'], row['B'], row['repeat'])].add((row['T'], row['S'], row['A']))
    assert all(cells == set(CELLS) for cells in bycase.values())
    definitions = effect_definitions()
    write_json(output / 'effect_definitions.json', definitions)
    scopes = [('primary_budget_mean', None, rows)] + [('prespecified_budget', B, [r for r in rows if r['B'] == B])
                                                       for B in protocol['executed_budgets']]
    effects, variants, attribution = [], [], []
    main_metrics = ['mae', 'p95_abs_error', 'workload_loss', 'query_loss', 'publication_projection_fraction',
                    'cache_projection_fraction', 'miss_rate', 'aoi_mean']
    identity_errors = []
    for scope, B, rr in scopes:
        for metric in main_metrics:
            panel = make_panel(rr, metric)
            if panel is None:
                continue
            for definition in definitions:
                values = panel @ weight_vector(definition)
                ci = paired_interval(values, protocol['bootstrap'], protocol['seed'])
                r = {'scope': scope, 'B': B, 'metric': metric, 'effect': definition['effect'], 'family': definition['family'],
                     'n_patients': panel.shape[0], 'repeats': panel.shape[1], 'mean': float(values.mean())}
                for name in ['patient', 'hierarchical', 'mechanism']:
                    r[name + '_ci_low'], r[name + '_ci_high'] = ci[name + '_ci']
                r.update({k: ci[k] for k in ['variance_mechanism_mean', 'variance_between_mean']})
                r['direction'] = ('negative' if r['patient_ci_high'] < 0 and r['hierarchical_ci_high'] < 0 else
                                  'positive' if r['patient_ci_low'] > 0 and r['hierarchical_ci_low'] > 0 else 'insufficient_evidence')
                effects.append(r)
        # Every diagnostic is reported per cell, including per-root publications/budget.
        excluded = {'subject_id', 'trajectory_id', 'template', 'B', 'repeat', 'parameter', 'method', 'T', 'S', 'A'}
        names = [name for name in rr[0] if name not in excluded]
        for cell in CELLS:
            vv = [r for r in rr if (r['T'], r['S'], r['A']) == cell]
            variant = {'scope': scope, 'B': B, 'T': cell[0], 'S': cell[1], 'A': cell[2]}
            for name in names:
                patient_means = []
                for sid in sorted({r['subject_id'] for r in vv}):
                    nums = [r[name] for r in vv if r['subject_id'] == sid and r.get(name) is not None]
                    if nums:
                        patient_means.append(float(np.mean(nums)))
                variant[name] = float(np.mean(patient_means)) if patient_means else None
            variants.append(variant)
        mae = make_panel(rr, 'mae')
        values_by_name = {d['effect']: mae @ weight_vector(d) for d in definitions}
        total = values_by_name['original_M1_minus_matched_uniform']
        full = values_by_name['full_cube_M1Q_minus_uniformUH']
        for prefix, expected in [('direct_quota_shapley_', total), ('contrast_shapley_', total), ('full_shapley_', full)]:
            sum_ = sum(values_by_name[prefix + f] for f in ['T', 'S', 'A'])
            error = float(np.max(np.abs(sum_ - expected)))
            assert error < 1e-10
            identity_errors.append(error)
        n, R, _ = mae.shape
        rng = np.random.default_rng(protocol['seed'])
        pi = rng.integers(n, size=(protocol['bootstrap'], n))
        ri = rng.integers(R, size=(protocol['bootstrap'], n, R))
        for prefix, endpoint in [('direct_quota_shapley_', 'original_M1_minus_matched_uniform'),
                                 ('contrast_shapley_', 'original_M1_minus_matched_uniform'),
                                 ('full_shapley_', 'full_cube_M1Q_minus_uniformUH')]:
            denom = values_by_name[endpoint]
            denominator_samples = denom[pi[:, :, None], ri].mean(axis=(1, 2))
            good = np.abs(denominator_samples) > 1e-12
            for factor in ['T', 'S', 'A']:
                values = values_by_name[prefix + factor]
                sample = values[pi[:, :, None], ri].mean(axis=(1, 2))
                shares = sample[good] / denominator_samples[good]
                attribution.append({'scope': scope, 'B': B, 'scheme': prefix[:-1], 'factor': factor,
                                    'contribution': float(values.mean()), 'total_gap': float(denom.mean()),
                                    'share': float(values.mean() / denom.mean()) if abs(denom.mean()) > 1e-12 else None,
                                    'share_ci_low': float(np.quantile(shares, .025)) if len(shares) else None,
                                    'share_ci_high': float(np.quantile(shares, .975)) if len(shares) else None,
                                    'share_defined_fraction': float(good.mean())})
    csv_rows(output / 'effect_summary.csv', effects)
    csv_rows(output / 'variant_summary.csv', variants)
    csv_rows(output / 'shapley_attribution.csv', attribution)
    write_json(output / 'attribution_audit.json', {'passed': True, 'maximum_identity_error': max(identity_errors)})
    patient_effects = []
    for sid in sorted({r['subject_id'] for r in rows}):
        pp = make_panel([r for r in rows if r['subject_id'] == sid], 'mae')
        for d in definitions:
            patient_effects.append({'subject_id': sid, 'effect': d['effect'], 'mean': float((pp @ weight_vector(d)).mean())})
    csv_rows(output / 'patient_effects.csv', patient_effects)
    write_report(output, effects, variants, attribution, protocol)


def write_report(output, effects, variants, attribution, protocol):
    primary = {r['effect']: r for r in effects if r['metric'] == 'mae' and r['scope'] == 'primary_budget_mean'}
    fmt = lambda v: 'NA' if v is None else f'{v:.4f}'
    ci = lambda r, name: f"[{fmt(r[name + '_ci_low'])}, {fmt(r[name + '_ci_high'])}]"
    table_name = lambda name: name.replace('|', ' / ')
    direction = {'negative': '显著为负', 'positive': '显著为正', 'insufficient_evidence': '证据不足'}
    s, a, interaction = (primary[k] for k in ['S_main', 'A_main', 'S_A_interaction_mean'])
    text = ['# 根集合 × 逐根预算因素拆分实验', '', '## 主要判断', '',
            f"根集合主效应 M1_roots−uniform_roots={fmt(s['mean'])}（{direction[s['direction']]}）；"
            f"分配主效应 quota−sqrt_h={fmt(a['mean'])}（{direction[a['direction']]}）；"
            f"根集合×分配交互={fmt(interaction['mean'])}（{direction[interaction['direction']]}）。", '',
            '负MAE差表示前者更好。主效应分别平均另两个因素；交互是quota下根集合差减去sqrt_h下根集合差，再平均两个时间水平。'
            '条件效应与原始端点差另行报告，不以主效应替代特定方法比较。', '']
    if s['direction'] == 'negative':
        text += ['根动作包具有正面证据，后续可优先设计并验证显式根选择模块。', '']
    elif s['direction'] == 'positive':
        text += ['根集合影响显著，但方向有利于uniform_roots，不能宣称M1当前选根更好。', '']
    if a['direction'] == 'negative':
        text += ['quota分配有正面证据，可重点研究逐根预算分配。', '']
    elif a['direction'] == 'positive':
        text += ['quota分配显著劣于sqrt_h，优化方向应包含简化或替换软配额，而不是继续增加quota复杂度。', '']
    if interaction['direction'] != 'insufficient_evidence':
        text += ['根集合与分配存在交互，应将选根与分配联合考察；交互为负表示quota增强M1_roots相对优势，为正表示削弱，不能将任意显著交互都称为正向协同。', '']
    if all(r['direction'] == 'insufficient_evidence' for r in [s, a, interaction]):
        text += ['三个预指定核心效应均未排除0，暂缓增加机制复杂度，优先扩患者与任务。', '']
    text += ['## 复现、冻结与定义', '',
             f"- 纯时间来源：`{protocol['pure_source']}`；匹配均匀来源：`{protocol['matched_source']}`。",
             f"- R={protocol['repeats']}，bootstrap={protocol['bootstrap']}；精确预算沿用原网格：{', '.join(str(b) for b in protocol['executed_budgets'])}。",
             '- 冻结数据划分、公式、查询、公开统计、M1参数、初始化、K、有序动作编号及每次总epsilon。八组合仅改变T/S/A，不重新门控或用test选参。',
             '- M1_roots为M1原动作根；uniform_roots为原匹配均匀基线对应动作实际测量且发布的根。两种时间表完全复用历史日程。',
             '- 跨时间执行时使用截至当批次已观测的最新根值；不要求本批次重测全部根，不使用未来观测。',
             '- quota使用冻结gamma；sqrt_h仅令gamma=0。各S/A有独立软配额账本，初始化先计费；该账本不依赖时间/真值，因此两个时间重放具有相同逐根epsilon。',
             '- 原Cache.allocate/_publish用于账本和发布；相同投影/覆盖更新。单位Laplace按(seed,subject,trajectory,B,repeat,action_index,root)配对，未发布根不使用。',
             '- 三条预指定复现：(M1_time,M1_roots,quota)→M1；(uniform_time,M1_roots,quota)→纯时间均匀；(uniform_time,uniform_roots,quota)→原匹配均匀。容差1e-10，逐根epsilon/值另审计。',
             '- 第3项原消息公式不完整，本轮明确采用 I(T)=[Y(T,S_M,Q)−Y(T,S_U,Q)]−[Y(T,S_M,H)−Y(T,S_U,H)]，Q=quota、H=sqrt_h。',
             '- 主检验：预指定预算患者内平均，再患者均权；患者级配对、患者＋机制两层、固定患者机制区间均保存。逐预算/条件效应区间为名义点态诊断；未按test挑预算或效应。', '',
             '## 八个组合：预指定预算聚合', '',
             '|时间|根集合|分配|MAE|p95|发布投影率|MissRate|AoI|根集合平均大小|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    vv = [r for r in variants if r['scope'] == 'primary_budget_mean']
    for r in vv:
        text.append(f"|{r['T']}|{r['S']}|{r['A']}|{fmt(r['mae'])}|{fmt(r['p95_abs_error'])}|{fmt(r['publication_projection_fraction'])}|"
                    f"{fmt(r['miss_rate'])}|{fmt(r['aoi_mean'])}|{fmt(r['root_set_size_mean'])}|")
    text += ['', '根集合效应包含根数量变化带来的每根预算集中，不能把它直接解读为在根数量/逐根预算固定时的选根质量。'
             'quota−sqrt_h才是在固定根动作包及总epsilon下的分配规则比较。', '',
             '## 配对主效应及交互', '', '|效应|MAE差|患者95% CI|两层95% CI|判断|', '|---|---:|---|---|---|']
    selected = ['S_main', 'S_mean_at_quota', 'S_mean_at_sqrt_h', 'A_main', 'T_main', 'S_A_interaction_mean',
                'S_A_interaction|T=M1_time', 'S_A_interaction|T=uniform_time', 'original_M1_minus_matched_uniform']
    for name in selected:
        r = primary[name]
        text.append(f"|{table_name(name)}|{fmt(r['mean'])}|{ci(r, 'patient')}|{ci(r, 'hierarchical')}|{direction[r['direction']]}|")
    text += ['', '## 条件效应', '', '|效应|MAE差|患者95% CI|两层95% CI|', '|---|---:|---|---|']
    for name, r in primary.items():
        if r['family'] in ['root_conditional', 'allocation_conditional', 'time_conditional']:
            text.append(f"|{table_name(name)}|{fmt(r['mean'])}|{ci(r, 'patient')}|{ci(r, 'hierarchical')}|")
    text += ['', '## 原方法总差与Shapley归因', '',
             '原M1和原匹配均匀都采用quota。故原端点直接Shapley只能改变T和S，直接A贡献为0。'
             '三因素全立方Shapley的起点则是(uniform_time,uniform_roots,sqrt_h)，它分解的不是原方法差。', '',
             '另报告共同sqrt_h基点的Shapley差分：对两个原方法分别相对共同基点归因，再相减。'
             '此时A贡献=全立方phi_A−[Y(U,U,Q)−Y(U,U,H)]，反映分配收益随T/S变化的交互，不是原端点切换了分配规则。', '',
             '|方案|因素|MAE贡献|占对应总差比例|比例两层95% CI|', '|---|---|---:|---:|---|']
    for r in attribution:
        if r['scope'] == 'primary_budget_mean':
            text.append(f"|{r['scheme']}|{r['factor']}|{fmt(r['contribution'])}|{fmt(r['share'])}|[{fmt(r['share_ci_low'])}, {fmt(r['share_ci_high'])}]|")
    text += ['', '所有路径均匀平均；直接归因及共同基点差分分别严格加和为原M1−匹配均匀，全立方归因严格加和为不同起点的全立方总差。'
             '贡献比例不截断负值，不强迫全部因素表现为正向收益；分母为0记NA。', '',
             '## 每根发布与累计epsilon', '',
             '|时间|根集合|分配|根|后续发布次数|累计epsilon（含初始化）|', '|---|---|---|---|---:|---:|']
    for r in vv:
        for root in ['bil', 'inr', 'cr', 'na']:
            text.append(f"|{r['T']}|{r['S']}|{r['A']}|{root}|{fmt(r['root_releases__'+root])}|{fmt(r['root_total_epsilon__'+root])}|")
    text += ['', '## 刷新收益及根重叠', '',
             '|时间|根集合|分配|根Jaccard|共同根/当前包|每次即时下降|每epsilon即时下降|每次长期平均下降|每epsilon长期累计下降|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in vv:
        text.append(f"|{r['T']}|{r['S']}|{r['A']}|{fmt(r['root_jaccard_mean'])}|{fmt(r['common_root_fraction'])}|"
                    f"{fmt(r['immediate_error_drop_per_refresh'])}|{fmt(r['immediate_error_drop_per_epsilon'])}|"
                    f"{fmt(r['long_term_error_drop_per_refresh'])}|{fmt(r['long_term_cumulative_drop_per_epsilon'])}|")
    text += ['', '下降为正表示发布有益。长期量沿用纯时间实验：仅跳过该次发布，保持后续动作、输入与噪声不变，比较该轮至T。'
             '动作长期效应可重叠，不能相加解释总体优势。Jaccard为|两包交集|/|两包并集|，共同根比例另按当前包大小归一化。', '',
             '## 文件与局限', '',
             '- `factorial_metrics.jsonl`：八组合完整重复指标及逐根次数/累计epsilon。',
             '- `factorial_actions.jsonl`：逐次根、epsilon、单位噪声、发布值、输入/观测来源、分配权重和独立软配额账本。',
             '- `effect_summary.csv`：主效应、条件效应、交互及归因的患者/两层/机制区间；`effect_definitions.json`列出全部线性系数。',
             '- `variant_summary.csv`、`shapley_attribution.csv`、`patient_effects.csv`：各组合、归因和患者异质性。',
             '- `audit.json`、`attribution_audit.json`：基线复现、总支出<1e-10、合法时间/值及Shapley加和断言。', '',
             '本轮只有5位test患者、一个任务，且test已用于前轮分析。结果用于机制诊断，不能直接外推为总体因果结论或论文级确认；不据此在本轮修改门控或调参。', '']
    (output / 'REPORT-zh.md').write_text('\n'.join(text))
