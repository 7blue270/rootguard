"""Build an evidence-linked Chinese first-round report, without retuning test."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .analyze import frontiers, csv_rows, write_json

METHODS = ['B0', 'B2', 'B2p', 'M1-a', 'M1-g']


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def percent(value, digits=1):
    return '—' if value is None else f'{100 * value:.{digits}f}%'


def interval(v, pct=False):
    if v is None:
        return '—'
    f = percent if pct else number
    return f'[{f(v[0])}, {f(v[1])}]'


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
        '| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]


def build(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text())
    stats = json.loads((directory / 'public_stats.json').read_text())['MELD_NA']
    summary = json.loads((directory / 'summary.json').read_text())
    stability = json.loads((directory / 'selection_stability.json').read_text())
    targets = meta['targets']['MELD_NA']
    ff = frontiers(summary, meta['selections'], meta['targets'])
    csv_rows(directory / 'frontiers.csv', ff)
    selected = meta['selections']['MELD_NA']
    audit = json.loads((directory / 'audit.json').read_text()) if (directory / 'audit.json').exists() else {}
    test = [r for r in summary if r['split'] == 'test' and r['template'] == 'MELD_NA']
    budgets = sorted(meta['config']['budgets'])

    def curve(tau, method):
        parameter = selected[str(tau)][method]['parameter']
        return sorted([r for r in test if r['method'] == method and r['parameter'] == parameter], key=lambda r: r['B'])

    def at(tau, method, B):
        return next(r for r in curve(tau, method) if r['B'] == B)

    def frontier(tau, method, criterion='mean'):
        return next(r for r in ff if r['tau'] == tau and r['method'] == method and r['criterion'] == criterion)

    diagnostics, risk, main_rows, robust_rows = [], [], [], []
    for tau in targets:
        ref = frontier(tau, 'B2p')['B_min']
        B = ref if ref is not None else budgets[-1]
        a, b, g = (at(tau, m, B) for m in ['M1-a', 'B2p', 'M1-g'])
        diff = np.array(a['bootstrap_means']) - b['bootstrap_means']
        age_effect = np.array(g['bootstrap_means']) - a['bootstrap_means']
        row = {'tau': tau, 'comparison_B': B, 'q': selected[str(tau)]['M1-a']['parameter'],
               'MAE_M1a': a['mean'], 'MAE_B2p': b['mean'], 'MAE_M1g': g['mean'],
               'delta_MAE_M1a_minus_B2p': a['mean'] - b['mean'],
               'delta_MAE_ci_low': float(np.quantile(diff, .025)), 'delta_MAE_ci_high': float(np.quantile(diff, .975)),
               'age_effect_M1g_minus_M1a': g['mean'] - a['mean'],
               'age_effect_ci_low': float(np.quantile(age_effect, .025)), 'age_effect_ci_high': float(np.quantile(age_effect, .975)),
               'M1a_MissRate': a['miss_rate'], 'B2p_MissRate': b['miss_rate'],
               'M1a_refreshes': a['refresh_count'], 'B2p_refreshes': b['refresh_count'],
               'M1a_spent': a['spent'], 'B2p_spent': b['spent'], 'M1a_frac_age': a['frac_age'],
               'M1a_freeze_rate': a['early_budget_freeze'], 'B2p_freeze_rate': b['early_budget_freeze'],
               'M1a_publication_projection': a['publication_projection_fraction'],
               'B2p_publication_projection': b['publication_projection_fraction']}
        diagnostics.append(row)
        for method in METHODS:
            f = frontier(tau, method)
            bm = f['B_min']
            main_rows.append([number(tau), method, f['parameter'], number(bm), percent(f['saving_vs_B2p']),
                              interval(f.get('saving_ci_conditional_on_both_reached'), True),
                              percent(f.get('bootstrap_both_reached_fraction'))])
            r = frontier(tau, method, 'ucb95')
            robust_rows.append([number(tau), method, number(r['B_min']), percent(r['saving_vs_B2p'])])
            own = at(tau, method, bm if bm is not None else budgets[-1])
            risk.append({'tau': tau, 'method': method, 'at_B': own['B'], 'reached': bm is not None,
                         **{k: own[k] for k in ['mean', 'ci_low', 'ci_high', 'miss_rate', 'refresh_count', 'spent',
                              'remaining_fraction', 'frac_age', 'early_budget_freeze', 'post_stop_mae',
                              'cache_projection_fraction', 'publication_projection_fraction', 'input_clipping_fraction',
                              'input_clipping_mae', 'reason_gate', 'reason_age_force', 'reason_both', 'reason_periodic',
                              'reason_none', 'reason_empty', 'reason_frozen']}})
    csv_rows(directory / 'matched_budget_diagnostics.csv', diagnostics)
    csv_rows(directory / 'cost_risk_at_frontier.csv', risk)
    csv_rows(directory / 'test_curves.csv', [{k: v for k, v in r.items() if k != 'bootstrap_means'} for r in test])

    wins = ties = losses = censored = 0
    saving_values = []
    for tau in targets:
        a, b = frontier(tau, 'M1-a'), frontier(tau, 'B2p')
        if a['B_min'] is None and b['B_min'] is None:
            censored += 1
        elif a['B_min'] is not None and (b['B_min'] is None or a['B_min'] < b['B_min']):
            wins += 1
        elif a['B_min'] is not None and a['B_min'] == b['B_min']:
            ties += 1
        else:
            losses += 1
        if a['saving_vs_B2p'] is not None:
            saving_values.append(a['saving_vs_B2p'])
    if wins == len(targets):
        if len(saving_values) == len(targets):
            answer1 = f'M1-a 在三个目标的均值口径下均以更小网格预算达标；Saving 为 {percent(min(saving_values))}–{percent(max(saving_values))}。'
        elif saving_values:
            saving_text = percent(saving_values[0]) if len(saving_values) == 1 else f'{percent(min(saving_values))}–{percent(max(saving_values))}'
            answer1 = (f'网格点估计有利于M1-a：只有 {len(saving_values)} 个目标能精确比较Saving，为 '
                       f'{saving_text}；其余目标M1-a达标而B2p在960内未达标，不能给精确节省百分比。')
        else:
            answer1 = '三个目标均为M1-a达标而B2p在960内未达标，只能报告网格可达性优势，不能给精确Saving。'
    elif wins:
        answer1 = f'M1-a 只在 {wins}/{len(targets)} 个目标省预算，另有 {ties} 个持平、{losses} 个更差、{censored} 个双方未达标；不能称为全面打赢 B2p。'
    else:
        answer1 = f'M1-a 未在任何目标以更小网格预算打赢 B2p：{ties} 个持平、{losses} 个更差、{censored} 个双方未达标。'
    comparable = [frontier(t, 'M1-a') for t in targets if frontier(t, 'M1-a')['saving_vs_B2p'] is not None]
    if comparable and all(r.get('saving_ci_conditional_on_both_reached') is not None and
                          r['saving_ci_conditional_on_both_reached'][0] <= 0 <= r['saving_ci_conditional_on_both_reached'][1]
                          for r in comparable):
        answer1 += ' 可比较目标的Saving区间均跨0，尚不能说已稳定打赢。'
    no_age = all((r['M1a_frac_age'] or 0) == 0 and abs(r['age_effect_M1g_minus_M1a']) < 1e-10 for r in diagnostics)
    age_help = [r for r in diagnostics if r['age_effect_ci_low'] > 0]
    if no_age:
        answer2 = '同预算下 M1-a 与 M1-g 相同且无年龄强制刷新，观察到的差异来自损失门控的公开调度，而非年龄上限。'
        lengths = audit.get('test_measurement_batches_per_patient_sorted', [])
        chosen_qs = {float(selected[str(t)]['M1-a']['parameter'].split('=')[1]) for t in targets}
        minimum_G = min((np.floor(1 / c) for q in chosen_qs for c in stats['C'][str(q)] if c > 0), default=float('inf'))
        if lengths and max(lengths) < minimum_G:
            answer2 += f' 但测试轨迹只有{min(lengths)}–{max(lengths)}个更新批次，最小年龄阈值为{int(minimum_G)}，本轮年龄分支不可能触发，不能据此认定它一般无效。'
    elif age_help:
        answer2 = f'去掉年龄强制后，{len(age_help)}/{len(targets)} 个同预算对照的误差显著变大（配对区间高于0）；年龄策略有可测作用，但其占比不能仅由触发次数证明。'
    else:
        answer2 = 'M1-g 对照未给出稳定的年龄强制增益；损失与年龄作用须按各目标的消融差值判断，不能仅凭触发次数归因。'
    if not wins:
        answer2 = '没有已证实的获胜收益可归因；' + answer2
    more_miss = sum(r['M1a_MissRate'] is not None and r['B2p_MissRate'] is not None and r['M1a_MissRate'] > r['B2p_MissRate'] + 1e-12 for r in diagnostics)
    more_pub = sum(r['M1a_refreshes'] > r['B2p_refreshes'] + 1e-12 for r in diagnostics)
    freezes = any((r['M1a_freeze_rate'] or 0) > 0 for r in diagnostics)
    answer3 = (f'同预算诊断中，M1-a 在 {more_miss}/{len(targets)} 个目标有更多 Miss、在 {more_pub}/{len(targets)} 个目标有更多刷新；'
               + ('存在预算提前冻结，需同时看冻结后误差。' if freezes else '未出现预算提前冻结，差异主要体现在刷新选择、噪声和实际预算支出，而非提前耗尽。'))
    if more_pub == len(targets) and not more_miss and not freezes:
        answer3 = '收益伴随更频繁刷新和更高的同预算实际epsilon支出；Miss反而减少，未提前冻结，因此不是靠增加Miss，也不是仅仅更合理地跳过发布。'

    lines = ['# MELD-Na 首轮有限缓存实验报告', '', '日期：2026-11-10（Asia/Shanghai）。用途：第一轮机制验证，非论文最终结论。', '',
             '## 优先回答的三句话', '', f'1. **是否打赢：**{answer1}', f'2. **收益来源：**{answer2}', f'3. **付出什么：**{answer3}', '',
             '## 1. 本轮固定协议', '']
    counts = meta['patient_counts']['MELD_NA']
    lines += [f'- 数据：20 位患者，每人一次住院内的 40 轮有限轨迹；train/validation/test = {counts["train"]}/{counts["validation"]}/{counts["test"]} 位，测试集不是 20×R 个独立患者。',
              '- 化验值真实、查询日程合成；按最早合格住院及最早窗口选取，包含部分测量和测得未变值。根状态为最新已知测量。',
              f'- 方法：{", ".join(METHODS)}。B2/B2p 的 P 分别在 validation 调优；M1-a 的 q 在 validation 选择，M1-g 复用该 q。',
              f'- 预算：80–960，几何比值1.1，含上端960，共 {len(budgets)} 个点；未另加1.05细化。R={meta["config"]["repeats"]}，bootstrap={meta["config"]["bootstrap"]}。',
              '- 主显著阈值：归一化活动根单批次变化 >0.05。MissRate 是这些事件中未覆盖其全部显著变化根的比例，不是分类漏诊率。',
              '- MAE：最终数值节点在全40轮的平均绝对误差，含输入轮和冻结后轮，以原始最新已知根真值为参照。',
              '- 预算单位：标准归一化 Laplace epsilon；单会话全根、全发布时间相加，初始化也计费。',
              '- 核心数值实现：固定公开域；加噪根投影到域内后计算；固定旧pilot连续MELD钠修正公式，不含完整临床透析/舍入协议。',
              '- 20次重复采用独立噪声流、方法间同轮同根配对；公开调度与噪声独立，批量实现已与逐次参考实现逐项对齐。',
              f'- 逐批次完整账本记录前 {meta["config"].get("log_repeats", meta["config"]["repeats"])} 次重复；全部20次保留指标及支出。各重复共享相同公开动作/预算，未共享噪声。', '']
    val_b2 = [r for r in summary if r['split'] == 'validation' and r['method'] == 'B2']
    target_rows = []
    for ref in meta['config']['target_reference_budgets']:
        B = min(budgets, key=lambda v: abs(v - ref))
        rr = min([r for r in val_b2 if r['B'] == B], key=lambda r: (r['mean'], r['parameter']))
        target_rows.append([ref, number(B), number(rr['mean'], 6), rr['parameter']])
    lines += ['## 2. 三个目标及验证集选择', '',
              '目标来自 validation 的 B2 曲线：在指定参考预算最近的格点上，取候选周期中的最低均值。目标固定后才运行 test，不通过 test 调整目标。', '']
    lines += table(['参考B', '实际格点B', '生成τ', '该点B2周期'], target_rows)
    lines += [''] + table(['τ', 'B2 P*', 'B2p P*', 'M1-a q*（M1-g共用）'],
                         [[number(t), selected[str(t)]['B2']['parameter'], selected[str(t)]['B2p']['parameter'], selected[str(t)]['M1-a']['parameter']] for t in targets])
    lines += ['', '验证集可达性也需要保留：', '']
    lines += table(['τ', 'B2p validation达标', 'M1-a validation达标'],
                   [[number(t), str(selected[str(t)]['B2p']['validation_reached']),
                     str(selected[str(t)]['M1-a']['validation_reached'])] for t in targets])
    lines += ['', '候选全部未达标时，程序按预先定义的整条validation曲线平均误差回退选择，再按参数打破平局；选出q不等于已在validation满足该τ。', '']
    lines += ['', '公开根域与训练统计如下；C 是经验参数，不是概率上界。', '']
    qs = sorted({float(selected[str(t)]['M1-a']['parameter'].split('=')[1]) for t in targets})
    root_rows = []
    for i, root in enumerate(stats['roots']):
        vals = []
        for q in qs:
            c = stats['C'][str(q)][i]
            vals.append(f'{c:.6f} / ' + (str(int(np.floor(1 / c))) if c > 0 else '∞'))
        root_rows.append([root, str(stats['domains'][root]), number(stats['h'][i], 6), *vals])
    lines += table(['根', '固定域', 'ĥ', *[f'C / G（q={q}）' for q in qs]], root_rows)

    lines += ['', '## 3. 主结果：网格B_min与Saving', '',
              'Saving = 1 − B_min(method)/B_min(B2p)。负数表示比B2p更费预算；“—”表示网格内未达标或无法计算，不能读成0或真正无穷大。', '',
              '区间基于配对患者/机制bootstrap，条件为双方重采样都达标；必须同时看双方达标比例。B_min=80为网格左端，真实所需预算可能更低。', '']
    lines += table(['τ', '方法', '验证集配置', '均值B_min', 'Saving', 'Saving条件95%区间', '双方达标比例'], main_rows)
    lines += ['', '### UCB稳健性口径', '', '采用单侧95%误差上界独立判定达标，再单独重算Saving；没有混用均值口径预算。', '']
    lines += table(['τ', '方法', 'UCB B_min', 'UCB Saving'], robust_rows)
    lines += ['', '### 强基线B2与静态B0不能省略', '',
              'M1-a相对B2p的优势没有转化为对B2的优势。下表仅比较网格点估计；正的额外预算比例表示M1-a更费预算。', '']
    strong_rows = []
    for tau in targets:
        a, b, static = (frontier(tau, m)['B_min'] for m in ['M1-a', 'B2', 'B0'])
        extra = a / b - 1 if a is not None and b is not None else None
        strong_rows.append([number(tau), number(b), number(a), percent(extra), number(static)])
    lines += table(['τ', 'B2 B_min', 'M1-a B_min', 'M1-a相对B2额外预算', 'B0 B_min'], strong_rows)
    lines += ['', '在B=960、P=12时，本测试队列的周期方法每人仅刷新1次。B2固定刷新额度为480/⌊39/12⌋=160；B2p额度为min(0.125×480,40)=40。首次刷新前所有根的配额比例相同，配额因子归一化后抵消。因此本对照显示，当前因果h/单次上限让周期发布更弱；不能将超过B2p等同于超过充分利用预算的周期方法。',
              '', '三个τ来自同一批5名测试患者、同一组曲线，不是三组独立验证；点估计首次过线也不表示后续所有更高预算都达标。', '']

    lines += ['', '## 4. 同预算对照：收益来源与代价', '',
              '各τ优先在B2p的均值B_min比较所有策略；若B2p未达标则用960。ΔMAE=M1-a−B2p，负数有利于M1-a；年龄效应=M1-g−M1-a，正数表示保留年龄规则有利。', '']
    lines += table(['τ', '比较B', 'M1-a MAE', 'B2p MAE', 'ΔMAE与95%区间', '年龄效应与95%区间'],
                   [[number(r['tau']), number(r['comparison_B']), number(r['MAE_M1a']), number(r['MAE_B2p']),
                     number(r['delta_MAE_M1a_minus_B2p']) + ' ' + interval([r['delta_MAE_ci_low'], r['delta_MAE_ci_high']]),
                     number(r['age_effect_M1g_minus_M1a']) + ' ' + interval([r['age_effect_ci_low'], r['age_effect_ci_high']])] for r in diagnostics])
    for tau in targets:
        d = next(r for r in diagnostics if r['tau'] == tau)
        lines += ['', f'### τ={tau:.6f}，共同B={d["comparison_B"]:.3f}', '']
        lines += table(['方法', 'MAE', 'MissRate', '刷新次数', '实际ε支出', '预算使用率', '发布投影率', '缓存投影率', '预算提前冻结率', '冻结后MAE'],
                       [[m, number(at(tau, m, d['comparison_B'])['mean']), percent(at(tau, m, d['comparison_B'])['miss_rate']),
                         number(at(tau, m, d['comparison_B'])['refresh_count'], 2), number(at(tau, m, d['comparison_B'])['spent']),
                         percent(at(tau, m, d['comparison_B'])['spent'] / d['comparison_B']),
                         percent(at(tau, m, d['comparison_B'])['publication_projection_fraction']),
                         percent(at(tau, m, d['comparison_B'])['cache_projection_fraction']),
                         percent(at(tau, m, d['comparison_B'])['early_budget_freeze']),
                         number(at(tau, m, d['comparison_B'])['post_stop_mae'])] for m in METHODS])
        lines += ['', '刷新原因及跳过次数（每条轨迹的平均次数；不含初始化）：', '']
        lines += table(['方法', '仅损失gate', '仅年龄', '两者同时', '周期', '跳过none', '空集合', '冻结跳过'],
                       [[m, *[number(at(tau, m, d['comparison_B'])['reason_' + k], 2) for k in
                              ['gate', 'age_force', 'both', 'periodic', 'none', 'empty', 'frozen']]] for m in METHODS])
    lines += ['', '不同方法各自在达标预算处的全部风险/成本指标另见 [cost_risk_at_frontier.csv](cost_risk_at_frontier.csv)。', '',
              '## 5. 指标解释与证据边界', '',
              '- **投影率有两种：**发布投影率是实际发布根值中越出[0,1]的比例；缓存投影率是所有轮次使用的根缓存越界比例，重复使用同一个缓存会重复计入后者。',
              '- 输入超界也单独记录。原始输入截断引入的任务误差不能归为刷新收益，详见成本表中的input_clipping_fraction/input_clipping_mae。',
              '- 动态方法未预算冻结时，冻结后MAE为空，不能记为0。B0从初始化后永不更新，其post_stop_mae描述静态缓存后续误差，不表示预算提前耗尽。',
              '- frac_age把both也计入；它是行为描述，主要归因依赖M1-g的同预算误差差值和区间。',
              '- 本轮P/q由validation选择；选择稳定性见 [selection_stability.json](selection_stability.json)。若选择频率分散，报告的是当前划分上的pilot结果。',
              '- 只有5位test患者；R=20用于降低机制噪声误差，不能增加独立患者数。宽区间、网格删失与左端截断均应保留。',
              '- 本轮不根据结果扩预算、重选τ、改显著阈值或改数据窗口。静态B0也按实际可达性参与对照。', '',
              '## 6. 文件与复现', '',
              '- [固定配置](config.json)、[患者划分](splits.json)、[公开训练统计](public_stats.json)、[目标](targets.json)、[选择](selections.json)。',
              '- [逐机制指标](metrics.jsonl)、[公共调度及发布账本](ledger.jsonl)、[全部测试曲线](test_curves.csv)。',
              '- [主前沿](frontiers.csv)、[同预算诊断](matched_budget_diagnostics.csv)、[达标处风险成本](cost_risk_at_frontier.csv)。',
              '- [源码、输入与方案哈希](metadata.json)。日志与数据保留本地，受.gitignore规则保护。', '',
              '```bash', '/root/miniconda3/envs/rootguard/bin/python -B -m experiments.workflow_extensions.finite_cache.run \\',
              '  --config experiments/workflow_extensions/finite_cache/configs/round1_meld.json \\',
              '  --output experiments/workflow_extensions/outputs/finite_cache/round1_meld_repeat_new', '```', '']
    # Insert experiment-specific audit findings before reproducibility details.
    insert_at = lines.index('## 6. 文件与复现')
    additions = []
    revision = meta.get('evaluation_revision')
    if revision:
        ev = revision['cohort_event_counts']['test']
        additions += ['### 严格0.05事件阈值的数值边界', '',
                      f'测试集真实显著事件为 **{ev["after"]} 个**，不是20次随机重复产生的新独立事件。原始双精度直接比较误计为{ev["before"]}个，现排除恰等于0.05的浮点边界误计；阈值仍为严格大于0.05，容差为1e-12。',
                      '仅重算显著事件与Miss计数；所有MAE、噪声发布、预算、参数选择与三个τ逐字段保持不变。原始指标保存在metrics_fp_original.jsonl，修订记录见 [evaluation_revision.json](evaluation_revision.json)。', '']
    if audit.get('noiseless_static_reference_mae') is not None:
        additions += [f'无噪声冻结初始真值的陈旧误差参考为 **{audit["noiseless_static_reference_mae"]:.3f} 分**。这是本测试队列的参考，不是所有有限预算下误差的严格下界；也不能沿用旧跨住院轨迹约6分的饱和值。', '']
    # Distinguish cap savings from actual average privacy charges at own B_min.
    for tau in targets:
        a = next(r for r in risk if r['tau'] == tau and r['method'] == 'M1-a')
        b = next(r for r in risk if r['tau'] == tau and r['method'] == 'B2p')
        if a['reached'] and b['reached']:
            additions += [f'在可精确比较的τ={tau:.6f}处，M1-a/B2p各自达标时的实际平均ε支出为 **{a["spent"]:.3f}/{b["spent"]:.3f}**，平均实际支出降低 **{100*(1-a["spent"]/b["spent"]):.1f}%**；它与预算上限B的Saving是两个口径。', '']
    additions += ['### 本轮之后的优先判断', '',
                  '先保留“动态维护在严格目标有作用”的观察，但不宣告门控机制已成熟：相对B2p的点估计优势不稳定，B2在三个目标上更好或持平，年龄分支本轮未有效受检。下一轮优先检查因果h/单次额度上限与强周期的预算使用差异；本轮不追加改参或新方法实验。', '']
    lines[insert_at:insert_at] = additions
    (directory / 'REPORT-zh.md').write_text('\n'.join(lines))
    # Reporting is postprocessing; preserve the original execution-source snapshot.
    report_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta.setdefault('postprocessing_code_sha256', {})[str(Path(__file__).relative_to(Path(__file__).resolve().parents[3]))] = report_hash
    write_json(directory / 'metadata.json', meta)
    write_json(directory / 'answers.json', {'answers': [answer1, answer2, answer3], 'wins': wins, 'ties': ties,
                                            'losses': losses, 'both_censored': censored,
                                            'finite_saving_comparisons': len(saving_values),
                                            'report_script_sha256': report_hash})
    print(json.dumps({'report': str(directory / 'REPORT-zh.md'), 'answers': [answer1, answer2, answer3]}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    build(a.directory)


if __name__ == '__main__':
    main()
