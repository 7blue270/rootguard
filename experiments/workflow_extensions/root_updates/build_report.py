"""Audit saved runs and produce the complete Chinese exploration report."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from mechanisms import original, run
from data import real


def table(rows, fields):
    lines = ['| '+' | '.join(label for _,label in fields)+' |',
             '| '+' | '.join('---' for _ in fields)+' |']
    for row in rows:
        values = []
        for key,_ in fields:
            value = row.get(key)
            if isinstance(value, float): value = f'{value:.3f}'
            if isinstance(value, list): value = '['+', '.join(f'{v:.3f}' for v in value)+']'
            values.append('—' if value is None else str(value))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    default = Path(__file__).resolve().parents[1]/'outputs/root_updates'
    ap.add_argument('--base', type=Path, default=default)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(); base = args.base.resolve()
    out = args.out or base/'report_20261012'; out.mkdir(parents=True, exist_ok=True)
    names = ['exploration_20261012','standard_ldp_20261012','confirmation_20261012']
    results = {}; metas = {}; audit = {}; evidence = []
    for name in names:
        path = base/name
        rows = json.loads((path/'analysis.json').read_text())
        meta = json.loads((path/'metadata.json').read_text())
        mismatches = [p for p,h in meta['sources_sha256'].items()
                      if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
        certified = [r for r in rows if r['certified']]
        assert not mismatches, mismatches
        assert len(rows)==meta['cell_count']
        assert all(math.isfinite(r['wmape']) and math.isfinite(r['rce']) for r in certified)
        assert all(r['spent_max']<=r['actual_budget_bound']+1e-8 for r in certified)
        refs = {(r['dataset'],r['budget_label'],r['method']):r for r in rows}
        for r in rows:
            if '_no_updates_' in r['dataset'] and r['method']=='root_opt':
                a=refs[r['dataset'],r['budget_label'],'static']
                assert r['wmape']==a['wmape'] and r['rce']==a['rce']
        audit[name] = dict(cells=len(rows), certified_cells=len(certified),
             source_hashes_checked=len(meta['sources_sha256']), source_mismatches=mismatches,
             finite_certified_metrics=True, budget_cap_pass=True,
             uncertified_nan_cells=sum(not math.isfinite(r['wmape']) for r in rows),
             elapsed_seconds=meta['elapsed_seconds'], task_evaluations=meta['task_evaluations'])
        results[name]=rows; metas[name]=meta
        for r in rows:
            if r['same_budget']:
                evidence.append(dict(run=name, **{k:r[k] for k in ['dataset','method','budget_label','privacy_track',
                    'wmape','rce','wmape_ci95','rce_ci95','change_miss_rate','false_change_rate',
                    'delta_wmape_vs_root_opt_ci95','delta_rce_vs_static_ci95']}))
    with (out/'evidence.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(evidence[0]));w.writeheader();w.writerows(evidence)
    a = results[names[0]]; s=results[names[1]]; c=results[names[2]]
    change_rows=[]
    # Correctness at actual transition times, supplementing the change-signal metric.
    # Reproduce this cell from its seed in float64 instead of thresholding the
    # float32 archive, which can round values onto a class boundary.
    d = real(8,40)
    cfg = json.loads((base/names[0]/'config.json').read_text())
    di = list(metas[names[0]]['datasets']).index('real_ANEMIA_H8')
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'],di]))
    u=rng.random((40,cfg['repeats'],8,3));lap=rng.laplace(size=(40,cfg['repeats'],8))
    truth=original.compute('ANEMIA',d['x']);gt=original.classes('ANEMIA',truth)
    for method in ['static','full_opt','period2','period4','predictive30','fixed_per_version']:
        B=2.4 if method=='fixed_per_version' else .3
        implementation='full_opt' if method=='fixed_per_version' else method
        period=2 if method=='period2' else 4 if method=='period4' else 1
        z,_=run('ANEMIA',d['x'],d['plan'],d['mu'],B,implementation,u,lap,period=period)
        pc=original.classes('ANEMIA',original.compute('ANEMIA',z))
        saved=next(r for r in a if r['dataset']=='real_ANEMIA_H8' and r['budget_label']==.3 and r['method']==method)
        assert abs(100*(pc!=gt[:,None,:]).mean()-saved['rce'])<1e-10
        changed=gt[:,1:]!=gt[:,:-1]
        change_rows.append(dict(method=method,transition_rce=float(100*(pc[:,:,1:]!=gt[:,None,1:])[np.broadcast_to(changed[:,None,:],pc[:,:,1:].shape)].mean())))
    (out/'validation.json').write_text(json.dumps(audit,indent=2))
    (out/'transition_time_checks.json').write_text(json.dumps(change_rows,indent=2))
    def select(rows,dataset,budget): return [r for r in rows if r['dataset']==dataset and r['budget_label']==budget]
    realrows=select(a,'real_ANEMIA_H8',.3)
    budgetrows=[r for r in a if r['dataset']=='real_ANEMIA_H8' and r['method'] in ['static','full_opt','period4']]
    fields=[('method','方法'),('wmape','wMAPE %'),('rce','RCE %'),('change_miss_rate','变化信号漏报 %'),('false_change_rate','稳定期误报 %'),('spent_mean','平均费用 B')]
    fair=[r for r in realrows if r['same_budget']]
    cohort=[dict(run=n, cells=audit[n]['cells'], draws=audit[n]['task_evaluations'], seconds=audit[n]['elapsed_seconds']) for n in names]
    matched=[r for r in c if r['source']=='real' and r['budget_label']==.3 and r['method'] in ['static','full_opt','period2','period4']]
    synthetic=[]
    for ds in ['syn_FIB4_sparse_H8_a0.05','syn_HOMA_sparse_H8_a0.05','syn_ANEMIA_jump_H8_a0.05','syn_ANEMIA_constant_H8_a0.0','syn_ANEMIA_irrelevant_H8_a0.05','syn_ANEMIA_correlated_H8_a0.05']:
        synthetic += [r for r in select(a,ds,.3) if r['method'] in ['static','full_opt','root_opt','period4']]
    standard=[r for r in s if r['dataset']=='real_ANEMIA_H8' and r['method'] in ['static','full_opt','period4']]
    jump=[r for r in select(c,'syn_confirm_ANEMIA_jump_H8_a0.05',.3) if r['method'] in ['static','root_opt','period4']]
    attack=list(csv.DictReader((base/'attack_20261012/reconstruction.csv').open()))
    attack=[dict(H=int(r['H']),method=r['method'],charged_budget=float(r['charged_budget']),mae_index=float(r['mae_index']),within_5_index_pct=float(r['within_5_index_pct'])) for r in attack if float(r['B'])==.3]
    meta=metas[names[0]]
    realaudit=meta['datasets']['real_ANEMIA_H8']
    doc=f'''# RootGuard 动态根属性隐私：探索实验完整报告

日期：2026-10-12。代码工作区：`/home/rootguard-research`，分支 `research/workflow-extensions`，运行时 HEAD `{meta['git_head']}`。本报告是研究方向筛查，不是新方法论文的性能结论。

## 1. 结论与研究价值判断

**这个方向有继续探索的价值，但本次证据支持的是“新鲜度、任务变化与跨版本隐私预算之间的权衡”，尚不能证明直接迁移轨迹 LDP 就足以构成创新。**

真实复测中，旧缓存平均误差较小，却完全不能发出新的类别变化信号；固定总预算的逐版本刷新使噪声和误报明显增大。允许总预算随版本数增长时可改善数值误差，但不满足同预算比较。只复用同一噪声并平移真实更新则直接泄露真实差分。

更值得关注的是“根数值改变不等于当前任务改变”：无关根更新、保持 Hb/Hct 比值的相关更新、同值新测量，都会使朴素逐版本刷新浪费任务效用。公共更新计划下仅刷新相关根，以及简单定期刷新，已经能取得很大收益。今后的创新必须对这些基线仍有优势，而不能只击败明显过期或反复加噪的弱基线。

**建议保留该方向，下一阶段集中验证任务相关变化与刷新成本；暂不声称已有可投稿的新算法。** 若成熟纵向机制或调优后的简单定期刷新已充分解决问题，应缩小或停止方法创新部分。本轮没有完成成熟 CGM/LOLOHA/RAPPOR 的严格重实现，不能据此宣称优于这些方法。

## 2. 研究问题与隐私目标

原有静态缓存思路在固定私有根 X 下将后续任务转化为缓存的后处理。我们改变的条件是同一根在多轮产生新的数值版本 X_t，仍需要回答当前版本的任务 g(X_t)。旧缓存是否过期是工程问题；刷新当前根时如何在完整历史上控制隐私费用，同时保持任务正确性，才是研究问题。

本轮保护一个用户的有限数值轨迹，公共信息为 H、根集合、测量时刻、更新计划、任务公式、根域和参考人口均值。攻击者可见所有已发布私有化根和由其计算的任务结果；私有测试的刷新决定也属于 transcript。没有原始值驱动的免费刷新决定，没有原始 prompt 或原始条件路由进入该发布接口。

定义归一化根索引 z_it=(m-1)(x_it-lo_i)/(hi_i-lo_i)，d_inf=max_it |z_it-z'_it|。各指数机制 epsilon_i 距离隐私顺序组合，账本 sum epsilon_i<=B，完整发布历史满足 B 距离隐私。其概率为 exp(-epsilon*|u-z|/2)/Z(z)，三角不等式对分子和归一化因子分别给出 epsilon*d/2 的界。私有最大偏差测试在固定过去输出下 1-Lipschitz，使用 Laplace 并计费；条件刷新费用也保守预留。

由于 d_inf<=m-1，B=E/(m-1) **足以**提供标准纯 E-轨迹 LDP。主实验 m=1000，B=0.1/0.3/0.7 对应充分标准界 E=99.9/299.7/699.3；这些不能解释成标准 epsilon=0.1/0.3/0.7 的强 LDP。标准轨迹实验单独使用 E=1/4/8。这个转换是保守充分校准，不是最紧界，也不是所有轨迹机制的效用下界。

当前不保护更新是否发生、时间戳、根属性新增、私有类别标签的额外直出、无限/未知长度以及非数值文本。患者筛选是离线评估数据处理，未声称其本身满足隐私。保护合同中的邻居轨迹均处于固定公开根域且结构一致。

## 3. 修改了什么代码

所有实验实现位于 `experiments/workflow_extensions/root_updates/`；更新该目录 README 和上级工作流 README。**没有改动 RootGuard 核心 sanitizer、optimizer、risk_class、既有数据或 target_mismatch 的代码。**

| 文件 | 修改/新增用途 |
|---|---|
| mechanisms.py | 有版本根缓存、公开刷新计划、预算受控机制、私有测试参考实现、账本和指标 |
| data.py | NHANES 合成轨迹和真实同标本 CBC 筛选 |
| run_experiments.py | 网格实验、配置快照、输入 SHA256、患者级结果、选定原始轨迹 |
| confirm.py | 24 患者同一住院截断序列，200 次噪声复核 |
| attack_probe.py | 精确似然 MAP 标量重构诊断 |
| analyze.py | 患者配对 bootstrap、固定患者噪声不确定性 |
| build_report.py | 输出审计、证据表与本报告 |
| configs/exploration.json | 主实验参数 |
| configs/standard_ldp.json | 标准轨迹 LDP 校准参数 |
| configs/confirmation.json | 独立复核参数 |
| tests/test_dynamic.py | 8 项有针对性的机制验证 |
| README.md、上级 README.md | 隐私合同、方法、数据和复现命令 |

代码复用上一轮 core.py 的原始公式、原始 optimizer 函数与离散指数采样；没有调用 LLM，也没有导入整套推理模型。动态 Opt 的每根总预算 beta_i、公共发布次数 n_i 及每次 epsilon_i=beta_i/n_i，用原绝对误差代理目标优化，sum beta_i=B。此处是公开次数感知的参考扩展，不作为新算法。为允许小总预算/多版本，采用比例下限 1e-6/k，区别于旧静态优化的固定最小 epsilon；原静态实现未被修改。

## 4. 数据与筛选流程

合成数据使用仓库 NHANES 每模板前 100 个样本，三个模板 FIB4、HOMA、ANEMIA，H=4/8/16。控制无新版本、同值新测量、渐变、突变、振荡、轮换单根稀疏更新、仅 RBC 无关根更新、Hb/Hct 同比例更新；最后两类仅 ANEMIA。变化幅度为公开域宽的 1%/5%；FIB4 年龄保持不变。越域值裁剪并记入 metadata。**这些是机制压力测试，不是医学合理的真实纵向轨迹。** H=4 振荡采样点发生混叠，振荡效果应看 H=8/16。

真实数据使用 [MIMIC-IV Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/) 的 labevents：itemid 51222/51221/51279，单位分别 g/dL、%、m/uL，按 subject、admission、specimen、charttime 取唯一完整 Hb/Hct/RBC 组；越出原公开 NHANES 域的整组剔除，同时间按公共 specimen ID 决定唯一组。一个患者只选择一次符合长度的住院，按 charttime 取前 H 组。

读取数值根行 {realaudit['numeric_root_rows']}，唯一完整面板 {realaudit['complete_unique_panels']}；排除无住院面板 {realaudit['no_admission_panels']}、越域面板 {realaudit['out_of_domain_panels']}。H=8 时可用患者 {realaudit['eligible_subjects']}，选 40 人；H=16 可用并选 24 人。主实验 H=4/8/16 的人群可能不同，因此额外取相同 24 人、相同住院的 H=16 序列并截断，作匹配人群复核。参考均值固定沿用 NHANES holdout，没有用测试患者原始轨迹调预算。

真实 H=8 共 320 个版本，类别分布 {realaudit['class_counts']}，真实类别变化 {realaudit['real_class_changes']} 次。无噪声旧值相对当前任务 wMAPE={realaudit['noiseless_stale_wmape']:.3f}%，RCE={realaudit['noiseless_stale_rce']:.3f}%。即使不考虑 DP，旧值也确实造成任务错误；但平均数值变化较小，所以强噪声可能超过时效性误差。

ANEMIA 延用原实验的 100*Hb/Hct 及 32/36 分类，仅为机制 benchmark；不解释为临床诊断性能。当前使用 charttime 而非 storetime，不模拟异步到达、补录或医生决策。

## 5. 实验设置、方法与统计

主实验 105 个数据场景、3 个总距离预算、11 个方法、每条件 30 次随机噪声重复；标准轨迹实验 45 个场景、E=1/4/8、7 方法、30 次重复；确认实验 15 个场景、3 预算、7 方法、200 次重复。另做 36 个标量重构条件。

{table(cohort,[('run','运行'),('cells','条件数'),('draws','逻辑任务评估数'),('seconds','运行秒数')])}

逻辑任务评估数包括同患者不同版本/方法/噪声重复，**不代表独立数据样本量**。正式效用条件总数为 {sum(v['cells'] for v in audit.values())}。主噪声种子 20261015，标准实验 20261016，确认 20261017，攻击诊断 20261018；合成轨迹种子单独固定为 481516+模板序号。共同随机数用于同数据同预算的方法对照；确认使用独立种子。运行环境 Python {meta['python']}、NumPy {meta['numpy']}、SciPy {meta['scipy']}。

| 方法 | 行为 | 费用 |
|---|---|---|
| static | 首次发布后永远复用 | 总 B |
| full_uniform/full_opt | 每版本全部根刷新 | 固定总 B |
| root_uniform/root_opt | 仅公共计划标记的根刷新 | 固定总 B |
| period2/period4 | 每 2/4 个版本刷新全部根 | 固定总 B |
| predictive30/predictive100 | 上一缓存作预测；私有偏差测试决定刷新，阈值 30/100 网格点，每步测试占 15% | 最坏总 B；测试收费 |
| fixed_per_version | 每版本都用原 B | 总 H*B；非公平成本参照 |
| unsafe_shared_noise | 把第一次误差原样平移到真实后续值 | 无有效轨迹隐私保证，仅泄漏反例 |

predictive 仅借鉴 [Predictive DP for Mobility Traces](https://arxiv.org/abs/1311.4008) 的测试/发布框架，是简化参考实现，未复现原完整算法或调优策略，不能用其较差结果否定成熟预测机制。

wMAPE 为完整当前任务绝对误差和 / 当前真实任务绝对值和；RCE 为使用原阈值的当前类别错误率。辅助指标包括按域宽归一化根误差、真实类别变化而预测无变化的信号漏报、真实类别稳定而预测变化的误报、到新类别匹配的版本延迟。未在下一个真实类别变化前匹配时，延迟设为该段长度，是截断/删失指标，不是自然时间延迟。漏报不衡量新类别是否预测正确，故另计算选定真实案例的实际转折点类别错误率。

每方法先对同一患者的版本和随机重复汇总，再做配对患者 bootstrap（主/标准 1000 次、确认 2000 次）。另对独立噪声重复给出固定人群下配对 t 区间。两类区间回答不同不确定性，不能把版本和噪声重复算成新患者；均为逐项 95%，未做多重比较校正。这是探索筛查，不按大量检验挑选最显著条件作结论。

## 6. 真实复测结果：固定总距离预算

以下为真实 H=8、40 人、B=0.3、30 次重复。标准轨迹充分界 E=299.7；勿解释为标准 E=0.3。

{table(fair,fields)}

旧缓存的低平均误差伴随 100% 变化信号漏报；逐版本全刷新 RCE 达 62.740%，稳定期误报 63.142%。period4 将 wMAPE 降至 4.971%，但漏报仍为 85.926%。真实面板所有根每次均被测量，root_opt 与 full_opt 完全一致，无法靠只刷新已有公共更新根解决这个场景。

同一真实 H=8 场景的全部总距离预算对照，避免仅凭 B=0.3 判断：

{table(budgetrows,[('budget_label','总 B'),('method','方法'),('wmape','wMAPE %'),('rce','RCE %'),('change_miss_rate','变化信号漏报 %')])}

成本参照 fixed_per_version 的 wMAPE=2.370%、RCE=18.062%，总费用 2.4，是公平实验 B=0.3 的 8 倍，不能宣称其同预算更好。unsafe_shared_noise 不进入公平比较。

实际转折点上的类别错误率补充如下，避免将噪声导致的“类别改变”误当真实更新被正确识别。报告生成时按原始配置、数据索引与噪声种子重新计算这六个条件，使用 float64 和原分类函数，并确认整体 RCE 与已保存结果一致；没有用 float32 归档的边界舍入替代正式分类：

{table(change_rows,[('method','方法'),('transition_rce','真实转折时刻 RCE %')])}

## 7. 固定患者复核：长度与噪声代价

同一 24 位患者、同一住院序列、200 次独立噪声重复，B=0.3：

{table(matched,[('H','长度'),('method','方法'),('wmape','wMAPE %'),('rce','RCE %'),('change_miss_rate','变化信号漏报 %')])}

full_opt 的误差随 H 从 4 到 16 明显增大；同一人群下仍成立，排除了主实验队列改变这一解释。不过截断长度也改变任务时间范围，并非完全孤立的因果长度实验。static 虽然平均误差低，仍完全漏报；period4 仍是有竞争力的简单参考，不能略去。

## 8. 合成机制控制：哪些变化值得刷新

主实验 H=8、B=0.3 的代表性结果：

{table(synthetic,[('dataset','场景'),('method','方法'),('wmape','wMAPE %'),('rce','RCE %')])}

解释：无关 RBC 更新时，root_opt 的效用与 static 一致，说明已有公共计划与任务敏感度信息就能避免部分浪费。Hb/Hct 同比改变时真实比值基本不变，static 有优势，但按根逐次刷新打破真实关联，引入新的任务噪声。同值新测量也显示“版本事件”与“有用的信息变化”不能等同。稀疏 FIB4/HOMA 中 root_opt 显著优于 full_opt，却不必优于 static/period4，不能只展示弱对照。

ANEMIA 突变的独立 200 次噪声复核：

{table(jump,[('method','方法'),('wmape','wMAPE %'),('rce','RCE %'),('delta_rce_vs_static_ci95','相对 static 的 RCE 差 95% 区间')])}

root_opt 的 RCE 优于 static，但 wMAPE 差的患者区间跨 0，收益是任务指标相关的。这里突变恰好落在 period4 的公开刷新点上，root_opt 与 period4 相同，不能当作击败周期策略的证据。尚未做突变相位随机化和周期参数调优，必须在下一轮补上。

## 9. 标准轨迹 LDP 的单独校准

真实 H=8、40 人、E=1/4/8，B=E/999：

{table(standard,[('budget_label','标准 E'),('method','方法'),('wmape','wMAPE %'),('rce','RCE %')])}

在本次公开域、网格和独立根指数机制下，强标准轨迹保护的效用明显较差。**这只说明当前机制经过该充分校准后的性能，不能推导所有轨迹 LDP 都不可用，也不能宣称已证明理论最优下界。** 后续若选标准轨迹 LDP，需以成熟轨迹机制作同目标比较；若选距离隐私或窗口保护，需明确换了什么隐私合同。

## 10. 重构诊断与共享噪声反例

标量潜在初值在 [200,800]、步长 5 的候选集上取 100 个样本，30 次随机重复；后续偏移为攻击者已知的公共正弦偏移，用实际离散 PMF 精确似然 MAP 重构。它是攻击者有利的受控诊断，不是完整真实医疗轨迹攻击。cache 只重复首次私有化值，并不提供最新真实变化。下表 B=0.3：

{table(attack,[('H','长度'),('method','方法'),('charged_budget','账本总 B'),('mae_index','重构 MAE 网格点'),('within_5_index_pct','5 网格点内成功率 %')])}

每版本恒定预算时，H 增长让重构显著更准确，同时总预算增长；固定总预算下每次噪声变大，没有出现相同的重构增强趋势。这说明不能用每版本预算相同替代轨迹总预算相同。不是通过“攻击失败”证明 DP，正式保证来自机制及组合证明。

共享噪声反例 Y_t=Y_0+(X_t-X_0) 给出 Y_t-Y_(t-1)=X_t-X_(t-1)，真实差分被精确恢复。不同增量轨迹的输出支持可不相交，因此单次发布的保证不能自动扩展到这种联合发布。8 项测试包含该反例。该诊断可能输出越域数值并在 FIB4 产生 NaN，主运行记录了 {audit[names[0]]['uncertified_nan_cells']} 个这样的非法诊断条件；没有任何认证机制的 wMAPE/RCE 非有限。非法诊断不参与研究效用结论，原始 JSON 中保留的 NaN 需要 Python 等兼容读取器，严格 JSON 使用者应过滤这类记录。

## 11. 检查结果、产物与复现

8 项 unittest 检查覆盖公开计划预算上限、未更新根缓存一致性、真实面板单位与患者去重、无更新 static/root_opt 相等、重复查询 1/4/8 次的后处理不变性、共享误差泄漏差分、指数机制相邻索引似然界、私有测试敏感度与标准 E 校准。测试日志见 `tests.log`。数学证明不能由有限单元测试代替。

报告脚本逐一重算所有正式运行记录的源文件 SHA256，确认与运行时一致；检查认证机制有限数值、费用上限和条件总数，输出 `validation.json`。本报告生成前的正式运行统计为：

{table([dict(run=n,**audit[n]) for n in names],[('run','运行'),('certified_cells','认证条件数'),('source_hashes_checked','核对 SHA 数'),('budget_cap_pass','预算通过'),('finite_certified_metrics','有限数值通过')])}

每组运行均保留 config、metadata、cells、患者级结果、analysis、summary CSV 和 Markdown。metadata 含输入 SHA、源码 SHA、环境、种子对应配置、数据筛选记录和时间。主实验选定真实 H=8 及合成突变案例另保存完整缓存轨迹；其他条件保存每患者汇总与每次噪声汇总，未保存所有随机轨迹。

本报告目录另含 `evidence.csv`（全部同预算认证条件及区间）、`transition_time_checks.json`、`validation.json`、`tests.log`。原始结果目录：

{chr(10).join('- ['+n+']('+str(base/n/'SUMMARY.md')+')' for n in names)}

- [标量重构结果]({base/'attack_20261012/reconstruction.csv'})
- [复现命令及隐私合同]({Path(__file__).parent/'README.md'})

所有生成产物在 Git 忽略的 outputs 下；代码和配置可由用户提交。未执行 commit 或 push。报告没有触发 LLM/API 调用，也未将患者数值发往远程模型。

## 12. 下一阶段如何判断是否值得做方法论文

本轮已经确认动态版本下存在稳定的任务与预算权衡；尚未确认新机制能优于成熟方法。下一轮应先做三件验证，而不是立即把现有参考扩展包装成新算法：

1. 对简单周期基线作公开参数调优、突变相位随机化，并报告不同新鲜度要求下的误差/误报/费用曲线。突变恰好对齐周期的案例不足以体现自适应价值。
2. 明确最终选标准轨迹、距离或窗口隐私，接入可适配的成熟纵向基线，在同合同同预算下比较。当前简化 predictive 结果只是参考，不足以证明成熟框架无法满足场景。
3. 扩大观测任务和样本，优先找“任务确实变化且要求及时响应”的工作流；当前 ANEMIA 比值相对稳定，不适合单独承担普遍临床动态保护结论。新增根、私有更新时间、未知序列长度和完整 prompt 评测应另设实验，不混入现有保证。

若这些对照仍留下显著的时效性与效用缺口，再考虑针对根依赖、变化与预算的改进。可投稿价值需要机制贡献、清楚的隐私证明和公平证据共同支持，不能保证 CCF-C/EI 接收，也不能用“首次动态 LLM prompt DP”替代贡献论证。

## 13. 相关研究边界

[RootGuard 原论文](https://arxiv.org/abs/2605.03188) 提供依赖感知静态根私有化背景。本实验是其固定缓存思想的机制级压力测试，不声称原论文已实现动态根版本更新。

[RAPPOR](https://research.google/pubs/rappor-randomized-aggregatable-privacy-preserving-ordinal-response/) 与 [LOLOHA](https://openproceedings.org/2023/conf/edbt/3-paper-41.pdf) 提供长期 memoization/类别频率相关背景；[Local DP for Evolving Data](https://arxiv.org/abs/1802.07128) 面向变化人群分布；[CGM](https://vldb.org/pvldb/vol14/p2258-bao.pdf) 提供相关噪声的持续数据流 LDP；[LDP-IDS](https://arxiv.org/abs/2204.00526) 采用窗口事件保护。这些保护目标、输出对象与本次单用户数值根轨迹任务未必相同，需要适配或明确不同合同，不能直接把公式名字当公平基线。

[Predictive DP for Mobility Traces](https://arxiv.org/abs/1311.4008) 为本次收费私有测试参考提供思路；本次并未复现其完整算法。因此本报告不提出“动态保护从未有人研究”的新颖性断言，也不对新近 LLM 论文作穷尽性排查。
'''
    (out/'REPORT.md').write_text(doc)
    print(out/'REPORT.md')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
