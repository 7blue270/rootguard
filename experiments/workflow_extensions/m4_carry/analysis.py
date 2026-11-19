"""Paired patient bootstrap and paired patient-plus-mechanism bootstrap."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .core import METHODS, key_seed


def write_csv(path, rows):
    if not rows:
        return
    names = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        wr = csv.DictWriter(f, fieldnames=names)
        wr.writeheader()
        wr.writerows(rows)


def q95(x):
    """Empirical endpoints work with +inf censoring, without inf interpolation."""
    x = np.sort(np.asarray(x))
    return [float(x[max(0, int(np.ceil(p*len(x)))-1)]) for p in (.025, .975)]


def finite_interval(x):
    x = np.asarray(x)
    return q95(x[np.isfinite(x)]) if np.isfinite(x).any() else [None, None]


def bmin(curves, budgets, tau):
    # No monotonic envelope: take min over every passing grid point.
    return np.min(np.where(curves <= tau, np.asarray(budgets), np.inf), axis=-1)


def bootstrap_curves(x, draws, seed):
    # x: method,budget,patient,repeat. One set of sampled indices for ALL curves.
    M, B, N, R = x.shape
    rng = np.random.default_rng(seed)
    pd = rng.integers(N, size=(draws, N))
    rd = rng.integers(R, size=(draws, N, R))
    patient = np.empty((M, B, draws))
    nested = np.empty_like(patient)
    means = x.mean(axis=-1)
    for start in range(0, draws, 50):
        stop = min(start+50, draws)
        p = pd[start:stop]
        r = rd[start:stop]
        patient[..., start:stop] = means[:, :, p].mean(axis=-1)
        nested[..., start:stop] = x[:, :, p[..., None], r].mean(axis=(-1, -2))
    return patient, nested


def interval_text(v, digits=3):
    if v is None or not np.isfinite(v):
        return '未达标/删失'
    return f'{v:.{digits}f}'


def mean_na(values):
    a = np.asarray([np.nan if v is None else v for v in values], float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else None


def analyze(out):
    out = Path(out)
    import hashlib
    sources = [Path(__file__), Path(__file__).with_name('plotting.py')]
    (out/'postprocessing_source_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    cfg = json.loads((out/'config.json').read_text())
    targets = json.loads((out/'targets.json').read_text())
    audit = json.loads((out/'audit.json').read_text())
    selections = json.loads((out/'selections.json').read_text())
    budgets, R, D = cfg['evaluated_budgets'], cfg['repeats'], cfg['bootstrap']
    bytemplate = {}
    for line in (out/'metrics.jsonl').open():
        r = json.loads(line)
        bytemplate.setdefault(r['template'], []).append(r)
    summary, paired, targets_rows, bmin_contrasts, curves_to_plot = [], [], [], [], {}
    for template, rows in bytemplate.items():
        subjects = sorted({r['subject'] for r in rows})
        # Average multiple trajectories within each patient and repetition first.
        group = {}
        for r in rows:
            group.setdefault((r['method'], r['B'], r['subject'], r['repeat']), []).append(r)
        x = np.empty((len(METHODS), len(budgets), len(subjects), R))
        for mi, method in enumerate(METHODS):
            for bi, B in enumerate(budgets):
                for ni, subject in enumerate(subjects):
                    for rep in range(R):
                        x[mi, bi, ni, rep] = np.mean([r['mae'] for r in group[method, B, subject, rep]])
        patient, nested = bootstrap_curves(x, D, key_seed(cfg['bootstrap_seed'], template))
        mean = x.mean(axis=(-1, -2))
        curves_to_plot[template] = dict(mean=mean, patient=patient, nested=nested)
        identity = {'template','subject','trajectory','B','method','repeat','stop_reason','mae'}
        # Include metrics whose first observation is NA (e.g. static frac_veto).
        metric_names = sorted({k for r in rows for k in r if k not in identity})
        for mi, method in enumerate(METHODS):
            for bi, B in enumerate(budgets):
                ss = dict(template=template, method=method, B=B, n_patients=len(subjects), repeats=R, mae=float(mean[mi, bi]))
                for prefix, boots in [('patient',patient),('two_layer',nested)]:
                    a,b = q95(boots[mi, bi])
                    ss[f'mae_{prefix}_lo'], ss[f'mae_{prefix}_hi'] = a,b
                    ss[f'mae_{prefix}_ucb95'] = float(np.quantile(boots[mi,bi], .95))
                for metric in metric_names:
                    # Preserve patient weighting, excluding undefined denominators.
                    pv = [mean_na([mean_na([rr.get(metric) for rr in group[method,B,s,rep]]) for rep in range(R)]) for s in subjects]
                    ss[metric] = mean_na(pv)
                summary.append(ss)
        for m, base in [('M4-carry','B2-val'),('M4-carry','M4'),('M4-carry-replan','M4-carry'),('M4-carry','B0')]:
            a,b = METHODS.index(m), METHODS.index(base)
            for bi,B in enumerate(budgets):
                pr = dict(template=template, B=B, method=m, reference=base, delta_mae=float(mean[a,bi]-mean[b,bi]),
                          mae_ratio=float(mean[a,bi]/mean[b,bi]))
                for prefix, boots in [('patient',patient),('two_layer',nested)]:
                    lo,hi = q95(boots[a,bi]-boots[b,bi])
                    pr[f'{prefix}_lo'], pr[f'{prefix}_hi'] = lo,hi
                paired.append(pr)
        for scenario, ts in targets[template].items():
            for target in cfg['target_multipliers']:
                tau = ts['tau_'+target]
                bm = bmin(mean, budgets, tau)
                pbs = bmin(patient.transpose(0,2,1), budgets, tau)
                nbs = bmin(nested.transpose(0,2,1), budgets, tau)
                for mi, method in enumerate(METHODS):
                    row = dict(template=template, scenario=scenario, delta=ts['delta'], target=target, tau=tau,
                               method=method, B_min_mean=float(bm[mi]) if np.isfinite(bm[mi]) else None,
                               reached=bool(np.isfinite(bm[mi])), censored=not np.isfinite(bm[mi]), max_grid=max(budgets))
                    for prefix, boots, raw in [('patient',pbs,patient),('two_layer',nbs,nested)]:
                        row[f'{prefix}_reach_rate'] = float(np.isfinite(boots[mi]).mean())
                        lo,hi=q95(boots[mi])
                        row[f'{prefix}_lo'] = lo if np.isfinite(lo) else None
                        row[f'{prefix}_hi'] = hi if np.isfinite(hi) else None
                        ucb = np.quantile(raw[mi], .95, axis=-1)
                        ub = bmin(ucb, budgets, tau)
                        row[f'B_min_{prefix}_ucb95'] = float(ub) if np.isfinite(ub) else None
                    targets_rows.append(row)
                for m,base in [('M4-carry','B2-val'),('M4-carry','M4'),('M4-carry-replan','B2-val')]:
                    ai,bi=METHODS.index(m),METHODS.index(base)
                    both=np.isfinite(bm[[ai,bi]]).all()
                    row=dict(template=template,scenario=scenario,target=target,tau=tau,method=m,reference=base,
                             difference=float(bm[ai]-bm[bi]) if both else None, ratio=float(bm[ai]/bm[bi]) if both else None,
                             method_reached=bool(np.isfinite(bm[ai])),reference_reached=bool(np.isfinite(bm[bi])))
                    for prefix,boots in [('patient',pbs),('two_layer',nbs)]:
                        ok=np.isfinite(boots[ai]) & np.isfinite(boots[bi])
                        row[f'{prefix}_both_reach_rate']=float(ok.mean())
                        differences=boots[ai,ok]-boots[bi,ok]
                        ratios=boots[ai,ok]/boots[bi,ok]
                        for name,vals in [('difference',differences),('ratio',ratios)]:
                            lo,hi=finite_interval(vals)
                            row[f'{prefix}_{name}_conditional_lo']=lo
                            row[f'{prefix}_{name}_conditional_hi']=hi
                    bmin_contrasts.append(row)
    write_csv(out/'summary.csv',summary)
    write_csv(out/'paired_mae.csv',paired)
    write_csv(out/'bmin.csv',targets_rows)
    write_csv(out/'bmin_paired.csv',bmin_contrasts)
    write_report(out,cfg,audit,selections,targets,summary,paired,targets_rows,bmin_contrasts)
    plot_results(out,cfg,curves_to_plot,targets)


def write_report(out,cfg,audit,selections,targets,summary,paired,bmins,contrasts):
    lines=['# M4-carry 真实有限轨迹实验报告','',
           '**性质：内部机制诊断；数据曾用于历史分析。当前结果不构成新的独立泛化验证。**','',
           f'正式设置：R={cfg["repeats"]}；患者级配对 bootstrap={cfg["bootstrap"]}，另报患者＋机制两层区间；共 {len(cfg["evaluated_budgets"])} 个预算点。预检查={cfg["precheck"]}。', '',
           '## 1. 数据、冻结规则与实现','',
           '复用 finite-cache-v1 的真实 MIMIC demo 化验轨迹及既有任务公式/域；对话轮次与查询安排为合成。跨模板按患者统一 train/validation/test=0.5/0.25/0.25。独立患者数不随模板、轮次或机制重复相乘。', '',
           '|模板|train 患者|validation 患者|test 患者|', '|---|---:|---:|---:|']
    headline=['## 先看结论','',
              '判断以carry−M4的结转对照、carry−B2-val的主对照及是否退化为B0共同确定；相对B2的优势不自动证明结转有效。','',
              '|模板|B=960 carry/B2-val MAE比|两层差值95% CI|应如何解读|',
              '|---|---:|---|---|']
    for t in targets:
        r=next(r for r in paired if r['template']==t and r['B']==960 and r['method']=='M4-carry' and r['reference']=='B2-val')
        if selections[t]['960.0']['M4-carry']['K']==0:
            label='K=0，完全等于B0；不是动态结转增益'
        elif r['patient_lo']>0 and r['two_layer_lo']>0:
            label='carry更差，两种区间方向一致'
        elif r['patient_hi']<0 and r['two_layer_hi']<0:
            label='该点误差更小，仍需检查是否来自结转'
        else:
            label='点估计改善，尚未通过两层区间确认' if r['delta_mae']<0 else '区间无法区分'
        headline.append(f'|{t}|{r["mae_ratio"]:.3f}|[{r["two_layer_lo"]:.4f}, {r["two_layer_hi"]:.4f}]|{label}|')
    same_at_max=all(abs(r['delta_mae'])<1e-10 for r in paired if r['B']==960 and r['method']=='M4-carry' and r['reference']=='M4')
    same_bmin=all(next(z for z in bmins if all(z[k]==r[k] for k in ('template','scenario','target')) and z['method']=='M4')['B_min_mean']==r['B_min_mean'] for r in bmins if r['method']=='M4-carry')
    if same_bmin:
        headline += ['', '**本轮全部预设目标中，结转没有改变同计划M4的B_min；尚未验证结转本身带来预算节省。**']
    headline += ['',f'结转归因核对：四模板在B=960的carry与M4患者均权MAE相同={same_at_max}；全部预设目标的carry与M4的B_min（含删失）相同={same_bmin}。因此不能把相对B2的改善自动归因结转。','']
    for r in contrasts:
        if r['method']!='M4-carry' or r['reference']!='B2-val' or r['ratio'] is None:
            continue
        if not (r['patient_both_reach_rate']>=.95 and r['two_layer_both_reach_rate']>=.95 and r['patient_ratio_conditional_hi']<1 and r['two_layer_ratio_conditional_hi']<1):
            continue
        bm=next(z for z in bmins if all(z[k]==r[k] for k in ('template','scenario','target')) and z['method']=='M4-carry')['B_min_mean']
        bm0=next(z for z in bmins if all(z[k]==r[k] for k in ('template','scenario','target')) and z['method']=='B2-val')['B_min_mean']
        plan=selections[r['template']][str(float(bm))]['M4-carry']
        label='与B0完全相同，属于静态选择，不能归因结转' if plan['K']==0 else '仍需与同计划M4对照归因'
        headline += [f'- {r["template"]} {r["scenario"]}/{r["target"]}：B_min {bm0:.2f}→{bm:.2f}（{(r["ratio"]-1)*100:.1f}%），两层比值区间[{r["two_layer_ratio_conditional_lo"]:.3f}, {r["two_layer_ratio_conditional_hi"]:.3f}]；K={plan["K"]}、ρ={plan["rho"]}，{label}。']
    headline += ['', '下一步优先保留静态退化能力和强B2对照，核查公开规划与真实任务效用的偏差，再扩患者或外层验证。本轮不根据test改Δ、τ或门控，也不补M2/M5实验。','']
    lines[4:4]=headline
    from collections import Counter
    gc=Counter(json.loads((out/'split.json').read_text()).values())
    lines.insert(lines.index('|模板|train 患者|validation 患者|test 患者|'),f'跨模板去重后的总患者数={sum(gc.values())}：train={gc["train"]}、validation={gc["validation"]}、test={gc["test"]}；下表不可直接相加。\n')
    for t,c in audit['counts'].items():
        lines.append(f'|{t}|{c["train"]["patients"]}|{c["validation"]["patients"]}|{c["test"]["patients"]}|')
    lines += ['', f'公开MC每Δ {cfg["public_trajectories_per_delta"]}条轨迹、每候选{cfg["public_repeats"]}次机制重复；τ参考校准{cfg["calibration_repeats"]}次重复。所有计划、τ和 validation 选择均在第一个 test 误差计算前完成（phase_a_complete.json）。Δ=[0.005,0.02,0.06] 是 config 常量，单位为投影前每公开测量批次的归一化最大根变化期望。公开人群只从 train 初值/联合变化方向与测量模式生成；并非真实新增患者，也不是按 test 漂移分档。规划使用三个 Δ 的等权混合均值风险；各 Δ 只另生成 τ，并未复制 test 样本来扩大 n。', '',
              'ĥ来自最终任务的解析/保守导数界（非 train 网格割线）；μ为 train 轮次均值，C̄为 train 实际测量根变化的90%分位数。目标权重固定为最终节点1、其余0。分配统一 √ĥ，沿用原型的 h≥1e−9 正数下限；如 ANEMIA 的 rbc 对最终节点无直接影响，仍获极小正预算。C_OP=0.5，固定32个公开 MC 样本，不调参。', '',
              '公开候选 K∈{0,1,2}、ρ∈{0.5,0.7,0.85}（K=0强制ρ=1）。在 config 的9个相对轮次中枚举时刻，并枚举全根/高h半数根/单根动作集合；每(K,ρ,全根或子集)组由公开解析代理保留1个，再用相同公开随机样本作任务空间 MC 精选。这是有限候选规划，不声称全空间最优。时刻通过严格递增的最小距离匹配对齐每条轨迹的公开测量机会。', '',
              'M4/carry使用公开选出的同一固定包计划。carry的veto使用当前 R_left/n_left；跳过只减机会数，金额全额结转。replan逐候选用carry仿真选公开时刻/根集合，再仅在validation选K/ρ，其相对carry的变化包含重规划及额外validation选择。未增加M2/M5或比例消融。', '',
              'B2周期单位为测量机会，P∈{2,3,4,5,6,8,10,12}，ρ=0.5；不截断后续刷新。所有方法均可用公开完整机会表；根发布使用截至执行轮次最新已知值，即使该批次未重测。年龄g只在该根被测时增长。B2-val按validation最终MAE选P；B2-pub按同一公开混合人群选P。每个B独立选择。', '',
              '粗网格80–960、倍率1.1（含960）；事先给每个区间加1.05倍点，全部方法共享，不用test挑细化位置。逐点扫描，不做单调包络/二分。实际最大相邻比值='+f'{cfg["actual_max_grid_ratio"]:.6f}。', '',
              '配对发布噪声键：(seed,phase,subject,trajectory,B,repeat,实际turn,root)，方法不入键；public_calibration/public_plan/validation/test四流独立。固定种子只用于离线复现；部署时真实发布噪声必须对服务端保密。公开计划/MC种子与发布噪声流分开。', '',
              '## 2. B=960：实际水平与主比较','',
              '|模板|方法|MAE|患者95% CI|两层95% CI|刷新m|支出%|投影%|MissRate|', '|---|---|---:|---|---|---:|---:|---:|---:|']
    for s in summary:
        if s['B']!=960:
            continue
        def fmt(k,m=1):
            return 'NA' if s.get(k) is None else f'{s[k]*m:.3f}'
        lines.append(f'|{s["template"]}|{s["method"]}|{fmt("mae")}|[{fmt("mae_patient_lo")}, {fmt("mae_patient_hi")}]|[{fmt("mae_two_layer_lo")}, {fmt("mae_two_layer_hi")}]|{fmt("m")}|{fmt("spend_fraction",100)}|{fmt("projection_rate",100)}|{fmt("MissRate")}|')
    lines += ['', '差值为前者−后者，负值更好。以下区间均为点态配对95%，未做多模板/多预算家族校正。', '',
              '|模板|比较（B=960）|MAE差|患者95% CI|两层95% CI|', '|---|---|---:|---|---|']
    for s in paired:
        if s['B']==960:
            lines.append(f'|{s["template"]}|{s["method"]} − {s["reference"]}|{s["delta_mae"]:.4f}|[{s["patient_lo"]:.4f}, {s["patient_hi"]:.4f}]|[{s["two_layer_lo"]:.4f}, {s["two_layer_hi"]:.4f}]|')
    lines += ['', '## 3. B_min：网格达标与删失','',
              '各τ=公开合成人群在B=960的B0全轨迹MAE×0.85/1.15。此参考值不是数学下界；不证明不可达。未达标表示扫描网格内没有达标点，按右删失记录，不丢弃、不填960。', '',
              '|模板|Δ档|τ_lo / τ_hi|', '|---|---|---:|']
    for t,sc in targets.items():
        for n,v in sc.items():
            lines.append(f'|{t}|{n}|{v["tau_lo"]:.4f} / {v["tau_hi"]:.4f}|')
    lines += ['', '|模板|Δ档|目标|B0|B2-val|B2-pub|M4|M4-carry|replan|', '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for t in targets:
        for sc in cfg['delta_scenarios']:
            for target in cfg['target_multipliers']:
                vals=[]
                for method in METHODS:
                    r=next(r for r in bmins if r['template']==t and r['scenario']==sc and r['target']==target and r['method']==method)
                    vals.append(interval_text(r['B_min_mean'],2))
                lines.append('|'+ '|'.join([t,sc,target]+vals)+'|')
    lines += ['', '|模板|Δ档/目标|carry/B2-val B_min比|患者 / 两层双方达标率|条件两层比值95% CI|', '|---|---|---:|---:|---|']
    for r in contrasts:
        if r['method']!='M4-carry':
            continue
        if r['reference']!='B2-val':
            continue
        ci=(r['two_layer_ratio_conditional_lo'],r['two_layer_ratio_conditional_hi'])
        lines.append(f'|{r["template"]}|{r["scenario"]}/{r["target"]}|{interval_text(r["ratio"])}|{r["patient_both_reach_rate"]:.3f} / {r["two_layer_both_reach_rate"]:.3f}|[{interval_text(ci[0])}, {interval_text(ci[1])}]|')
    lines += ['', 'B_min的患者/两层区间、达标比例和单侧95% UCB口径完整见bmin.csv；差值/比值见bmin_paired.csv。区间条件于已选配置与τ，不包含再次选参的不确定性。条件区间仅使用双方都达标的重采样，必须连同双方达标率解读，不能把少量达标重采样当作总体显著性。UCB是逐预算点上界，并非全网格同时保证。', '',
              '## 4. 选择、退化与结转诊断','',
              '|模板|B=960 的 B2-val / B2-pub P|M4/carry K,ρ|replan K,ρ|carry实际刷新m|carry静态计划比例（全B）|', '|---|---|---|---|---:|---:|']
    for t,bs in selections.items():
        z=bs['960.0']
        s=next(r for r in summary if r['template']==t and r['B']==960 and r['method']=='M4-carry')
        static=np.mean([b['M4-carry']['K']==0 for b in bs.values()])
        lines.append(f'|{t}|{z["B2-val"]["period"]} / {z["B2-pub"]["period"]}|{z["M4"]["K"]}, {z["M4"]["rho"]}|{z["M4-carry-replan"]["K"]}, {z["M4-carry-replan"]["rho"]}|{s["m"]:.3f}|{static:.1%}|')
    from collections import Counter
    lines += ['', '|模板|全部预算的 B2-val P选择次数|', '|---|---|']
    for t,bs in selections.items():
        counts=Counter(z['B2-val']['period'] for z in bs.values())
        lines.append('|'+t+'|'+', '.join(f'P={p}: {n}/{len(bs)}' for p,n in sorted(counts.items()))+'|')
    lines += ['', '完整逐B计划、P和K/ρ见selections.json，validation各候选误差及公开评分见selection_scores.jsonl。逐机会根集合、候选/实际逐根ε、预算阶梯、发布前后缓存和即时收益见actions.jsonl，含初始化与全否机会。候选与执行保持动作总ε，选根后重新分配，逐根ε通常不同。', '',
              'p95为每轨迹全T误差95分位数再患者均权；post-stop只取t>stop，分母max(T−stop,1)，同时记录n_post；停止区分静态/机会耗尽，支出耗尽亦可由actions余额识别。MissRate为显著根更新在同轮未获覆盖比例，显著阈值固定0.05，无事件记NA。有害刷新为相同当轮真值下误差上升，无刷新记NA。投影率在裁剪前统计且含初始化。danger是根向下扰动0.08的数值放大超过train μ处2倍的轮次，不是临床标签；无危险轮danger_mae=NA。完整次要轴见summary.csv和metrics.jsonl。', '',
              '## 5. 内部 go/no-go 判断','']
    for t in targets:
        r=next(r for r in paired if r['template']==t and r['B']==960 and r['method']=='M4-carry' and r['reference']=='B2-val')
        z=selections[t]['960.0']['M4-carry']
        if r['patient_hi']<0 and r['two_layer_hi']<0:
            verdict='该预指定预算点支持carry较B2-val误差更小'
        elif r['patient_lo']>0 and r['two_layer_lo']>0:
            verdict='该预算点B2-val更好'
        else:
            verdict='该预算点证据不足以区分'
        extra='；但计划K=0，收益属于静态复用而非动态结转' if z['K']==0 else ''
        lines.append(f'- {t}：{verdict}（MAE比={r["mae_ratio"]:.3f}）{extra}。')
    wins=[r for r in contrasts if r['method']=='M4-carry' and r['reference']=='B2-val' and r['ratio'] is not None and r['ratio']<1
          and r['patient_both_reach_rate']>=.95 and r['two_layer_both_reach_rate']>=.95
          and r['patient_ratio_conditional_hi'] is not None and r['patient_ratio_conditional_hi']<1
          and r['two_layer_ratio_conditional_hi'] is not None and r['two_layer_ratio_conditional_hi']<1]
    lines += ['',f'预指定所有模板×Δ目标中，双方达标率≥95%且两种配对区间均支持B_min比<1的格数：{len(wins)}。点态诊断不能替代扩患者确认；不得由一个有利格推出普遍优越。', '',
              '结转全网格诊断（探索性汇总，非独立确认；不同模板的绝对MAE不可相加）：', '',
              '|模板|54个预算中carry较M4均值更低/更高/相同|两种区间均支持carry更差的预算点数|', '|---|---|---:|']
    for t in targets:
        rr=[r for r in paired if r['template']==t and r['method']=='M4-carry' and r['reference']=='M4']
        better=sum(r['delta_mae'] < -1e-10 for r in rr)
        worse=sum(r['delta_mae'] > 1e-10 for r in rr)
        bad=sum(r['patient_lo']>0 and r['two_layer_lo']>0 for r in rr)
        lines.append(f'|{t}|{better} / {worse} / {len(rr)-better-worse}|{bad}|')
    lines += ['', 'FIB4绝对MAE仍很大，不能只看相对百分比；既有宽公开域、全域导数界分配与分母非线性下存在严重任务误差。本轮没有据test缩域或换分配。ANEMIA投影率包含对最终节点无作用、只分得极小预算的rbc，不能把该比例直接解读为最终任务失效率。', '',
              '重点区分：carry−M4才对应结转作用；carry−B0用于判断是否真正超越静态；replan−carry包含重规划和validation选择。若优势主要是K=0或ρ接近1，不归因为动态根刷新。后续是否扩展机制，应同时看这三组配对比较和B_min删失情况。', '',
              '## 6. 审计、局限与复现','',
              f'共审计{audit["evaluated_mechanism_paths"]}条机制路径；ε>0、含初始化Σε≤B、carry/M4同计划与因果发布断言通过。最大数值超支={audit["max_budget_excess"]:.3g}。账本核验不是独立的DP完整证明；隐私保证仍以同一公开测量/查询/计划侧信息及保密发布噪声为条件。', '',
              '主要局限：每模板原始20名患者，test患者更少；Δ是公开假设而非真实患者标签；公共模拟规模有限；候选K≤2且MC短名单有限；先前使用过同批患者；逐格点态CI且未校正多重比较。本轮只作机制go/no-go，不作论文发表或临床有效性结论。', '',
              '复现（在仓库根目录，输出目录必须为空；不会覆盖旧实验）：','', '```bash',
              'cd /home/rootguard-research',
              '/root/miniconda3/envs/rootguard/bin/python -m unittest experiments.workflow_extensions.m4_carry.test_core -v',
              '/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m4_carry.run --precheck --output experiments/workflow_extensions/outputs/m4_carry/precheck_reproduce',
              '/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m4_carry.run --output experiments/workflow_extensions/outputs/m4_carry/real_v1_reproduce',
              '# 从已完成原始日志重新生成统计和图：',
              f'/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m4_carry.analysis --output {out}',
              '```', '', '配置：experiments/workflow_extensions/m4_carry/configs/real_v1.json。input_hashes.json/source_hashes.json记录复现输入与程序版本。', '',
              '## 7. 结果图','']
    for t in targets:
        lines += [f'![{t} 全轨迹最终任务MAE](figures/{t}_mae.png)','']
    (out/'REPORT-zh.md').write_text('\n'.join(lines)+'\n')


def plot_results(out,cfg,curves,targets):
    from .plotting import figures
    figures(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args()
    if args.report_only:
        out=Path(args.output)
        def read_rows(name):
            def convert(v):
                if v=='': return None
                if v in ('True','False'): return v=='True'
                try: return float(v)
                except ValueError: return v
            return [{k:convert(v) for k,v in r.items()} for r in csv.DictReader((out/name).open())]
        write_report(out, *[json.loads((out/f).read_text()) for f in ('config.json','audit.json','selections.json','targets.json')],
                     *[read_rows(f) for f in ('summary.csv','paired_mae.csv','bmin.csv','bmin_paired.csv')])
    else:
        analyze(Path(args.output))
