"""Build a source-linked Chinese report from completed experimental runs."""
import argparse
import csv
import json
from pathlib import Path
from core import REPO, sha

RUN_IDS = ['pilot_checked_20261012', 'confirmation_a_20261012', 'confirmation_b_20261012']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = REPO / 'experiments/workflow_extensions/outputs/target_mismatch'
    analyses = [json.loads((base / rid / 'analysis.json').read_text()) for rid in RUN_IDS]
    metas = [json.loads((base / rid / 'metadata.json').read_text()) for rid in RUN_IDS]
    a, b = analyses[1:]
    def find(data, name, B, method='opt_proxy', root='', level=0):
        return next(c for c in data if c['template'] == name and c['B'] == B and c['method'] == method
                    and (method != 'opt_proxy' or c['family'] == 'power' and c['root'] == root and c['level'] == level))
    def ci(values):
        return f"[{values[0]:.3f}, {values[1]:.3f}]"
    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                         ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])
    lines = ['# RootGuard 目标函数失配探索实验报告', '',
    '日期：2026-10-12（北京时间）。工作目录：`/home/rootguard-research`；分支：`research/workflow-extensions`。', '',
    '**结论：值得继续研究，但问题应收窄为“近似函数的输出准确性与敏感度可靠性脱节，导致 M-Opt 预算错配”。本实验没有支持“轻微敏感度偏差普遍导致崩溃”；相反，轻微扰动下保留了优化收益。严重低估关键属性可以稳定造成劣于均匀分配的效用，且非零敏感度条件下同样存在。**', '',
    '## 1. 要回答的问题与证据边界', '',
    '本阶段验证：近似函数用于预算分配时，错误属性敏感度是否会使真实任务效用下降；是否劣于均匀 M-Roots；这一现象是否跨预算和独立噪声稳定。只研究目标函数知识失配，不开展目标切换、NER 或根更新。', '',
    '真实任务记为 T=h(g(X))。分配使用 g 或 ghat 在固定公开均值 mu 处的绝对梯度；最终评价始终使用真实 g 与固定 h。这里的问题属于 M-Opt 效用优化，不是根缓存复用失效。没有新增隐私攻击或重新证明隐私；沿用原指数机制的索引空间度量及公开信息固定分配假设。', '',
    '本实验是机制级数值实验，不是完整 LLM Agent 端到端实验。近似函数人为构造，未从公共数据训练；不能据此估计现实错误率或宣布新颖性已成立。', '',
    '## 2. 修改了什么', '',
    '原版基线为 `8eb2560`，研究工作树 HEAD 为 `df576c5`。原核心 Python 源码、输入数据、既有结果未修改。新增实验目录 `experiments/workflow_extensions/target_mismatch/`，更新上级 README 的当前优先方向。没有创建提交或推送远端。', '',
    table(['文件', '职责'], [
       ['core.py', '从原文件提取原预算优化器与静态域；提供真实函数、近似函数、原指数概率的批量采样'],
       ['run_pilot.py', '核对数据，运行扫描，保存逐患者和逐噪声重复的误差及源文件哈希'],
       ['analyze.py', '患者配对 bootstrap；另报固定队列的噪声 Monte Carlo 不确定性'],
       ['configs/pilot.json', '200 样本/模板、30 噪声重复、两个失配家族'],
       ['configs/confirmation_a.json、confirmation_b.json', '各 200 噪声重复，独立种子，增加非零但接近零的敏感度条件'],
       ['tests/test_invariants.py', '六项有效性检查'],
       ['build_report.py', '从正式结果生成本报告和关键证据表'],
       ['README.md', '公式、参数、运行方法、统计口径和限制']]), '',
    '采用轻量 AST 适配，执行原文件中未经改写的优化器定义，避免无关的 NER/模型/姓名数据库导入。最初检查发现适配命名空间缺少 random，导致原梯度核对失败；补齐依赖后六项检查全部通过，随后重新运行正式 pilot。最初探索输出 `pilot_20261012/` 保留供审计，但不作为本报告正式证据。', '',
    '## 3. 基线一致性与数据核对', '',
    '代码核对发现 RQ4 的模板域与原 RQ1 不同，且 RQ4 指数采样先将真实索引取整。本次选用原 RQ1 的逐模板域及未取整指数概率。优化器采用原预计算默认：mode=abs、网格中点噪声矩。该设置在所有组固定；不是针对各患者重新优化。', '',
    '采样概率为 P(s|x) 正比于 exp(-epsilon_i*|t(x)-s|/2)，s 为 0..999。使用批量逆 CDF 采样，逐次对照原方法通过。真实函数的公共均值梯度逐项对照原 DAG 自动微分通过。', '',
    '输入数据：仓库 NHANES 2017–2018 benchmark，每模板 200 个唯一 ID。公开均值来自 holdout_population_means.json（该文件说明排除了测试样本，本次未重新下载 NHANES 验证这一数据构建过程）。三个模板合计 600 条模板记录，不保证是 600 名互不重叠的自然人；分析分别按模板进行。', '',
    table(['模板', '根属性与取值域', '类别 0/1/2 数量', '越界样本'], [
       [n, '; '.join(f'{r}: {lo}–{hi}' for r,(lo,hi) in metas[1]['data_audit'][n]['bounds'].items()),
        '/'.join(map(str, metas[1]['data_audit'][n]['class_counts'])), metas[1]['data_audit'][n]['n']-metas[1]['data_audit'][n]['in_domain_n']]
        for n in ['FIB4','HOMA','ANEMIA']]), '',
    '真实函数：FIB4=age*AST/(PLT*sqrt(ALT))；HOMA=Glu*Ins/405；ANEMIA=MCHC=100*Hb/Hct，RBC 仍为工作流根但不影响本目标。重算真实目标与数据中存储的数值在 1e-8 容差内一致；无噪声对自身真值误差为零。分类口径固定采用 utils/risk_class.py，ANEMIA 的分类评价为 32/36 三分类，HOMA 为 1/2.5 三分类，不使用 DAG 内的二分类表达式。', '',
    'ANEMIA 188/200 为正常类别，且无类别 2：它提供清晰的机制例子，但分类效应不能推广到均衡患者群体或临床应用。', '',
    '## 4. 实验设计', '',
    '三个主组：M-Roots 均分 B/k；真实 M-Opt 使用真实均值梯度；近似 M-Opt 使用失配梯度。A2 只是完整知识参考，局部线性/中点噪声优化不保证真实非线性任务或分类效用全局最优。', '',
    '所有组固定相同患者、公开 mu、根集合、逐模板域、网格 m=1000、预算下限 0.001、总预算 B∈{0.1,0.3,0.7}。这里 B 是索引度量下的根预算参数之和，不是论文每次调用的 epsilon，也不是任意两条记录的标准 LDP 参数。', '',
    '真实公式写成幂乘式 g(mu)*product((x_i/mu_i)^p_i)，三个指数向量分别为 FIB4=(1,1,-0.5,-1)、HOMA=(1,1)、ANEMIA=(1,-1,0)。近似函数只改变一个指数 q_j=p_j*(1+level)，其余不变；零指数 RBC 则引入 q=max(level,0)。这样近似函数在公开均值处值完全正确，但可以具有错误的属性敏感度。', '',
    '直接敏感度扰动 weights 仅是诊断；power 是有明确公式的近似函数。非零根的两个家族产生相同敏感度和分配，不能视为两份独立实验证据。函数输出误差在原始测试输入上测量；近似函数不参与加噪后任务计算。', '',
    '零失配、±10%、±25%、更大偏差和完全遗漏均扫描。确认实验在首轮观察后增加 -90%、-95%、-99% 等仍非零的严重低估条件，属于探索后确认，未预注册。', '',
    table(['正式运行', '噪声重复/患者', '随机种子', '比较单元', '名义任务评估数', '运行秒数'], [
       [rid, 30 if i==0 else 200, 20261012+i, m['cell_count'], f"{m['task_draw_count']:,}", f"{m['elapsed_seconds']:.2f}"]
       for i,(rid,m) in enumerate(zip(RUN_IDS,metas))]), '',
    f"正式运行合计 {sum(m['cell_count'] for m in metas):,} 个比较单元、{sum(m['task_draw_count'] for m in metas):,} 次名义任务评估（包括零失配和相同分配单元的缓存复用，不能当作独立样本数）。两组确认共享原队列，独立的是噪声，不是患者。", '',
    'wMAPE=100*sum(患者平均绝对误差)/sum(|真实值|)；RCE=100*平均分类错误率。差值以百分点表示，正数表示更差。每患者先平均噪声，再做 2000 次配对患者 bootstrap；另报告固定患者队列的噪声 Monte Carlo t 区间。所有 CI 是逐单元探索性区间，未作多重比较校正。', '',
    '## 5. 结果：准确目标知识确实有收益', '',
    '以下为确认 A；确认 B 的方向一致。所有值为百分数。', '',
    table(['模板','B','均匀 wMAPE','真实 M-Opt wMAPE','均匀 RCE','真实 M-Opt RCE'], [
       [n, B, f"{find(a,n,B,'roots')['wmape']:.3f}", f"{find(a,n,B,'opt_true')['wmape']:.3f}",
        f"{find(a,n,B,'roots')['rce']:.3f}", f"{find(a,n,B,'opt_true')['rce']:.3f}"]
       for n in ['FIB4','HOMA','ANEMIA'] for B in [.1,.3,.7]]), '',
    '准确知识组在三个模板、三个预算下均优于均匀组，表明本实验有可供失配损失的优化收益，基线比较不是“原本优化就无效”。', '',
    '## 6. 结果：轻微偏差下较鲁棒', '',
    '确认 A/B 中，level∈{-0.25,-0.1,0,0.1,0.25} 的全部 129 个 power 单元，近似 M-Opt 的数值 wMAPE 均低于均匀 M-Roots。FIB4/HOMA 的非零根对应 ±10%/±25% 均值敏感度偏差；ANEMIA RBC 的正 level 是新增虚假相关指数，不是百分比误差。', '',
    '这不能概括为“任意近似函数误差下都鲁棒”，但明确反对将当前结果包装成“轻微敏感度误差普遍摧毁效用”。少数失配可在某个指标上略优于真实知识组，也符合局部优化并非全局上界的解释。', '',
    '## 7. 结果：严重但非零的敏感度低估也可劣于均匀', '',
    '下表固定 B=0.3，采用确认 A；括号是近似组减均匀组 wMAPE 的患者配对 95% CI。', '',
    table(['模板/根','level','近似函数输出 wMAPE','加噪后任务 wMAPE','任务 RCE','比均匀差多少（百分点）','患者 CI'], [
       [n+'/'+r, level, f"{c['proxy_wmape']:.3f}", f"{c['wmape']:.3f}", f"{c['rce']:.3f}",
        f"{c['delta_wmape_vs_roots']:.3f}", ci(c['delta_wmape_vs_roots_ci95'])]
       for n,r,level in [('FIB4','plt',-.75),('FIB4','plt',-.9),('HOMA','ins',-.9),
                         ('HOMA','ins',-.95),('ANEMIA','hct',-.95),('ANEMIA','hct',-.99),('ANEMIA','hct',-1.)]
       for c in [find(a,n,.3,root=r,level=level)]]), '',
    'FIB4 的 PLT 敏感度低估 75% 就出现数值任务劣于均匀，但该点分类 RCE 仍优于均匀；因此不能仅据数值误差声称所有任务指标都变差。进一步低估至 90% 后，数值与分类均退化。HOMA 的 Ins 低估 90% 后也劣于均匀，但此时近似函数输出误差已经较大，现实合理性需要额外论证。', '',
    '## 8. 关键例子：输出误差较小，却缺少正确梯度', '',
    'ANEMIA 的近似函数把 Hct 指数从 -1 改为 -0.05（level=-0.95）。它在公开均值处数值正确，测试原始输入上的输出 wMAPE 仅 6.726%，但 Hct 的均值敏感度只有真实值的 5%。这不是“小梯度误差”，而是“小输出误差隐藏大梯度失配”。', '',
    table(['B=0.3 的分配','Hb 预算','Hct 预算','RBC 预算'], [
       [label] + [f"{c['eps'][r]:.6f}" for r in ['hb','hct','rbc']]
       for label,c in [('均匀',find(a,'ANEMIA',.3,'roots')),('真实 M-Opt',find(a,'ANEMIA',.3,'opt_true')),
                       ('近似 M-Opt，Hct -95%',find(a,'ANEMIA',.3,root='hct',level=-.95)),
                       ('遗漏 Hct',find(a,'ANEMIA',.3,root='hct',level=-1.))]]), '',
    'Hct 被严重低估，预算偏向 Hb；最终仍用真实 Hb/Hct 计算 MCHC，因此较噪的 Hct 会破坏真实任务。完全遗漏 Hct 的近似函数等价于用公开均值替代分母，其原始输入输出误差仍仅 7.071%，但预算触及下限后任务错误显著放大。严重失配的危害不是近似函数被直接用于最终预测造成的。', '',
    table(['关键条件','确认 A 的 wMAPE / RCE','确认 B 的 wMAPE / RCE','A 比均匀的患者 CI','B 比均匀的患者 CI'], [
       [f'{n}/{r} {level}', f"{ca['wmape']:.3f} / {ca['rce']:.3f}", f"{cb['wmape']:.3f} / {cb['rce']:.3f}",
        ci(ca['delta_wmape_vs_roots_ci95']),ci(cb['delta_wmape_vs_roots_ci95'])]
       for n,r,level in [('ANEMIA','hct',-.95),('ANEMIA','hct',-.99),('FIB4','plt',-.9),('HOMA','ins',-.9)]
       for ca,cb in [(find(a,n,.3,root=r,level=level),find(b,n,.3,root=r,level=level))]]), '',
    'ANEMIA -95% 条件，确认 A 的近似组 RCE 为 20.928%，均匀组为 15.125%，真实 M-Opt 为 7.897%；确认 B 分别为 20.785%、14.992%、7.975%。固定队列的独立噪声重复支持该效应。', '',
    '## 9. 失效边界与预算依赖', '',
    '以下仅给出已扫描点：level=-0.9 表示保留 10% 真实敏感度，-0.95 保留 5%。不能在离散扫描点间宣称精确临界值。', '',
    table(['ANEMIA/Hct，确认 A','B','函数输出 wMAPE','任务 wMAPE','均匀 wMAPE','差值患者 CI'], [
       [level,B,f"{c['proxy_wmape']:.3f}",f"{c['wmape']:.3f}",f"{find(a,'ANEMIA',B,'roots')['wmape']:.3f}",ci(c['delta_wmape_vs_roots_ci95'])]
       for level in [-.9,-.95,-.99] for B in [.1,.3,.7]
       for c in [find(a,'ANEMIA',B,root='hct',level=level)]]), '',
    '近零敏感度和预算下限会使分配出现强烈变化；例如 FIB4/PLT 的 -99% 与完全遗漏在本次配置中得到相同分配和任务结果。这是下限约束的作用，不能将完全一致误认为两个独立现象。', '',
    '## 10. 均匀回退能否解决问题，以及研究价值', '',
    '本实验确认均匀回退避免了所展示严重错配，并留下准确知识的收益。以 ANEMIA B=0.3 为例，真实 M-Opt 相比均匀使 wMAPE 从 2.555% 降至 1.709%（约 33.1% 相对降低），RCE 从 15.125% 降至 7.897%（约 47.8% 相对降低）。', '',
    '**不能由此断言均匀回退“无法满足需求”**：本阶段没有预先给出应用允许的误差上限、成本或效用要求。现有证据说明回退会丢失可观的优化收益，而不是证明这些收益对真实应用必不可少。', '',
    '研究动机目前为：输出预测误差不足以判断目标知识是否适合用于预算优化；在原始输入变化较窄的队列中，遗漏/弱化属性可保持较小输出误差，却导致严重敏感度错配。这个机制现象值得下一步检验，尚不是完整可发表改进。', '',
    '下一阶段应优先验证：独立公共训练集学习的近似函数是否自然出现同样的输出—梯度脱节；在新队列/更多任务中是否重现；在明确应用效用要求下均匀回退损失是否值得优化。不据当前结果直接开展具体修补算法。', '',
    '## 11. 检查、复现与产物', '',
    '六项测试通过：①原始真值和零失配恢复；②近似函数解析梯度对照有限差分；③真实梯度对照原 DAG；④预算守恒/下限与零敏感度根；⑤批量采样逐次对照原方法；⑥使用原 RootGuard 缓存检查重复访问/后处理隔离。每个正式运行 checks.json 还检查零失配单元任务结果精确恢复真实 M-Opt，患者 ID 顺序完全配对。', '',
    '运行环境：已有 rootguard Conda 环境，Python '+metas[1]['python']+'、NumPy '+metas[1]['numpy']+'、SciPy '+metas[1]['scipy']+'；未新增安装依赖。配置、原源码和输入 SHA-256 在各 metadata.json 中。git diff --check 与核心源码差异核对另存于 validation.txt。', '',
    '运行命令：', '', '```bash',
    'PYTHON=/root/miniconda3/envs/rootguard/bin/python',
    '$PYTHON -m unittest discover -s experiments/workflow_extensions/target_mismatch/tests -v',
    '# 替换 <config> 为 pilot / confirmation_a / confirmation_b，<new_id> 必须未存在',
    '$PYTHON experiments/workflow_extensions/target_mismatch/run_pilot.py \\',
    '  --config experiments/workflow_extensions/target_mismatch/configs/<config>.json \\',
    '  --out experiments/workflow_extensions/outputs/target_mismatch/<new_id>',
    '$PYTHON experiments/workflow_extensions/target_mismatch/analyze.py \\',
    '  experiments/workflow_extensions/outputs/target_mismatch/<new_id>',
    '$PYTHON experiments/workflow_extensions/target_mismatch/build_report.py \\',
    '  --out experiments/workflow_extensions/outputs/target_mismatch/report_20261012', '```', '',
    '本次已完成的运行：', '',
    '\n'.join(f'- [{rid} 汇总](../{rid}/SUMMARY.md)：同目录含完整单元、逐患者结果、逐重复误差、配置和来源。' for rid in RUN_IDS), '',
    '关键证据表：`key_evidence.csv`；完整确认扫描：`all_confirmation_cells.csv`。表格保留每一个 level 的结果，不只保留正向现象。此报告未生成科研图；失配轨迹可直接从 summary.csv 作图。', '',
    '## 12. 原研究对应关系', '',
    '本报告仅做实验，不做新的文献新颖性检索。原论文与代码分别见 [Dependency-Aware Privacy for Multi-turn Agents](https://arxiv.org/abs/2605.03188) 和 [RootGuard 原仓库](https://github.com/danshumaan/rootguard)。前期已讨论的“学习近似目标函数”方向需要真实学习实验来承接；本次受控结果不能替代这一步。', '']
    (args.out / 'REPORT.md').write_text('\n'.join(lines))
    fields = ['run','template','B','method','family','root','level','proxy_wmape','importance_l1','wmape','rce',
              'delta_wmape_vs_roots','delta_wmape_vs_roots_ci95','delta_wmape_vs_roots_mc_ci95',
              'delta_rce_vs_roots','delta_rce_vs_roots_ci95','eps']
    all_rows = [dict(c, run=rid) for rid,data in zip(RUN_IDS[1:],analyses[1:]) for c in data]
    for filename, rows in [('all_confirmation_cells.csv', all_rows),('key_evidence.csv', [r for r in all_rows
        if r['B']==.3 and (r['method']!='opt_proxy' or (r['template'],r['root']) in [('ANEMIA','hct'),('FIB4','plt'),('HOMA','ins')])])]:
        with (args.out / filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)
    manifest = {'runs': RUN_IDS, 'report_builder_sha256': sha(Path(__file__)),
                'report_sha256': sha(args.out/'REPORT.md'),
                'analysis_sha256': {rid: sha(base/rid/'analysis.json') for rid in RUN_IDS}}
    (args.out / 'report_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(args.out / 'REPORT.md')

if __name__ == '__main__':
    main()
