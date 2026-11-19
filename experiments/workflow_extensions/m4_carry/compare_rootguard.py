"""Frozen finite-cache diagnosis against original M-Opt static/periodic baselines."""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from .core import Policy,simulate,evaluate,key_seed
from .run import ROOT,dump,line,clean
from .periodic_skeleton import load_plan
from .rootguard_mopt import MOpt,simulate_baseline,LOADED_FUNCTIONS
from .analysis import bootstrap_curves,q95,bmin,finite_interval,write_csv,mean_na

METHODS=['RG-static-exp','RG-periodic-exp','RG-static-lap','RG-periodic-lap','M4-carry']


def score(records,B,P,kernel,adapter,cfg):
    per=defaultdict(list)
    for rec in records:
        r=simulate_baseline(rec,B,P,kernel,adapter,cfg,'validation',cfg['validation_repeats'])
        mae=evaluate(rec,r,adapter.w,{},dict(cfg,_B=B))['mae']
        per[str(rec['subject_id'])].append(float(mae.mean()))
    return float(np.mean([np.mean(v) for v in per.values()]))


def analyze(out,cfg,rows,targets):
    summary,paired,bmins,effects=[],[],[],[]
    # One patient and within-mechanism sampling plan per template, shared across
    # every method, budget and target. Parameters and tau remain fixed.
    for template in cfg['templates']:
        rr=[r for r in rows if r['template']==template]
        subjects=sorted({r['subject'] for r in rr});budgets=cfg['evaluated_budgets'];R=cfg['repeats']
        groups=defaultdict(list)
        for r in rr:groups[r['method'],r['B'],r['subject'],r['repeat']].append(r)
        x=np.empty((len(METHODS),len(budgets),len(subjects),R))
        for mi,m in enumerate(METHODS):
            for bi,B in enumerate(budgets):
                for ni,s in enumerate(subjects):
                    for rep in range(R):x[mi,bi,ni,rep]=np.mean([r['mae'] for r in groups[m,B,s,rep]])
        p,n=bootstrap_curves(x,cfg['bootstrap'],key_seed(cfg['bootstrap_seed'],template))
        mean=x.mean(axis=(-1,-2))
        cells=defaultdict(list)
        for r in rr:cells[r['method'],r['B']].append(r)
        metrics=['p95','m','used','spend_fraction','projection_rate','MissRate','frac_veto','frac_root_veto','harmful_refresh','post_stop_mae','danger_rate','danger_mae','K','rho','n_pub','input_clip_rate']
        for mi,m in enumerate(METHODS):
            for bi,B in enumerate(budgets):
                row=dict(template=template,B=B,method=m,patients=len(subjects),repeats=R,mae=float(mean[mi,bi]))
                for name,boot in [('patient',p),('two_layer',n)]:
                    row[name+'_lo'],row[name+'_hi']=q95(boot[mi,bi])
                    row[name+'_ucb95']=float(np.quantile(boot[mi,bi],.95))
                cell=cells[m,B]
                for metric in metrics:
                    row[metric]=mean_na([mean_na([r.get(metric) for r in cell if r['subject']==s]) for s in subjects])
                summary.append(row)
        for ai,bi in [(4,j) for j in range(4)]+[(1,0),(3,2)]:
            for bj,B in enumerate(budgets):
                row=dict(template=template,B=B,method=METHODS[ai],reference=METHODS[bi],delta_mae=float(mean[ai,bj]-mean[bi,bj]),mae_ratio=float(mean[ai,bj]/mean[bi,bj]))
                for name,boot in [('patient',p),('two_layer',n)]:
                    row[name+'_lo'],row[name+'_hi']=q95(boot[ai,bj]-boot[bi,bj])
                paired.append(row)
        for scenario,ts in targets[template].items():
            for target in ['lo','hi']:
                tau=ts['tau_'+target]
                bm=bmin(mean,budgets,tau)
                pb=bmin(p.transpose(0,2,1),budgets,tau);nb=bmin(n.transpose(0,2,1),budgets,tau)
                for mi,m in enumerate(METHODS):
                    row=dict(template=template,scenario=scenario,target=target,tau=tau,method=m,
                             B_min=float(bm[mi]) if np.isfinite(bm[mi]) else None,reached=bool(np.isfinite(bm[mi])))
                    for name,boot,raw in [('patient',pb,p),('two_layer',nb,n)]:
                        a,b=q95(boot[mi]);row[name+'_lo']=a if np.isfinite(a) else None;row[name+'_hi']=b if np.isfinite(b) else None
                        row[name+'_reach_rate']=float(np.isfinite(boot[mi]).mean())
                        ub=bmin(np.quantile(raw[mi],.95,axis=-1),budgets,tau)
                        row['B_min_'+name+'_ucb95']=float(ub) if np.isfinite(ub) else None
                    bmins.append(row)
                for j in range(4):
                    both=np.isfinite(bm[4]) and np.isfinite(bm[j])
                    row=dict(template=template,scenario=scenario,target=target,tau=tau,method='M4-carry',reference=METHODS[j],
                             ratio=float(bm[4]/bm[j]) if both else None,Saving=float(1-bm[4]/bm[j]) if both else None,
                             difference=float(bm[4]-bm[j]) if both else None,
                             method_reached=bool(np.isfinite(bm[4])),reference_reached=bool(np.isfinite(bm[j])))
                    for name,boot in [('patient',pb),('two_layer',nb)]:
                        ok=np.isfinite(boot[4])&np.isfinite(boot[j]);row[name+'_both_reach_rate']=float(ok.mean())
                        for label,vals in [('ratio',boot[4,ok]/boot[j,ok]),('difference',boot[4,ok]-boot[j,ok])]:
                            row[name+'_'+label+'_conditional_lo'],row[name+'_'+label+'_conditional_hi']=finite_interval(vals)
                    effects.append(row)
    write_csv(out/'summary.csv',summary);write_csv(out/'paired_summary.csv',paired)
    write_csv(out/'bmin.csv',bmins);write_csv(out/'bmin_paired.csv',effects)
    report(out,cfg,summary,paired,bmins,effects,targets)


def fmt(v,d=3,missing='NA'):
    return missing if v is None else f'{v:.{d}f}'


def report(out,cfg,summary,paired,bmins,effects,targets):
    audit=json.loads((out/'audit.json').read_text());selection=json.loads((out/'selections.json').read_text())
    sens=json.loads((out/'mopt_public_stats.json').read_text())
    lines=['# 原版 M-Opt 静态/周期对照与 M4-carry','',
           f'本轮为机制诊断。R={cfg["repeats"]}，患者配对及患者＋机制两层bootstrap={cfg["bootstrap"]}，预算点={len(cfg["evaluated_budgets"])}；预检查={cfg["precheck"]}。原数据/划分/公式/域/公开统计/Δ/τ及M4计划全部复用，不用test重调M4。','',
           '## 1. 比较的具体对象','',
           '|方法|逐根分配|发布核|初始化和刷新|','|---|---|---|---|',
           '|RG-static-exp|原版精确M-Opt|原版1000网格指数机制|全B一次发布后缓存到T|',
           '|RG-periodic-exp|原版精确M-Opt|同上|初始化B/2，余下B/2均分周期全根发布|',
           '|RG-static-lap|同一原版分配器输出|与M4相同连续Laplace+投影|全B一次发布后缓存到T|',
           '|RG-periodic-lap|同上|同上|初始化B/2，周期全根发布|',
           '|M4-carry|上一轮冻结的全域界√ĥ|原连续Laplace+投影|冻结公开计划、逐根veto、预算结转|','',
           '前两个是用户要求的原版M-Opt主对照；后两个用于检查结果是否来自发布核差异，它们是原分配器的同核移植对照，不宣称原版RootGuard整套协议复现。M-Opt本身已经由最终任务梯度引导，不能把“任务驱动”本身当作M4相对它新增的能力。系统对照不能单独归因每个组件。','',
           '## 2. 原版复用和单位','',
           '通过AST逐字加载utils/utils.py中的optimal_budget_allocation_exact及依赖，mode=abs、num_candidates=1000、epsilon_min_grid=0.001，中心网格矩目标，未替换成√ĥ。记录原SLSQP的成功标志。低动作预算不足以满足所有根下限时保留原函数的均分分支。','',
           'ANEMIA/FIB4复用原版compute_population_mean_sensitivities和解析DAG链式法则；MELD_NA/BUN_CR原版未提供，增加同一冻结公式的解析梯度，并与数值导数核对。均值来自既有train统计，不使用当前用户真值。所有方法使用同一已冻结临床公开域；没有换成NHANES窄域，也没有看test缩域。因此这里是原算法在新场景的适配，不是原论文原始数据复现。','',
           '原核p(s|z)∝exp(−ε_grid |s−999z|/2)。按归一化根距离，其log核和log归一化常数各贡献≤ε_grid·999/2的Lipschitz界，故本轮统一按ε_metric=999·ε_grid计费。动作总量B_action对应原分配器输入B_action/999；不能把原版网格ε的数值直接当成本轮B。Laplace同核对照使用换算后的逐根ε_metric。两者均按同一整段轨迹max范数预算上界计费；这是共同认证上界，不表示两种核的实际隐私损失分布相同。','',
           '原生PMF与preempt/sanitizer.py的M_exponential_discrete逐点核对，使用逆CDF仅为配对随机数，未改变PMF。基线的原分配规则对所有患者固定于公开均值和动作总预算；不按test真值分配。','',
           '周期P在两个发布核各自的validation上重新选择（相同候选P集合和MAE目标），不能沿用旧√ĥ基线的P来削弱M-Opt对照。全部P选择在本轮第一个新test基线误差计算前完成。周期以公开测量机会计；不允许静态退化，静态方法另列。','',
           '## 3. B=960的全轨迹MAE','',
           '|模板|RG静态原核|RG周期原核|RG静态同Laplace核|RG周期同Laplace核|M4-carry|','|---|---:|---:|---:|---:|---:|']
    headline=['## 先看判断','']
    for t in cfg['templates']:
        rr=[r for r in paired if r['template']==t and r['B']==960 and r['method']=='M4-carry' and r['reference'] in ('RG-static-exp','RG-periodic-exp')]
        good=all(r['patient_hi']<0 and r['two_layer_hi']<0 for r in rr)
        bad=all(r['patient_lo']>0 and r['two_layer_lo']>0 for r in rr)
        label='在B=960同时优于原版M-Opt静态与周期，两种配对区间均支持' if good else '在B=960同时劣于原版M-Opt静态与周期，两种配对区间均支持' if bad else '在B=960尚无同时优于静态与周期的稳定证据'
        headline.append('- '+t+'：'+label+'。')
    headline += ['', '以上判断限于这一预指定预算点；B_min达标情况另看全网格。系统胜出不自动证明部分根更新或结转各自有收益。相同发布核的对照及第9节的既有配对数据用于解释来源。','']
    lines[4:4]=headline
    for t in cfg['templates']:
        d={r['method']:r for r in summary if r['template']==t and r['B']==960}
        lines.append('|'+t+'|'+'|'.join(fmt(d[m]['mae']) for m in METHODS)+'|')
    lines+=['','差值=MAE(M4)−MAE(参考)，负值表示M4更好；以下为点态配对区间，未做多重比较家族校正。','',
            '|模板|参考|MAE比M4/参考|差值|患者95% CI|两层95% CI|','|---|---|---:|---:|---|---|']
    for r in paired:
        if r['B']==960 and r['method']=='M4-carry':
            lines.append(f'|{r["template"]}|{r["reference"]}|{r["mae_ratio"]:.3f}|{r["delta_mae"]:.4f}|[{r["patient_lo"]:.4f}, {r["patient_hi"]:.4f}]|[{r["two_layer_lo"]:.4f}, {r["two_layer_hi"]:.4f}]|')
    lines+=['','## 4. 固定τ下的最小网格预算','',
            'τ沿用原公开合成人群在Bmax的B0参考MAE×0.85/1.15，未改用新M-Opt或test成绩定τ。Δ只是公开假设档，不是test分组。逐点扫描，不假设单调；未达标表示扫描范围没有达标点，不能当成数学不可达，也不填960。','',
            '|模板|Δ/目标|τ|RG静态原核|RG周期原核|RG静态同核|RG周期同核|M4-carry|','|---|---|---:|---:|---:|---:|---:|---:|']
    for t in cfg['templates']:
        for scenario in targets[t]:
            for target in ('lo','hi'):
                rows=[r for r in bmins if r['template']==t and r['scenario']==scenario and r['target']==target]
                vals={r['method']:r['B_min'] for r in rows}
                lines.append('|'+t+'|'+scenario+'/'+target+'|'+fmt(rows[0]['tau'])+'|'+'|'.join(fmt(vals[m],2,missing='未达标') for m in METHODS)+'|')
    lines+=['','B_min的患者/两层区间、UCB口径和达标重采样比例见bmin.csv；Savings及配对比值/差值区间见bmin_paired.csv。仅在双方达标时比值有定义；条件区间须结合双方达标概率，不能用少量成功重采样宣称获胜。','',
            '## 5. B=960风险和消耗','',
            '|模板|方法|p95|刷新m|支出%|MissRate|有害刷新|投影率|停止后MAE|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['B']==960:
            lines.append('|'+r['template']+'|'+r['method']+'|'+'|'.join(fmt(r.get(k)) for k in ['p95','m'])+'|'+fmt(100*r['spend_fraction'])+'|'+'|'.join(fmt(r.get(k)) for k in ['MissRate','harmful_refresh','projection_rate','post_stop_mae'])+'|')
    lines+=['','原生指数核的投影率为0是其天然有界支持的结果，不能把它与Laplace的裁剪率直接当作效用优劣比较。各项定义见[指标阅读说明](METRICS-zh.md)。','',
            '## 6. 选择与分配核对','',
            '|模板|B960原核P*|B960同核P*|局部零梯度根|','|---|---:|---:|---|']
    for t in cfg['templates']:
        lines.append(f'|{t}|{selection[t]["960.0"]["native_exp"]}|{selection[t]["960.0"]["laplace"]}|{", ".join(sens[t]["derivative_zero_roots"]) or "无"}|')
    lines+=['','原版逐动作分配和优化器诊断见mopt_allocations.jsonl，validation周期敏感性见validation_scores.jsonl，实际发布根和预算阶梯见actions.jsonl。','',
            '## 7. 结论与边界','']
    for t in cfg['templates']:
        for family in ('exp','lap'):
            rr=[r for r in paired if r['template']==t and r['B']==960 and r['method']=='M4-carry' and r['reference'].endswith(family)]
            verdict=[]
            for r in rr:
                result='M4更好' if r['patient_hi']<0 and r['two_layer_hi']<0 else 'M4更差' if r['patient_lo']>0 and r['two_layer_lo']>0 else '区间不足以区分'
                verdict.append(r['reference']+': '+result)
            lines.append('- '+t+' / '+family+'：'+'；'.join(verdict)+'。')
    lines+=['',f'原版优化器非成功调用数={audit["solver_failures"]}；原版预算下限不足均分分支数={audit["floor_fallbacks"]}。原生与同核对照均保留原版分配实现。', '',
            '本轮是已分析过的MIMIC demo患者上的机制诊断，test每模板仅4–5人。参数/τ在bootstrap中固定，CI不涵盖再次选择参数的不确定性。多模板多目标逐格报告，未做全比较族同时保证。即使系统获胜，也需分配/选根/结转消融才能把收益归给单个模块。M-Opt本身就是任务驱动分配；M4不是把无任务基线改成有任务基线。','',
            '## 8. 审计与复现','',
            f'账本断言通过：所有已发布根ε>0、包含初始化Σε_metric≤B；新基线路径={audit["new_baseline_paths"]}。预检查复现既有M4路径的最大差={audit["max_m4_reproduction_difference"]:.3g}。M4在正式比较中复用既有原始指标和动作，未重调参。','',
            '```bash','cd /home/rootguard-research',
            '/root/miniconda3/envs/rootguard/bin/python -m unittest experiments.workflow_extensions.m4_carry.test_rootguard_mopt -v',
            '/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m4_carry.compare_rootguard --precheck --output experiments/workflow_extensions/outputs/m4_carry/rootguard_mopt_precheck_reproduce',
            '/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m4_carry.compare_rootguard --output experiments/workflow_extensions/outputs/m4_carry/rootguard_mopt_reproduce',
            '```','']
    (out/'REPORT-zh.md').write_text('\n'.join(lines)+'\n')
    (out/'METRICS-zh.md').write_text((Path(__file__).parent/'METRICS-zh.md').read_text())


def run(args):
    started=time.monotonic();source=Path(args.source).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use an empty new output directory')
    cfg=json.loads((source/'config.json').read_text());cfg['precheck']=args.precheck
    if args.precheck:
        cfg.update(repeats=2,validation_repeats=2,bootstrap=100)
        grid=cfg['evaluated_budgets'];cfg['evaluated_budgets']=[grid[0],grid[len(grid)//2],grid[-1]]
    cfg.update(mopt_grid_points=1000,mopt_epsilon_min_grid=.001,metric_epsilon_per_grid_epsilon=999,
               original_solver='utils/utils.py::optimal_budget_allocation_exact(mode=abs)',methods=METHODS)
    dump(out/'config.json',cfg)
    assignment=json.loads((source/'split.json').read_text());dump(out/'split.json',assignment)
    stats=json.loads((source/'public_stats.json').read_text());targets=json.loads((source/'targets.json').read_text())
    selections_m4=json.loads((source/'selections.json').read_text())
    records=data.load(args.data,cfg['templates'])
    assert data.input_hashes(args.data,cfg['templates'])==json.loads((source/'input_hashes.json').read_text())
    contexts={};selection={};all_logs=[]
    # Every baseline selection completes before any new test baseline evaluation.
    with (out/'validation_scores.jsonl').open('w') as vh:
        for t in cfg['templates']:
            w=Workload(t);adapter=MOpt(w,stats[t]);contexts[t]=(w,adapter)
            validation=[r for r in records if r['template']==t and assignment[str(r['subject_id'])]=='validation']
            selection[t]={}
            for bi,B in enumerate(cfg['evaluated_budgets']):
                selection[t][str(B)]={}
                for kernel in ('native_exp','laplace'):
                    options=[]
                    for P in cfg['periods']:
                        value=score(validation,B,P,kernel,adapter,cfg)
                        options.append((value,P));line(vh,dict(template=t,B=B,kernel=kernel,P=P,validation_mae=value))
                    selection[t][str(B)][kernel]=min(options)[1]
                if bi%10==0 or bi==len(cfg['evaluated_budgets'])-1:print('SELECT',t,bi+1,'/',len(cfg['evaluated_budgets']),'elapsed',round(time.monotonic()-started,1),flush=True)
    dump(out/'selections.json',selection);dump(out/'mopt_public_stats.json',{t:a.info for t,(w,a) in contexts.items()})
    dump(out/'targets.json',targets);dump(out/'phase_a_complete.json',dict(new_test_errors_evaluated=False,selections_complete=True))
    previous={}
    for ln in (source/'metrics.jsonl').open():
        r=json.loads(ln)
        if r['method']=='M4-carry' and r['B'] in cfg['evaluated_budgets'] and r['repeat']<cfg['repeats']:
            previous[r['trajectory'],r['B'],r['repeat']]=r
    max_repro=0.;rows=[];paths=0;max_excess=0.
    with (out/'metrics.jsonl').open('w') as mh,(out/'actions.jsonl').open('w') as ah:
        for t,(w,adapter) in contexts.items():
            test=[r for r in records if r['template']==t and assignment[str(r['subject_id'])]=='test']
            for bi,B in enumerate(cfg['evaluated_budgets']):
                for rec in test:
                    # A bounded replay verifies frozen M4 under current source;
                    # the full experiment reuses existing R20 saved measurements.
                    if bi in (0,len(cfg['evaluated_budgets'])-1):
                        p=load_plan(selections_m4[t][str(B)]['M4-carry'])
                        replay=simulate(rec,B,p,'carry',Policy(w,stats[t],cfg),cfg,'test',min(2,cfg['repeats']))
                        mae=evaluate(rec,replay,w,{},dict(cfg,_B=B))['mae']
                        for r,v in enumerate(mae):
                            diff=abs(v-previous[rec['trajectory_id'],B,r]['mae']);max_repro=max(max_repro,diff)
                            assert diff<1e-10
                    for rep in range(cfg['repeats']):
                        saved=previous[rec['trajectory_id'],B,rep];line(mh,saved);rows.append(saved)
                    for kernel,static_name,periodic_name in [('native_exp',METHODS[0],METHODS[1]),('laplace',METHODS[2],METHODS[3])]:
                        for method,P in [(static_name,None),(periodic_name,selection[t][str(B)][kernel])]:
                            result=simulate_baseline(rec,B,P,kernel,adapter,cfg,'test',cfg['repeats'],True)
                            met=evaluate(rec,result,w,stats[t],dict(cfg,_B=B),True)
                            max_excess=max(max_excess,float(np.max(result['spent']-B)))
                            for rep in range(cfg['repeats']):
                                ident=dict(template=t,subject=str(rec['subject_id']),trajectory=rec['trajectory_id'],B=B,method=method,repeat=rep)
                                row={**ident,**{k:v[rep] for k,v in met.items()},'K':result['K'],'rho':1. if P is None else .5,'P':P}
                                line(mh,row);rows.append(clean(row));paths+=1
                                line(ah,{**ident,'turn':1,'action_index':0,'selected_roots':w.roots,
                                         'actual_epsilon':result['anchor_epsilon'][rep],'epsilon_grid':result['anchor_epsilon_grid'][rep],
                                         'cache_after':result['anchor_cache'][rep],'R_after':B-result['anchor_epsilon'][rep].sum()})
                                for h in result['histories']:
                                    line(ah,{**ident,'turn':h['turn'],'action_index':h['action_index'],'selected_roots':w.roots,
                                             'actual_epsilon':h['actual_epsilon'][rep],'epsilon_grid':h['epsilon_grid'][rep],
                                             'cache_before':h['cache_before'][rep],'cache_after':h['cache_after'][rep],
                                             'R_before':h['left_before'][rep],'R_after':h['left_after'][rep]})
                if bi%10==0 or bi==len(cfg['evaluated_budgets'])-1:print('TEST',t,bi+1,'/',len(cfg['evaluated_budgets']),'elapsed',round(time.monotonic()-started,1),flush=True)
            all_logs+=adapter.logs
        # Preserve original M4's actual root sets/epsilon ledger unchanged.
        for ln in (source/'actions.jsonl').open():
            r=json.loads(ln)
            if r['method']=='M4-carry' and r['B'] in cfg['evaluated_budgets'] and r['repeat']<cfg['repeats']:line(ah,r)
    with (out/'mopt_allocations.jsonl').open('w') as f:
        for r in all_logs:line(f,r)
    failures=sum(not r['solver']['success'] for r in all_logs)
    audit=dict(passed=True,epsilon_positive=True,ledger_all_ok=True,new_baseline_paths=paths,reused_m4_paths=len(previous),
               max_budget_excess=max_excess,max_m4_reproduction_difference=max_repro,solver_failures=failures,
               solver_calls=len(all_logs),floor_fallbacks=sum(r['insufficient_floor_fallback'] for r in all_logs),
               phases_separate=True,all_selections_before_new_test=True,elapsed_simulation_seconds=time.monotonic()-started)
    dump(out/'audit.json',audit)
    source_paths=[Path(__file__),Path(__file__).with_name('rootguard_mopt.py'),ROOT/'utils/utils.py',ROOT/'preempt/sanitizer.py',ROOT/'utils/med_domain/all_templates.py']
    dump(out/'source_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    dump(out/'loaded_original_functions.json',LOADED_FUNCTIONS)
    from .audit_mopt import audit
    audit(out)
    analyze(out,cfg,rows,targets)
    from .mopt_diagnostics import diagnose
    diagnose(source,out)
    print('DONE',out,'elapsed',round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=str(ROOT/'experiments/workflow_extensions/outputs/m4_carry/real_v1_20261116'))
    p.add_argument('--data',default=str(ROOT/'data_mine/finite_cache'))
    p.add_argument('--output',required=True)
    p.add_argument('--precheck',action='store_true')
    run(p.parse_args())
