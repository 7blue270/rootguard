"""Root changes, task staleness and noise/freshness diagnostics; no tuning."""
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.core import key_seed,task
from experiments.workflow_extensions.m4_carry.analysis import write_csv,bootstrap_curves,q95
from experiments.workflow_extensions.m4_carry.run import ROOT,dump,line

TEMPLATES=['MELD_NA','ANEMIA','FIB4','BUN_CR']
PARTS=['all','train','validation','test']


def quantile(v,w,p):
    v=np.asarray(v);w=np.asarray(w)
    if not len(v):return None
    ix=np.argsort(v);cw=np.cumsum(w[ix]);j=np.searchsorted(cw,p*cw[-1],side='left')
    return float(v[ix[min(j,len(v)-1)]])


def mean(v):return float(np.mean(v)) if len(v) else None


def stats(records,assignment,out):
    roots_out=[];nodes_out=[];events_out=[];patients_out=[];pairs_out=[];cohorts=[]
    for template in TEMPLATES:
        w=Workload(template);full=[r for r in records if r['template']==template]
        for part in PARTS:
            selected=[r for r in full if part=='all' or assignment[str(r['subject_id'])]==part]
            assert len({str(r['subject_id']) for r in selected})==len(selected),'This audit expects one trajectory per patient'
            if not selected:continue
            duration=[];root_patient=defaultdict(list);node_patient=defaultdict(list);event_by_node=defaultdict(list);pair_by_roots=defaultdict(list)
            for rec in selected:
                x=data.replay(rec);values=w.evaluate(x);sid=str(rec['subject_id']);trajectory=rec['trajectory_id']
                days=(datetime.fromisoformat(rec['events'][-1]['available_time'])-datetime.fromisoformat(rec['initial_available_time'])).total_seconds()/86400
                duration.append(days);seq=np.array([e['turn']-1 for e in rec['events']])
                for i,root in enumerate(w.roots):
                    ix=[e['turn']-1 for e in rec['events'] if root in e['measured_roots']]
                    delta=x[ix,i]-x[np.array(ix,dtype=int)-1,i] if ix else np.array([])
                    prev=x[np.array(ix,dtype=int)-1,i] if ix else np.array([])
                    absd=abs(delta);rel=100*absd/prev;domain=100*absd/w.width[i]
                    tv=float(abs(np.diff(x[:,i])).sum());net=float(abs(x[-1,i]-x[0,i]))
                    rr=dict(template=template,split=assignment[sid],subject=sid,trajectory=trajectory,root=root,
                            measurements=len(ix),changed=int(np.sum(absd>1e-10)),step_abs_mean=mean(absd),
                            step_relative_pct_mean=mean(rel),mean_value=float(x[:,i].mean()),
                            initial_value=float(x[0,i]),end_value=float(x[-1,i]),total_variation=tv,net_displacement=net,
                            directionality=net/tv if tv>1e-10 else None,
                            frozen_root_mae=float(abs(x[:,i]-x[0,i]).mean()),
                            max_displacement=float(abs(x[:,i]-x[0,i]).max()),
                            max_displacement_pct_initial=float(100*abs(x[:,i]-x[0,i]).max()/x[0,i]),
                            range_pct_initial=float(100*np.ptp(x[:,i])/x[0,i]),
                            clip_fraction=float(np.mean((x[:,i]<w.lo[i])|(x[:,i]>w.hi[i]))))
                    root_patient[root].append((rr,absd,rel,domain))
                    if part=='all':patients_out.append(rr)
                for j,node in enumerate(w.nodes):
                    y=values[node];stale=abs(y-y[0]);ix=[q['turn']-1 for q in rec['queries'] if q['node']==node]
                    nr=dict(subject=sid,trajectory=trajectory,mae=float(stale.mean()),truth_abs=float(abs(y).mean()),
                            p95=float(np.quantile(stale,.95)),initial=float(y[0]),mean_value=float(y.mean()),
                            max_displacement=float(stale.max()),max_relative_pct=float(100*stale.max()/abs(y[0])) if abs(y[0])>1e-12 else None,
                            query_mae=mean(stale[ix]),query_truth_abs=mean(abs(y[ix])),
                            directionality=float(abs(y[-1]-y[0])/abs(np.diff(y)).sum()) if abs(np.diff(y)).sum()>1e-12 else None,
                            changed_batch_fraction=float(np.mean(abs(y[seq]-y[seq-1])>1e-10)))
                    if node==w.target and w.classes(y) is not None:nr['rce']=float(100*np.mean(abs(w.classes(y)-w.classes(y[0]))/2))
                    node_patient[node].append(nr)
                for e in rec['events']:
                    idx=e['turn']-1;old=x[idx-1];new=x[idx];oldvalues={n:values[n][idx-1] for n in w.nodes}
                    changed=bool(np.any(abs(new-old)>1e-10))
                    for node in w.nodes:
                        joint=abs(values[node][idx]-oldvalues[node]);individual=0.
                        for i,root in enumerate(w.roots):
                            if root in e['measured_roots']:
                                one=old.copy();one[i]=new[i];individual+=abs(w.evaluate(one)[node]-oldvalues[node])
                        er=dict(subject=sid,trajectory=trajectory,turn=e['turn'],node=node,joint=float(joint),individual=float(individual),root_changed=changed,
                                task_unchanged=bool(joint<1e-10),measured_roots=e['measured_roots'])
                        event_by_node[node].append(er)
                        if part=='all':events_out.append(dict(template=template,split=assignment[sid],**er))
                for i in range(len(w.roots)):
                    for j in range(i+1,len(w.roots)):
                        ix=[e['turn']-1 for e in rec['events'] if w.roots[i] in e['measured_roots'] and w.roots[j] in e['measured_roots']]
                        if ix:
                            ds=x[ix]-x[np.array(ix)-1]
                            pair_by_roots[i,j].append((sid,ds[:,i],ds[:,j]))
            cohorts.append(dict(template=template,split=part,patients=len(selected),trajectories=len(selected),
                                batches=sum(len(r['events']) for r in selected),batches_median=float(np.median([len(r['events']) for r in selected])),
                                duration_days_median=float(np.median(duration)),duration_days_min=float(min(duration)),duration_days_max=float(max(duration))))
            for i,root in enumerate(w.roots):
                pp=root_patient[root];v=[];rel=[];dom=[];ww=[]
                for rr,a,b,c in pp:
                    if len(a):v.extend(a);rel.extend(b);dom.extend(c);ww.extend([1/len(a)]*len(a))
                unit=selected[0].get('units',{}).get(root,'years' if root=='age' else '')
                if isinstance(unit,list):unit=', '.join(unit)
                r=dict(template=template,split=part,root=root,unit=unit,
                       patients=len(pp),patients_measured=sum(len(a)>0 for rr,a,b,c in pp),measurements=sum(len(a) for rr,a,b,c in pp),
                       changed_measurements=sum(rr['changed'] for rr,a,b,c in pp),
                       mean_value=float(np.mean([rr['mean_value'] for rr,a,b,c in pp])),domain_width=float(w.width[i]),
                       step_abs_mean=float(np.average(v,weights=ww)) if v else None,
                       step_abs_median=quantile(v,ww,.5),step_abs_p90=quantile(v,ww,.9),
                       step_relative_pct_median=quantile(rel,ww,.5),step_relative_pct_p90=quantile(rel,ww,.9),
                       step_domain_pct_mean=float(np.average(dom,weights=ww)) if v else None,
                       step_domain_pct_p90=quantile(dom,ww,.9),
                       changed_fraction=float(np.average(np.asarray(v)>1e-10,weights=ww)) if v else None,
                       relative_over10_fraction=float(np.average(np.asarray(rel)>10,weights=ww)) if v else None,
                       domain_over5_fraction=float(np.average(np.asarray(dom)>5,weights=ww)) if v else None,
                       domain5_threshold_raw=float(.05*w.width[i]),
                       max_displacement_pct_initial_median=float(np.median([rr['max_displacement_pct_initial'] for rr,a,b,c in pp])),
                       range_pct_initial_median=float(np.median([rr['range_pct_initial'] for rr,a,b,c in pp])),
                       directionality_median=float(np.median([rr['directionality'] for rr,a,b,c in pp if rr['directionality'] is not None])) if any(rr['directionality'] is not None for rr,a,b,c in pp) else None,
                       frozen_root_mae=float(np.mean([rr['frozen_root_mae'] for rr,a,b,c in pp])),
                       clip_fraction=float(np.mean([rr['clip_fraction'] for rr,a,b,c in pp])))
                r['frozen_root_wmape']=100*r['frozen_root_mae']/r['mean_value'];roots_out.append(r)
            for node,pp in node_patient.items():
                bysubject=defaultdict(list)
                for e in event_by_node[node]:bysubject[e['subject']].append(e)
                joint=np.array([np.mean([e['joint'] for e in ee]) for ee in bysubject.values()]);individual=np.array([np.mean([e['individual'] for e in ee]) for ee in bysubject.values()])
                changed_unchanged=[]
                for ee in bysubject.values():
                    valid=[e for e in ee if e['root_changed']]
                    if valid:changed_unchanged.append(np.mean([e['task_unchanged'] for e in valid]))
                mae=np.array([r['mae'] for r in pp]);den=np.array([r['truth_abs'] for r in pp]);rng=np.random.default_rng(key_seed(20261117,template,part,node))
                ix=rng.integers(len(pp),size=(2000,len(pp)));boot=mae[ix].mean(axis=1)
                r=dict(template=template,split=part,node=node,final=node==w.target,patients=len(pp),
                       true_mean=float(den.mean()),clean_static_mae=float(mae.mean()),clean_static_wmape=float(100*mae.mean()/den.mean()) if den.mean()>1e-12 else None,
                       clean_static_p95=float(np.mean([r['p95'] for r in pp])),
                       clean_static_query_mae=mean([r['query_mae'] for r in pp if r['query_mae'] is not None]),
                       step_joint_abs_mean=float(joint.mean()),step_separate_abs_mean=float(individual.mean()),
                       joint_to_separate_ratio=float(joint.mean()/individual.mean()) if individual.mean()>1e-12 else None,
                       root_changed_task_unchanged_fraction=mean(changed_unchanged),
                       task_directionality_median=float(np.median([r['directionality'] for r in pp if r['directionality'] is not None])) if any(r['directionality'] is not None for r in pp) else None,
                       clean_static_rce=mean([r['rce'] for r in pp if 'rce' in r]))
                r['clean_static_mae_patient_lo'],r['clean_static_mae_patient_hi']=q95(boot);nodes_out.append(r)
            for (i,j),pp in pair_by_roots.items():
                # Equal total mass per patient, then equal mass per co-measured event.
                a=np.concatenate([x for s,x,y in pp]);b=np.concatenate([y for s,x,y in pp]);wt=np.concatenate([np.full(len(x),1/len(x)) for s,x,y in pp]);wt/=wt.sum()
                aa=a-np.sum(wt*a);bb=b-np.sum(wt*b);den=np.sqrt(np.sum(wt*aa**2)*np.sum(wt*bb**2))
                pairs_out.append(dict(template=template,split=part,root_i=w.roots[i],root_j=w.roots[j],patients=len(pp),co_measured_events=len(a),correlation=float(np.sum(wt*aa*bb)/den) if den>1e-14 else None))
    write_csv(out/'cohort_summary.csv',cohorts);write_csv(out/'root_summary.csv',roots_out);write_csv(out/'node_summary.csv',nodes_out)
    write_csv(out/'root_patients.csv',patients_out);write_csv(out/'root_correlations.csv',pairs_out)
    with (out/'task_event_changes.jsonl').open('w') as f:
        for r in events_out:line(f,r)
    return cohorts,roots_out,nodes_out,pairs_out


def noise_diagnostic(records,source,assignment,out):
    cfg=json.loads((source/'config.json').read_text());R=cfg['repeats'];budgets=cfg['evaluated_budgets'];wanted={'RG-static-lap','RG-periodic-lap'}
    raw=defaultdict(list);saved={};recs={r['trajectory_id']:r for r in records if assignment[str(r['subject_id'])]=='test'}
    for ln in (source/'actions.jsonl').open():
        a=json.loads(ln)
        if a['method'] in wanted:raw[a['trajectory'],a['B'],a['method'],a['repeat']].append(a)
    for ln in (source/'metrics.jsonl').open():
        r=json.loads(ln)
        if r['method'] in wanted:saved[r['trajectory'],r['B'],r['method'],r['repeat']]=r['mae']
    paths=[];scale_rows=[];maxdiff=0.
    for template in TEMPLATES:
        w=Workload(template)
        for rec in [r for r in recs.values() if r['template']==template]:
            x=data.replay(rec);truth=w.evaluate(x)[w.target];clean_static=float(abs(truth-truth[0]).mean())
            for B in budgets:
                for rep in range(R):
                    def cache_stream(method,clean_anchor=False,clean_refresh=False):
                        acts=sorted(raw[rec['trajectory_id'],B,method,rep],key=lambda a:a['turn'])
                        c=np.empty((rec['T'],len(w.roots)))
                        for j,a in enumerate(acts):
                            assert a['selected_roots']==w.roots
                            t=a['turn'];end=acts[j+1]['turn']-1 if j+1<len(acts) else rec['T']
                            value=w.normalize(x[t-1]) if ((j==0 and clean_anchor) or (j>0 and clean_refresh)) else np.array(a['cache_after'])
                            c[t-1:end]=value
                        return c
                    def error(c):return float(abs(task(w,c)-truth).mean())
                    st=error(cache_stream('RG-static-lap'));period=error(cache_stream('RG-periodic-lap'))
                    clean=error(cache_stream('RG-periodic-lap',True,True));init=error(cache_stream('RG-periodic-lap',False,True))
                    updates=error(cache_stream('RG-periodic-lap',True,False))
                    for method,val in [('RG-static-lap',st),('RG-periodic-lap',period)]:
                        diff=abs(val-saved[rec['trajectory_id'],B,method,rep]);maxdiff=max(maxdiff,diff);assert diff<1e-9
                    rr=dict(template=template,subject=str(rec['subject_id']),trajectory=rec['trajectory_id'],B=B,repeat=rep,
                            clean_static=clean_static,clean_periodic=clean,noisy_initial_clean_refresh=init,
                            clean_initial_noisy_refresh=updates,noisy_static=st,noisy_periodic=period,
                            freshness_gain=clean_static-clean,initial_noise_cost=init-clean,
                            refresh_noise_cost=period-init,static_noise_cost=st-clean_static,
                            observed_periodic_minus_static=period-st)
                    identity=-rr['freshness_gain']+rr['initial_noise_cost']+rr['refresh_noise_cost']-rr['static_noise_cost']
                    assert abs(identity-rr['observed_periodic_minus_static'])<1e-9
                    paths.append(rr)
                    if B==960 and rep==0:
                        acts=sorted(raw[rec['trajectory_id'],B,'RG-periodic-lap',rep],key=lambda a:a['turn'])
                        for j,a in enumerate(acts[1:],1):
                            t=a['turn'];prev=acts[j-1]['turn'];delta=abs(x[t-1]-x[prev-1]);eps=np.array(a['actual_epsilon']);scale=w.width/eps
                            for i,root in enumerate(w.roots):
                                scale_rows.append(dict(template=template,subject=str(rec['subject_id']),trajectory=rec['trajectory_id'],turn=t,root=root,
                                                       stale_abs=float(delta[i]),laplace_scale=float(scale[i]),
                                                       stale_to_noise=float(delta[i]/scale[i]),epsilon=float(eps[i])))
    summary=[]
    for t in TEMPLATES:
        rows=[r for r in paths if r['template']==t];subjects=sorted({r['subject'] for r in rows});lookup={(r['B'],r['subject'],r['repeat']):r for r in rows}
        fields=[k for k in rows[0] if k not in ['template','subject','trajectory','B','repeat']]
        x=np.array([[[[lookup[B,s,rep][f] for rep in range(R)] for s in subjects] for B in budgets] for f in fields])
        pb,nb=bootstrap_curves(x,2000,key_seed(20261117,t,'drift_diagnostic'))
        for bi,B in enumerate(budgets):
            rr=dict(template=t,B=B,patients=len(subjects),repeats=R)
            for fi,f in enumerate(fields):
                rr[f]=float(x[fi,bi].mean())
                for name,boot in [('patient',pb),('two_layer',nb)]:rr[f+'_'+name+'_lo'],rr[f+'_'+name+'_hi']=q95(boot[fi,bi])
            summary.append(rr)
    write_csv(out/'noise_freshness_summary.csv',summary);write_csv(out/'refresh_root_noise_scales.csv',scale_rows)
    with (out/'noise_freshness_paths.jsonl').open('w') as f:
        for r in paths:line(f,r)
    return summary,maxdiff


def fmt(v,d=2):return 'NA' if v is None else f'{v:.{d}f}'


def report(out,cohorts,roots,nodes,pairs,noise,audit):
    lines=['# 四模板根漂移、任务陈旧与刷新噪声诊断','',
           '仅统计冻结的MIMIC demo有限轨迹，不修改方法、参数、τ或test划分。每模板20名患者、同一次住院一条40轮轨迹；四模板患者重叠，总共24个不同患者，不能当作80名独立患者。全量与train/validation/test分别报告，实验解释对齐test。','',
           '## 1. 数据和统计口径','',
           '根的单步变化仅在该根实际被重测时计算，包含真实重测但值未变的情况；未测轮不混入分母。每患者先给相同总权重，患者内各次重测均权，再求加权中位数/P90。相对变化=100|x_new−x_previous_known|/|x_previous_known|；域宽比例另列，不能据公开域宽百分比很小就判定实际变化很小。最大偏移是每患者max_t|x_t−x_1|/|x_1|，再取患者中位数。','',
           '本轮是描述性事后诊断，读test只为解释既有结果，不将统计返回规划器或重选工作区间。10%变化门槛仅作描述，不是临床显著阈值。','',
           '|模板|全量患者|批次数|每人批次中位数|自然时间跨度天中位数|test患者|','|---|---:|---:|---:|---:|---:|']
    for t in TEMPLATES:
        c=next(r for r in cohorts if r['template']==t and r['split']=='all');test=next(r for r in cohorts if r['template']==t and r['split']=='test')
        lines.append(f'|{t}|{c["patients"]}|{c["batches"]}|{c["batches_median"]:.1f}|{c["duration_days_median"]:.2f}|{test["patients"]}|')
    lines+=['','## 2. 全量20名患者的逐根变化','',
            '|模板/根|原单位|单次绝对变化中位数|相对变化中位数%|相对变化P90%|患者最大偏移中位数%|值改变的重测比例%|','|---|---|---:|---:|---:|---:|---:|']
    for r in roots:
        if r['split']=='all':lines.append('|'+r['template']+'/'+r['root']+'|'+r['unit']+'|'+'|'.join(fmt(r[k]) for k in ['step_abs_median','step_relative_pct_median','step_relative_pct_p90','max_displacement_pct_initial_median'])+'|'+fmt(100*r['changed_fraction'] if r['changed_fraction'] is not None else None)+'|')
    lines+=['','age在住院内固定、不作为后续测量根，因此其单步变化是NA，最大偏移为0；不是统计缺失。完整的原单位/域宽变化、趋势性、零变重测比例及各划分数据见root_summary.csv和root_patients.csv。','',
            '## 3. 根漂移传到任务后的实际影响','',
            '把第一轮真实根固定到最后、不加任何噪声，得到纯陈旧MAE。这是离线诊断，不满足当前DP发布协议，不能拿来当可部署基线。下面使用test，与既有实验人群相同。','',
            '|模板|干净静态MAE|患者95% CI|干净静态wMAPE%|联合变化/逐根绝对变化和|根变而最终任务不变%|','|---|---:|---|---:|---:|---:|']
    for r in nodes:
        if r['split']=='test' and r['final']:
            lines.append(f'|{r["template"]}|{r["clean_static_mae"]:.3f}|[{r["clean_static_mae_patient_lo"]:.3f},{r["clean_static_mae_patient_hi"]:.3f}]|'+fmt(r['clean_static_wmape'])+'|'+fmt(r['joint_to_separate_ratio'],3)+'|'+fmt(100*r['root_changed_task_unchanged_fraction'])+'|')
    lines+=['','“联合/逐根和”按真实批次做有限改变量比较，可提示抵消/叠加；非线性任务存在交互，因此不是严格归因比例。中间节点的同类结果和干净静态RCE见node_summary.csv，不能把最终节点稳定当作所有查询稳定。','',
            '## 4. 固定预算下：新鲜度收益与噪声代价','',
            '复用原M-Opt同Laplace核静态/周期对照的冻结动作、逐根epsilon、公开周期及R=20。以下只离线去掉指定发布的噪声，不重跑门控、不重新选择周期、不发布新私有值。全54个预算点见noise_freshness_summary.csv，主表沿用预指定B=960。','',
            '|模板|干净静态D0|干净周期D1|静态实际误差|周期实际误差|周期−静态|两层95% CI|','|---|---:|---:|---:|---:|---:|---|']
    for r in noise:
        if r['B']==960:
            lines.append('|'+r['template']+'|'+'|'.join(fmt(r[k],3) for k in ['clean_static','clean_periodic','noisy_static','noisy_periodic','observed_periodic_minus_static'])+f'|[{r["observed_periodic_minus_static_two_layer_lo"]:.3f},{r["observed_periodic_minus_static_two_layer_hi"]:.3f}]|')
    lines+=['','可核对的有序差分恒等式：','',
            '`周期实际−静态实际 = −(D0−D1) + (D2−D1) + (D3−D2) − (静态实际−D0)`','',
            'D2=带噪初始化+干净周期刷新，D3=完整带噪周期。四项依次是负的新鲜度收益、周期初始化噪声影响、周期刷新噪声影响、减去静态噪声影响。它是指定顺序的精确有符号差分，不是唯一的因果/方差分解；噪声有时抵消陈旧，分项可以为负。','',
            '|模板|新鲜度收益D0−D1|初始化噪声影响|刷新噪声影响|静态噪声影响|','|---|---:|---:|---:|---:|']
    for r in noise:
        if r['B']==960:lines.append('|'+r['template']+'|'+'|'.join(fmt(r[k],3) for k in ['freshness_gain','initial_noise_cost','refresh_noise_cost','static_noise_cost'])+'|')
    lines+=['','## 5. 对“为什么不更新反而更好”的判断','',
            '不能统一归因为根变化小。FIB4的AST/ALT相对前值单步中位变化约22%/17%，患者最大偏移中位数约84%/79%；ANEMIA根的最大偏移约25%–26%，但三个血液比值任务的干净静态wMAPE只有约2%–4%。主要差别是变化如何传到任务，而非是否发生根变化。','',
            '- ANEMIA：test干净冻结MCHC误差0.791（2.48%）；联合变化只有逐根绝对变化叠加的约16.5%，MCV/MCH也明显抵消。当前冻结周期即便完全去掉噪声也为0.946，说明回摆和刷新位置使本周期没有净新鲜度收益；带噪后再增加误差。不能据此说所有日程均无收益，既有M4另一日程的干净误差更低。','',
            '- FIB4：test干净冻结误差16.948（121.8%），明确不是最终任务稳定；干净周期可降到10.023。但预算拆分使初始化和后续发布精度下降，乘除结构/小分母放大长尾，完整同核周期329.187反而高于静态75.456。这里是噪声成本压过真实更新收益。该数值诊断专指Laplace移植对照，原生指数核的绝对误差不同。','',
            '- MELD_NA：test干净冻结误差3.346（19.9%），干净周期2.265，实际同核周期5.579也优于静态6.530。不能把约6分静态误差全部叫作不可消除的纯陈旧下限；它还含分配及噪声误差。M5选择静态不证明数据不需要动态维护。','',
            '- BUN_CR：test干净冻结误差10.358（47.1%），干净周期7.118，实际同核周期7.600优于静态10.356；相关变化有部分抵消，但仍有充分漂移影响比值。','',
            '固定周期可能在暂态极值附近更新，之后轨迹回到原水平；缓存更近的一次测量不保证整个后续区间的平均误差更低。相同B下再次发布会分走初始化预算，并给已知静态根（如年龄）重新加噪。这些现象共同影响比较。','',
            '## 6. 公开域归一化与显著变化阈值','',
            '当前MissRate的显著变化阈值为公开域宽5%，不是相对前值5%。AST/ALT的公开域[1,20000]意味着门槛约1000 IU/L。AST单次绝对变化中位数8 IU/L，只占公开域0.040%，但相对前值中位数为21.9%。所以“归一化变化小”或“显著事件少”不能直接解释为根实际稳定。','',
            '|根（全量）|相对前值变化>10%的重测比例%|超过公开域5%的重测比例%|公开域5%对应原单位幅度|','|---|---:|---:|---:|']
    for r in roots:
        if r['split']=='all' and r['root']!='age':
            lines.append('|'+r['template']+'/'+r['root']+'|'+fmt(100*r['relative_over10_fraction'])+'|'+fmt(100*r['domain_over5_fraction'])+'|'+fmt(r['domain5_threshold_raw'])+'|')
    lines+=['','## 7. 解释边界与复现','',
            '构建器选最早符合条件的住院和最早有限窗口，按采样次数筛选，不按最终任务漂移筛选。40轮是合成对话时钟，化验批次保留其自然时间；不是40个等间隔自然时间点。16–24个批次不等于每个根都发生16–24次数值变化。FIB4年龄固定，ANEMIA等保留重测但未变值，BUN_CR额外要求至少16个批次数值改变。','',
            '小幅单步变化、相关根抵消、轨迹回摆/初值代表性、根的噪声大小和非线性放大必须区分。公开域归一化很小不等于临床幅度小；短窗口有漂移也不保证每个固定周期都改善全轨迹MAE。统计支持的具体解释见数据表，不据此挑选更容易胜出的test子组。','',
            f'输入哈希与原实验一致；两种同核基线全路径误差重放最大差={audit["max_saved_mae_difference"]:.3g}；有序差分恒等式逐路径通过。原始数据和结果没有被覆盖。','',
            '```bash','cd /home/rootguard-research',
            '/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.trajectory_drift.analyze --output experiments/workflow_extensions/outputs/trajectory_drift/reproduce_20261117','```','']
    (out/'REPORT-zh.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--data',default=str(ROOT/'data_mine/finite_cache'))
    p.add_argument('--source',default=str(ROOT/'experiments/workflow_extensions/outputs/m5/real_v2_20261117'))
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Output must be empty')
    source=Path(a.source);records=data.load(a.data,TEMPLATES);assignment=json.loads((source/'split.json').read_text())
    hashes=data.input_hashes(a.data,TEMPLATES);assert hashes==json.loads((source/'input_hashes.json').read_text())
    cohorts,roots,nodes,pairs=stats(records,assignment,out)
    print('ROOT AND TASK STATISTICS COMPLETE',flush=True)
    noise,diff=noise_diagnostic(records,source,assignment,out)
    audit=dict(input_hashes=hashes,independent_patients=len({str(r['subject_id']) for r in records}),trajectories=len(records),
               saved_noise_diagnostic_paths=sum(1 for _ in (out/'noise_freshness_paths.jsonl').open()),
               max_saved_mae_difference=diff,no_parameter_changes=True,no_new_private_releases=True,identity_checked_each_path=True,
               analysis_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(out/'audit.json',audit);report(out,cohorts,roots,nodes,pairs,noise,audit)
    print('DONE',out,flush=True)


if __name__=='__main__':main()
