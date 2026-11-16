"""Internal go/no-go report: M1-a versus strong periodic B2, no test tuning."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from . import data
from .analyze import csv_rows, write_json, first_reached, select
from .run import grid, refinement_points
from .workload import Workload

METHODS = ['B0','B2','B2p','M1-a','M1-g']


def protocol_audit(output,config,ss,selections,targets,data_directory):
    """Reconstruct tuning and fine-grid provenance using validation alone."""
    base=json.loads((Path(__file__).with_name('configs')/'round1_meld.json').read_text())
    assert {k for k in set(base)|set(config) if base.get(k)!=config.get(k)}=={'templates','refine','budgets'}
    coarse=grid(**base['budget_grid']);crossings=[]
    for template in config['templates']:
        val=[r for r in ss if r['split']=='validation' and r['template']==template]
        initial=[r for r in val if r['B'] in coarse]
        expected_targets=sorted(set(min(r['mean'] for r in initial if r['method']=='B2' and
                                       r['B']==min(coarse,key=lambda b:abs(b-ref)))
                                    for ref in config['target_reference_budgets']))
        np.testing.assert_allclose(targets[template],expected_targets,rtol=0,atol=1e-12)
        for tau in targets[template]:
            for method in METHODS:
                if method=='M1-g':
                    param=select(initial,template,'M1-a',tau)['parameter']
                    crossings.append(first_reached([r for r in initial if r['method']==method and r['parameter']==param],tau))
                    assert selections[template][str(tau)][method]['parameter']==selections[template][str(tau)]['M1-a']['parameter']
                else:
                    crossings.append(select(initial,template,method,tau)['validation_B_min'])
                    assert selections[template][str(tau)][method]==select(val,template,method,tau)
        diagnostic={int(r['parameter'].split('=')[1]) for r in ss if r['split']=='test_period_sensitivity' and r['template']==template}
        assert diagnostic==set(range(2,13))
    assert config['budgets']==sorted(set(coarse)|set(refinement_points(coarse,crossings)))
    metadata=json.loads((output/'metadata.json').read_text())
    protocol=json.loads((output/'protocol.json').read_text())
    assert metadata['input_sha256']==data.input_hashes(data_directory,config['templates'])
    assert hashlib.sha256(Path(__file__).with_name('mechanisms.py').read_bytes()).hexdigest()==protocol['mechanisms_sha256']
    root=Path(__file__).resolve().parents[3]
    changes={name:{'execution_snapshot':sha,'current':hashlib.sha256((root/name).read_bytes()).hexdigest()}
             for name,sha in metadata['code_sha256'].items()
             if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha}
    allowed={'experiments/workflow_extensions/finite_cache/round2_report.py',
             'experiments/workflow_extensions/finite_cache/round2_figures.py'}
    assert set(changes)<=allowed, 'Experimental code changed after execution snapshot'
    write_json(output/'postprocessing_revisions.json',{'source_changes':changes,
               'reason':'display right-censored frontiers as scan-window lower limits; preserve unidentified point ratios',
               'raw_results_splits_parameters_unchanged':True})
    return {'validation_targets_reconstructed':True,'validation_only_P_q_and_fine_grid_reconstructed':True,
            'integer_period_diagnostic_complete':True,'input_hashes_preserved':True,'DP_mechanism_hash_unchanged':True,
            'postprocessing_source_changes':changes}


def interval(values):
    return [float(v) for v in np.quantile(values,[.025,.975])] if len(values) else None


def crossing_draws(budgets, means, tau):
    # Never assume monotonicity in B.
    return [min((b for b,v in zip(budgets,column) if v<=tau),default=None) for column in np.asarray(means).T]


def front(curve, patient_draws, tau):
    curve=sorted(curve,key=lambda r:r['B'])
    budgets=[r['B'] for r in curve]
    draws=crossing_draws(budgets,[r['bootstrap_means'] for r in curve],tau)
    patient=crossing_draws(budgets,patient_draws,tau)
    finite=[x for x in draws if x is not None]; pp=[x for x in patient if x is not None]
    return {'B_min':first_reached(curve,tau),'hierarchical_draws':draws,'patient_draws':patient,
            'hierarchical_ci_conditional':interval(finite),'patient_ci_conditional':interval(pp),
            'hierarchical_reached_fraction':len(finite)/len(draws),'patient_reached_fraction':len(pp)/len(patient)}


def ratios(numerator, denominator):
    result={}
    for kind in ['patient','hierarchical']:
        pairs=[(a,b) for a,b in zip(numerator[kind+'_draws'],denominator[kind+'_draws']) if a is not None and b is not None]
        result[kind+'_ratio_ci_conditional']=interval([a/b for a,b in pairs])
        result[kind+'_both_reached_fraction']=len(pairs)/len(numerator[kind+'_draws'])
    a,b=numerator['B_min'],denominator['B_min']
    result['ratio']=a/b if a is not None and b is not None else None
    cis=[result[k+'_ratio_ci_conditional'] for k in ['patient','hierarchical']]
    reliable=all(result[k+'_both_reached_fraction']>=.95 for k in ['patient','hierarchical'])
    result['significance']=('M1_better' if reliable and all(ci and ci[1]<1 for ci in cis) else
                            'B2_better' if reliable and all(ci and ci[0]>1 for ci in cis) else 'insufficient_or_censored')
    return result


def resampling_plan(n,R,draws,seed):
    # Reproduce analyze.summaries RNG ordering, retaining patient-only samples.
    rng=np.random.default_rng(seed); pi=[]; ri=[]
    for _ in range(draws):
        indices=rng.integers(n,size=n);pi.append(indices)
        ri.append([rng.integers(R,size=R) for _ in indices])
        for _ in range(n):rng.integers(R,size=R)  # mechanism-only draw in original summary
    return np.array(pi),np.array(ri)


def patient_curve_draws(curve, raw, config):
    curve=sorted(curve,key=lambda r:r['B'])
    example=curve[0];key=example['template'],example['method'],example['parameter']
    by=raw[key];ids=sorted({sid for b,sid,repeat in by})
    R=config['repeats'];pi,ri=resampling_plan(len(ids),R,config['bootstrap'],config['seed'])
    matrix=np.array([[[np.mean(by[(r['B'],sid,repeat)]) for repeat in range(R)] for sid in ids] for r in curve])
    # Exact paired plans are common across B/methods; verify the original hierarchy.
    expected=matrix[0][pi[:,:,None],ri].mean(axis=(1,2))
    np.testing.assert_allclose(expected,curve[0]['bootstrap_means'],rtol=0,atol=1e-10)
    return matrix.mean(axis=2)[:,pi].mean(axis=2)


def drift_and_floors(records,assignment,template,w,seed,draws):
    train=data.partition(records,assignment,'train',template)
    drift=[]
    for rec in train:
        z=w.normalize(data.replay(rec)); previous=z[0];dd=[]
        for e in rec['events']:
            now=z[e['turn']-1];dd.append(float(np.max(np.abs(now-previous))));previous=now
        drift.append(float(np.mean(dd)))
    rng=np.random.default_rng(seed)
    bootstrap=np.mean(np.array(drift)[rng.integers(len(drift),size=(draws,len(drift)))],axis=1)
    test=data.partition(records,assignment,'test',template);floors=[];initial=[]
    for rec in test:
        y=w.evaluate(data.replay(rec))[w.target]
        floors.append(float(np.abs(y-np.median(y)).mean()))
        value=float(w.from_cache(w.normalize([rec['initial_state'][r] for r in w.roots]))[w.target])
        initial.append(float(np.abs(y-value).mean()))
    return {'template':template,'train_normalized_drift':float(np.mean(drift)),
            'train_drift_patient_ci':interval(bootstrap),'strict_constant_output_floor':float(np.mean(floors)),
            'noiseless_frozen_initial_limit':float(np.mean(initial)),'train_patients':len(train),'test_patients':len(test)}


def analyze(output,data_directory):
    output=Path(output);config=json.loads((output/'config.json').read_text())
    ss=json.loads((output/'summary.json').read_text()); selections=json.loads((output/'selections.json').read_text())
    targets=json.loads((output/'targets.json').read_text());assignment=json.loads((output/'splits.json').read_text())
    stability=json.loads((output/'selection_stability.json').read_text())
    records=data.load(data_directory,config['templates'])
    raw=defaultdict(lambda:defaultdict(list));total_rows=0;maxratio=0.;costs=defaultdict(list)
    with (output/'metrics.jsonl').open() as f:
        for line in f:
            r=json.loads(line);total_rows+=1
            assert r['spent']<=r['B']+1e-8
            maxratio=max(maxratio,r['spent']/r['B'])
            if r['split']=='test':
                raw[(r['template'],r['method'],r['parameter'])][(r['B'],r['subject_id'],r['repeat'])].append(r['mae'])
            costs[(r['split'],r['trajectory_id'],r['method'],r['parameter'],r['B'])].append(r['spent'])
    assert all(max(v)-min(v)<1e-10 for v in costs.values())
    csv_rows(output/'summary.csv',[{k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in r.items() if k!='bootstrap_means'} for r in ss])
    info={t:drift_and_floors(records,assignment,t,Workload(t),config['seed'],config['bootstrap']) for t in config['templates']}
    write_json(output/'drift_and_static_bounds.json',info)
    frontier_rows=[];cross=[];budget960=[];periods=[]
    cache={}
    def curve(template,method,parameter):
        rr=sorted([r for r in ss if r['split']=='test' and r['template']==template and r['method']==method and r['parameter']==parameter],key=lambda r:r['B'])
        key=template,method,parameter
        if key not in cache:cache[key]=patient_curve_draws(rr,raw,config)
        return rr,cache[key]
    coarse=grid(**config['budget_grid'])
    for template in config['templates']:
        for tau in targets[template]:
            selected=selections[template][str(tau)];ff={}
            for method in METHODS:
                rr,pp=curve(template,method,selected[method]['parameter']);ff[method]=front(rr,pp,tau)
                item={'template':template,'tau':tau,'method':method,'parameter':selected[method]['parameter'],
                      **{k:v for k,v in ff[method].items() if not k.endswith('_draws')},
                      'B_min_status':'reached' if ff[method]['B_min'] is not None else 'not_reached_in_grid'}
                if method=='B0':
                    item['tau_below_strict_constant_floor']=tau<info[template]['strict_constant_output_floor']-1e-12
                    item['tau_below_noiseless_initial_limit']=tau<info[template]['noiseless_frozen_initial_limit']-1e-12
                    if item['tau_below_strict_constant_floor']:
                        assert item['B_min'] is None
                        item['B_min_status']='impossible_for_any_budget_constant_output_bound'
                frontier_rows.append(item)
                last=next(r for r in rr if r['B']==960.)
                budget960.append({'template':template,'tau':tau,'method':method,'parameter':selected[method]['parameter'],
                                  'mae':last['mean'],'mae_patient_ci':last['patient_ci'],'mae_hierarchical_ci':[last['ci_low'],last['ci_high']],
                                  'MissRate':last['miss_rate'],'frac_age':last['frac_age'],'m':last['refresh_count'],
                                  'rho_used':last['spent']/960.,'spent':last['spent'],
                                  'publication_projection_rate':last['publication_projection_fraction'],
                                  'cache_projection_rate':last['cache_projection_fraction']})
            ratio=ratios(ff['M1-a'],ff['B2']);ratio_p=ratios(ff['B2p'],ff['B2'])
            lower=(max(config['budgets'])/ff['B2']['B_min']
                   if ff['M1-a']['B_min'] is None and ff['B2']['B_min'] is not None else None)
            stab=next(r for r in stability if r['template']==template and r['method']=='B2' and r['tau']==tau)
            anchors=[]
            for ref in config['target_reference_budgets']:
                b=min(coarse,key=lambda x:abs(x-ref))
                val=min(r['mean'] for r in ss if r['split']=='validation' and r['template']==template and r['method']=='B2' and r['B']==b)
                if abs(val-tau)<1e-10:anchors.append(ref)
            cross.append({'template':template,'tau':tau,'tau_reference_budgets':anchors,
                          'drift':info[template]['train_normalized_drift'],
                          'B2_P_star':selected['B2']['parameter'],'B2_P_selection_frequency':stab['selection_frequency'].get(selected['B2']['parameter'],0.),
                          'B2_P_all_frequencies':stab['selection_frequency'],
                          'M1_q':selected['M1-a']['parameter'],'B2_Bmin':ff['B2']['B_min'],'M1_Bmin':ff['M1-a']['B_min'],
                          'Bmin_difference':ff['M1-a']['B_min']-ff['B2']['B_min'] if ff['M1-a']['B_min'] is not None and ff['B2']['B_min'] is not None else None,
                          'scan_window_ratio_lower_limit':lower,
                          'scan_window_difference_lower_limit':max(config['budgets'])-ff['B2']['B_min'] if lower is not None else None,
                          **ratio,'B2p_over_B2':ratio_p['ratio'],
                          'B2p_B2_patient_ci':ratio_p['patient_ratio_ci_conditional'],
                          'B2p_B2_hierarchical_ci':ratio_p['hierarchical_ratio_ci_conditional'],
                          'B2p_B2_joint_reach':ratio_p['hierarchical_both_reached_fraction']})
        for r in ss:
            if r['template']==template and r['split'] in ['validation','test_period_sensitivity'] and r['method']=='B2':
                periods.append({'template':template,'split':r['split'],'P':int(r['parameter'].split('=')[1]),'B':r['B'],
                                'mae':r['mean'],'ci_low':r['ci_low'],'ci_high':r['ci_high']})
    def serialize(rows):
        return [{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows]
    csv_rows(output/'Bmin_mean.csv',serialize(frontier_rows));csv_rows(output/'cross_template.csv',serialize(cross))
    csv_rows(output/'budget960.csv',serialize(budget960));csv_rows(output/'period_sensitivity.csv',periods)
    write_json(output/'cross_template.json',cross);write_json(output/'Bmin_mean.json',frontier_rows)
    # Ledger audit uses repeat0 because schedule/charges are identical across all R.
    publications=0; schedules=0
    with (output/'ledger.jsonl').open() as f:
        for line in f:
            r=json.loads(line);schedules+=1
            eps=[e for release in r['transcript'] for e in release['eps']]
            assert all(e>0 for e in eps)
            assert sum(eps)<=r['B']+1e-8
            publications+=len(eps)
    audit={'passed':True,'rows':total_rows,'unique_public_schedules':schedules,'root_publications_checked':publications,
           'max_spent_over_B':maxratio,'all_eps_positive':True,'all_repeats_equal_spend':True,
           'test_not_used_for_selection':True,'patient_splits':json.loads((output/'metadata.json').read_text())['patient_counts']}
    audit.update(protocol_audit(output,config,ss,selections,targets,data_directory))
    write_json(output/'audit.json',audit)
    write_report(output,config,info,cross,frontier_rows,budget960)
    from .round2_figures import render
    render(output,cross,info,periods)


def write_report(output,config,info,cross,frontiers,budget960):
    f=lambda v:'未达≤960' if v is None else f'{v:.3f}'
    metric=lambda v:'NA' if v is None else f'{v:.3f}'
    ci=lambda v:'NA' if v is None else f'[{v[0]:.3f},{v[1]:.3f}]'
    labels={'M1_better':'M1显著更省','B2_better':'B2显著更省','insufficient_or_censored':'无法判断（删失/证据不足）'}
    text=['# Round 2 跨模板受控实验：内部go/no-go','',
          '仅用于检查MELD上M1不及强周期B2的现象是否可推广，不写论文结论。所有选择在validation完成，test仅评估。','',
          '## 跨模板主表','',
          '|模板|train漂移δ|τ|B2 B_min|M1 B_min|M1/B2|相对差距|患者CI|两层CI|是否显著|B2 P*（选中率）|',
          '|---|---:|---:|---:|---:|---:|---:|---|---|---|---|']
    for r in cross:
        gap=(f">{100*(r['scan_window_ratio_lower_limit']-1):+.1f}%（删失界）"
             if r['scan_window_ratio_lower_limit'] is not None else
             'NA' if r['ratio'] is None else f"{100*(r['ratio']-1):+.1f}%")
        text.append(f"|{r['template']}|{r['drift']:.5f}|{r['tau']:.3f}|{f(r['B2_Bmin'])}|{f(r['M1_Bmin'])}|{f(r['ratio']) if r['ratio'] is not None else 'NA'}|"
                    f"{gap}|{ci(r['patient_ratio_ci_conditional'])}|{ci(r['hierarchical_ratio_ci_conditional'])}|{labels[r['significance']]}|"
                    f"{r['B2_P_star']}（{r['B2_P_selection_frequency']:.1%}）|")
    text += ['', '比值<1有利于M1。B_min定义为扫描网格内最小达标预算，不假设曲线单调。'
             '区间以相应bootstrap中双方都达标为条件；稳定胜出要求患者及两层区间均排除1，且双方达标比例各≥95%。'
             '未达标只表示该网格≤960未达，不擅自替换成960或无限大。完整达标比例见cross_template.csv。','',
             '若只有M1未达而B2达标，额外列出扫描窗口下界960/B2_Bmin−1，以“>”标注：这是右删失界，不是已识别的预算比值或显著性结论。'
             '双方都未达时不计算差距。','',
             '![漂移量级与相对预算差距](drift_vs_gap.png)','',
             '图中δ是train患者均权的每批次归一化根最大变化；纵轴为(M1/B2−1)×100%，误差线为配对两层95%条件区间。'
             '空心点及“>”表示扫描窗口右删失下界，不是有限前沿估计；删失点不绘制伪造的比值区间。'
             '不同模板域宽不同，只有三个模板，图用于描述，不据此拟合或宣称漂移造成收益。','',
             '## 配置、患者划分及数据','',
             '- 复用round1_meld参数；仅templates替换为三模板，并按用户明确的1.05局部细化协议将refine从false设true。其余参数（包括eps_min=0.01、bootstrap=500、R=20）完全不变。',
             '- 局部细化只看validation的五方法前沿，取跨模板邻域并集；然后重新在validation选P/q，之后冻结test预算及选择。test不触发二次细化。',
             '- 所有患者跨模板去重后只划分一次，同人不跨train/validation/test；每模板总人数20，实际各集合人数见audit.json。FIB4 test仅4位，其他两者5位。',
             '- ANEMIA/FIB4原JSONL未改动。BUN_CR为同住院、charttime排序的40轮轨迹，≥16次数值真实变化；storetime用前缀最大值保证信息可用性，未使用未来测量。原输入及manifest/validation备份保留。',
             '- ANEMIA/FIB4沿用原先保留数值未变测量批次的口径；BUN_CR按本轮要求只计数值改变批次。δ是各自保留批次上的描述量，不能把这种筛选差异解释为机制效果。',
             '- BUN_CR公式ratio=bun/cr，未加入可选分类；公开域bun∈[1,200]、cr∈[0.2,15]预先固定，不按test拟合。',
             '- NLR=2、TYG=6，不满足≥10人，未运行；请求的三模板均达到≥10人。',
             '- DP机制/账本文件未修改，逐根epsilon>0且总支出≤B。全部R20保留指标，公开账本保存repeat0并核对所有R收费相同。','',
             '## 五方法的B_min^mean(τ)','',
             '|模板|τ|方法|参数|B_min|患者CI（达标比例）|两层CI（达标比例）|',
             '|---|---:|---|---|---:|---|---|']
    for r in frontiers:
        value='∞（恒定输出下界）' if r['B_min_status']=='impossible_for_any_budget_constant_output_bound' else f(r['B_min'])
        text.append(f"|{r['template']}|{r['tau']:.3f}|{r['method']}|{r['parameter']}|{value}|"
                    f"{ci(r['patient_ci_conditional'])}（{r['patient_reached_fraction']:.1%}）|{ci(r['hierarchical_ci_conditional'])}（{r['hierarchical_reached_fraction']:.1%}）|")
    text += ['', '## B0的严格下界与冻结初值极限','',
             '|模板|任意恒定输出的严格MAE下界|无噪声冻结初值极限|', '|---|---:|---:|']
    for t,v in info.items():
        text.append(f"|{t}|{v['strict_constant_output_floor']:.4f}|{v['noiseless_frozen_initial_limit']:.4f}|")
    text += ['', 'B0在整轨迹输出恒定，因此每位患者的时间中位数预测误差是任何B0输出的下界；τ低于该值时无论预算多大都无法达到。'
             '若τ介于严格下界和冻结初值极限之间，有限噪声可能偶然抵消偏差，仍按实际扫描结果报告；不把冻结初值极限当作所有有限预算的严格下界。'
             '高于下界也不保证≤960达标，域投影、初始缓存和有限噪声仍影响结果。','',
             '## B=960：误差与代价','',
             '|模板|τ|方法|参数|MAE（患者CI；两层CI）|MissRate|frac_age|m|ρ_used|发布投影率|缓存投影率|',
             '|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in budget960:
        text.append(f"|{r['template']}|{r['tau']:.3f}|{r['method']}|{r['parameter']}|{r['mae']:.3f}（{ci(r['mae_patient_ci'])}；{ci(r['mae_hierarchical_ci'])}）|"
                    f"{metric(r['MissRate'])}|{metric(r['frac_age'])}|{r['m']:.2f}|{r['rho_used']:.3f}|{r['publication_projection_rate']:.3f}|{r['cache_projection_rate']:.3f}|")
    text += ['', 'ρ_used=实际epsilon/B；MissRate沿用原框架的显著事件加权口径；frac_age为年龄相关刷新/全部刷新，无刷新时NA。','',
             '## 周期敏感性及P*稳定性','',
             '主选择P候选严格保持[2,3,4,5,6,8,10,12]；q候选及目标生成保持原样。'
             '完整整数P=2..12仅用于冻结后诊断，7/9/11不进入选择。逐点MAE与两层CI保存于period_sensitivity.csv，验证集选P频率保存于selection_stability.json。'
             '按执行前规则，选中率≥80%才称为稳定；跨τ是否相同另行报告。', '']
    for t in config['templates']:
        choices=[r for r in cross if r['template']==t]
        text += [f"{t}：P*为 "+'；'.join(f"τ={r['tau']:.3f}: {r['B2_P_star']}，选中率{r['B2_P_selection_frequency']:.1%}" for r in choices)+'。', '',
                 f'![{t}周期敏感性](period_sensitivity_{t}.png)','']
    with (output/'period_sensitivity.csv').open() as file:
        diagnostics=list(csv.DictReader(file))
    text += ['B=960的冻结后诊断（仅点估计，不据此重新选择周期）：','',
             '|模板|B2 P=10 MAE|P=11 MAE|P=12 MAE|M1-a MAE|','|---|---:|---:|---:|---:|']
    for t in config['templates']:
        values={int(r['P']):float(r['mae']) for r in diagnostics if r['template']==t and
                r['split']=='test_period_sensitivity' and float(r['B'])==960.}
        m=next(r['mae'] for r in budget960 if r['template']==t and r['method']=='M1-a')
        text.append(f"|{t}|{values[10]:.3f}|{values[11]:.3f}|{values[12]:.3f}|{m:.3f}|")
    text += ['', '这三个邻近长周期在B=960的MAE均低于M1-a，因而最高预算的点估计差异并非P=12孤立有利；'
             '这不替代B_min区间，也不证明跨患者的显著劣势。','']
    text += ['## B2p/B2与内部判断','', '|模板|τ|B2p/B2|患者CI|两层CI|双方达标率|', '|---|---:|---:|---|---|---:|']
    for r in cross:
        text.append(f"|{r['template']}|{r['tau']:.3f}|{f(r['B2p_over_B2']) if r['B2p_over_B2'] is not None else 'NA'}|{ci(r['B2p_B2_patient_ci'])}|"
                    f"{ci(r['B2p_B2_hierarchical_ci'])}|{r['B2p_B2_joint_reach']:.1%}|")
    wins=[r for r in cross if r['significance']=='M1_better']
    selected_only=[r for r in cross if r['ratio'] is None and r['hierarchical_ratio_ci_conditional']
                   and r['hierarchical_ratio_ci_conditional'][1]<1]
    if selected_only:
        text += ['', '注意删失筛选：'+ '；'.join(
            f"{r['template']} τ={r['tau']:.3f}的条件比值两层区间虽低于1，但双方达标仅占患者bootstrap的{r['patient_both_reached_fraction']:.1%}、两层的{r['hierarchical_both_reached_fraction']:.1%}"
            for r in selected_only)+'。这些区间只代表成功子样本，不能当作M1显著更省的证据。']
    text += ['', ('存在达到预指定稳定胜出规则的模板/目标：'+','.join(f"{r['template']} τ={r['tau']:.3f}, M1/B2={r['ratio']:.3f}" for r in wins)
                  if wins else '本轮没有模板/目标满足M1稳定优于强周期B2的预指定规则；有利点估计或删失对比不替代稳定胜出证据。'), '',
             '配置冲突处理、敏感性隔离、B0下界定义均在执行前protocol.json记录。置信区间仅在冻结validation选择条件下评估test不确定性，不涵盖重新取样后目标τ改变。'
             '原始运行代码快照保留于metadata.json；执行后为正确显示删失前沿所做的纯报告/绘图调整另记postprocessing_revisions.json，不改原始数据、选择和账本。'
             '样本小且数据为真实化验＋合成查询日程，不写论文结论；仅作为下一步go/no-go依据。','']
    (output/'REPORT-zh.md').write_text('\n'.join(text))
