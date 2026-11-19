"""Patient-equal ratio metrics, paired/nested CI, censored nonmonotone B_min."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from experiments.workflow_extensions.m4_carry.analysis import bootstrap_curves,q95,bmin,write_csv
from experiments.workflow_extensions.m4_carry.core import key_seed
from .run import METHODS


def interval(x):
    a=np.asarray(x);a=a[np.isfinite(a)]
    return q95(a) if len(a) else [None,None]


def ratio(a,b):
    return np.divide(a,b,out=np.full_like(a,np.nan),where=abs(b)>1e-14)


def fmt(x):return 'NA' if x is None or not np.isfinite(x) else f'{x:.3f}'


def tensors(rows,methods,budgets,subjects,R,field):
    g=defaultdict(list)
    for r in rows:
        v=r.get(field)
        if v is not None:g[r['method'],r['B'],r['subject'],r['repeat']].append(v)
    x=np.full((len(methods),len(budgets),len(subjects),R),np.nan)
    for mi,m in enumerate(methods):
        for bi,B in enumerate(budgets):
            for si,s in enumerate(subjects):
                for rep in range(R):
                    a=g.get((m,B,s,rep))
                    if a:x[mi,bi,si,rep]=np.mean(a)
    return x


def analyze(out):
    out=Path(out);cfg=json.loads((out/'config.json').read_text());stats=json.loads((out/'public_stats.json').read_text())
    targets=json.loads((out/'targets.json').read_text());budgets=cfg['evaluated_budgets'];R=cfg['repeats'];D=cfg['bootstrap']
    by=defaultdict(list)
    for ln in (out/'metrics.jsonl').open():
        r=json.loads(ln);by[r['template']].append(r)
    summary=[];paired=[];bmins=[];bmeffects=[]
    contrasts=[('M5',m) for m in METHODS if m!='M5']+[
        ('M5-alloc-static','M4-static'),('M5-trace-fixed','M5-local-veto-fixed'),('M5-atomic-fixed','M5-trace-fixed')]
    for t,rows in by.items():
        subjects=sorted({r['subject'] for r in rows});seed=key_seed(cfg['bootstrap_seed'],t)
        fields=['mae','final_truth_abs','query_loss','rce','cmr','p95','projection_rate','m','spend_fraction','MissRate','harmful_refresh','post_stop_mae','frac_veto']
        fields += [s+'__'+node for node in stats[t]['main']['nodes'] for s in ['node_num','node_den']]
        raw={};means={};pb={};nb={}
        for f in fields:
            x=tensors(rows,METHODS,budgets,subjects,R,f);raw[f]=x
            if np.isfinite(x).all():
                means[f]=x.mean(axis=(-1,-2));pb[f],nb[f]=bootstrap_curves(x,D,seed)
            else:
                # Entirely undefined RCE or undefined ratio/event denominators.
                count=np.isfinite(x).sum(axis=(-1,-2))
                means[f]=np.divide(np.nansum(x,axis=(-1,-2)),count,out=np.full(x.shape[:2],np.nan),where=count>0)
                # Event-ratio intervals are omitted rather than treating NA as zero.
        for dest in [means,pb,nb]:
            dest['wmape']=100*ratio(dest['mae'],dest['final_truth_abs'])
            for node in stats[t]['main']['nodes']:
                a='node_num__'+node;b='node_den__'+node
                if a in dest and b in dest:dest['node_wmape__'+node]=100*ratio(dest[a],dest[b])
            intermediate=[(j,n) for j,n in enumerate(stats[t]['main']['nodes']) if n!=__import__('experiments.workflow_extensions.finite_cache.workload',fromlist=['Workload']).Workload(t).target]
            q=stats[t]['main']['query_weights'];den=sum(q[j] for j,n in intermediate)
            if den:
                vals=[(q[j],dest['node_wmape__'+n]) for j,n in intermediate if 'node_wmape__'+n in dest]
                num=sum(weight*np.nan_to_num(v,nan=0.) for weight,v in vals)
                valid=sum(weight*np.isfinite(v) for weight,v in vals)
                dest['intermediate_wmape']=ratio(num,valid)
            else:dest['intermediate_wmape']=np.full_like(dest['mae'],np.nan)
        reported=['mae','wmape','rce','cmr','intermediate_wmape','query_loss','p95','projection_rate','m','spend_fraction','MissRate','harmful_refresh','post_stop_mae','frac_veto']+[k for k in means if k.startswith('node_wmape__')]
        for mi,m in enumerate(METHODS):
            for bi,B in enumerate(budgets):
                r=dict(template=t,method=m,B=B,n_patients=len(subjects),repeats=R)
                for f in reported:
                    v=means[f][mi,bi];r[f]=float(v) if np.isfinite(v) else None
                    if f in pb:
                        for name,boot in [('patient',pb),('two_layer',nb)]:
                            lo,hi=interval(boot[f][mi,bi]);r[f+'_'+name+'_lo']=lo;r[f+'_'+name+'_hi']=hi
                summary.append(r)
        for a,b in contrasts:
            ai=METHODS.index(a);bi=METHODS.index(b)
            for f in ['mae','wmape','rce','intermediate_wmape','query_loss']:
                if f not in pb:continue
                for bj,B in enumerate(budgets):
                    diff=means[f][ai,bj]-means[f][bi,bj]
                    if not np.isfinite(diff):continue
                    r=dict(template=t,B=B,metric=f,method=a,reference=b,delta=float(diff))
                    for name,boot in [('patient',pb),('two_layer',nb)]:r[name+'_lo'],r[name+'_hi']=interval(boot[f][ai,bj]-boot[f][bi,bj])
                    paired.append(r)
        for scenario,ts in targets[t].items():
            for target in ['lo','hi']:
                tau=ts['tau_'+target];bm=bmin(means['mae'],budgets,tau)
                bs={name:bmin(boot['mae'].transpose(0,2,1),budgets,tau) for name,boot in [('patient',pb),('two_layer',nb)]}
                for mi,m in enumerate(METHODS):
                    r=dict(template=t,scenario=scenario,target=target,tau=tau,method=m,B_min=float(bm[mi]) if np.isfinite(bm[mi]) else None,reached=bool(np.isfinite(bm[mi])))
                    for name,x in bs.items():
                        lo,hi=q95(x[mi]);r[name+'_lo']=lo if np.isfinite(lo) else None;r[name+'_hi']=hi if np.isfinite(hi) else None
                        r[name+'_reach_rate']=float(np.isfinite(x[mi]).mean())
                    bmins.append(r)
                ai=METHODS.index('M5')
                for base in FOCUS:
                    bi=METHODS.index(base);both=np.isfinite(bm[ai])&np.isfinite(bm[bi]);r=dict(template=t,scenario=scenario,target=target,reference=base,ratio=float(bm[ai]/bm[bi]) if both else None,Saving=float(1-bm[ai]/bm[bi]) if both else None)
                    for name,x in bs.items():
                        ok=np.isfinite(x[ai])&np.isfinite(x[bi]);r[name+'_both_reach_rate']=float(ok.mean())
                        r[name+'_ratio_conditional_lo'],r[name+'_ratio_conditional_hi']=interval(x[ai,ok]/x[bi,ok])
                    bmeffects.append(r)
    write_csv(out/'summary.csv',summary);write_csv(out/'paired_summary.csv',paired);write_csv(out/'bmin.csv',bmins);write_csv(out/'bmin_paired.csv',bmeffects)
    attack_summary,attack_pairs=analyze_attacks(out,cfg,stats)
    report(out,cfg,stats,summary,paired,bmins,attack_summary,attack_pairs)
    from .attack_checks import run as check_attack_endpoints
    check_attack_endpoints(out)


FOCUS=['RG-static-exp','RG-periodic-exp','RG-static-lap','RG-periodic-lap','M4-carry']


def analyze_attacks(out,cfg,stats):
    groups=defaultdict(list)
    for ln in (out/'attack_metrics.jsonl').open():
        r=json.loads(ln);groups[r['template'],r['attack'],r['prior']].append(r)
    summary=[];paired=[];budgets=cfg['evaluated_budgets'];methods=cfg['attack_methods'];R=cfg['repeats'];D=cfg['bootstrap']
    for (t,attack,prior),rows in groups.items():
        subjects=sorted({r['subject'] for r in rows});seed=key_seed(cfg['bootstrap_seed'],t)
        for i,root in enumerate(stats[t]['main']['roots']):
            for r in rows:r['num__'+root]=r['root_mae'][i];r['den__'+root]=r['root_truth_abs'][i]
        rootvals={};rootboots={};rootnested={}
        for root in stats[t]['main']['roots']:
            x=tensors(rows,methods,budgets,subjects,R,'num__'+root);y=tensors(rows,methods,budgets,subjects,R,'den__'+root)
            p,n=bootstrap_curves(x,D,seed);pd,nd=bootstrap_curves(y,D,seed)
            rootvals[root]=100*ratio(x.mean(axis=(-1,-2)),y.mean(axis=(-1,-2)))
            rootboots[root]=100*ratio(p,pd);rootnested[root]=100*ratio(n,nd)
        val=np.mean(list(rootvals.values()),axis=0);bp=np.mean(list(rootboots.values()),axis=0);bn=np.mean(list(rootnested.values()),axis=0)
        nx=tensors(rows,methods,budgets,subjects,R,'normalized_mae');succ=tensors(rows,methods,budgets,subjects,R,'success_005')
        for mi,m in enumerate(methods):
            for bi,B in enumerate(budgets):
                r=dict(template=t,attack=attack,prior=prior,B=B,method=m,root_macro_wmape=float(val[mi,bi]),normalized_mae=float(nx[mi,bi].mean()),success_005=float(succ[mi,bi].mean()),n_patients=len(subjects))
                for root,v in rootvals.items():r['wmape__'+root]=float(v[mi,bi])
                for name,boot in [('patient',bp),('two_layer',bn)]:r[name+'_lo'],r[name+'_hi']=interval(boot[mi,bi])
                summary.append(r)
        ai=methods.index('M5')
        for base in FOCUS:
            bi=methods.index(base)
            for bj,B in enumerate(budgets):
                r=dict(template=t,attack=attack,prior=prior,B=B,reference=base,delta_reconstruction_wmape=float(val[ai,bj]-val[bi,bj]))
                for name,boot in [('patient',bp),('two_layer',bn)]:r[name+'_lo'],r[name+'_hi']=interval(boot[ai,bj]-boot[bi,bj])
                paired.append(r)
    write_csv(out/'attack_summary.csv',summary);write_csv(out/'attack_paired.csv',paired)
    return summary,paired


def report(out,cfg,stats,summary,paired,bmins,attacks,attack_pairs):
    sel=json.loads((out/'selections.json').read_text());audit=json.loads((out/'independent_audit.json').read_text())
    lines=['# M5 工作区间分配、耦合迹门控与原子根包：机制诊断','',
           f'R={cfg["repeats"]}；预算80–960，原1.1网格及1.05细化共{len(cfg["evaluated_budgets"])}点；患者配对和患者＋机制两层bootstrap={cfg["bootstrap"]}。沿用旧患者划分、域、公式、Δ和τ；四模板test分别4–5人。既有test已参与研究诊断，本轮不作为全新泛化验证。','',
           '## 1. 实现与提前固定的选择','',
           'M5使用原连续Laplace+clip，根归一化恒等发布，仍按整条轨迹max范数、纯epsilon相加。新权重和协方差仅参与效用判断，未缩小隐私域或噪声敏感度。原M-Opt指数核与同Laplace核基线、原M4均从冻结动作重放，并逐路径复现MAE。','',
           '方案没有指定的数值提前固定：工作区间Q10–Q90、耦合阈值0.5、非线性度阈值1、方向持续性阈值0.5。工作区间不裁剪真实发布值。原文未给投影惩罚公式，因此不另造惩罚项。MELD的工作区间sup为角点加裁剪断点邻域的效用近似，不声称新的敏感度证明。','',
           '全部派生节点使用train查询频率作为权重，并除以train节点平均绝对值，避免不同中间节点的单位尺度支配矩阵。M5-final仅使用最终节点；它与M5共用已选时间计划，检验公开任务权重改变对中间查询的影响。','',
           'M5公开等间隔目标时间kT/(K+1)，对齐合法测量机会；K∈{0,1,2}、rho∈{0.5,0.7,0.85}及静态rho=1，仅在validation按查询加权尺度化MAE选。所有模板/B选择完成后才开始test。M5各动态消融使用相同K/rho/时间，不能把静态→动态差值当作纯门控单因素。','',
           '漂移协方差按公开测量批次拟合，未测根增量填0，年龄按全部公开批次数累计；随机游走用min(g_i,g_l)Sigma_il，持续方向根用g_i²E[Δz_i²]对角项。保留PSD并避免常量方向漂移被中心化消掉。原子包以实际执行epsilon计算G，入选后不再改变根集合或预算。','',
           '|模板|test患者|原子组|分配指数|非线性度|B960 K/rho|','|---|---:|---|---:|---:|---|']
    for t,p in stats.items():
        pp=p['main'];s=sel[t]['960.0'];roots=pp['roots'];n=next(r['n_patients'] for r in summary if r['template']==t)
        lines.append(f'|{t}|{n}|'+ '; '.join('{'+','.join(roots[i] for i in g)+'}' for g in pp['groups'])+f'|{pp["exponent"]:.3f}|{pp["nonlinearity"]:.3f}|{s["K"]}/{s["rho"]}|')
    lines+=['','## 2. B=960最终节点和中间查询结果','',
            'MAE/wMAPE/RCE越低越好。wMAPE=100×患者均权绝对误差/患者均权真实绝对值；RCE复用RQ1风险阈值，为100×mean(|预测类别−真类别|/(C−1))，ANEMIA/FIB4的C=3；普通分类错误率另列CMR。MELD_NA/BUN_CR没有现成类别，RCE=NA。中间查询wMAPE在实际查询轮计算，先逐节点求比值，再按train查询频率宏平均；逐节点结果均在summary.csv，不混加不同单位。','',
            '|模板|方法|MAE|wMAPE %|RCE %|中间查询wMAPE %|刷新m|支出比例|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['B']==960 and r['method'] in FOCUS+['M5','M5-final']:
            lines.append('|'+r['template']+'|'+r['method']+'|'+'|'.join(fmt(r.get(f)) for f in ['mae','wmape','rce','intermediate_wmape','m','spend_fraction'])+'|')
    lines+=['','## 3. M5配对差值与判定','',
            '差值=M5−参考，负值更好。95%区间为点态、未做全比较族校正；参数/公开统计/τ固定，CI不包括重新选参不确定性。','',
            '|模板|参考|MAE差|患者95% CI|两层95% CI|判断|','|---|---|---:|---|---|---|']
    for r in paired:
        if r['B']==960 and r['metric']=='mae' and r['method']=='M5' and r['reference'] in FOCUS:
            verdict='M5更好' if r['patient_hi']<0 and r['two_layer_hi']<0 else 'M5更差' if r['patient_lo']>0 and r['two_layer_lo']>0 else '证据不足'
            lines.append(f'|{r["template"]}|{r["reference"]}|{r["delta"]:.3f}|[{fmt(r["patient_lo"])},{fmt(r["patient_hi"])}]|[{fmt(r["two_layer_lo"])},{fmt(r["two_layer_hi"])}]|{verdict}|')
    lines+=['','## 4. 中间查询权重是否有效','',
            'M5−M5-final使用相同时间/初始化比例，公开工作量权重随之影响分配、矩阵和分组，是工作量建模整体消融，不是只改变单个矩阵元素。原M-Opt主基线仍按最终节点分配，故跨方法的中间节点结果还受目标差异影响。','',
            '|模板|中间wMAPE差|两层95% CI|最终MAE差|','|---|---:|---|---:|']
    for t in cfg['templates']:
        rows=[r for r in paired if r['template']==t and r['B']==960 and r['method']=='M5' and r['reference']=='M5-final']
        a=next((r for r in rows if r['metric']=='intermediate_wmape'),None);b=next(r for r in rows if r['metric']=='mae')
        if a:lines.append(f'|{t}|{a["delta"]:.3f}|[{fmt(a["two_layer_lo"])},{fmt(a["two_layer_hi"])}]|{b["delta"]:.3f}|')
        else:lines.append(f'|{t}|NA，无中间节点|NA|{b["delta"]:.3f}|')
    lines+=['','## 5. 分层消融','',
            'M5-alloc-static与M4-static比较分配规则整体替换，含工作区间、指数及多节点目标；M5-local-veto-fixed→M5-trace-fixed比较逐根L1门控与迹门控（两者都使用相同公开多节点查询目标；迹门控先用单根原子组、允许组并集，作为诊断对照）；再到M5-atomic-fixed引入耦合原子组；M5-atomic-fixed→M5只改因果结转。动态三步固定同一已选K/rho/公开时刻；如K=0或无否决，相关模块没有触发，不能据零差断言模块普遍无用。','',
            '|模板|方法−参考|MAE差|两层95% CI|','|---|---|---:|---|']
    ablations=[('M5-alloc-static','M4-static'),('M5-trace-fixed','M5-local-veto-fixed'),('M5-atomic-fixed','M5-trace-fixed'),('M5','M5-atomic-fixed')]
    for r in paired:
        if r['B']==960 and r['metric']=='mae' and (r['method'],r['reference']) in ablations:
            lines.append(f'|{r["template"]}|{r["method"]} − {r["reference"]}|{r["delta"]:.3f}|[{fmt(r["two_layer_lo"])},{fmt(r["two_layer_hi"])}]|')
    lines+=['','## 6. MAP重构攻击','',
            '保留RootGuard RQ2的1000候选网格MAP、均匀/独立人群Gaussian先验及重构wMAPE口径；先验和最大跳变只从train估计。为适配动态轨迹，snapshot MAP仅用当前版本最近一次实际发布，缓存重复查询不叠加独立似然。history MAP额外使用完整发布历史及公开测量掩码，在逐根有界均匀随机游走模型上作精确网格Viterbi；模型并非真实患者分布的已知正确模型，也未利用跨根联合先验。公开日程/根选择/epsilon按转录可测量量处理，不额外计作私有真值观测。攻击者不读取机制噪声种子或未来真值。','',
            'Laplace裁剪边界用尾概率质量、内点用密度；原生指数机制保留依赖候选根值的归一化常数。攻击误差越小代表重构越成功、经验隐私越弱；不能把任务误差下降当作隐私提升。重构wMAPE先逐根ratio-of-sums再宏平均，逐根值与归一化MAE、域宽5%成功率另表。','',
            '|模板|方法|snapshot uniform 重构wMAPE %|snapshot informed 重构wMAPE %|history informed 重构wMAPE %|','|---|---|---:|---:|---:|']
    for t in cfg['templates']:
        for m in cfg['attack_methods']:
            aa={r['attack']:r for r in attacks if r['template']==t and r['B']==960 and r['method']==m and r['prior']=='informed'}
            uniform=next(r for r in attacks if r['template']==t and r['B']==960 and r['method']==m and r['prior']=='uniform')
            lines.append('|'+t+'|'+m+'|'+fmt(uniform['root_macro_wmape'])+'|'+fmt(aa['snapshot_map']['root_macro_wmape'])+'|'+fmt(aa['history_map']['root_macro_wmape'])+'|')
    lines+=['','M5相对原M4的历史MAP差值（正值为攻击误差更高，不代表DP保证更强）：','',
            '|模板|重构wMAPE差|两层95% CI|','|---|---:|---|']
    for r in attack_pairs:
        if r['B']==960 and r['reference']=='M4-carry' and r['attack']=='history_map' and r['prior']=='informed':
            lines.append(f'|{r["template"]}|{r["delta_reconstruction_wmape"]:.3f}|[{fmt(r["two_layer_lo"])},{fmt(r["two_layer_hi"])}]|')
    lines+=['','## 7. 固定τ最小网格预算','',
            'τ完全沿用旧公开合成人群目标；只报告扫描网格内达标，未达标为删失，不证明下界。双方达标时才计算Saving；bootstrap删失概率和条件区间见bmin*.csv。','',
            '|模板|目标|τ|RG静态原核|RG周期原核|M4-carry|M5|','|---|---|---:|---:|---:|---:|---:|']
    for t in cfg['templates']:
        for sc in targets_for(out,t):
            for target in ['lo','hi']:
                rows=[r for r in bmins if r['template']==t and r['scenario']==sc and r['target']==target];rr={r['method']:r for r in rows}
                vals=[rr[m]['B_min'] for m in ['RG-static-exp','RG-periodic-exp','M4-carry','M5']]
                lines.append('|'+t+'|'+sc+'/'+target+'|'+fmt(rows[0]['tau'])+'|'+'|'.join('未达标' if x is None else f'{x:.2f}' for x in vals)+'|')
    lines+=['','## 8. 审计与复现','',
            f'独立转录审计通过：{audit["independent_paths"]}条路径，包含初始化的总支出、正epsilon、原子组、最新已观测真值和发布噪声重放均通过；最大发布重放差{audit["max_release_difference"]:.3g}。旧结果未覆盖。','',
            '实现对照、修正及单元测试口径见源码目录MECHANISM-REVIEW-zh.md。噪声流按旧框架离线配对；部署必须使用秘密随机性，不能公开真实噪声种子。','',
            '```bash','cd /home/rootguard-research',
            '/root/miniconda3/envs/rootguard/bin/python -m unittest experiments.workflow_extensions.m5.test_m5 -v',
            '/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m5.run --precheck --output experiments/workflow_extensions/outputs/m5/precheck_reproduce',
            '/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m5.run --output experiments/workflow_extensions/outputs/m5/real_reproduce','```','']
    conclusions=['## 主要判断（预指定B=960）','']
    for t in cfg['templates']:
        s={r['method']:r for r in summary if r['template']==t and r['B']==960}
        effects=[r for r in paired if r['template']==t and r['B']==960 and r['metric']=='mae' and r['method']=='M5']
        old=next(r for r in effects if r['reference']=='M4-carry')
        delta=100*(s['M5']['mae']/s['M4-carry']['mae']-1)
        direction='降低' if delta<0 else '上升'
        oldverdict='两种区间均支持改善' if old['patient_hi']<0 and old['two_layer_hi']<0 else '两种区间均支持退化' if old['patient_lo']>0 and old['two_layer_lo']>0 else '区间尚不能区分'
        refs=[r for r in effects if r['reference'] in ('RG-static-exp','RG-periodic-exp')]
        both=all(r['patient_hi']<0 and r['two_layer_hi']<0 for r in refs)
        conclusions.append(f'- {t}：相对M4-carry，MAE{direction}{abs(delta):.1f}%，{oldverdict}；'+('同时优于两个原核M-Opt对照。' if both else '未形成同时优于两个原核M-Opt对照的证据。'))
    conclusions+=['','H1要求ANEMIA与FIB4均同时胜过两个原M-Opt对照；本轮逐模板判断见上，不能只报告改善幅度而忽略强基线。H2的MELD不劣化也须独立检查，不能以其他模板收益抵消。主目标MAE改善不保证RCE同步改善。','',
                 'B=960前三模板K=0，相关门控、分组和结转不执行；FIB4改善可定位为分配规则整体替换，ANEMIA改善还包括避免原来的两次刷新。当前结果没有建立因果结转的独立收益。对MELD的退化主要需检查为何公开多节点目标在validation选择静态；不在本轮据test重调。','']
    lines[4:4]=conclusions
    (out/'REPORT-zh.md').write_text('\n'.join(lines)+'\n')


def targets_for(out,t):return json.loads((out/'targets.json').read_text())[t]
