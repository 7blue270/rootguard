"""Build the M9 report from archived scores and locked policies."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'---|'*len(header)]+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    s=pd.read_csv(OUT/'summary.csv',dtype={'B':str}).set_index(['cohort','group','B','method'])
    e=pd.read_csv(OUT/'effects.csv').set_index(['cohort','group','method','ref'])
    iso=pd.read_csv(OUT/'iso.csv');routes=pd.read_csv(OUT/'routes.csv',dtype={'B':str})
    groups=list(json.loads((OUT/'protocol.json').read_text())['cohorts'])
    def gain(n,m,b,g='all',ci=False):
        r=e.loc[n,g,m,b]
        return f'{r.gain:+.2f}'+(f' [{r.lo:+.2f}, {r.hi:+.2f}]' if ci else '')
    def num(x):return '—' if pd.isna(x) else f'{x:.3f}'
    cfg={n:json.loads((OUT/n/'frozen.json').read_text()) for n in groups}
    lines=['# M9 端到端探索性回测','', '以下均为已使用过的测试记录。主效应按患者/住院记录配对，四个预算锚点等权。不同 AST 的 MAE 不跨队列相加。百分比效应为误差相对降幅，正值较好；区间为 1500 次记录级配对 bootstrap 的点态 95% 区间，条件于当前训练划分和随机数。旧模板每组仅有 4–5 条测试记录，区间不能充分反映训练和分流策略的不确定性。','', '## 全轨迹 MAE 与静动态分流','']
    ms=['rootguard','m8','static','period','full','package','m9']
    lines.append(table(['队列','训练/验证/测试','原版','M8','任务静态','周期全根','单次全根','单次根包','M9','选动态%'],[[n,'/'.join(map(str,cfg[n]['split_counts']))]+[num(s.loc[n,'all','avg',m].mae) for m in ms]+[f"{100*s.loc[n,'all','avg','m9'].refresh:.1f}"] for n in groups]))
    lines+=['','选动态比例按实际发生一次刷新计算。单次全根和单次根包均以第 21 轮为目标、首发一半预算。完整 M9 的详细分支分布在 routes.csv。','',table(['队列','相对 M8 [95% 区间]','相对任务静态 [95% 区间]','高波动相对静态','低波动相对静态','相对强制动态','相对无密度分组'],[[n,gain(n,'m9','m8',ci=True),gain(n,'m9','static',ci=True),gain(n,'m9','static','high'),gain(n,'m9','static','low'),gain(n,'m9','force_dynamic'),gain(n,'m9','pooled')] for n in groups]),'','## 效用、效率与重构','', '响应指标为超过训练集非零绝对变化中位数（含等于）的变化轮及下一轮的 MAE，重叠去重。P95 在单条轨迹内部计算再平均。RCE 沿用仓库的序数风险等级定义，BUN/CR 与 MELD-Na 无该指标。下表为 M9，其他所有方法相同口径的结果见 summary.csv。MAP 为归一化全根全轨迹误差，较大表示该离线重构器误差较大。','',table(['队列','高/低波动 MAE','响应 MAE','wMAPE %','P95','RCE %','刷新次数','付费根次数（含首发）','平均支出','MAP：M9 / M8'],[[n,' / '.join(num(s.loc[n,g,'avg','m9'].mae) for g in ('high','low'))]+[num(s.loc[n,'all','avg','m9'][k]) for k in ('response','wmape','p95','rce','refresh','roots','spent')]+[' / '.join(num(s.loc[n,'all','avg',m].map_error) for m in ('m9','m8'))] for n in groups]),'','## 控制其他因素的组件比较','', '按列依次为完整路由上用整套估计替代边际覆盖、缩小根包、将预约提前、增加首发份额、均分刷新根预算。后四项在强制单次动态下比较。百分比表示变更带来的 MAE 降幅。','',table(['队列','估计模块','部分根/全根','早/中预约','75%/50% 首发','均分/m-opt'],[[n]+[gain(n,a,b) for a,b in [('m9','no_estimator'),('package','full'),('early','package'),('anchor75','package'),('equal','package')]] for n in groups])]
    extra=[]
    for n in groups:
        paths=pd.read_csv(OUT/n/'paths.csv',usecols=['subject','B','method','mae'])
        m=paths[(paths.method=='m9')&(paths.B.isin([80.,318.99,651.22,960.]))].groupby('subject').mae.mean()
        for label,g in pd.read_csv(OUT/n/'estimator.csv').groupby('method'):
            ref=g.groupby('subject').mae.mean().loc[m.index]
            ix=np.random.default_rng(211).integers(len(m),size=(1500,len(m)))
            x=m.to_numpy();y=ref.to_numpy()
            lo,hi=np.quantile(100*(1-x[ix].mean(1)/y[ix].mean(1)),[.025,.975])
            extra.append([n,label,f'{100*(1-x.mean()/y.mean()):+.2f} [{lo:+.2f}, {hi:+.2f}]'])
    lines+=['','同一路由、付费动作和基础后验下，独立移除校正，或在 2012 队列用同训练样本的逐根损失替代终端 AST 损失。','',table(['队列','对照','M9 全程 MAE 降幅 %'],extra),'','## 同等效用的 B_min','', '下表使用各队列预设的目标域验证静态在 B=651.22 时的验证 MAE 作为统一阈值，并在整个测试组上找首次达标预算。对每一行所有机制使用同一阈值。— 表示至 960 未达，80 表示网格下限即已达。原始三档历史阈值各有上下界，以及各组、各法的结果均列在 iso.csv。54 点主系统扫描与配对逐记录实现交叉校验；此处收益只精确到网格。','']
    ir=[]
    for n in groups:
        a=iso[(iso.cohort==n)&(iso.group=='all')&(iso.target=='validation_static_651.22')].set_index('method')
        ir.append([n,num(a.tau.iloc[0])]+['—' if pd.isna(a.loc[m,'B_min']) else str(a.loc[m,'B_min']) for m in ('rootguard','m8','static','full','package','m9')])
    lines.append(table(['队列','目标 MAE','原版','M8','任务静态','单次全根','单次根包','M9'],ir))
    lines+=['','[机制、威胁模型及命令](README.md) · [完整指标](results/summary.csv) · [配对效应](results/effects.csv) · [预算阈值](results/iso.csv) · [路由占比](results/routes.csv) · [发布与读取审计](results/audit.json)']
    judgment=HERE/'INTERPRETATION.md'
    if judgment.exists():lines[2:2]=[judgment.read_text().strip(),'']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
