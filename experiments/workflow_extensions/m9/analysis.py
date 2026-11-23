"""Paired record-level inference, grid minima and validation-locked local targets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
ANCHORS=[80.,318.99,651.22,960.]
METRICS=['mae','p95','response','den','rce','refresh','roots','spent','map_error']
PAIRS=[('m9',x) for x in ('rootguard','m8','static','period','full','package','pooled','force_dynamic','no_estimator')]+[('package','full'),('early','package'),('anchor75','package'),('equal','package'),('full','static')]
def main():
    rows=[];effects=[];iso=[];distributions=[];checks=[]
    old=json.loads((HERE.parent/'outputs/m4_carry/real_v1_20261116/targets.json').read_text())
    for name,(_,t) in json.loads((OUT/'protocol.json').read_text())['cohorts'].items():
        dest=OUT/name;cfg=json.loads((dest/'frozen.json').read_text())
        df=pd.read_csv(dest/'paths.csv')
        assert not df.duplicated(['subject','B','method']).any()
        assert df.subject.nunique()==cfg['split_counts'][2]
        assert (df.spent<=df.B+1e-8).all()
        full=pd.concat([pd.read_csv(dest/filename) for filename in ('grid54.csv','extra54.csv')])
        val=pd.read_csv(dest/'iterations.csv')
        targets={f'{level}_{key}':v[key] for level,v in old[t].items() for key in ('tau_lo','tau_hi')}
        for B in (318.99,651.22):
            vv=val[(val.B==B)&(val.arm=='static')];targets['validation_static_'+str(B)]=np.average(vv.mae,weights=vv.validation_n)
        for group,frame in [('all',df),('high',df[df.high==1]),('low',df[df.high==0])]:
            if frame.empty:continue
            grids=frame.groupby(['method','B']).mae.mean()
            for m in frame.method.unique():
                curve=full[(full.group==group)&(full.method==m)].sort_values('B')
                coarse=False
                if curve.empty:
                    curve=grids.loc[m].reset_index();coarse=True
                else:
                    for B in ANCHORS:
                        diff=abs(float(curve[curve.B==B].mae.iloc[0])-grids.loc[m,B])
                        checks.append(diff)
                        if diff>1e-8:raise ValueError((name,group,m,B,diff))
                for target,tau in targets.items():
                    hits=curve[curve.mae<=tau]
                    iso.append(dict(cohort=name,group=group,method=m,target=target,tau=tau,grid=len(curve),B_min=None if hits.empty else hits.B.min()))
            for budget,part in [('avg',frame[frame.B.isin(ANCHORS)]),*frame.groupby('B')]:
                pt=part.groupby(['method','subject'])[METRICS].mean()
                for m,r in pt.groupby(level=0).mean().iterrows():
                    rows.append(dict(cohort=name,group=group,B=budget,n=part.subject.nunique(),method=m,wmape=100*r.mae/r.den,**r))
                for (m,a),gg in part[part.method=='m9'].groupby(['method','arm']):
                    distributions.append(dict(cohort=name,group=group,B=budget,arm=a,fraction=len(gg)/sum(part.method=='m9')))
                if budget!='avg':continue
                rng=np.random.default_rng(211);n=part.subject.nunique();ix=rng.integers(n,size=(1500,n))
                for a,b in PAIRS:
                    x,y=(pt.loc[m,'mae'].to_numpy() for m in (a,b))
                    ci=np.quantile(100*(1-x[ix].mean(1)/y[ix].mean(1)),[.025,.975])
                    effects.append(dict(cohort=name,group=group,method=a,ref=b,gain=100*(1-x.mean()/y.mean()),lo=ci[0],hi=ci[1]))
    for filename,items in [('summary',rows),('effects',effects),('iso',iso),('routes',distributions)]:
        pd.DataFrame(items).to_csv(OUT/(filename+'.csv'),index=False)
    (OUT/'grid_parity.json').write_text(json.dumps(dict(cells=len(checks),max_abs=max(checks))))
if __name__=='__main__':main()
