"""Patient-record paired effects with pointwise bootstrap intervals."""
import numpy as np
import pandas as pd
from pathlib import Path
OUT=Path(__file__).resolve().parent.parent/'outputs/physionet2012'
METRICS=['mae','p95','response','refreshes','spent','den']
def main():
    summaries=[];effects=[]
    for task in ('BUN_CR','FIB4'):
        df=pd.concat([pd.read_csv(OUT/(task+s+'.csv')) for s in ('','_local')])
        assert not df.duplicated(['subject','B','method']).any()
        assert (df.spent-df.B).abs().max()<1e-8
        groups={'all':df,'high':df[df.high==1],'low':df[df.high==0],'dense':df[df.dense==1]}
        for gr,g in groups.items():
            for lab,f in [('avg',g)]+list(g.groupby('B')):
                patient=f.groupby(['subject','method'])[METRICS].mean()
                avg=patient.groupby('method').mean()
                for m,r in avg.iterrows():
                    summaries.append(dict(task=task,group=gr,B=lab,method=m,n=f.subject.nunique(),**r,wmape=100*r.mae/r.den))
                bootstrap={}
                pairs=[('task',m) for m in avg.index if m!='task']+[('new_static','cal_static'),('new_static','local_static'),('old_continuous','joint'),('old_analog','joint'),('local_joint','joint')]
                for a,b in pairs:
                    for metric in ('mae','response','p95'):
                        both=patient[metric].unstack()[[a,b]].dropna()
                        x,y=both.to_numpy().T
                        if not len(x):continue
                        if len(x) not in bootstrap:
                            ix=np.random.default_rng(113).integers(len(x),size=(1000,len(x)))
                            bootstrap[len(x)]=np.stack([np.bincount(z,minlength=len(x)) for z in ix]).astype(float)
                        weights=bootstrap[len(x)]
                        boot=100*(1-(weights@x)/(weights@y))
                        lo,hi=np.quantile(boot,[.025,.975])
                        effects.append(dict(task=task,group=gr,B=lab,method=a,reference=b,metric=metric,n=len(x),gain=100*(1-x.mean()/y.mean()),lo=lo,hi=hi))
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(effects).to_csv(OUT/'effects.csv',index=False)
if __name__=='__main__':main()
