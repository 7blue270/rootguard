"""Summarize patient-matched differences and within-template utility."""
from collections import defaultdict
from pathlib import Path
import csv
import json
from ..m7.run import DEFAULT_SPLIT
import numpy as np
from .study import OUT,BUDGETS,write
from ..m7.engine import Context
from ..m7.analysis import derived
from ..finite_cache import data

def summarize(kind):
    panels=defaultdict(dict)
    for path in sorted(OUT.glob(f'*-{kind}.csv')):
        with path.open() as stream:
            for r in csv.DictReader(stream):
                label=r['dataset'].split('-fold')[0];B=float(r['B']);s=r['subject'];method=r['method']
                rec={k:float(v) for k,v in r.items() if k not in ('subject','B','method','dataset','template') and v}
                panels[label,B,method][s]=rec
                # Identical public branch gives identical paired measurements.
                source='forced' if B==960. else 'm8'
                if method==source and label!='three_batches':panels[label,B,'ceiling'][s]=rec
    out=[]
    for label in sorted({t for t,_,_ in panels}):
        weights={}
        if label in ('MELD_NA','ANEMIA','FIB4','TYG','NLR'):
            recs=data.load(Path('data_mine/finite_cache'),[label])
            split=json.loads(DEFAULT_SPLIT.read_text())
            for r in recs:
                refs=[z for z in recs if z!=r] if label=='NLR' else [z for z in recs if split.get(z['subject_id'])=='train']
                c=Context(label,refs,713)
                weights[r['subject_id']]={node:w for node,w in zip(c.nodes,c.query_weights) if node!=c.work.target and w>0}
        rng=np.random.default_rng(713)
        for group in ('all','high','low'):
            ids=sorted(s for s,r in panels[label,BUDGETS[0],'static'].items() if group=='all' or (r['high']==1)==(group=='high'))
            if not ids:continue
            ix=rng.integers(len(ids),size=(1000,len(ids)))
            for budget,bs in [('avg',BUDGETS)]+[(str(b),[b]) for b in BUDGETS]:
                methods=[m for t,b,m in panels if t==label and b==bs[0]]
                metrics=['mae','wmape','p95','refreshes','spent','rce'] if kind=='paths' else ['history_map_informed','snapshot_map_informed','mean_all_hit','kalman_all_hit','changepoint_all_hit']
                if weights and kind=='paths':metrics.append('intermediate_wmape')
                for k in metrics:
                    values={}
                    for m in methods:
                        akey='mae' if k=='wmape' else k
                        table=panels[label,bs[0],m]
                        if k=='intermediate_wmape':
                            # Use train query frequencies; LOPO has a distinct reference per fold.
                            vals=[];boots=[]
                            for b in bs:
                                tab=panels[label,b,m]; vv=0.;zz=np.zeros(len(ix));denw=0;bw=np.zeros(len(ix))
                                for node in weights[ids[0]]:
                                    w=np.array([weights[s][node] for s in ids])
                                    num=np.array([tab[s]['num:'+node] for s in ids])
                                    den=np.array([tab[s]['den:'+node] for s in ids])
                                    if den.mean()>0:
                                        vv+=w.mean()*100*num.mean()/den.mean()
                                        denw+=w.mean()
                                    d=den[ix].mean(axis=1); ww=w[ix].mean(axis=1)
                                    zz+=np.divide(ww*100*num[ix].mean(axis=1),d,out=np.zeros_like(d),where=d>0)
                                    bw+=ww*(d>0)
                                vals.append(vv/denw if denw else np.nan);boots.append(np.divide(zz,bw,out=np.full_like(zz,np.nan),where=bw>0))
                            values[m]=np.mean(vals),np.mean(boots,axis=0)
                            continue
                        if ids[0] not in table or akey not in table[ids[0]]:continue
                        a=np.array([[panels[label,b,m][s][akey] for b in bs] for s in ids]).mean(axis=1)
                        v=a.mean();draw=a[ix].mean(axis=1)
                        if k=='wmape':
                            den=np.array([table[s]['final_truth_abs'] for s in ids])
                            v=100*v/den.mean();draw=100*draw/den[ix].mean(axis=1)
                        values[m]=v,draw
                    for m,(v,z) in values.items():
                        for ref in ('static','periodic','m7','m8_original'):
                            if ref not in values:continue
                            v0,z0=values[ref];lo,hi=np.nanquantile(z-z0,[.025,.975])
                            out.append(dict(dataset=label,group=group,B=budget,n=len(ids),method=m,metric=k,
                                value=v,reference=ref,delta=v-v0,lo=lo,hi=hi))
    write(OUT/f'{kind}_summary.csv',out)
if __name__=='__main__':
    summarize('paths');summarize('privacy')
