"""Within-template uncertainty for paired patient means and relative reduction."""
import csv
import numpy as np
from .study import OUT,load,write

def main(out=OUT, cohorts=None, primary='task', extra_pairs=None):
    summary=[];effects=[]
    for name,t in cohorts or [('PhysioNet','BUN_CR'),('MIMIC_BUN','BUN_CR'),('MELD_NA','MELD_NA'),('ANEMIA','ANEMIA'),('FIB4','FIB4')]:
        ctx,*_=load(t,name.startswith('PhysioNet'))
        with (out/(name+'.csv')).open() as file:rows=list(csv.DictReader(file))
        idx={(r['subject'],r['B'],r['method']):r for r in rows}
        methods=sorted({r['method'] for r in rows})
        budgets=sorted({r['B'] for r in rows},key=float)
        for group in ('all','high','low'):
            ids=sorted({r['subject'] for r in rows if group=='all' or int(r['high'])==(group=='high')})
            if not ids:continue
            rng=np.random.default_rng(113);ix=rng.integers(len(ids),size=(1000,len(ids)))
            for label,bs in [('avg',budgets)]+[(b,[b]) for b in budgets]:
                computed={}
                for m in methods:
                    keys=['mae','response','jump_error','p95','final_truth_abs','rce']
                    keys += [k for k in rows[0] if k.startswith(('den:','num:'))]
                    vals={}
                    for k in keys:
                        mat=np.array([[float(idx[s,b,m].get(k) or 'nan') for b in bs] for s in ids])
                        if not np.isfinite(mat).any():continue
                        a=mat.mean(axis=1)
                        sample=a[ix]; valid=np.isfinite(sample).sum(axis=1)
                        vals[k]=(np.nanmean(a),np.divide(np.nansum(sample,axis=1),valid,out=np.full(len(ix),np.nan),where=valid>0))
                    def ratio(a,b):
                        return tuple(np.divide(100*np.asarray(x), y, out=np.full_like(np.asarray(x),np.nan), where=np.asarray(y)>0) for x,y in zip(vals[a],vals[b]))
                    vals['wmape']=ratio('mae','final_truth_abs')
                    nodes=[(node,w) for node,w in zip(ctx.nodes,ctx.query_weights) if node!=ctx.work.target and w>0 and 'num:'+node in vals]
                    if nodes:
                        def macro(j):
                            parts=np.asarray([ratio('num:'+node,'den:'+node)[j] for node,_ in nodes])
                            ws=np.asarray([w for _,w in nodes]).reshape((-1,)+(1,)*(parts.ndim-1))
                            norm=(ws*np.isfinite(parts)).sum(axis=0)
                            return np.divide(np.nansum(ws*parts,axis=0),norm,out=np.full_like(norm,np.nan),where=norm>0)
                        vals['intermediate']=tuple(macro(j) for j in (0,1))
                    for k,(v,z) in vals.items():
                        if ':' in k or k=='final_truth_abs':continue
                        computed[m,k]=v,z
                        summary.append(dict(dataset=name,group=group,n=len(ids),B=label,method=m,metric=k,value=v))
                pairs=[(primary,ref) for ref in methods if ref!=primary]+(extra_pairs if extra_pairs is not None else [('joint','independent'),('independent','overwrite'),('emp_median','emp_mean')])
                for a,b in pairs:
                    for m,k in computed:
                        if m!=a:continue
                        x,u=computed[a,k];y,v=computed[b,k]
                        diff=x-y;lo,hi=np.nanquantile(u-v,[.025,.975])
                        reduction=1-x/y if y>0 else None
                        ok=v>0
                        l,h=np.nanquantile(1-u[ok]/v[ok],[.025,.975]) if ok.any() else (None,None)
                        effects.append(dict(dataset=name,group=group,B=label,method=a,reference=b,metric=k,delta=diff,lo=lo,hi=hi,reduction=reduction,reduction_lo=l,reduction_hi=h))
    write(out/'summary.csv',summary);write(out/'effects.csv',effects)
if __name__=='__main__':main()
