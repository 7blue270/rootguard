"""Final-task and actual-query metrics; RootGuard RQ1 ordinal RCE, not CMR."""
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.m4_carry.core import evaluate as old_evaluate


def evaluate(rec,result,w,old_stats,cfg,public):
    out=old_evaluate(rec,result,w,old_stats,cfg,detailed=True)
    truth=w.evaluate(data.replay(rec));pred=w.from_cache(result['caches']);R=result['caches'].shape[0]
    out['final_truth_abs']=np.full(R,np.mean(abs(truth[w.target])))
    out['wmape']=100*out['mae']/out['final_truth_abs']
    classes=w.classes(truth[w.target])
    if classes is None:
        out['rce']=np.full(R,np.nan);out['cmr']=np.full(R,np.nan)
    else:
        pc=w.classes(pred[w.target]);out['rce']=100*np.mean(abs(pc-classes)/2,axis=1)
        out['cmr']=100*np.mean(pc!=classes,axis=1)
    # Per-node numerator/denominator preserved for ratio-of-sums bootstrapping.
    query_loss=np.zeros(R)
    for j,node in enumerate(w.nodes):
        ix=[q['turn']-1 for q in rec['queries'] if q['node']==node]
        if ix:
            num=np.mean(abs(pred[node][:,ix]-truth[node][ix]),axis=1)
            den=np.full(R,np.mean(abs(truth[node][ix])))
        else:num=np.full(R,np.nan);den=np.full(R,np.nan)
        out['node_num__'+node]=num;out['node_den__'+node]=den
        if ix:query_loss+=public['query_weights'][j]*num/public['node_scales'][j]
    out['query_loss']=query_loss
    return out
