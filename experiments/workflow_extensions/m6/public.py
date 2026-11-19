"""All fitting is train-only; constants are specified before fitting/test."""
import ast
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.m5.public import fit_public as old_fit,jacobian,allocate

# Canonical workload ASTs: reference derived nodes instead of flattening the DAG.
EXPRESSIONS={
 'ANEMIA':['10*hct/rbc','10*hb/rbc','100*hb/hct'],
 'FIB4':['age*ast','sqrt(alt)*plt','fib4_prod/fib4_denom'],
 'BUN_CR':['bun/cr'],
 'MELD_NA':['3.78*log(maximum(bil,1))','11.2*log(maximum(inr,1))',
            '9.57*log(clip(cr,1,4))','bil_term+inr_term+cr_term+6.43',
            'clip(meld+1.32*(137-clip(na,125,137))-.033*meld*(137-clip(na,125,137)),6,40)']}


def ast_groups(w):
    """Merge root pairs inside smallest multiplicative AST subexpressions.

    Derived-node identifiers are boundaries. Overlapping groups are closed
    transitively. This intentionally merges ALL three ANEMIA roots because the
    workload queries all three overlapping ratios; it is not weight-thresholded.
    """
    parent=list(range(len(w.roots)));evidence=[]
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    def walk(node):
        children=[walk(c) for c in ast.iter_child_nodes(node)]
        roots={w.roots.index(n.id) for n in ast.walk(node) if isinstance(n,ast.Name) and n.id in w.roots}
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Mult,ast.Div,ast.Pow)) and len(roots)>1 and not any(len(s)>1 for s in children):
            ids=sorted(roots)
            for i in ids[1:]:parent[find(i)]=find(ids[0])
            evidence.append(dict(expression=ast.unparse(node),roots=ids))
        return roots
    for expression in EXPRESSIONS[w.template]:walk(ast.parse(expression,mode='eval'))
    groups={}
    for i in range(len(w.roots)):groups.setdefault(find(i),[]).append(i)
    return list(groups.values()),evidence


def fit_public(records,w,cfg,final_only=False):
    p=old_fit(records,w,cfg,final_only)
    rng=np.random.default_rng(cfg['skeleton_seed'])
    points=rng.uniform(p['work_lo'],p['work_hi'],size=(cfg['skeleton_points'],len(w.roots)))
    weights=np.array(p['query_weights']);scales=np.array(p['node_scales'])
    gradients=np.sum(weights[None,:,None]*abs(jacobian(w,points))/scales[None,:,None],axis=1)
    a_mean=np.sum(weights[:,None]*abs(jacobian(w,np.array(p['mu'])))/scales[:,None],axis=0)
    a_q=np.quantile(gradients,cfg['gradient_quantile'],axis=0)
    gm=gradients.mean(axis=0);cv=np.divide(gradients.var(axis=0),gm**2,out=np.zeros_like(gm),where=gm>1e-14)
    nu=float(np.dot(a_mean,cv)/max(a_mean.sum(),1e-30))
    M=np.array(p['M']);cov=np.array(p['covariance']);den=np.sum(np.diag(M)*np.diag(cov))
    cancellation=float(1-np.sum(M*cov)/den) if den>1e-20 else 0.
    groups,evidence=ast_groups(w)
    from experiments.workflow_extensions.finite_cache.workload import MONOMIALS
    powers=[v[1] for v in MONOMIALS.get(w.template,{}).values()]
    denominator=np.any(np.array(powers)<0,axis=0) if powers else np.zeros(len(w.roots),bool)
    # A denominator is high-gradient at the lower train working quantile. This
    # predeclared 0.5 threshold is the median (no fitting against test outcomes).
    states=np.array([w.normalize(data.replay(r)) for r in records]);flat=states.reshape(-1,len(w.roots))
    threshold=np.quantile(flat,cfg['danger_quantile'],axis=0)
    gridpoints=np.clip(np.ceil(cfg['grid_base']*(1+cv)).astype(int),cfg['grid_base'],cfg['grid_max'])
    p.update(a_mean=a_mean.tolist(),a_quantile=a_q.tolist(),nu_i=cv.tolist(),nonlinearity=nu,
             cancellation=cancellation,allocation_weights=np.maximum(a_q if nu>cfg['nu_grid_threshold'] else a_mean,cfg['allocation_weight_floor']).tolist(),
             allocation_mode='quantile' if nu>cfg['nu_grid_threshold'] else 'mean',exponent=.5,
             regime='STATIC' if cancellation>cfg['c_threshold'] or nu>cfg['nu_threshold'] else 'DYNAMIC',
             groups=groups,group_ast_evidence=evidence,denominator_roots=denominator.tolist(),
             danger_threshold=threshold.tolist(),grid_points=gridpoints.tolist(),
             # Declared public scenario library; no current/test patient included.
             scenario_paths=states.tolist(),scenario_subjects=[str(r['subject_id']) for r in records])
    return p
