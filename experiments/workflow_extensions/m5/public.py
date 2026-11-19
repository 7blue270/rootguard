"""Train-only utility quantities. None changes the identity-release sensitivity."""
from itertools import product
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload, MONOMIALS


def jacobian(w, z):
    """Signed analytic Jacobian of all AST nodes in normalized root coordinates."""
    z=np.asarray(z,float); x=w.lo+np.clip(z,0,1)*w.width
    out=np.zeros(x.shape[:-1]+(len(w.nodes),len(w.roots)))
    for node,(coef,powers) in MONOMIALS.get(w.template,{}).items():
        p=np.array(powers); value=coef*np.prod(x**p,axis=-1)
        out[...,w.nodes.index(node),:]=value[...,None]*p/x*w.width
    if w.template=='MELD_NA':
        b,i,c,n=np.moveaxis(x,-1,0)
        terms=[3.78*np.log(np.maximum(b,1)),11.2*np.log(np.maximum(i,1)),9.57*np.log(np.clip(c,1,4))]
        ders=[np.where(b>1,3.78/b,0),np.where(i>1,11.2/i,0),np.where((c>1)&(c<4),9.57/c,0)]
        for j in range(3):out[...,j,j]=ders[j]*w.width[j]
        out[...,3,:]=out[...,:3,:].sum(axis=-2)
        meld=sum(terms)+6.43;d=137-np.clip(n,125,137);y=meld+1.32*d-.033*meld*d
        out[...,4,:]=out[...,3,:]*(1-.033*d)[...,None]
        out[...,4,3]=np.where((n>125)&(n<137),-(1.32-.033*meld)*w.width[3],0)
        out[...,4,:]*=((y>6)&(y<40))[...,None]
    return out


def connected_groups(M,cov,theta):
    k=len(M); groups=[];remaining=set(range(k));chi=np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            den=np.sqrt(max(0,M[i,i]*M[j,j]*cov[i,i]*cov[j,j]))
            if den>0:chi[i,j]=abs(M[i,j]*cov[i,j])/den
    while remaining:
        component={min(remaining)}; frontier=list(component);remaining-=component
        while frontier:
            i=frontier.pop();new={j for j in remaining if chi[i,j]>theta}
            component|=new;remaining-=new;frontier.extend(sorted(new))
        groups.append(sorted(component))
    return groups,chi


def fit_public(records,w,cfg,final_only=False):
    assert records
    arrays=[w.normalize(data.replay(r)) for r in records]
    # Equal patient/trajectory weights; this cohort has one trajectory/patient.
    flat=np.concatenate(arrays);mu=np.mean([z.mean(axis=0) for z in arrays],axis=0)
    a,b=np.quantile(flat,[cfg['work_quantile'],1-cfg['work_quantile']],axis=0)
    # Quantile intervals can collapse for constant age. No artificial privacy domain.
    counts=np.zeros(len(w.nodes)); scales=[]
    for rec in records:
        for q in rec['queries']:counts[w.nodes.index(q['node'])]+=1
    weights=counts/counts.sum()
    if final_only:
        weights[:]=0;weights[w.nodes.index(w.target)]=1
    for node in w.nodes:
        scales.append(max(cfg['node_scale_floor'],float(np.mean([np.mean(abs(w.from_cache(z)[node])) for z in arrays]))))
    scales=np.array(scales)
    grids=[]
    breakpoints=[[],[],[],[]]
    if w.template=='MELD_NA':breakpoints=[[1],[1],[1,4],[125,137]]
    for i,(lo,hi) in enumerate(zip(a,b)):
        g=[lo,hi,mu[i]]
        if w.template=='MELD_NA':
            for raw in breakpoints[i]:
                v=(raw-w.lo[i])/w.width[i]
                if lo<=v<=hi:g.extend([np.clip(v-1e-9,lo,hi),np.clip(v+1e-9,lo,hi)])
        grids.append(sorted(set(float(np.clip(v,lo,hi)) for v in g)))
    points=np.array(list(product(*grids)))
    local=jacobian(w,mu); J=jacobian(w,points)
    # Jacobians already multiply full-domain widths D. Work correction rescales
    # by dW/D; positive constant-domain roots retain their local-times-D term.
    mean_abs=np.sum(weights[:,None]*abs(local)/scales[:,None],axis=0)
    sup_abs=np.sum(weights[:,None]*np.max(abs(J),axis=0)/scales[:,None],axis=0)
    score=np.maximum(mean_abs,sup_abs*(b-a))
    sample=np.sum(weights[None,:,None]*abs(J)/scales[None,:,None],axis=1)
    grad_mean=sample.mean(axis=0)
    cv2=np.divide(sample.var(axis=0),grad_mean**2,out=np.zeros_like(grad_mean),where=grad_mean>1e-14)
    nu=float(np.sum(score*cv2)/max(score.sum(),1e-30))
    exponent=2/3 if nu>cfg['nonlinearity_threshold'] else .5
    normalized=local/scales[:,None]
    M=normalized.T@(weights[:,None]*normalized)
    increments=[];event_weights=[]
    for rec,z in zip(records,arrays):
        ds=np.array([z[e['turn']-1]-z[e['turn']-2] for e in rec['events']])
        increments.extend(ds);event_weights.extend([1/(len(records)*len(ds))]*len(ds))
    increments=np.array(increments);event_weights=np.array(event_weights)
    drift_mean=np.sum(event_weights[:,None]*increments,axis=0)
    second=(increments*event_weights[:,None]).T@increments
    cov=second-np.outer(drift_mean,drift_mean);cov=(cov+cov.T)/2
    eig,V=np.linalg.eigh(cov);cov=(V*np.maximum(eig,0))@V.T
    mean_abs_d=np.sum(event_weights[:,None]*abs(increments),axis=0)
    direction=np.divide(abs(drift_mean),mean_abs_d,out=np.zeros_like(drift_mean),where=mean_abs_d>1e-14)
    groups,chi=connected_groups(M,cov,cfg['coupling_threshold'])
    root_std=np.sqrt(np.mean([np.mean((z-mu)**2,axis=0) for z in arrays],axis=0))
    return dict(template=w.template,roots=w.roots,nodes=w.nodes,mu=mu.tolist(),work_lo=a.tolist(),work_hi=b.tolist(),
                query_weights=weights.tolist(),query_counts=counts.tolist(),node_scales=scales.tolist(),
                allocation_weights=np.maximum(score,cfg['allocation_weight_floor']).tolist(),unfloored_weights=score.tolist(),
                exponent=exponent,nonlinearity=nu,M=M.tolist(),covariance=cov.tolist(),second_moment=second.tolist(),
                directional_ratio=direction.tolist(),groups=groups,coupling=chi.tolist(),
                root_std=root_std.tolist(),max_train_jump=np.max(abs(increments),axis=0).tolist(),
                train_subjects=sorted({str(r['subject_id']) for r in records}),
                train_trajectories=[r['trajectory_id'] for r in records],final_only=final_only)


def allocate(total,mask,public):
    mask=np.asarray(mask,bool); v=np.asarray(public['allocation_weights'])**public['exponent']
    a=mask*v;denom=a.sum(axis=-1,keepdims=True)
    return np.asarray(total)[...,None]*a/np.where(denom>0,denom,1)


def drift_covariance(ages,public,cfg):
    """Per-public-batch ages; min(age_i,age_j) counts shared RW increments.

    Persistent coordinates use diagonal g^2 E[delta^2], retaining mean drift;
    mixed model is PSD and reduces to g*Sigma for common random-walk ages.
    """
    ages=np.asarray(ages,float);cov=np.asarray(public['covariance'])
    persistent=np.asarray(public['directional_ratio'])>cfg['directional_threshold']
    rw=~persistent
    out=np.minimum(ages[..., :,None],ages[...,None,:])*cov*rw[:,None]*rw[None,:]
    ii=np.arange(len(cov))
    out[...,ii,ii]+=ages**2*np.diag(public['second_moment'])*persistent
    return out
