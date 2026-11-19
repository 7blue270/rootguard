"""Public policy, trusted mixed-kernel owner, common causal carry executor."""
from itertools import combinations
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.core import paired_noise,realize,key_seed
from .public import allocate
from .kernels import emit,danger


class PrivateSource:
    def __init__(self,w,rec,public):
        self.__states=w.normalize(data.replay(rec));self.public=public;self.release_reads=[]

    def release(self,turn,mask,epsilon,noise,cache,kernels):
        assert np.all(epsilon[mask]>0) and np.isfinite(epsilon).all()
        truth=self.__states[turn-1]
        values,projection=emit(np.broadcast_to(truth,cache.shape),epsilon,noise,kernels,self.public)
        out=cache.copy();out[mask]=values[mask]
        outside=((truth<np.array(self.public['work_lo']))|(truth>np.array(self.public['work_hi'])))&mask&kernels
        self.release_reads.append(turn)
        return out,projection&mask,outside


class Policy:
    """No record, private value, or private evaluation is accepted by decide().

    Risk uses an explicitly approximate particle posterior over TRAIN-derived
    synthetic paths, with variance 2/epsilon_last² + age*Sigma_ii. Rollouts hold
    the candidate cache to T (terminal-hold lookahead, no oracle future control).
    This is a utility model, not a DP theorem or a calibrated clinical posterior.
    """
    def __init__(self,w,public,cfg,groups=True,mixed=True,horizon=True,force=False):
        self.w,self.public,self.cfg=w,public,cfg;self.mixed=mixed;self.horizon=horizon;self.force=force
        atoms=public['groups'] if groups else [[i] for i in range(len(w.roots))]
        self.packages=[]
        for n in range(1,len(atoms)+1):
            for ids in combinations(range(len(atoms)),n):
                mask=np.zeros(len(w.roots),bool)
                for j in ids:mask[atoms[j]]=True
                self.packages.append(mask)
        rng=np.random.default_rng(key_seed(cfg['public_mc_seed'],w.template,'M6-risk'))
        paths=np.array(public['scenario_paths']);ix=rng.integers(len(paths),size=cfg['n_mc'])
        jitter=rng.normal(size=(cfg['n_mc'],1,len(w.roots)))*np.array(public['root_std'])*cfg['scenario_jitter']
        self.paths=np.clip(paths[ix]+jitter,0,1)
        self.noise=rng.laplace(size=(cfg['n_mc'],len(w.roots)))
        self.values=w.from_cache(self.paths)

    def loss(self,cache,turn,end):
        # cache [R,MC,k] -> loss [R,MC,horizon].
        vals=self.w.from_cache(cache);p=self.public
        return sum(p['query_weights'][j]*abs(vals[node][...,None]-self.values[node][None,:,turn-1:end])/p['node_scales'][j]
                   for j,node in enumerate(self.w.nodes))

    def decide(self,turn,cache,eps_last,ages,candidate_mask,candidate_epsilon):
        R,k=cache.shape;MC=len(self.paths);end=len(self.paths[0]) if self.horizon else turn
        variance=2/eps_last**2+ages*np.diag(self.public['covariance'])+self.cfg['posterior_variance_floor']
        logw=-.5*np.sum((cache[:,None,:]-self.paths[None,:,turn-1,:])**2/variance[:,None,:],axis=-1)
        weights=np.exp(logw-logw.max(axis=1,keepdims=True));weights/=weights.sum(axis=1,keepdims=True)
        base=np.broadcast_to(cache[:,None,:],(R,MC,k));old=self.loss(base,turn,end);oldsum=old.sum(axis=-1)
        skip=np.sum(weights*oldsum,axis=1)+self.cfg['eta_tail']*np.quantile(old.reshape(R,-1),.95,axis=1)
        best=np.full(R,np.inf) if self.force else skip.copy()
        selected=np.zeros((R,k),bool);actual=np.zeros((R,k));winbad=np.zeros(R);wingain=np.zeros(R)
        kernels=danger(cache,self.public,self.mixed)
        for mask in self.packages:
            legal=np.all(~mask|candidate_mask,axis=1)
            eps=allocate(candidate_epsilon,np.broadcast_to(mask,(R,k)),self.public)
            latent=np.broadcast_to(self.paths[None,:,turn-1,:],(R,MC,k))
            values,_=emit(latent,eps[:,None,:],self.noise[None,:,:],kernels[:,None,:],self.public)
            trial=np.where(mask,values,base)
            loss=self.loss(trial,turn,end);sums=loss.sum(axis=-1)
            bad=np.sum(weights*(sums>oldsum+1e-12),axis=1)
            risk=np.sum(weights*sums,axis=1)+self.cfg['lambda_harm']*bad+self.cfg['eta_tail']*np.quantile(loss.reshape(R,-1),.95,axis=1)
            win=legal&(risk<best-self.cfg['gain_tolerance'])
            selected[win]=mask;actual[win]=eps[win];best[win]=risk[win];winbad[win]=bad[win];wingain[win]=skip[win]-risk[win]
        return selected,actual,dict(risk_skip=skip,risk_selected=best,predicted_harm=winbad,predicted_gain=wingain)


def simulate(rec,B,plan,public,cfg,phase,R,groups=True,mixed=True,horizon=True,force=False,carry=True,detailed=False,policy=None):
    w=Workload(rec['template']);k=len(w.roots);T=rec['T']
    if policy is None:policy=Policy(w,public,cfg,groups,mixed,horizon,force)
    # No future private event VALUES cross this boundary.
    events={e['turn']:tuple(w.roots.index(r) for r in e['measured_roots']) for e in rec['events']}
    schedule=realize(plan,sorted(events),T);K=len(schedule);assert K or plan.rho==1
    assert all(a[0]<b[0] for a,b in zip(schedule,schedule[1:])) and len({t for t,_ in schedule})==K
    noise=paired_noise(cfg['noise_seed'],phase,str(rec['subject_id']),str(rec['trajectory_id']),float(B),R,T,tuple(w.roots))
    owner=PrivateSource(w,rec,public);mask=np.ones((R,k),bool)
    eps_last=allocate(np.full(R,plan.rho*B),mask,public)
    ak=danger(np.broadcast_to(public['mu'],(R,k)),public,mixed)
    cache,pr,outside=owner.release(1,mask,eps_last,noise[1],np.full((R,k),.5),ak)
    anchor=cache.copy();anchor_eps=eps_last.copy();spent=eps_last.sum(axis=1);left=B-spent
    ages=np.zeros((R,k));n_left=K;n_pub=np.full(R,k);n_proj=pr.sum(axis=1);n_grid=ak.sum(axis=1);n_out=outside.sum(axis=1)
    m=np.zeros(R);empty=np.zeros(R);veto=np.zeros(R);count=np.zeros(R);hist=[];stream=np.empty((R,T,k))
    mapping={t:(j,roots) for j,(t,roots) in enumerate(schedule)}
    for t in range(1,T+1):
        if t in events:ages+=1
        if t in mapping:
            j,roots=mapping[t];n_before=n_left
            total=left/n_left if carry else np.full(R,(1-plan.rho)*B/K);n_left-=1
            cm=np.zeros((R,k),bool);cm[:,list(roots)]=True
            selected,actual,diagnostic=policy.decide(t,cache,eps_last,ages,cm,total)
            publishing=selected.any(axis=1);charge=actual.sum(axis=1)
            assert np.all(actual[selected]>0) and np.all(actual[~selected]==0)
            np.testing.assert_allclose(charge[publishing],total[publishing],atol=1e-10,rtol=0)
            for group in public['groups'] if groups else [[i] for i in range(k)]:assert np.all(selected[:,group]==selected[:,group[:1]])
            before=cache.copy();left_before=left.copy();kernels=danger(cache,public,mixed)
            cache,pr,outs=owner.release(t,selected,actual,noise[t],cache,kernels)
            eps_last[selected]=actual[selected];ages[selected]=0;spent+=charge;left-=charge
            m+=publishing;empty+=~publishing;veto+=(cm&~selected).sum(axis=1);count+=cm.sum(axis=1)
            n_pub+=selected.sum(axis=1);n_proj+=pr.sum(axis=1);n_grid+=(kernels&selected).sum(axis=1);n_out+=outs.sum(axis=1)
            assert np.all(spent<=B+1e-10) and np.all(left>=-1e-10)
            if detailed:hist.append(dict(turn=t,action_index=j+1,planned_roots=roots,selected=selected,actual_epsilon=actual,
                    candidate_epsilon=actual.copy(),eps_can=total,left_before=left_before,left_after=left.copy(),
                    n_before=n_before,n_after=n_left,cache_before=before,cache_after=cache.copy(),kernels=kernels,**diagnostic))
        stream[:,t-1]=cache
    assert n_left==0
    np.testing.assert_allclose(spent+left,B,atol=1e-10,rtol=0)
    return dict(caches=stream,spent=spent,left=left,m=m,n_pub=n_pub,n_proj=n_proj,n_grid=n_grid,n_outside_grid=n_out,
                empty=empty,root_veto=veto,root_count=count,K=K,stop=schedule[-1][0] if K else 1,histories=hist,
                anchor_epsilon=anchor_eps,anchor_cache=anchor,anchor_kernels=ak,release_reads=owner.release_reads,
                stop_reason='static' if not K else 'opportunities_exhausted')
