"""M5 executor: public policy / private release / offline evaluation are separate."""
from itertools import combinations
import numpy as np
from experiments.workflow_extensions.m4_carry.core import PrivateSource, paired_noise, realize, Policy as OldPolicy
from .public import allocate,drift_covariance


class Policy:
    def __init__(self,w,public,cfg,groups=True):
        self.w,self.public,self.cfg=w,public,cfg
        atomic=public['groups'] if groups else [[i] for i in range(len(w.roots))]
        self.packages=[]
        for n in range(1,len(atomic)+1):
            for group_ids in combinations(range(len(atomic)),n):
                mask=np.zeros(len(w.roots),bool)
                for j in group_ids:mask[atomic[j]]=True
                self.packages.append(mask)

    def decide(self,cache,eps_last,ages,candidate_mask,candidate_epsilon):
        R,k=cache.shape;best=np.zeros(R);selected=np.zeros((R,k),bool);eps=np.zeros((R,k))
        M=np.asarray(self.public['M']); diag=np.diag(M); dc=drift_covariance(ages,self.public,self.cfg)
        for mask in self.packages:
            legal=np.all(~mask|candidate_mask,axis=1)
            new=allocate(candidate_epsilon,np.broadcast_to(mask,(R,k)),self.public)
            old_noise=np.sum(np.where(mask,2*diag/eps_last**2,0),axis=1)
            new_noise=np.sum(np.divide(2*diag,new**2,out=np.zeros_like(new),where=mask[None,:]),axis=1)
            stale=np.einsum('ij,rij->r',M*mask[:,None]*mask[None,:],dc)
            gain=stale+old_noise-(1+self.cfg['c_op'])*new_noise
            win=legal&(gain>best+self.cfg['gain_tolerance'])
            selected[win]=mask;eps[win]=new[win];best[win]=gain[win]
        # eps is exactly the epsilon vector used in the winning comparison.
        return selected,eps


class LocalVeto:
    """Former coordinatewise L1 gate, extended to the same public query objective."""
    def __init__(self,w,public,old_stats,cfg):
        self.public=public
        stats={**old_stats,'h':(np.asarray(public['allocation_weights'])**(2*public['exponent'])).tolist()}
        self.policy=OldPolicy(w,stats,cfg)

    def decide(self,cache,eps_last,ages,candidate_mask,candidate_epsilon):
        cand=allocate(candidate_epsilon,candidate_mask,self.public)
        mask=np.zeros_like(candidate_mask,bool);w=self.policy.w
        base=w.from_cache(cache)
        def loss(z,active):
            vals=w.from_cache(z);value=np.zeros(len(z))
            for j,node in enumerate(w.nodes):
                value+=self.public['query_weights'][j]*np.mean(abs(vals[node]-base[node][active,None]),axis=1)/self.public['node_scales'][j]
            return value
        for i in range(cache.shape[1]):
            active=candidate_mask[:,i]
            if not active.any():continue
            c=cache[active];z=np.broadcast_to(c[:,None,:],(len(c),self.policy.cfg['n_mc'],cache.shape[1])).copy()
            z[...,i]+=self.policy.eta/eps_last[active,i,None]
            old=loss(z,active)
            z[...,i]=c[:,i,None]+np.sign(self.policy.drift)*np.minimum(abs(self.policy.drift)*.5,2)*ages[active,i,None]*self.policy.cbar[i]*.5
            old+=loss(z,active)
            z[...,i]=c[:,i,None]+self.policy.eta/cand[active,i,None]
            new=loss(z,active)
            mask[active,i]=old-(1+self.policy.cfg['c_op'])*new>0
        return mask,allocate(candidate_epsilon,mask,self.public)


def simulate(rec,B,plan,public,cfg,phase,R,gate='atomic',carry=True,old_stats=None,detailed=False):
    from experiments.workflow_extensions.finite_cache.workload import Workload
    w=Workload(rec['template']);k=len(w.roots);T=rec['T']
    policy=LocalVeto(w,public,old_stats,cfg) if gate=='local' else Policy(w,public,cfg,groups=gate=='atomic')
    public_events={e['turn']:tuple(w.roots.index(r) for r in e['measured_roots']) for e in rec['events']}
    schedule=realize(plan,sorted(public_events),T);K=len(schedule)
    assert all(a[0]<b[0] for a,b in zip(schedule,schedule[1:]))
    assert K or plan.rho==1
    noise=paired_noise(cfg['noise_seed'],phase,str(rec['subject_id']),str(rec['trajectory_id']),float(B),R,T,tuple(w.roots))
    owner=PrivateSource(w,rec);mask=np.ones((R,k),bool)
    eps_last=allocate(np.full(R,plan.rho*B),mask,public)
    cache,proj=owner.release(1,mask,eps_last,noise[1],np.full((R,k),.5));anchor=cache.copy();anchor_eps=eps_last.copy()
    spent=eps_last.sum(axis=1);left=B-spent;ages=np.zeros((R,k));n_left=K
    n_pub=np.full(R,k);n_proj=proj.sum(axis=1);m=np.zeros(R);empty=np.zeros(R);veto=np.zeros(R);count=np.zeros(R)
    cache_stream=np.empty((R,T,k));hist=[];mapping={t:(j,roots) for j,(t,roots) in enumerate(schedule)}
    for t in range(1,T+1):
        # Covariance is fitted per global public batch, including zero changes.
        if t in public_events:ages+=1
        if t in mapping:
            j,roots=mapping[t];n_before=n_left
            total=left/n_left if carry else np.full(R,(1-plan.rho)*B/K)
            n_left-=1;cm=np.zeros((R,k),bool);cm[:,list(roots)]=True
            assert np.all(total>0)
            selected,actual=policy.decide(cache,eps_last,ages,cm,total)
            publishing=selected.any(axis=1)
            assert np.all(actual[selected]>0) and np.all(actual[~selected]==0)
            charge=actual.sum(axis=1)
            assert np.allclose(charge[publishing],total[publishing],atol=1e-10,rtol=0)
            before=cache.copy();left_before=left.copy()
            if publishing.any():
                cache,pr=owner.release(t,selected,actual,noise[t],cache)
                eps_last[selected]=actual[selected];ages[selected]=0
                spent+=charge;left-=charge;n_proj+=pr.sum(axis=1);n_pub+=selected.sum(axis=1)
            m+=publishing;empty+=~publishing;veto+=(cm&~selected).sum(axis=1);count+=cm.sum(axis=1)
            assert np.all(spent<=B+1e-10) and np.all(left>=-1e-10)
            if detailed:hist.append(dict(turn=t,action_index=j+1,planned_roots=roots,selected=selected,actual_epsilon=actual,
                                        candidate_epsilon=actual.copy(),eps_can=total,left_before=left_before,left_after=left.copy(),
                                        n_before=n_before,n_after=n_left,cache_before=before,cache_after=cache.copy()))
        cache_stream[:,t-1]=cache
    assert n_left==0 and np.allclose(spent+left,B,atol=1e-10,rtol=0)
    return dict(caches=cache_stream,spent=spent,left=left,m=m,n_pub=n_pub,n_proj=n_proj,empty=empty,
                root_veto=veto,root_count=count,K=K,stop=schedule[-1][0] if K else 1,histories=hist,
                anchor_epsilon=anchor_eps,anchor_cache=anchor,release_reads=owner.release_reads,
                stop_reason='static' if not K else 'opportunities_exhausted')
