import copy
import inspect
import itertools
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.m4_carry.core import Plan,paired_noise
from experiments.workflow_extensions.m4_carry.run import ROOT
from .run import config,make_plan
from .public import jacobian,fit_public,allocate,connected_groups,drift_covariance
from .core import Policy,LocalVeto,simulate
from .attack import exp_logZ,log_likelihood,reconstruct,prior_scores
from .metrics import evaluate

SOURCE=ROOT/'experiments/workflow_extensions/outputs/m4_carry/real_v1_20261116'
CFG=config(SOURCE,True)


def record():
    return dict(template='ANEMIA',subject_id='unit',trajectory_id='unit:1',T=40,
                initial_state=dict(hb=12.,hct=36.,rbc=4.),queries=[dict(turn=t,node=n) for t in [3,5,8,20] for n in ['mcv','mch','mchc']],
                events=[dict(turn=4,measured_roots=['hb','hct'],values=dict(hb=11.,hct=33.)),
                        dict(turn=8,measured_roots=['hb','hct'],values=dict(hb=10.,hct=30.)),
                        dict(turn=30,measured_roots=['hb','hct'],values=dict(hb=11.,hct=34.))])


def public():return fit_public([record()],Workload('ANEMIA'),CFG)


class Tests(unittest.TestCase):
    def test_analytic_jacobian_all_nodes(self):
        for t in CFG['templates']:
            w=Workload(t);z=np.random.default_rng(2).uniform(.1,.9,(7,len(w.roots)))
            j=jacobian(w,z)
            for i in range(len(w.roots)):
                a=z.copy();b=z.copy();a[:,i]-=1e-7;b[:,i]+=1e-7
                va=w.from_cache(a);vb=w.from_cache(b)
                num=np.stack([(vb[n]-va[n])/2e-7 for n in w.nodes],axis=-1)
                np.testing.assert_allclose(j[:,:,i],num,rtol=1e-5,atol=1e-5)

    def test_fit_weights_and_positive_epsilon(self):
        p=public();self.assertAlmostEqual(sum(p['query_weights']),1)
        self.assertTrue(all(x>0 for x in p['allocation_weights']))
        for total in [1.,80.,960.]:
            e=allocate(total,np.array([True,True,True]),p)
            self.assertTrue(np.all(e>0));self.assertAlmostEqual(e.sum(),total)

    def test_allocation_uses_full_domain_not_work_width_only(self):
        p=fit_public([record()],Workload('ANEMIA'),CFG,True)
        self.assertGreater(p['allocation_weights'][0],0)
        w=Workload('ANEMIA');mu=np.array(p['mu']);local=abs(jacobian(w,mu)[2])/p['node_scales'][2]
        self.assertTrue(np.all(np.array(p['allocation_weights'])>=local-1e-12))

    def test_trace_cancellation_and_group(self):
        w=Workload('ANEMIA');mu=w.normalize([12.,36.,4.]);a=jacobian(w,mu)[2]
        delta=np.array([1/23,3/70,0]);cov=np.outer(delta,delta);M=np.outer(a,a)
        self.assertLess(abs(np.sum(M*cov)),1e-10)
        self.assertGreater(np.sum(np.diag(M)*np.diag(cov)),1)
        groups,chi=connected_groups(M,cov,.5)
        self.assertIn([0,1],groups);self.assertAlmostEqual(chi[0,1],1)

    def test_covariance_age_model_psd(self):
        p=public();ages=np.array([[1,2,3],[9,1,0]])
        for cutoff in [0.,.5,1.]:
            dc=drift_covariance(ages,p,dict(CFG,directional_threshold=cutoff))
            self.assertGreaterEqual(np.linalg.eigvalsh(dc).min(),-1e-12)

    def test_common_age_random_walk_exact(self):
        p=public();dc=drift_covariance(np.array([[3,3,3]]),p,dict(CFG,directional_threshold=1.))
        np.testing.assert_allclose(dc[0],3*np.array(p['covariance']))

    def test_gate_epsilon_matches_winning_candidate(self):
        p=public();policy=Policy(Workload('ANEMIA'),p,CFG);cache=np.full((2,3),.4)
        last=allocate(np.array([40.,40.]),np.ones((2,3),bool),p)
        selected,eps=policy.decide(cache,last,np.full((2,3),30),np.ones((2,3),bool),np.array([200.,200.]))
        np.testing.assert_allclose(eps,allocate(np.array([200.,200.]),selected,p))
        for g in p['groups']:self.assertTrue(np.all(selected[:,g]==selected[:,g[:1]]))

    def test_carry_skip_and_budget(self):
        p=public();observed=[]
        class Script:
            def __init__(self,*args,**kwargs):self.step=0
            def decide(self,cache,eps_last,ages,mask,total):
                observed.append(total.copy());self.step+=1
                selected=mask if self.step==2 else np.zeros_like(mask)
                return selected,allocate(total,selected,p)
        plan=Plan(.5,(.1,.75),((0,1,2),)*2)
        with patch('experiments.workflow_extensions.m5.core.Policy',Script):
            r=simulate(record(),320.,plan,p,CFG,'unit',2,detailed=True)
        np.testing.assert_allclose(observed,[np.full(2,80),np.full(2,160)])
        np.testing.assert_allclose(r['spent'],320);self.assertEqual(r['histories'][0]['n_after'],1)

    def test_fixed_skip_has_no_carry(self):
        p=public();observed=[]
        class Script:
            def __init__(self,*a,**kw):self.step=0
            def decide(self,cache,last,ages,mask,total):
                observed.append(total.copy());self.step+=1
                selected=mask if self.step==2 else np.zeros_like(mask)
                return selected,allocate(total,selected,p)
        with patch('experiments.workflow_extensions.m5.core.Policy',Script):
            r=simulate(record(),320,Plan(.5,(.1,.75),((0,1,2),)*2),p,CFG,'unit',2,carry=False,detailed=True)
        np.testing.assert_allclose(observed,80);np.testing.assert_allclose(r['spent'],240)

    def test_causal_future_mutation(self):
        rec=record();future=copy.deepcopy(rec);future['events'][-1]['values']=dict(hb=22.,hct=70.)
        p=public();plan=Plan(.5,(.1,.75),((0,1,2),)*2)
        a=simulate(rec,960,plan,p,CFG,'unit',2,detailed=True);b=simulate(future,960,plan,p,CFG,'unit',2,detailed=True)
        np.testing.assert_array_equal(a['caches'][:,:29],b['caches'][:,:29])
        self.assertNotIn('truth',inspect.signature(Policy.decide).parameters)

    def test_actual_publication_value_noise_and_ledger(self):
        rec=record();w=Workload('ANEMIA');p=public();r=simulate(rec,960,make_plan(2,.5,3),p,CFG,'unit',2,detailed=True)
        noise=paired_noise(CFG['noise_seed'],'unit','unit','unit:1',960.,2,40,tuple(w.roots));truth=w.normalize(data.replay(rec))
        np.testing.assert_allclose(r['anchor_cache'],np.clip(truth[0]+noise[1]/r['anchor_epsilon'],0,1))
        total=r['anchor_epsilon'].sum(axis=1)
        for h in r['histories']:
            mask=h['selected'];eps=h['actual_epsilon'];raw=truth[h['turn']-1]+np.divide(noise[h['turn']],eps,out=np.zeros_like(eps),where=mask)
            np.testing.assert_allclose(h['cache_after'][mask],np.clip(raw,0,1)[mask]);total+=eps.sum(axis=1)
        np.testing.assert_allclose(total,r['spent']);self.assertTrue(np.all(total<=960+1e-10))

    def test_initial_and_static_cache(self):
        p=public();r=simulate(record(),960,Plan(1.),p,CFG,'unit',2,detailed=True)
        self.assertEqual(r['release_reads'],[1]);self.assertEqual(r['K'],0)
        np.testing.assert_array_equal(r['caches'],np.repeat(r['anchor_cache'][:,None,:],40,axis=1))

    def test_rce_is_ordinal_not_class_mismatch(self):
        w=Workload('ANEMIA');p=public();rec=record();r=simulate(rec,960,Plan(1.),p,CFG,'unit',2,detailed=True)
        old=json.loads((SOURCE/'public_stats.json').read_text())['ANEMIA'];m=evaluate(rec,r,w,old,dict(CFG,_B=960),p)
        true=w.classes(w.evaluate(data.replay(rec))[w.target]);pred=w.classes(w.from_cache(r['caches'])[w.target])
        np.testing.assert_allclose(m['rce'],100*np.mean(abs(pred-true)/2,axis=1))
        self.assertTrue(np.all(m['rce']<=m['cmr']))

    def test_exp_log_normalizer_and_kernel(self):
        for m in [7,21]:
            grid=np.linspace(0,1,m);eps=9.
            z=np.log(np.sum(np.exp(-eps/2*abs(grid[:,None]-grid[None,:])),axis=1))
            np.testing.assert_allclose(exp_logZ(eps,m),z,atol=1e-14)
            ll=log_likelihood(grid[3],eps,'native_exp',m)
            np.testing.assert_allclose(ll,-eps/2*abs(grid-grid[3])-z)

    def test_kernel_privacy_bound_and_clip_mass(self):
        eps=7.;m=31;grid=np.linspace(0,1,m)
        for kernel in ['native_exp','laplace']:
            for y in [0.,.4,1.]:
                ll=log_likelihood(y,eps,kernel,m)
                self.assertTrue(np.all(abs(np.diff(ll))<=eps/(m-1)+1e-12))
        self.assertAlmostEqual(np.exp(log_likelihood(0.,eps,'laplace',m)[0]),.5)

    def test_history_map_matches_exhaustive_tiny_grid(self):
        m=5;grid=np.linspace(0,1,m);pub=dict(roots=['a'],mu=[.4],root_std=[.2],max_train_jump=[.25])
        view=dict(T=3,events=[dict(turn=2,measured_roots=['a']),dict(turn=3,measured_roots=['a'])])
        hist=[dict(turn=3,selected=np.ones((1,1),bool),cache_after=np.array([[.75]]),actual_epsilon=np.array([[7.]]))]
        got=reconstruct(view,hist,np.array([[.25]]),np.array([[8.]]),pub,'laplace',dict(CFG,attack_grid_points=m),'informed',True)[0,:,0]
        prior=prior_scores(pub,'informed',m)[0];best=None;score=-np.inf
        for states in itertools.product(range(m),repeat=3):
            if any(abs(a-b)>1 for a,b in zip(states,states[1:])):continue
            s=prior[states[0]]+log_likelihood(.25,8,'laplace',m)[states[0]]+log_likelihood(.75,7,'laplace',m)[states[2]]
            for prev in states[:-1]:s-=np.log(min(4,prev+1)-max(0,prev-1)+1)
            if s>score:score=s;best=states
        np.testing.assert_array_equal(got,grid[list(best)])

    def test_cached_answers_do_not_become_new_likelihood(self):
        pub=dict(roots=['a'],mu=[.5],root_std=[.2],max_train_jump=[0.])
        view=dict(T=8,events=[]);initial=np.array([[.7]]);eps=np.array([[10.]])
        a=reconstruct(view,[],initial,eps,pub,'laplace',dict(CFG,attack_grid_points=101),'informed',True)
        b=reconstruct(dict(T=1,events=[]),[],initial,eps,pub,'laplace',dict(CFG,attack_grid_points=101),'informed',True)
        np.testing.assert_array_equal(a,np.repeat(b,8,axis=1))

    def test_intermediate_query_metrics_use_query_turns(self):
        p=public();w=Workload('ANEMIA');rec=record();r=simulate(rec,960,Plan(1.),p,CFG,'unit',2,detailed=True)
        old=json.loads((SOURCE/'public_stats.json').read_text())['ANEMIA'];m=evaluate(rec,r,w,old,dict(CFG,_B=960),p)
        truth=w.evaluate(data.replay(rec));pred=w.from_cache(r['caches'])
        ix=[q['turn']-1 for q in rec['queries'] if q['node']=='mcv']
        np.testing.assert_allclose(m['node_num__mcv'],np.mean(abs(pred['mcv'][:,ix]-truth['mcv'][ix]),axis=1))

    def test_local_gate_ablation_keeps_intermediate_query_objective(self):
        p=public();w=Workload('ANEMIA')
        old=json.loads((SOURCE/'public_stats.json').read_text())['ANEMIA']
        gate=LocalVeto(w,p,old,CFG);cache=w.normalize(np.array([[12.,36.,4.]]))
        # RBC is irrelevant to MCHC but relevant to MCV/MCH. With bad old RBC
        # precision and very accurate candidate, its intermediate utility wins.
        selected,_=gate.decide(cache,np.array([[100.,100.,.01]]),np.zeros((1,3)),np.ones((1,3),bool),np.array([10000.]))
        self.assertTrue(selected[0,2])


if __name__=='__main__':unittest.main()
