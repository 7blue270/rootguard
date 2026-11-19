"""Frozen protocol constants. No values in this file are estimated from test."""
from pathlib import Path
import numpy as np
from experiments.workflow_extensions.finite_cache.workload import Workload
from experiments.workflow_extensions.m4_carry.core import key_seed

ROOT=Path(__file__).resolve().parents[3]
TEMPLATES=['MELD_NA','ANEMIA','FIB4','BUN_CR']
SELECTORS=['S','D','Default','TrainLookup','B2-val','L0','L1','L2',
           'Local-L0','Local-L1','Local-L2','NoAST-L0','NoBudget-L0','Nonlinearity']
# Factual reference intervals, NOT fitted population distributions. Pooled-sex
# unions are explicit modeling assumptions. Age uses the existing public domain.
REFERENCE={
 'bil':[.1,1.2],'inr':[.8,1.1],'cr':[.6,1.3],'na':[135.,145.],
 'hb':[12.,18.],'hct':[36.,55.],'rbc':[4.2,6.2],
 'ast':[8.,33.],'alt':[4.,36.],'plt':[140.,450.],'age':[18.,110.]}
REFERENCE_SOURCES={
 'chemistry':dict(url='https://medlineplus.gov/ency/article/003468.htm',roots=['bil','cr','na','ast','alt','bun']),
 'cbc':dict(url='https://medlineplus.gov/ency/article/003642.htm',roots=['hb','hct','rbc']),
 'platelets':dict(url='https://www.nhlbi.nih.gov/health/blood-tests',roots=['plt'],conversion='140000-450000 cells/microliter -> 140-450 K/microliter'),
 'coagulation':dict(url='https://medlineplus.gov/ency/article/003652.htm',roots=['inr']),
 'age':dict(source='existing benchmark public domain; not an external normal-age interval'),
 'access_date':'2026-11-18',
 'interpretation':'independent marginal uniform priors on public adult reference intervals; not a fitted ICU population; input domains unchanged'}
REFERENCE['bun']=[6.,20.]


def config(precheck=False):
    return dict(templates=TEMPLATES,budgets=[80.,160.,320.,480.,640.,960.],T=40,
       refresh_turns=[14,27],refresh_rule='ceil(j*T/3), j=1,2; public release clock, distinct rounds; latest arrived values only',
       schedule_design='equal mixture of frozen public measurement/query calendars per template; same L0/L1 prediction for every patient in a template-budget cell',
       prediction_scope='template x B x frozen public calendar design, not a value-dependent patient classifier',
       query_rule='existing queries to final numeric task node only, equal weight within session; sessions within patient, then patients equally',
       repeats=2 if precheck else 20,train_repeats=2 if precheck else 20,
       n_sim=400 if precheck else 12000,bootstrap=100 if precheck else 1000,
       public_seed=20261118,noise_seed=72861118,bootstrap_seed=20261118,validation_seed=20261118,
       delta=.010,component=0.,deltas=[.002,.005,.010,.020,.040],components=[0.,.5,.9],
       models=['linear','redraw'],primary_model='linear',levels=['L0','L1','L2'],
       periods=[2,3,4,5,6,8,10,12],period_unit='actual public measurement batch; anchor B/2, remaining B/2 equal across realized refreshes',
       allocation='uniform across all roots',kernel='continuous Laplace(scale=1/epsilon_root) + full public domain clipping',
       metric='normalized whole-trajectory max norm; conditional on public calendar and unprotected training statistics',
       default_rule='outer-training patient mean MAE, equal across the six budgets within each template; ties S',
       lookup_rule='outer-training mean S/D loss per template and B; ties S',
       inner_validation_fraction=.25,nonlinearity_threshold=3.,
       uncertainty='paired patient bootstrap and patient-plus-mechanism bootstrap; conditional on frozen outer-fold rules, no retraining inside bootstrap',
       eta=None,eta_reason='user requested continuous effects and intervals; no externally justified practical threshold',
       fixed_no_budget=320.,selectors=SELECTORS,precheck=precheck,
       limitations='previously explored cohort; patient LOPO is exploratory rechecking, not independent confirmation or unseen-formula transfer')


def calendars(records,template):
    w=Workload(template);out=[]
    for rec in sorted([r for r in records if r['template']==template],key=lambda r:r['trajectory_id']):
        out.append(dict(T=rec['T'],events=[dict(turn=e['turn'],roots=[w.roots.index(s) for s in e['measured_roots']]) for e in rec['events']],
                        query_turns=[q['turn'] for q in rec['queries'] if q['node']==w.target]))
    return out


def fold_validation(subjects,heldout,cfg):
    train=[s for s in subjects if s!=heldout]
    ranked=sorted(train,key=lambda s:key_seed(cfg['validation_seed'],heldout,s,'inner'))
    n=max(1,int(len(ranked)*cfg['inner_validation_fraction']))
    return sorted(train),ranked[:n]


def prior_bounds(w,level):
    if level=='L0':return np.zeros(len(w.roots)),np.ones(len(w.roots))
    lo=np.maximum([REFERENCE[r][0] for r in w.roots],w.lo)
    hi=np.minimum([REFERENCE[r][1] for r in w.roots],w.hi)
    assert np.all(hi>=lo)
    return (lo-w.lo)/w.width,(hi-w.lo)/w.width
