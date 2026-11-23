"""Fit template-level routing on disjoint references and calibration patients."""
from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np

from ..m4_carry.core import key_seed
from .engine import BASE_PERIODIC, Branch, branches, execute
from .metrics import utility

DYNAMIC_PERIODS=(4,8,12)
BASELINE_PERIODS=(2,3,4,5,6,8,10,12)
RULES=('sign','patient_guard','smoothed','reference_pool','median_mix')


@dataclass(frozen=True)
class Route:
    """Only one branch is selected before the target transcript starts."""
    branches: dict[float,tuple[Branch,Branch]]
    choices: dict[float,bool]
    rule: str

    @classmethod
    def from_frozen(cls, config):
        pairs={float(B):tuple(Branch(**branch) for branch in pair)
               for B,pair in config['branches'].items()}
        return cls(pairs,{float(B):bool(x) for B,x in config['choices'].items()},config['rule'])

    def choose(self, budget):
        pair=self.branches[float(budget)]
        return pair[int(self.choices[float(budget)])]


def loss(ctx, records, budget, branch, phase, repeats, objective):
    key='query_objective' if objective=='workload' else 'mae'
    return np.array([utility(ctx,record,execute(record,budget,branch,ctx,phase,repeats))[key].mean()
                     for record in records])


def select_period(ctx, records, budget, branch, periods, phase, repeats, objective):
    candidates=[(period,loss(ctx,records,budget,replace(branch,period=period),phase,repeats,objective))
                for period in periods]
    p,values=min(candidates,key=lambda trial:(trial[1].mean(),trial[0]))
    return replace(branch,period=p),values


def fit(ctx, references, calibration, budgets, profile, objective, repeats, phase):
    if len(calibration)<2:
        raise ValueError('At least two independent calibration patients are required')
    if {str(r['subject_id']) for r in references} & {str(r['subject_id']) for r in calibration}:
        raise ValueError('Reference and calibration patients must be disjoint')
    S,D=branches(profile,ctx.work.template,objective)
    pairs,rs,rd,ps,pd={},[],[],[],[]
    for B in budgets:
        dynamic,values=select_period(ctx,references,B,D,DYNAMIC_PERIODS,phase+'-ref',repeats,objective)
        pairs[B]=(S,dynamic)
        rs.append(loss(ctx,references,B,S,phase+'-ref',repeats,objective))
        rd.append(values)
        ps.append(loss(ctx,calibration,B,S,phase+'-pilot',repeats,objective))
        pd.append(loss(ctx,calibration,B,dynamic,phase+'-pilot',repeats,objective))
    return pairs,*(np.array(a) for a in (rs,rd,ps,pd))


def choices(features):
    _,rs,rd,ps,pd=features
    paired=ps-pd
    denom=np.maximum(ps.mean(axis=1),1e-12)
    gain=paired.mean(axis=1)/denom
    reference=(rs-rd).mean(axis=1)/np.maximum(rs.mean(axis=1),1e-12)
    se=paired.std(axis=1,ddof=1)/np.sqrt(paired.shape[1])/denom
    neighbors=np.r_[gain[1],(gain[:-2]+gain[2:])/2,gain[-2]] if len(gain)>1 else gain
    median=np.median(paired,axis=1)/denom
    return dict(sign=gain>0,patient_guard=gain>.65*se,smoothed=.75*gain+.25*neighbors>0,
                reference_pool=.75*gain+.25*reference>0,median_mix=.65*gain+.35*median>0)


def make_route(features,rule):
    pairs=features[0]
    return Route(pairs,dict(zip(pairs,choices(features)[rule])),rule)


def source_scores(template, train, validation, budgets, profile, objective, repeats, make_context):
    ranked=sorted(train,key=lambda rec:key_seed('m7-source',template,str(rec['subject_id'])))
    pilot,refs=ranked[:3],ranked[3:]
    ctx=make_context(template,refs)
    features=fit(ctx,refs,pilot,budgets,profile,objective,repeats,'m7-source')
    choices_by_rule=choices(features)
    records=[]
    for i,(B,(S,D)) in enumerate(features[0].items()):
        a=loss(ctx,validation,B,S,'m7-source-score',repeats,objective)
        b=loss(ctx,validation,B,D,'m7-source-score',repeats,objective)
        denominator=max((a.mean()+b.mean())/2,1e-12)
        records.append({rule:float(np.mean(np.where(mask[i],b-a,0))/denominator)
                        for rule,mask in choices_by_rule.items()})
    return {rule:float(np.mean([r[rule] for r in records])) for rule in RULES}


def select_rule(target, scores):
    others=[values for template,values in scores.items() if template!=target]
    if not others:
        return 'sign'
    return min(RULES,key=lambda rule:np.mean([s[rule] for s in others]))


def periodic_controls(ctx, history, B, repeats):
    raw,_=select_period(ctx,history,B,BASE_PERIODIC,BASELINE_PERIODS,'m7-controls',repeats,'final')
    adjusted=replace(BASE_PERIODIC,calibration='marginal',shrink=.85)
    cal,_=select_period(ctx,history,B,adjusted,BASELINE_PERIODS,'m7-controls',repeats,'final')
    return raw,cal
