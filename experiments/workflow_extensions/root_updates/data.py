"""Public benchmark extraction and controlled versus observed trajectories."""
import csv
import gzip
import json
import hashlib
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from mechanisms import original,bounds

MIMIC_DIR=Path('/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2')

def nhanes(name,n):
    records=json.loads((original.REPO/'data/nhanes_benchmark_200.json').read_text())[name][:n]
    rows=[]
    for rec in records:
        v={}
        for turn in rec['turns']:v.update(turn.get('private_value',{}))
        rows.append([float(v[r]) for r in original.ROOTS[name]])
    means=json.loads((original.REPO/'data/holdout_population_means.json').read_text())['per_template_means'][name]
    return np.array(rows),np.array([means[r] for r in original.ROOTS[name]]),[r['id'] for r in records]

def synthetic(name,n,H,scenario,amplitude,seed):
    base,mu,ids=nhanes(name,n)
    lo,hi=bounds(name)
    k=len(lo); x=np.repeat(base[:,None,:],H,axis=1)
    plan=np.zeros((H,k),bool);plan[0]=True
    mutable=[i for i,r in enumerate(original.ROOTS[name]) if r!='age']
    rng=np.random.default_rng(seed)
    directions=rng.choice([-1.,1.],size=(n,k))
    if scenario=='no_updates':pass
    elif scenario=='irrelevant':
        j=original.ROOTS[name].index('rbc')
        plan[1:,j]=True
        x[:,:,j]+=directions[:,j,None]*amplitude*(hi[j]-lo[j])*np.linspace(0,1,H)
    elif scenario=='jump':
        plan[H//2,mutable]=True
        x[:,H//2:,mutable]+=directions[:,None,mutable]*amplitude*(hi-lo)[mutable]
    elif scenario=='sparse':
        for t in range(1,H):
            x[:,t]=x[:,t-1]
            j=mutable[(t-1)%len(mutable)]
            plan[t,j]=True
            x[:,t,j]+=directions[:,j]*amplitude*(hi[j]-lo[j])/max(1,(H-1)/len(mutable))
    elif scenario=='correlated':
        if name!='ANEMIA':raise ValueError('correlated case is ANEMIA only')
        factor=1+amplitude*np.linspace(0,1,H)
        x[:,:,0:2]=base[:,None,0:2]*factor[None,:,None]
        plan[1:,0:2]=True
    else:
        plan[1:,mutable]=True
        wave=np.linspace(0,1,H) if scenario=='drift' else np.sin(np.linspace(0,3*np.pi,H)) if scenario=='oscillate' else np.zeros(H)
        x[:,:,mutable]+=directions[:,None,mutable]*amplitude*(hi-lo)[mutable]*wave[None,:,None]
    clipped=(x<lo)|(x>hi)
    x=np.clip(x,lo,hi)
    return dict(name=name,x=x,mu=mu,ids=ids,plan=plan,
        audit={'source':'synthetic_from_NHANES_not_observed_longitudinal','clipped_root_values':int(clipped.sum()),
               'scenario':scenario,'amplitude':amplitude})

def real(H,n,data_dir=MIMIC_DIR):
    p=data_dir/'labevents.csv.gz'
    groups=defaultdict(lambda:defaultdict(list))
    audit=Counter()
    unit={'51222':'g/dL','51221':'%','51279':'m/uL'}
    with gzip.open(p,'rt') as f:
        for row in csv.DictReader(f):
            if row['itemid'] not in unit or not row['valuenum']:continue
            audit['numeric_root_rows']+=1
            if row['valueuom']!=unit[row['itemid']]:audit['wrong_unit_rows']+=1;continue
            key=(row['subject_id'],row['hadm_id'],row['specimen_id'],row['charttime'])
            groups[key][row['itemid']].append(float(row['valuenum']))
    stays=defaultdict(list)
    lo,hi=bounds('ANEMIA')
    for key,vals in groups.items():
        if len(vals)!=3:continue
        if any(len(v)!=1 for v in vals.values()):audit['duplicate_panels']+=1;continue
        audit['complete_unique_panels']+=1
        v=np.array([vals[j][0] for j in ['51222','51221','51279']])
        if not key[1]:audit['no_admission_panels']+=1;continue
        if np.any(v<lo)|np.any(v>hi):audit['out_of_domain_panels']+=1;continue
        stays[key[:2]].append((key[3],key[2],v))
    by_subject=defaultdict(list)
    for key,rows in stays.items():
        # One panel per timestamp; tie-break by public specimen ID, not value.
        rows.sort(key=lambda r:(r[0],r[1]))
        unique=[];seen=set()
        for row in rows:
            if row[0] in seen:continue
            unique.append(row);seen.add(row[0])
        if len(unique)>=H:by_subject[key[0]].append((key[1],unique))
    ids=sorted(by_subject,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:n]
    sequences=[];times=[]
    for pid in ids:
        stay,rows=sorted(by_subject[pid],key=lambda pair:pair[0])[0]
        sequences.append([r[2] for r in rows[:H]])
        times.append([r[0] for r in rows[:H]])
    if not sequences:raise ValueError('No eligible real trajectories')
    x=np.array(sequences)
    _,mu,_=nhanes('ANEMIA',1)
    audit['eligible_subjects']=len(by_subject)
    audit['selected_subjects']=len(ids)
    audit['source']='MIMIC-IV Demo 2.2 observed panels, charttime ordered'
    audit['timestamps']=times
    return dict(name='ANEMIA',x=x,mu=mu,ids=['MIMIC_'+s for s in ids],plan=np.ones((H,3),bool),audit=dict(audit))
