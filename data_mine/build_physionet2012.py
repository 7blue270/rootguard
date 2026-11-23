"""Build 40-round episodic workloads from actual, minute-stamped ICU labs."""
import csv
import hashlib
import json
import re
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.stats import spearmanr
from data_mine.finite_trajectories import SCHEMA, ROOTS, UNITS, NODES, sha
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload

HOME=Path(__file__).resolve().parent
DEST=HOME/'finite_cache_physionet2012'
COL={'Age':'age','AST':'ast','ALT':'alt','Platelets':'plt','BUN':'bun','Creatinine':'cr'}
TASKS=('BUN_CR','FIB4')

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda v:v.item()))
def quant(x):return np.quantile(np.asarray(x,float),[0,.1,.5,.9,1]).tolist() if len(x) else []
def main():
    archive=HOME/'challenge-2012-1.0.0.zip'
    DEST.mkdir(exist_ok=True)
    counts=Counter();stats=defaultdict(list);pairs=defaultdict(list);rows=defaultdict(list);seen=set();split={}
    with ZipFile(archive) as z:
        for member in sorted(z.namelist()):
            m=re.fullmatch(r'[^/]+/(set-[ab])/(\d+)\.txt',member)
            if not m:continue
            group,sid=m.groups()
            assert sid not in seen
            seen.add(sid)
            name='pn12-'+sid
            raw=z.read(member);path=HOME/'sources'/'challenge2012'/group/(sid+'.txt')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            batches=defaultdict(dict)
            for line,r in enumerate(csv.DictReader(raw.decode().splitlines()),2):
                col=r['Parameter']
                if col=='RecordID':assert r['Value']==sid
                if col not in COL:continue
                val=float(r['Value'])
                if not np.isfinite(val) or val<=0:continue
                h,minute=map(int,r['Time'].split(':'));tm=h*60+minute
                key=COL[col]
                if key in batches[tm]:counts['duplicate_same_time']+=1
                batches[tm][key]=(val,line)
            counts[group]+=1
            def source(k,tm,v):return dict(file=member,minute=tm,row_1based=v[1],valuenum=v[0],valueuom=(['years'] if k=='age' else UNITS[k])[0])
            for t in TASKS:
                keys=ROOTS[t];dynamic=set(keys)-{'age'}
                seq=sorted((tm,{k:v for k,v in batch.items() if k in keys}) for tm,batch in batches.items() if set(batch)&set(keys))
                last={};following=[];initial=None
                for tm,obs in seq:
                    if initial is not None:following.append((tm,obs))
                    last.update({k:(tm,v) for k,v in obs.items()})
                    if initial is None and set(last)==set(keys):start=tm;initial=last.copy()
                for k in keys:
                    ns=sum(k in obs for _,obs in seq)
                    counts[f'{t}.present.{k}']+=ns>0
                    counts[f'{t}.repeated.{k}']+=ns>=2
                    if ns:stats[f'{t}.measurements.{k}'].append(ns)
                if initial is None:continue
                counts[t+'.complete']+=1
                for m0 in (1,2,4,8):counts[f'{t}.later{m0}']+=len(following)>=m0
                follow=following[:24]
                if len(follow)<2 or not dynamic<=set().union(*(set(v) for _,v in follow)):continue
                counts[t+'.eligible.'+group]+=1
                split[name]='test' if group=='set-b' else ('validation' if int(hashlib.sha256(name.encode()).hexdigest(),16)%5==0 else 'train')
                clock=lambda tm:f'ICU-minute-{tm:04d}'
                events=[]
                for j,(tm,obs) in enumerate(follow):
                    ideal=1+int(np.ceil(39*(tm-start)/max(1,2880-start)))
                    turn=min(40-(len(follow)-1-j),max(events[-1]['turn']+1 if events else 2,ideal))
                    events.append(dict(turn=turn,batch_id=f'{sid}-{tm}',available_time=clock(tm),measurement_time=clock(tm),values={k:v[0] for k,v in obs.items()},measured_roots=sorted(obs),source_rows={k:source(k,tm,v) for k,v in obs.items()}))
                turns=[i for i in range(2,41) if i not in {e['turn'] for e in events}]
                rec=dict(schema_version=SCHEMA,template=t,T=40,trajectory_id=t+':'+name,subject_id=name,admission_id=name,source_group=group,initial_state={k:v[0] for k,(tm,v) in initial.items()},initial_source_rows={k:source(k,tm,v) for k,(tm,v) in initial.items()},initial_available_time=clock(start),initial_measurement_time=clock(start),events=events,queries=[dict(turn=i,node=NODES[t][j%len(NODES[t])]) for j,i in enumerate(turns)],units={k:(['years'] if k=='age' else UNITS[k]) for k in keys})
                data.validate(rec);rows[t].append(rec)
                x=data.replay(rec);w=Workload(t);y=w.evaluate(x)[w.target]
                changes=np.ptp(x,axis=0)
                for k,v in zip(keys,changes):stats[t+'.range.'+k].append(v)
                for k,v in dict(hours=(follow[-1][0]-start)/60,anchor_gap=(start-min(tm for tm,_ in initial.values()))/60,batches=len(follow),task_span=np.ptp(y),relative_span=np.ptp(y)/y[0],changed_roots=sum(changes>0),outside=np.any((x<w.lo)|(x>w.hi))).items():stats[t+'.'+k].append(v)
                for a,b in combinations(sorted(dynamic),2):
                    joint=[(obs[a][0],obs[b][0]) for _,obs in seq if a in obs and b in obs]
                    if joint:pairs[f'{t}.{a}/{b}.level'].extend(joint)
                    if len(joint)>1:pairs[f'{t}.{a}/{b}.change'].extend(np.diff(joint,axis=0))
    for t,recs in rows.items():
        (DEST/(t+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in recs))
        assert len(data.load(DEST,[t]))==len(recs)
    dump(DEST/'split.json',split)
    dump(DEST/'audit.json',dict(source_sha256=sha(archive),counts=counts,quantiles={k:quant(v) for k,v in stats.items()},splits={t:Counter(split[r['subject_id']] for r in recs) for t,recs in rows.items()},multi_change={t:sum(v>=2 for v in stats[t+'.changed_roots']) for t in TASKS},dense={t:sum(v>=4 for v in stats[t+'.batches']) for t in TASKS},outside={t:int(sum(stats[t+'.outside'])) for t in TASKS},spearman={k:dict(n=len(v),rho=float(spearmanr(*np.array(v).T).statistic)) for k,v in pairs.items()}))
if __name__=='__main__':main()
