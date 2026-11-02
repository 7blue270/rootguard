"""Audit actual attribute movement; no resampling or release run."""
import csv,json
import numpy as np
from data_cases import inventory,split
from common import original,events


def build(out,cfg):
    audit=[];changed_ids={}
    for did,d,source in inventory(cfg):
        if 'only_cases' in cfg and did not in cfg['only_cases']:continue
        v,t=split(d,source)
        for part,pd in [('validation',v),('test',t)]:
            x=pd['x'];movement=np.max(abs(np.diff(x,axis=1)),axis=(1,2))>1e-8;y,c,e=events(pd)
            attr_step=np.max(abs(np.diff(np.log(x),axis=1)),axis=-1)>1e-8
            audit.append(dict(dataset=did,source=source,split=part,patients=len(x),moving_patients=int(movement.sum()),
                class_events=int(e.sum()),event_patients=int((e.sum(axis=1)>0).sum()),
                changed_steps=int(attr_step.sum()),changed_steps_without_class_change=int((attr_step&~e[:,1:]).sum())))
            if part=='test' and 'stress_HOMA_stable' in did:changed_ids[did]={p for i,p in enumerate(pd['ids']) if movement[i]}
    (out/'generation_audit.json').write_text(json.dumps(audit,indent=2))
    with (out/'generation_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    # For the partly boundary-limited stable condition report each subgroup,
    # using existing patient aggregates rather than rerunning any method.
    buckets={}
    cells={r['cell']:r for r in json.loads((out/'cells.json').read_text())}
    with (out/'patients.csv').open() as f:
        for r in csv.DictReader(f):
            c=cells[r['cell']];did=c['dataset']
            if did not in changed_ids:continue
            subgroup='actual_attribute_change' if r['patient'] in changed_ids[did] else 'no_attribute_change'
            buckets.setdefault((r['cell'],subgroup),[]).append(r)
    sub=[]
    for (cell,group),rr in buckets.items():
        c=cells[cell];num=lambda k:sum(float(r[k]) for r in rr)
        sub.append(dict(dataset=c['dataset'],B=c['B'],comparison=c['comparison'],method=c['method'],subgroup=group,n=len(rr),
            rce=100*num('rce')/len(rr),cmr=100*num('cmr')/len(rr),wmape=100*num('error')/num('truth'),
            false_change=None if not num('stable_den') else 100*num('stable_false')/num('stable_den')))
    (out/'HOMA_stable_subgroups.json').write_text(json.dumps(sub,indent=2))
    return audit
