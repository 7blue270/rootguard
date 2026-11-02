"""Capacity for asynchronous latest-observed roots; no claim of contemporaneous clinical panels."""
import csv,gzip,json,importlib.util,math
from collections import defaultdict
from pathlib import Path
B=Path('/home/rootguard-research/data_mine');S=Path('/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2')
spec=importlib.util.spec_from_file_location('cap',B/'feasibility/audit_capacity.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Import re-runs the reproducible complete-panel audit, leaving review samples intact.
MAP={'51222':'hb','51221':'hct','51279':'rbc','50861':'alt','50878':'ast','51265':'plt',**m.MAP}
ROOTS={'ANEMIA':['hb','hct','rbc'],'FIB4':['age','ast','alt','plt'],'AIP':['tc','hdl','tg'],'TYG':['tg','glu'],'HOMA':['glu','ins'],'NLR':['neu','lym']}
UNITS={'hb':'g/dL','hct':'%','rbc':'m/uL','ast':'IU/L','alt':'IU/L','plt':'K/uL','tc':'mg/dL','hdl':'mg/dL','tg':'mg/dL','glu':'mg/dL','ins':'uIU/mL','neu':'K/uL','lym':'K/uL'}
g=defaultdict(lambda:defaultdict(list))
with gzip.open(S/'labevents.csv.gz','rt') as f:
 for r in csv.DictReader(f):
  if r['itemid'] not in MAP or not r['hadm_id'] or not r['valuenum']:continue
  n=MAP[r['itemid']]
  try:v=float(r['valuenum'])
  except:continue
  if math.isfinite(v) and v>0 and r['valueuom']==UNITS[n]:g[(r['subject_id'],r['hadm_id'],r['charttime'])][n].append(r)
by=defaultdict(list)
for (sid,hadm,time),attrs in g.items():
 clean={n:rs for n,rs in attrs.items() if len({float(r['valuenum']) for r in rs})==1}
 rows={n:max(rs,key=lambda r:(r['storetime'] or r['charttime'],int(r['labevent_id']))) for n,rs in clean.items()}
 if rows:by[(sid,hadm)].append((time,rows))
out={}
for t,roots in ROOTS.items():
 caps=defaultdict(int);events={n:set() for n in [16,20,24]};both={n:set() for n in events}
 for (sid,hadm),panels in m.pools[t].items():
  if not panels:continue
  initial=min(panels,key=lambda p:(p['available_time'],p['measurement_time']))
  x=dict(initial['values']);root_time={n:initial['measurement_time'] for n in roots};updates=0;has_event=False;has_small=False
  batches=[]
  for time,rows in by[(sid,hadm)]:
   rs={n:r for n,r in rows.items() if n in roots}
   if rs:batches.append((max(r['storetime'] or time for r in rs.values()),time,rs))
  for available,time,rs in sorted(batches,key=lambda p:(p[0],p[1])):
   if available<initial['available_time']:continue
   prev=dict(x);changed=[]
   for n,r in rs.items():
    if time<=root_time[n]:continue
    v=float(r['valuenum']);root_time[n]=time
    if x[n]!=v:changed.append(n);x[n]=v
   if not changed:continue
   updates+=1
   a=m.target(t,prev);b=m.target(t,x);ca=m.cls(t,a);cb=m.cls(t,b)
   has_event|=ca!=cb
   has_small|=ca==cb and 0<abs(b-a)/max(abs(a),1e-12)<=.2
  caps[sid]=max(caps[sid],updates)
  for n in events:
   if updates>=n and has_event:events[n].add(sid)
   if updates>=n and has_event and has_small:both[n].add(sid)
 out[t]={'patients_by_min_true_update_batches':{str(n):sum(v>=n for v in caps.values()) for n in events},
 'patients_also_with_latest_state_category_event':{str(n):len(events[n]) for n in events},
 'patients_also_with_event_and_small_class_preserving_change':{str(n):len(both[n]) for n in events},
 'max_update_batches_in_one_admission':max(caps.values(),default=0)}
result={'source':'MIMIC-IV Demo 2.2','definition':'Initialize from earliest complete same-time panel; thereafter merge actual available partial-root measurements, grouped by charttime, respecting per-root increasing measurement times and result availability; retain older unmeasured roots. Counts describe latest-observed state, not a contemporaneous clinical score. No interpolation, simulated values, splitting one charttime batch, or new independent patients. Event existence checked over whole available admission, not yet over selected 40-turn windows.', 'templates':out}
(B/'feasibility/partial_update_capacity.json').write_text(json.dumps(result,indent=2)+'\n')
print('PARTIAL_ROOT_AUDIT',json.dumps(result,indent=2))
