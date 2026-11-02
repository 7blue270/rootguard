"""Read-only source audit; does not regenerate or change review samples."""
import csv,gzip,json,math,importlib.util,hashlib
from collections import defaultdict,Counter
from pathlib import Path
BASE=Path('/home/rootguard-research/data_mine')
SRC=Path('/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2')
spec=importlib.util.spec_from_file_location('samples',BASE/'build_review_samples.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
pools,_=m.load()
MAP={'50907':'tc','50904':'hdl','51000':'tg','50931':'glu','51676':'ins','51133':'lym','52075':'neu'}
g=defaultdict(lambda:defaultdict(list));counts=Counter();subjects=defaultdict(set)
with gzip.open(SRC/'labevents.csv.gz','rt') as f:
 for r in csv.DictReader(f):
  if r['itemid'] not in MAP or not r['valuenum']:continue
  try:v=float(r['valuenum'])
  except ValueError:continue
  if not math.isfinite(v) or v<=0:continue
  n=MAP[r['itemid']];counts[n]+=1;subjects[n].add(r['subject_id'])
  if r['hadm_id']:g[(r['subject_id'],r['hadm_id'],r['charttime'])][n].append(r)
for t,roots in {'AIP':['tc','hdl','tg'],'TYG':['tg','glu'],'HOMA':['glu','ins'],'NLR':['neu','lym']}.items():
 pool=defaultdict(list)
 for (sid,hadm,time),values in g.items():
  if not all(n in values for n in roots):continue
  if any(len({float(r['valuenum']) for r in values[n]})!=1 for n in roots):continue
  rows={n:max(values[n],key=lambda r:(r['storetime'] or r['charttime'],int(r['labevent_id']))) for n in roots}
  unit='K/uL' if t=='NLR' else 'mg/dL'
  if any(rows[n]['valueuom']!=unit for n in roots):continue
  pool[(sid,hadm)].append({'values':{n:float(rows[n]['valuenum']) for n in roots},'measurement_time':time,'available_time':max(r['storetime'] or time for r in rows.values())})
 pools[t]=pool

def target(t,x):
 if t in ['ANEMIA','FIB4']:return m.score(t,x)
 if t=='NLR':return x['neu']/x['lym']
 if t=='AIP':return math.log10(x['tg']/x['hdl'])
 if t=='TYG':return math.log(x['tg']*x['glu']/2)
 return x['glu']*x['ins']/405

def cls(t,v):
 if t in ['ANEMIA','FIB4']:return m.risk(t,v)
 if t=='NLR':return int(v>=3)
 if t=='AIP':return 0 if v<.11 else 1 if v<=.21 else 2
 if t=='TYG':return int(v>8.5)
 return 0 if v<1 else 1 if v<2.5 else 2

result={}
for t,pool in pools.items():
 by_patient={};with_event=defaultdict(set);with_small=defaultdict(set)
 total=0
 for (sid,hadm),panels in pool.items():
  chosen=[]
  for p in sorted(panels,key=lambda p:(p['available_time'],p['measurement_time'])):
   if chosen and p['measurement_time']<=chosen[-1]['measurement_time']:continue
   if chosen and p['values']==chosen[-1]['values']:continue
   chosen.append(p)
  n=len(chosen);total+=n
  by_patient[sid]=max(by_patient.get(sid,0),n)
  cs=[cls(t,target(t,p['values'])) for p in chosen]
  has_event=any(a!=b for a,b in zip(cs,cs[1:]))
  small=any(a==b and 0<abs(target(t,q['values'])-target(t,p['values']))/max(abs(target(t,p['values'])),1e-12)<=.2 for p,q,a,b in zip(chosen,chosen[1:],cs,cs[1:]))
  for threshold in [17,21,25]:
   if n>=threshold and has_event:with_event[threshold].add(sid)
   if n>=threshold and has_event and small:with_small[threshold].add(sid)
 result[t]={'patients_with_complete_panels':len(by_patient),'changed_snapshots_same_admission_total':total,
   'patients_by_min_changed_snapshots':{str(n):sum(c>=n for c in by_patient.values()) for n in [17,21,25]},
   'patients_also_with_category_event':{str(n):len(with_event[n]) for n in [17,21,25]},
   'patients_also_with_category_event_and_small_class_preserving_change':{str(n):len(with_small[n]) for n in [17,21,25]},
   'max_changed_snapshots_one_admission':max(by_patient.values(),default=0)}
report={'source':'MIMIC-IV Clinical Database Demo 2.2','source_sha256':hashlib.sha256((SRC/'labevents.csv.gz').read_bytes()).hexdigest(),
 'rules':'one admission; complete panel at identical charttime; valid positive numeric values; expected units; conflicting duplicates rejected; order by availability and require increasing measurement time; adjacent identical root vectors excluded; maximum count over admissions per patient; 17/21/25 snapshots support 16/20/24 atomic-panel update rounds; these are capacity counts, not final selected 40-turn sequences',
 'small_change_definition':'0<relative target change<=20%, same class; descriptive only',
 'labs':{n:{'positive_numeric_rows':counts[n],'patients':len(subjects[n])} for n in MAP.values()},
 'templates':result,'not_audited':['VASCULAR: chartevents/omr not available locally','CONICITY: required waist series not established'],
 'caveats':['Full MIMIC not available locally; no extrapolated patient counts','No partial-root carry-forward or interpolation used','NLR uses absolute counts only, not percentage roots','Blood glucose fasting status not established; numeric TYG/HOMA formula feasibility is not clinical validity','Counts for different templates may overlap in patients']}
path=BASE/'feasibility/capacity_audit.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
