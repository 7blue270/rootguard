"""Build real-measurement, synthetic-dialogue datasets without adding patients or values."""
import csv,gzip,json,math,hashlib,importlib.util
from collections import defaultdict,Counter
from pathlib import Path

BASE=Path(__file__).resolve().parent
SOURCE=Path('/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2')
ROOTS={'ANEMIA':['hb','hct','rbc'],'FIB4':['age','ast','alt','plt'],'NLR':['neu','lym'],'TYG':['tg','glu']}
ITEMS={'51222':'hb','51221':'hct','51279':'rbc','50861':'alt','50878':'ast','51265':'plt','51133':'lym','52075':'neu','51000':'tg','50931':'glu'}
UNITS={'hb':'g/dL','hct':'%','rbc':'m/uL','ast':'IU/L','alt':'IU/L','plt':'K/uL','lym':'K/uL','neu':'K/uL','tg':'mg/dL','glu':'mg/dL'}
TARGETS={'ANEMIA':('mchc','anemia_class'),'FIB4':('fib4','fib4_risk'),'NLR':('nlr','nlr_class'),'TYG':('tyg','tyg_class')}
LABELS={'ANEMIA':['Hypochromic','Normochromic (Normal Color)','Hyperchromic'],'FIB4':['Low','Indeterminate','High'],'NLR':['Normal','Elevated'],'TYG':['Normal','Elevated']}
FORMULAS={
 'mcv':('10*hct/rbc',['hct','rbc']), 'mch':('10*hb/rbc',['hb','rbc']), 'mchc':('100*hb/hct',['hb','hct']),
 'fib4_prod':('age*ast',['age','ast']), 'fib4_denom':('plt*sqrt(alt)',['plt','alt']), 'fib4':('fib4_prod/fib4_denom',['fib4_prod','fib4_denom']),
 'nlr_sum':('neu+lym',['neu','lym']), 'nlr_diff':('abs(neu-lym)',['neu','lym']), 'nlr':('neu/lym',['neu','lym']),
 'tyg_prod':('tg*glu',['tg','glu']), 'tyg_half':('tyg_prod/2',['tyg_prod']), 'tyg':('ln(tyg_half)',['tyg_half'])}
NODES={'ANEMIA':['mcv','mch','mchc'],'FIB4':['fib4_prod','fib4_denom','fib4'],'NLR':['nlr_sum','nlr_diff','nlr'],'TYG':['tyg_prod','tyg_half','tyg']}

def calc(t,x):
 if t=='ANEMIA':return {'mcv':10*x['hct']/x['rbc'],'mch':10*x['hb']/x['rbc'],'mchc':100*x['hb']/x['hct']}
 if t=='FIB4':
  a=x['age']*x['ast'];b=x['plt']*math.sqrt(x['alt']);return {'fib4_prod':a,'fib4_denom':b,'fib4':a/b}
 if t=='NLR':return {'nlr_sum':x['neu']+x['lym'],'nlr_diff':abs(x['neu']-x['lym']),'nlr':x['neu']/x['lym']}
 a=x['tg']*x['glu'];return {'tyg_prod':a,'tyg_half':a/2,'tyg':math.log(a/2)}

def category(t,v):
 if t=='ANEMIA':return 0 if v<32 else 2 if v>36 else 1
 if t=='FIB4':return 0 if v<1.30 else 1 if v<=2.67 else 2
 if t=='NLR':return int(v>=3)
 return int(v>8.5)

def describe(t,a,b):
 tk,_=TARGETS[t];old=calc(t,a)[tk];new=calc(t,b)[tk];ca=category(t,old);cb=category(t,new)
 changed=[n for n in ROOTS[t] if a[n]!=b[n]];delta=abs(new-old)/max(abs(old),1e-12)
 irrelevant=t=='ANEMIA' and set(changed)=={'rbc'}
 kind='task_irrelevant_root_update' if irrelevant else 'class_change' if ca!=cb else 'class_preserving_small_target_change' if delta<=.20 else 'class_preserving_large_target_change'
 return {'changed_roots':changed,'type':kind,'old_target':old,'new_target':new,'relative_target_change':delta,'old_class':ca,'new_class':cb}

def load():
 with gzip.open(BASE/'sources/patients.csv.gz','rt') as f:demo={r['subject_id']:r for r in csv.DictReader(f)}
 g=defaultdict(lambda:defaultdict(list))
 with gzip.open(SOURCE/'labevents.csv.gz','rt') as f:
  for r in csv.DictReader(f):
   if r['itemid'] not in ITEMS or not r['hadm_id'] or not r['valuenum']:continue
   n=ITEMS[r['itemid']]
   try:v=float(r['valuenum'])
   except ValueError:continue
   if math.isfinite(v) and v>0 and r['valueuom']==UNITS[n]:g[(r['subject_id'],r['hadm_id'],r['charttime'])][n].append(r)
 by=defaultdict(list)
 for (sid,hadm,time),attrs in g.items():
  rows={n:max(rs,key=lambda r:(r['storetime'] or r['charttime'],int(r['labevent_id']))) for n,rs in attrs.items() if len({float(r['valuenum']) for r in rs})==1}
  if rows:by[(sid,hadm)].append((time,rows))
 return demo,by

def timeline(t,key,batches,demo):
 sid,hadm=key;labs=[n for n in ROOTS[t] if n!='age'];events=[];initials=[]
 for time,rs in batches:
  rows={n:rs[n] for n in labs if n in rs}
  if not rows:continue
  available=max(r['storetime'] or time for r in rows.values())
  event={'measurement_time':time,'available_time':available,'rows':rows}
  events.append(event)
  if set(rows)==set(labs):
   if t=='ANEMIA' and len({r['specimen_id'] for r in rows.values()})!=1:continue
   initials.append(event)
 if not initials:return []
 initial=min(initials,key=lambda p:(p['available_time'],p['measurement_time']))
 x={n:float(r['valuenum']) for n,r in initial['rows'].items()};timestamps={n:initial['measurement_time'] for n in labs};sources={n:r['labevent_id'] for n,r in initial['rows'].items()}
 if t=='FIB4':
  # Fixed within one admission: year-resolution age at initial measurement.
  x['age']=int(demo[sid]['anchor_age'])+int(initial['measurement_time'][:4])-int(demo[sid]['anchor_year'])
  if x['age']<=0:return []
  timestamps['age']=initial['measurement_time'];sources['age']='patients:'+sid
 states=[{**initial,'values':dict(x),'root_measurement_times':dict(timestamps),'root_source_ids':dict(sources),'age_reference_time':initial['measurement_time']}]
 for e in sorted(events,key=lambda p:(p['available_time'],p['measurement_time'])):
  if e['available_time']<initial['available_time']:continue
  accepted={n:r for n,r in e['rows'].items() if r['charttime']>timestamps[n]}
  if not accepted:continue
  before=dict(x)
  for n,r in accepted.items():x[n]=float(r['valuenum']);timestamps[n]=r['charttime'];sources[n]=r['labevent_id']
  if x==before:continue
  states.append({**e,'rows':accepted,'values':dict(x),'root_measurement_times':dict(timestamps),'root_source_ids':dict(sources),'age_reference_time':initial['measurement_time']})
 return states

def choose(t,by,demo):
 candidates=defaultdict(list)
 for key,batches in sorted(by.items()):
  states=timeline(t,key,batches,demo)
  if len(states)>=17:candidates[key[0]].append((key,states))
 result=[]
 for i,sid in enumerate(sorted(candidates)[:20]):
  # Requested update counts cycle from 16 to 24; cap at real available capacity.
  desired=16+i%9;options=[]
  for key,states in candidates[sid]:
   n=min(desired,len(states)-1)
   for start in range(len(states)-n):
    window=states[start:start+n+1]
    kinds={describe(t,a['values'],b['values'])['type'] for a,b in zip(window,window[1:])}
    # Fixed coverage rule, using no DP method outputs; earliest window breaks ties.
    priority=(n,'class_change' in kinds,'task_irrelevant_root_update' in kinds,len(kinds))
    options.append((priority,key,start,window))
  priority,key,start,window=max(options,key=lambda z:(z[0],-int(z[1][1]),-z[2]))
  result.append((key,start,window))
 return result,len(candidates)

def row_evidence(r):return {n:r[n] for n in ['labevent_id','subject_id','hadm_id','itemid','specimen_id','charttime','storetime','valuenum','valueuom']}

def build(t,key,start,states,demo):
 sid,hadm=key;tk,ck=TARGETS[t];n=len(states)-1
 # All root changes in a measurement batch are introduced atomically.
 update_turns={2+(j*38//(n-1)):j+1 for j in range(n)}
 assert len(update_turns)==n and max(update_turns)==40
 versions={};parents={};current={};current_state=None;turns=[];updates=[];query_i=0
 first=states[0]['values'];final=states[-1]['values']
 record={'id':f'{t}_{sid}','topic':{'ANEMIA':'What does my MCHC indicate about my red blood cells?','FIB4':'Do I have advanced liver fibrosis?','NLR':'What is my systemic inflammation status?','TYG':'What is my triglyceride-glucose index?'}[t],
  'target_attribute':ck,'initial_target_risk':LABELS[t][category(t,calc(t,first)[tk])],'target_risk':LABELS[t][category(t,calc(t,final)[tk])],
  'subject_id':sid,'hadm_id':hadm,'split':'unassigned','data_kind':'real_measurements_with_synthetic_dialogue_schedule',
  'source_dataset':'MIMIC-IV Clinical Database Demo 2.2','state_semantics':'latest observed values; roots may have different measurement times',
  'dialogue_turn_count':40,'update_count':n,'update_fraction':n/40,'window_start_in_changed_state_sequence':start,
  'selection_note':'Purposeful update/event coverage; not a representative clinical cohort. Patients are not duplicated to meet quotas.',
  'measurements':[],'turns':turns,'updates':updates}
 if t=='FIB4':record['age_provenance']={**{k:demo[sid][k] for k in ['anchor_age','anchor_year','anchor_year_group']},'age_reference_measurement_time':states[0]['age_reference_time'],'age_value_used':first['age'],'rule':'anchor_age + first_complete_panel_year_in_admission - anchor_year; year-resolution estimate, held fixed within admission'}
 for j,s in enumerate(states):
  # First window state can inherit earlier root results; preserve their actual source IDs.
  record['measurements'].append({'measurement_index':j,'measurement_time':s['measurement_time'],'available_time':s['available_time'],'root_state':s['values'],'root_measurement_times':s['root_measurement_times'],'root_source_ids':s['root_source_ids'],'batch_source_rows':{a:row_evidence(r) for a,r in s['rows'].items()}})
 def apply(s):
  nonlocal current,current_state
  old=dict(current);current=dict(s['values']);current_state=s
  changed=[a for a in ROOTS[t] if old.get(a)!=current[a]]
  for a in changed:versions[a]=versions.get(a,0)+1
  for node in NODES[t]+[ck]:
   ps=FORMULAS[node][1] if node!=ck else [tk]
   pvs={a:versions[a] for a in ps}
   if parents.get(node)!=pvs:versions[node]=versions.get(node,0)+1;parents[node]=pvs
  return changed
 for turn in range(1,41):
  index=0 if turn==1 else update_turns.get(turn)
  if index is not None:
   s=states[index];old=dict(current);changed=apply(s)
   event='root_initialization' if turn==1 else 'root_update'
   private={a:current[a] for a in changed}
   text='What are your initial measurements?' if turn==1 else 'What measurements have changed in the next available laboratory batch?'
   node=None
   if turn>1:updates.append({'turn':turn,'measurement_index':index,**describe(t,old,current)})
  else:
   nodes=NODES[t]+[ck,tk,ck];node=nodes[query_i%len(nodes)];query_i+=1
   event='task_classification' if node==ck else 'derived_query'
   value=LABELS[t][category(t,calc(t,current)[tk])] if node==ck else calc(t,current)[node]
   private={node:value};text=f'What is the current {node}, using the latest available measurements?'
  derived=calc(t,current);c=category(t,derived[tk]);label=LABELS[t][c]
  entry={'turn':turn,'speaker':'assistant','text':text,'user_response':'; '.join(f'{a} {v}' for a,v in private.items()),'private_value':private,'event_type':event,
   'measurement_index':0 if turn==1 else max((idx for pos,idx in update_turns.items() if pos<=turn),default=0),
   'information_available_time':current_state['available_time'],'root_state':dict(current),
   'root_measurement_times':dict(current_state['root_measurement_times']),'root_source_ids':dict(current_state['root_source_ids']),
   'root_versions':{a:versions[a] for a in ROOTS[t]},
   'reference_derived_state':derived,'reference_node_versions':{a:versions[a] for a in NODES[t]+[ck]},
   'reference_parent_versions':{a:dict(parents[a]) for a in NODES[t]+[ck]},
   'expected_task_output':{'target_value':derived[tk],'class_index':c,'class_label':label},
   'annotation_note':'Reference states and expected outputs are evaluation gold, not additional dialogue disclosures.'}
  if node:
   entry['node_version']=versions[node];entry['parent_versions']=dict(parents[node])
   entry['math_expression']=f'{node}={FORMULAS[node][0]}' if node!=ck else {'ANEMIA':'class=0 if mchc<32 else (2 if mchc>36 else 1)','FIB4':'class=0 if fib4<1.30 else (1 if fib4<=2.67 else 2)','NLR':'class=1 if nlr>=3.0 else 0','TYG':'class=1 if tyg>8.5 else 0'}[t]
  turns.append(entry)
 return record

def legacy_main(output):
 output=Path(output)
 output.mkdir(parents=True,exist_ok=True)
 if any((output/f'{t}.json').exists() for t in ROOTS):
  raise FileExistsError('Legacy export requires an empty output directory; existing dialogue files are preserved')
 demo,by=load();summary={};all_ids=set()
 for t in ROOTS:
  chosen,capacity=choose(t,by,demo);records=[build(t,key,start,states,demo) for key,start,states in chosen]
  (output/f'{t}.json').write_text(json.dumps({t:records},indent=2,ensure_ascii=False)+'\n')
  all_ids.update(r['subject_id'] for r in records)
  summary[t]={'patients':len(records),'eligible_patients':capacity,'requested_patients':20,'shortfall':20-len(records),'turns_per_patient':40,
   'total_turns':40*len(records),'update_count_range':[min(r['update_count'] for r in records),max(r['update_count'] for r in records)],
   'patients_with_category_change':sum(any(u['old_class']!=u['new_class'] for u in r['updates']) for r in records),
   'update_types':dict(Counter(u['type'] for r in records for u in r['updates']))}
 sourcefiles=[SOURCE/'labevents.csv.gz',SOURCE/'d_labitems.csv.gz',BASE/'sources/patients.csv.gz']
 (output/'dataset_manifest.json').write_text(json.dumps({'templates':summary,'unique_patients_across_templates':len(all_ids),
  'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcefiles},'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'updates_are_real':True,'dialogue_is_synthetic':True,'dp_or_llm_execution':False,
  'notes':['Same patients can occur in more than one template; no splits assigned.','NLR absolute counts only.','TYG uses recorded glucose without confirmed fasting status; numeric benchmark, not a verified clinical diagnosis.','No interpolated roots, no duplicate patients to meet quotas, no artificial value changes.','Window selection uses a fixed task-event coverage rule and no DP method output.','Small target change means relative change <=20%; descriptive benchmark label, not clinical standard.','Root ages for FIB4 are year-resolution estimates.']},ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(summary,indent=2))

def main():
 import argparse
 # Support both `python -m data_mine.build_datasets` and the legacy script entry.
 import sys
 sys.path.insert(0,str(BASE.parent))
 from data_mine.finite_trajectories import build,DEFAULT_OUT,DEFAULT_SOURCE
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--format',choices=['finite','legacy'],default='finite')
 p.add_argument('--output',type=Path,default=DEFAULT_OUT)
 p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
 p.add_argument('--templates',nargs='+',default=['ANEMIA','FIB4','NLR','TYG','MELD_NA'])
 p.add_argument('--turns',type=int,default=40)
 p.add_argument('--min-updates',type=int,default=16)
 p.add_argument('--max-updates',type=int,default=24)
 p.add_argument('--patients',type=int,default=20)
 p.add_argument('--overwrite',action='store_true')
 a=p.parse_args()
 if a.format=='legacy':
  global SOURCE
  SOURCE=a.source
  legacy_main(a.output)
 else:
  print(json.dumps(build(a.output,a.source,a.templates,a.turns,a.min_updates,a.max_updates,a.patients,a.overwrite),indent=2))

if __name__=='__main__':main()
