"""Independent source replay and canonical RootGuard formula validation."""
import csv,gzip,json,math,sys,hashlib,ast
from collections import Counter
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent))
tree=ast.parse((BASE.parent/'utils/med_domain/all_templates.py').read_text())
canonical={}
for statement in tree.body:
 if isinstance(statement,ast.Assign):
  for name in statement.targets:
   if isinstance(name,ast.Name) and name.id in ['template_expressions','template_nodes','template_target_keys']:canonical[name.id]=ast.literal_eval(statement.value)
template_expressions=canonical['template_expressions'];template_nodes=canonical['template_nodes'];template_target_keys=canonical['template_target_keys']
from utils.risk_class import get_risk_class
with gzip.open('/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2/labevents.csv.gz','rt') as f:raw={r['labevent_id']:r for r in csv.DictReader(f)}
with gzip.open(BASE/'sources/patients.csv.gz','rt') as f:patients={r['subject_id']:r for r in csv.DictReader(f)}
LABELS={'ANEMIA':['Hypochromic','Normochromic (Normal Color)','Hyperchromic'],'FIB4':['Low','Indeterminate','High'],'NLR':['Normal','Elevated'],'TYG':['Normal','Elevated']}
report={};total=0;sourcerefs=0
for t in LABELS:
 path=BASE/f'{t}.json';data=json.loads(path.read_text());assert list(data)==[t]
 records=data[t];assert len({r['subject_id'] for r in records})==len(records)
 for r in records:
  assert len(r['turns'])==40
  x={};rv={};nodeversions={};nodeparents={};lastavailable='';updates=0
  assert len(r['measurements'])==r['update_count']+1
  for i,q in enumerate(r['turns'],1):
   assert q['turn']==i and q['information_available_time']>=lastavailable
   lastavailable=q['information_available_time']
   roots=set(q['root_state']);old=dict(x)
   if q['event_type'] in ['root_initialization','root_update']:
    changed={a:v for a,v in q['private_value'].items() if a in roots}
    assert changed and len(changed)==len(q['private_value'])
    for a,v in changed.items():
     assert x.get(a)!=v;x[a]=v;rv[a]=rv.get(a,0)+1
    if q['event_type']=='root_update':updates+=1
   assert x==q['root_state'] and rv==q['root_versions']
   if t=='FIB4':
    age=r['age_provenance'];d=patients[r['subject_id']]
    assert x['age']==int(d['anchor_age'])+int(age['age_reference_measurement_time'][:4])-int(d['anchor_year'])
   for a,v in x.items():
    if a=='age':continue
    source=raw[q['root_source_ids'][a]]
    assert source['subject_id']==r['subject_id'] and source['hadm_id']==r['hadm_id']
    assert float(source['valuenum'])==v and source['charttime']==q['root_measurement_times'][a]
    assert (source['storetime'] or source['charttime'])<=q['information_available_time']
    sourcerefs+=1
   values=dict(x)
   for node,value in q['reference_derived_state'].items():
    canonical=eval(template_expressions[node],{'__builtins__':{},'np':math,'abs':abs},values)
    assert math.isclose(canonical,value,rel_tol=1e-12,abs_tol=1e-12)
    values[node]=canonical
   target=values[template_target_keys[t]];c=get_risk_class(t,target)
   assert math.isclose(q['expected_task_output']['target_value'],target,rel_tol=1e-12)
   assert q['expected_task_output']['class_index']==c
   assert q['expected_task_output']['class_label']==LABELS[t][c]
   for node in list(q['reference_derived_state'])+[r['target_attribute']]:
    ps=template_nodes[node];pv={a:rv[a] if a in rv else nodeversions[a] for a in ps}
    if nodeparents.get(node)!=pv:nodeversions[node]=nodeversions.get(node,0)+1;nodeparents[node]=pv
    assert q['reference_parent_versions'][node]==pv
    assert q['reference_node_versions'][node]==nodeversions[node]
   if q['event_type'] not in ['root_initialization','root_update']:
    for node,v in q['private_value'].items():
     if node==r['target_attribute']:assert v==LABELS[t][c]
     else:assert math.isclose(v,values[node],rel_tol=1e-12)
   for node,v in q['private_value'].items():assert f'{node} {v}' in q['user_response']
   total+=1
  assert updates==r['update_count'] and 16<=updates<=24 and updates/40==r['update_fraction']
  actual=[q['turn'] for q in r['turns'] if q['event_type']=='root_update']
  assert actual==[u['turn'] for u in r['updates']]
  for u in r['updates']:
   a=r['turns'][u['turn']-2]['expected_task_output'];b=r['turns'][u['turn']-1]['expected_task_output']
   assert u['old_class']==a['class_index'] and u['new_class']==b['class_index']
  assert r['target_risk']==r['turns'][-1]['expected_task_output']['class_label']
 report[t]={'patients':len(records),'turns':40*len(records),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
result={'passed':True,'total_turns':total,'source_references_checked':sourcerefs,'datasets':report,
 'checks':['unique patients within templates','40 sequential turns','16-24 real update batches','replay changed roots','no unavailable future results','original source patient/admission/value/time','canonical RootGuard formulas and classes','parent/node versions','text/private_value consistency','event annotations','file hashes']}
(BASE/'dataset_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
