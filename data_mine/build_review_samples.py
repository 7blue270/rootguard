"""Build four deliberately selected review examples; not a representative test cohort."""
import csv, gzip, hashlib, json, math
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCE = Path('/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2')
ITEMS = {'51222':'hb','51221':'hct','51279':'rbc','50861':'alt','50878':'ast','51265':'plt'}
ROOTS = {'ANEMIA':['hb','hct','rbc'], 'FIB4':['age','ast','alt','plt']}
TARGET = {'ANEMIA':('mchc','anemia_class'), 'FIB4':('fib4','fib4_risk')}
FORMULAS = {
 'mcv':('10*hct/rbc',['hct','rbc']),
 'mch':('10*hb/rbc',['hb','rbc']),
 'mchc':('100*hb/hct',['hb','hct']),
 'fib4_prod':('age*ast',['age','ast']),
 'fib4_denom':('plt*sqrt(alt)',['plt','alt']),
 'fib4':('fib4_prod/fib4_denom',['fib4_prod','fib4_denom']),
}
LABELS = {'ANEMIA':['Hypochromic','Normochromic (Normal Color)','Hyperchromic'],
          'FIB4':['Low','Indeterminate','High']}

def derived(template, x):
 if template == 'ANEMIA':
  return {'mcv':10*x['hct']/x['rbc'], 'mch':10*x['hb']/x['rbc'], 'mchc':100*x['hb']/x['hct']}
 prod=x['age']*x['ast']; den=x['plt']*math.sqrt(x['alt'])
 return {'fib4_prod':prod,'fib4_denom':den,'fib4':prod/den}

def risk(template, value):
 if template=='ANEMIA': return 0 if value<32 else 2 if value>36 else 1
 return 0 if value<1.30 else 1 if value<=2.67 else 2

def score(template,x):return derived(template,x)[TARGET[template][0]]

def load():
 with gzip.open(OUT/'sources/patients.csv.gz','rt') as f:
  demographics={r['subject_id']:r for r in csv.DictReader(f)}
 groups=defaultdict(lambda:defaultdict(list))
 with gzip.open(SOURCE/'labevents.csv.gz','rt') as f:
  for r in csv.DictReader(f):
   if r['itemid'] not in ITEMS or not r['hadm_id'] or not r['valuenum']:continue
   try:v=float(r['valuenum'])
   except ValueError:continue
   if not math.isfinite(v) or v<=0:continue
   groups[(r['subject_id'],r['hadm_id'],r['charttime'])][ITEMS[r['itemid']]].append(r)
 pools={t:defaultdict(list) for t in ROOTS}
 rejected=0
 for (sid,hadm,time),attrs in sorted(groups.items()):
  for t,names in ROOTS.items():
   labs=[n for n in names if n!='age']
   if not all(n in attrs for n in labs):continue
   # A panel is admitted only when each item has one unambiguous numeric result.
   if any(len({float(r['valuenum']) for r in attrs[n]})!=1 for n in labs):
    rejected+=1;continue
   rows={n:sorted(attrs[n],key=lambda r:(r['storetime'] or r['charttime'],int(r['labevent_id'])))[-1] for n in labs}
   expected={'hb':'g/dL','hct':'%','rbc':'m/uL','ast':'IU/L','alt':'IU/L','plt':'K/uL'}
   if any(rows[n]['valueuom']!=expected[n] for n in labs):continue
   if t=='ANEMIA' and len({rows[n]['specimen_id'] for n in labs})!=1:continue
   x={n:float(rows[n]['valuenum']) for n in labs}
   demo=demographics[sid]
   if t=='FIB4':x['age']=int(demo['anchor_age'])+int(time[:4])-int(demo['anchor_year'])
   if t=='FIB4' and x['age']<=0:continue
   panel={'subject_id':sid,'hadm_id':hadm,'measurement_time':time,
          'available_time':max(r['storetime'] or r['charttime'] for r in rows.values()),
          'values':x,'source_rows':{n:{k:r[k] for k in ['labevent_id','itemid','specimen_id','charttime','storetime','valueuom','valuenum']} for n,r in rows.items()}}
   if t=='FIB4':panel['age_provenance']={k:demo[k] for k in ['anchor_age','anchor_year','anchor_year_group']}
   pools[t][(sid,hadm)].append(panel)
 for p in pools.values():
  for k,ls in p.items():
   # Information becomes usable only after all selected results are available.
   ls.sort(key=lambda a:(a['available_time'],a['measurement_time']))
 return pools,rejected

def event_type(t,a,b):
 changed=[n for n in ROOTS[t] if a['values'][n]!=b['values'][n]]
 if not changed:return None
 if t=='ANEMIA' and set(changed)=={'rbc'}:return 'task_irrelevant_root_update'
 ca=risk(t,score(t,a['values']));cb=risk(t,score(t,b['values']))
 return 'class_change' if ca!=cb else 'class_preserving_relevant_update'

def select(t,pool):
 # Exactly four chronological panels: ANEMIA covers all three transition types;
 # FIB4 covers both possible types (all its roots affect the final score).
 need=7 if t=='ANEMIA' else 3
 bits={'class_change':1,'class_preserving_relevant_update':2,'task_irrelevant_root_update':4}
 selected=[];used=set()
 for key,panels in sorted(pool.items()):
  if key[0] in used or len(panels)<4:continue
  states={(i,1,0):[i] for i in range(len(panels))}
  for depth in range(1,4):
   for (last,d,mask),path in list(states.items()):
    if d!=depth:continue
    for j in range(last+1,len(panels)):
     if panels[j]['measurement_time']<=panels[last]['measurement_time']:continue
     typ=event_type(t,panels[last],panels[j])
     if typ is None:continue
     # Review examples explicitly include a modest score change without a class crossing.
     if typ=='class_preserving_relevant_update' and abs(score(t,panels[j]['values'])-score(t,panels[last]['values']))/max(abs(score(t,panels[last]['values'])),1e-12)>0.20:continue
     new=(j,d+1,mask|bits[typ])
     if new not in states:states[new]=path+[j]
  paths=[p for (i,d,m),p in states.items() if d==4 and m&need==need]
  if not paths:continue
  path=min(paths,key=lambda p:(p[-1]-p[0],p))
  selected.append([panels[i] for i in path]);used.add(key[0])
  if len(selected)==2:break
 if len(selected)!=2:raise RuntimeError(f'{t}: only {len(selected)} qualifying examples')
 return selected

def fmt(v):return format(v,'.12g') if isinstance(v,(int,float)) else v

def build(t,panels):
 sid=panels[0]['subject_id'];tk,ck=TARGET[t]
 out={'id':f'{t}_{sid}','topic': 'What does my MCHC indicate about my red blood cells?' if t=='ANEMIA' else 'Do I have advanced liver fibrosis?',
      'target_attribute':ck,'target_risk':None,'initial_target_risk':None,
      'data_kind':'real_MIMIC_IV_Demo_measurements_with_synthetic_template_dialogue',
      'source_dataset':'MIMIC-IV Clinical Database Demo 2.2','subject_id':sid,'hadm_id':panels[0]['hadm_id'],
      'review_only':True,'selection':'purposefully select four real panels to cover specified update types; class-preserving relevant transitions have <=20% relative target change; not prevalence estimation',
      'measurement_count':4,'dialogue_turn_count':40,
      'age_note':'FIB4 uses anchor_age + measurement_year - anchor_year: year-resolution age estimate, not exact birthday age.' if t=='FIB4' else None,
      'turns':[],'updates':[],'measurement_panels':panels}
 versions={};current={};nodevalues={};nodeparents={};nodevalid={}
 def emit(text,private,event,panel,expression=None,node=None):
  rec={'turn':len(out['turns'])+1,'speaker':'assistant','text':text,
       'user_response':'; '.join(f'{k} {fmt(v)}' for k,v in private.items()),'private_value':private,
       'event_type':event,'measurement_index':panel,'measurement_time':panels[panel]['measurement_time'],
       'information_available_time':panels[panel]['available_time'],
       'root_state':dict(current),'root_versions':{n:versions[n] for n in current},
       'truth_current_target':score(t,current),'truth_current_class':risk(t,score(t,current))}
  if expression:rec['math_expression']=expression
  if node:
   rec['node_version']=versions[node];rec['parent_versions']=nodeparents[node]
  out['turns'].append(rec)
 def compute(n,p):
  expr,parents=FORMULAS[n]
  vals=derived(t,current)
  assert all(nodevalid.get(a,False) for a in parents)
  nodevalues[n]=vals[n];versions[n]=versions.get(n,0)+1;nodevalid[n]=True
  nodeparents[n]={a:versions[a] for a in parents}
  emit(f'What is the current {n}?',{n:vals[n]},'derived_compute',p,f'{n}={expr}',n)
 def category(p):
  assert nodevalid.get(tk,False)
  c=risk(t,nodevalues[tk]);label=LABELS[t][c]
  versions[ck]=versions.get(ck,0)+1;nodevalid[ck]=True;nodeparents[ck]={tk:versions[tk]}
  emit(f'Based on the current {tk}, what is the category?',{ck:label},'task_classification',p,
       'class=0 if mchc<32 else (2 if mchc>36 else 1)' if t=='ANEMIA' else 'class=0 if fib4<1.30 else (1 if fib4<=2.67 else 2)',ck)
 for p,panel in enumerate(panels):
  new=panel['values'];changed={n:new[n] for n in ROOTS[t] if current.get(n)!=new[n]}
  before=dict(current)
  for n,v in changed.items():current[n]=v;versions[n]=versions.get(n,0)+1;nodevalid[n]=True
  # Invalidate descendants transitively, including branches that are not queried yet.
  invalid=set(changed)
  while True:
   add={n for n,(_,ps) in FORMULAS.items() if any(a in invalid for a in ps)}
   if tk in invalid:add.add(ck)
   if add<=invalid:break
   invalid|=add
  for n in invalid-set(changed):nodevalid[n]=False
  emit('Please provide the initial measurements.' if p==0 else 'Please provide the changed measurements from the next selected real laboratory panel.',
       changed,'root_initialization' if p==0 else 'root_update',p)
  if p:
   out['updates'].append({'turn':10*p+1,'from_measurement_index':p-1,'to_measurement_index':p,
       'changed_roots':list(changed),'type':event_type(t,panels[p-1],panel),
       'old_target':score(t,before),'new_target':score(t,current),
       'old_class':risk(t,score(t,before)),'new_class':risk(t,score(t,current)),
       'invalidated_nodes':sorted(invalid-set(changed))})
  emit('What are the latest available root measurements?',dict(current),'cached_root_query',p)
  for n in (['mcv','mch','mchc'] if t=='ANEMIA' else ['fib4_prod','fib4_denom','fib4']):compute(n,p)
  category(p)
  emit(f'What is the latest {tk} again?',{tk:nodevalues[tk]},'cached_derived_query',p,f'{tk}={FORMULAS[tk][0]}',tk)
  category(p)
  emit('Please repeat the current root measurements; no new measurement is supplied.',dict(current),'cached_root_query',p)
  category(p)
 out['initial_target_risk']=LABELS[t][risk(t,score(t,panels[0]['values']))]
 out['target_risk']=LABELS[t][risk(t,score(t,panels[-1]['values']))]
 return out

def validate(data):
 checks=0
 for t,records in data.items():
  assert len(records)==2 and len({r['subject_id'] for r in records})==2
  for r in records:
   assert len(r['turns'])==40
   seen={};pv={}
   for i,turn in enumerate(r['turns'],1):
    assert turn['turn']==i
    x=turn['root_state'];ds=derived(t,x)
    assert math.isclose(turn['truth_current_target'],score(t,x),rel_tol=1e-12)
    for n,v in turn['private_value'].items():
     if n in ROOTS[t]:assert v==x[n]
     elif n in ds:assert math.isclose(v,ds[n],rel_tol=1e-12)
     else:assert v==LABELS[t][risk(t,score(t,x))]
    if 'parent_versions' in turn:
     for n,v in turn['parent_versions'].items():assert pv[n]==v
    if 'node_version' in turn:pv[next(iter(turn['private_value']))]=turn['node_version']
    pv.update(turn['root_versions'])
    for n,v in turn['private_value'].items():assert f'{n} {fmt(v)}' in turn['user_response']
    checks+=1
   kinds={u['type'] for u in r['updates']}
   assert {'class_change','class_preserving_relevant_update'}<=kinds
   if t=='ANEMIA':assert 'task_irrelevant_root_update' in kinds
 return {'passed':True,'records':4,'turns':checks,'checks':['40 sequential turns per record','two distinct patients per template','numeric recomputation','classification thresholds','parent versions','text/value consistency','required update types']}

def main():
 pools,rejected=load()
 data={t:[build(t,p) for p in select(t,pool)] for t,pool in pools.items()}
 result=validate(data)
 (OUT/'review_samples.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
 for t,records in data.items():
  (OUT/f'{t.lower()}_review_samples.json').write_text(json.dumps({t:records},indent=2,ensure_ascii=False)+'\n')
 (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
 sources=[]
 for p in [SOURCE/'labevents.csv.gz',SOURCE/'d_labitems.csv.gz',OUT/'sources/patients.csv.gz']:
  sources.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 manifest={'generator':str(Path(__file__).resolve()),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'sources':sources,'patients_source_url':'https://physionet.org/files/mimic-iv-demo/2.2/hosp/patients.csv.gz',
  'selection_rule':'same admission, same charttime per complete panel; positive finite values, expected units, ambiguous duplicates excluded; CBC same specimen; chronological charttime and availability; deterministic selection of four panels covering required update types; class-preserving relevant transitions limited to <=20% relative target change',
  'ambiguous_template_panels_rejected':rejected,'deployed_dp':False,'llm_calls':0,
  'eligible_complete_panel_counts':{t:sum(map(len,p.values())) for t,p in pools.items()}}
 (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
 lines=['# 动态根节点审核样例','', '共4条记录：ANEMIA、FIB4各2位患者，每人40个问答单元（不是40次测量）。每人采用4个真实测量面板，更新轮为11、21、31。', '',
 '数值来自MIMIC-IV Demo 2.2；对话与轮次安排为确定性合成。样例为覆盖更新类型而有目的筛选，不代表自然发生比例，也不是独立验证/测试集。未执行DP发布或LLM调用。','',
 '类别保持的相关根更新额外限制目标数值相对变化不超过20%，仅为审核样例筛选条件，不是临床标准。', '', '## 更新概览','', '|模板|患者|更新轮|类型|旧目标→新目标|旧类别→新类别|变化根|','|---|---|---:|---|---|---|---|']
 for t,rs in data.items():
  for r in rs:
   for u in r['updates']:
    lines.append(f"|{t}|{r['subject_id']}|{u['turn']}|{u['type']}|{u['old_target']:.6f} → {u['new_target']:.6f}|{LABELS[t][u['old_class']]} → {LABELS[t][u['new_class']]}|{', '.join(u['changed_roots'])}|")
 lines+=['','## 阅读说明','',
 '- root_update仅包含实际改变的根值；root_state保留当轮完整状态，根值未变时不增加版本。',
 '- 更新后先使受影响派生节点失效，再按依赖顺序重算。重复询问使用最新有效结果。',
 '- 每个测量面板的全部变化根在一个更新轮原子引入，避免人为制造不同面板混合导致的中间类别事件。',
 '- FIB4的AST/ALT与血小板采用相同charttime，但可能来自不同标本；原始标本ID均保留。',
 '- 面板只有在所有结果可用后才引入；charttime和storetime均保留。不按对话轮次推断真实时间间隔。',
 '- FIB4年龄按anchor_age + 检验年份 - anchor_year估计，具有年级精度；不是根据生日得到的精确年龄。',
 '- ANEMIA最终任务是MCHC类别，不是一般临床贫血诊断；类别编号升高不等同于病情加重。',
 '- 仅RBC更新会改变MCV/MCH，但不影响MCHC和最终类别。FIB4的四个根均影响最终指标，未人为添加无关根。',
 '- 类别不变并不一定意味着变化幅度小，还可能是阈值距离或依赖抵消；更新类型仅按实际计算结果命名。',
 '- target_risk为最后状态分类，initial_target_risk为初始分类；逐轮truth_current_class为数值类别。',
 '- 这40轮主要包含计算与重复查询，尚不构成40次真实测量或自然采集的医患长对话。',
 '- 当前先审核4个样例，尚未扩展到8模板×20人；原始数据能否覆盖其余模板需另做可用性审计。','',
 '## 完整对话（数值文本为12位有效数字，JSON数值保留计算精度）','']
 for t,rs in data.items():
  for r in rs:
   lines += [f"### {r['id']}",'','|轮|类型|提问|回答|','|---:|---|---|---|']
   for q in r['turns']:lines.append(f"|{q['turn']}|{q['event_type']}|{q['text']}|{q['user_response']}|")
 (OUT/'REVIEW.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(result))
 print(json.dumps({t:[{'id':r['id'],'updates':r['updates']} for r in rs] for t,rs in data.items()},indent=2))

if __name__=='__main__':main()
