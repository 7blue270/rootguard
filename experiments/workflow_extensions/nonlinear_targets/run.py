"""Exploratory static-target benchmark. No new privacy mechanism; no LLM."""
import ast,csv,gzip,json,hashlib,math,sys,argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[2];SRC=Path('/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2')
parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=20261023);args=parser.parse_args()
OUT=P/('results' if args.seed==20261023 else f'results_seed_{args.seed}');OUT.mkdir(exist_ok=True)
# Load original allocation functions verbatim, excluding unrelated heavy imports.
tree=ast.parse((ROOT/'utils/utils.py').read_text());functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
needed={'optimal_budget_allocation_exact'}
while True:
 extra={n.id for k in needed for n in ast.walk(functions[k]) if isinstance(n,ast.Name) and n.id in functions}
 if extra<=needed:break
 needed|=extra
ns={'np':np,'math':math};exec(compile(ast.Module(body=[functions[k] for k in sorted(needed)],type_ignores=[]),'<original RootGuard allocation>','exec'),ns)
allocate=ns['optimal_budget_allocation_exact']
M=1000;SEED=args.seed;BUDGETS=[.03,.1,.3,.7,1.,3.];REPS=60
MAP={'51222':'hb','51221':'hct','51279':'rbc','50861':'alt','50878':'ast','51265':'plt','51133':'lym','52075':'neu','51000':'tg','50931':'glu','50912':'creat','50885':'bili','51237':'inr','50983':'na','50821':'po2','50818':'pco2','50816':'fio2'}
UNITS={'hb':'g/dL','hct':'%','rbc':'m/uL','ast':'IU/L','alt':'IU/L','plt':'K/uL','lym':'K/uL','neu':'K/uL','tg':'mg/dL','glu':'mg/dL','creat':'mg/dL','bili':'mg/dL','inr':None,'na':'mEq/L','po2':'mm Hg','pco2':'mm Hg','fio2':'%'}
g=defaultdict(lambda:defaultdict(list));specimens={}
with gzip.open(SRC/'labevents.csv.gz','rt') as f:
 for r in csv.DictReader(f):
  key=(r['subject_id'],r['hadm_id'],r['charttime'])
  if r['itemid']=='52033':specimens[r['specimen_id']]=r['value']
  if r['itemid'] not in MAP or not r['hadm_id'] or not r['valuenum']:continue
  n=MAP[r['itemid']]
  try:v=float(r['valuenum'])
  except:continue
  if not np.isfinite(v) or v<=0:continue
  if n=='fio2':
   if .2<v<=1:v*=100
   elif not 20<v<=100:continue
  elif UNITS[n] is not None and r['valueuom']!=UNITS[n]:continue
  g[key][n].append((v,r['labevent_id'],r['specimen_id']))
with gzip.open(ROOT/'data_mine/sources/patients.csv.gz','rt') as f:demo={r['subject_id']:r for r in csv.DictReader(f)}
allids=sorted(demo,key=lambda s:hashlib.sha256(f'{SEED}:{s}'.encode()).hexdigest());reference=set(allids[:50])

def meldbase(x):return .957*np.log(np.clip(x[...,0],1,4))+.378*np.log(np.maximum(x[...,1],1))+1.120*np.log(np.maximum(x[...,2],1))+.643

def meld(t,x):
 base=meldbase(x)
 # Official SQL rounds positive sum to 0.1 and multiplies by ten.
 score=np.where(base>4,40,np.floor(base*10+.5)) if t=='MELDNA_exact' else np.minimum(40,base*10)
 if t=='MELD_smooth':return score
 sodium=np.clip(137-x[...,3],0,12)
 return np.where(score>11,score+1.32*sodium-.033*score*sodium,score)

def sofa(x):
 c,b,p=x[...,0],x[...,1],x[...,2]
 return np.select([c>=5,c>=3.5,c>=2,c>=1.2],[4,3,2,1],0)+np.select([b>=12,b>=6,b>=2,b>=1.2],[4,3,2,1],0)+np.select([p<20,p<50,p<100,p<150],[4,3,2,1],0)

TASKS={
 'ANEMIA':(['hb','hct','rbc'],lambda x:100*x[...,0]/x[...,1]),
 'FIB4':(['age','ast','alt','plt'],lambda x:x[...,0]*x[...,1]/(x[...,3]*np.sqrt(x[...,2]))),
 'NLR':(['neu','lym'],lambda x:x[...,0]/x[...,1]),
 'TYG':(['tg','glu'],lambda x:np.log(x[...,0]*x[...,1]/2)),
 'AADO2_affine':(['fio2','pco2','po2'],lambda x:7.13*x[...,0]-x[...,1]/.8-x[...,2]),
 'PF_ratio':(['po2','fio2'],lambda x:100*x[...,0]/x[...,1]),
 'MELD_smooth':(['creat','bili','inr'],lambda x:meld('MELD_smooth',x)),
 'MELDNA_smooth':(['creat','bili','inr','na'],lambda x:meld('MELDNA_smooth',x)),
 'MELDNA_exact':(['creat','bili','inr','na'],lambda x:meld('MELDNA_exact',x)),
 'SOFA_lab3':(['creat','bili','plt'],sofa)}

def classes(name,y):
 if name=='ANEMIA':return np.where(y<32,0,np.where(y>36,2,1)),3
 if name=='FIB4':return np.where(y<1.3,0,np.where(y<=2.67,1,2)),3
 if name=='NLR':return (y>=3).astype(int),2
 if name=='TYG':return (y>8.5).astype(int),2
 return None,None

samples={};allocations={};results=[];patient_rows=[]
for ti,(name,(roots,fn)) in enumerate(TASKS.items()):
 candidates=defaultdict(list)
 for (sid,hadm,time),attrs in sorted(g.items()):
  labs=[r for r in roots if r!='age']
  if not all(r in attrs and len({v[0] for v in attrs[r]})==1 for r in labs):continue
  if name in ['PF_ratio','AADO2_affine'] and not all(specimens.get(attrs[r][0][2],'').upper().startswith('ART') for r in labs):continue
  vals={r:attrs[r][0][0] for r in labs}
  if 'age' in roots:
   vals['age']=int(demo[sid]['anchor_age'])+int(time[:4])-int(demo[sid]['anchor_year'])
   if vals['age']<18:continue
  candidates[sid].append({'subject_id':sid,'hadm_id':hadm,'charttime':time,'values':[vals[r] for r in roots], 'source_ids':{r:attrs[r][0][1] for r in labs}})
 # Earliest complete same-time row per patient, independent of outcomes.
 records=[ls[0] for _,ls in sorted(candidates.items())];samples[name]=records
 if not records:continue
 X=np.array([r['values'] for r in records]);mask=np.array([r['subject_id'] in reference for r in records]);ref=X[mask];test=X[~mask];ids=[r['subject_id'] for r in records if r['subject_id'] not in reference]
 if len(ref)<5 or len(test)<5:continue
 # Domain is a public released-Demo benchmark domain, not estimated secretly in deployment.
 lo=X.min(axis=0);hi=X.max(axis=0);hi=np.where(hi<=lo,lo+1,hi);width=hi-lo
 mu=ref.mean(axis=0);h=[];hcheck=[]
 for j in range(len(roots)):
  def diff(factor):
   step=1e-5*max(1,abs(mu[j]))*factor;a=mu.copy();b=mu.copy();a[j]+=step;b[j]-=step
   return abs(float(fn(a)-fn(b))/(2*step))
  h.append(diff(1));hcheck.append(diff(.1))
 assert np.allclose(h,hcheck,rtol=1e-4,atol=1e-6)
 y=fn(test);gtclass,C=classes(name,y);rng=np.random.default_rng(SEED+ti);U=rng.random((len(test),REPS,len(roots)))
 allocations[name]={'roots':roots,'reference_patients':len(ref),'test_patients':len(test),'mu':mu.tolist(),'h':h,'lo':lo.tolist(),'hi':hi.tolist(),'all_zero_gradient':bool(np.all(np.array(h)==0)), 'by_budget':{}}
 for B in BUDGETS:
  a=allocate(dict(zip(roots,h)),B,epsilon_min=.001,domain_widths=dict(zip(roots,width)),num_candidates=M)
  assert all(v>0 for v in a.values()) and abs(sum(a.values())-B)<1e-7
  allocations[name]['by_budget'][str(B)]=a
  predictions={}
  for method,budgets in [('M-Roots',np.full(len(roots),B/len(roots))),('M-Opt',np.array([a[r] for r in roots]))]:
   z=np.zeros((len(test),REPS,len(roots)))
   for i in range(len(test)):
    for j in range(len(roots)):
     idx=(test[i,j]-lo[j])/width[j]*(M-1)
     logp=-budgets[j]*np.abs(idx-np.arange(M))/2;logp-=logp.max();p=np.exp(logp);p/=p.sum()
     z[i,:,j]=lo[j]+np.searchsorted(np.cumsum(p),U[i,:,j],side='right').clip(0,M-1)*width[j]/(M-1)
   predictions[method]=fn(z)
  er={k:np.abs(v-y[:,None]) for k,v in predictions.items()};per={k:v.mean(axis=1) for k,v in er.items()}
  boots=np.random.default_rng(SEED).integers(0,len(test),(2000,len(test)))
  gap=per['M-Opt']-per['M-Roots'];ci=np.quantile(gap[boots].mean(axis=1),[.025,.975]);den=np.mean(np.abs(y))
  entry={'task':name,'B':B,'test_patients':len(test),'reference_patients':len(ref),'mae_ratio_opt_over_roots':float(per['M-Opt'].mean()/max(per['M-Roots'].mean(),1e-15)),'mae_gap_CI_low':float(ci[0]),'mae_gap_CI_high':float(ci[1]),'patient_win_fraction_opt':float(np.mean(gap<0)), 'methods':{}}
  for method in predictions:
   metrics={'MAE':float(er[method].mean()),'wMAPE':float(er[method].mean()/den*100),'p95_absolute_error':float(np.quantile(er[method],.95))}
   if C:
    pc,_=classes(name,predictions[method]);metrics.update(CMR=float(np.mean(pc!=gtclass[:,None])*100),RCE=float(np.mean(np.abs(pc-gtclass[:,None])/(C-1))*100))
   entry['methods'][method]=metrics
  results.append(entry)
  for i,sid in enumerate(ids):patient_rows.append({'task':name,'B':B,'subject_id':sid,'roots_MAE':float(per['M-Roots'][i]),'opt_MAE':float(per['M-Opt'][i])})
 print(name,len(ref),len(test),'h',np.round(h,5).tolist(),flush=True)
(OUT/'results.json').write_text(json.dumps(results,indent=2));(OUT/'allocations.json').write_text(json.dumps(allocations,indent=2));(OUT/'sample_provenance.json').write_text(json.dumps(samples,indent=2));(OUT/'patient_metrics.json').write_text(json.dumps(patient_rows,indent=2))
(OUT/'config.json').write_text(json.dumps({'seed':SEED,'budgets':BUDGETS,'repetitions':REPS,'grid':M,'reference_patient_ids':sorted(reference),'solver_functions':sorted(needed),'scope':'static target utility; not dynamic trajectory privacy','mechanism':'original preempt/sanitizer.py continuous-index discrete exponential PMF; vectorized inverse CDF','MOpt_extension':'same original exact solver; end-to-end partial derivatives at held-out population mean, finite-difference checked at two scales. New target differentiation is an extension, not out-of-box code support.','domains':'per-task min/max of one public released-Demo observation per patient; includes test values as public benchmark domain, reference mean excludes test patients','MELD_scope':'lab function at contemporaneous measurements, dialysis fixed absent; no first-day extrema or full ICU policy recreation; smooth variants remove only rounding','SOFA_scope':'sum of platelet, bilirubin and creatinine-only renal components; not full SOFA; no urine output','blood_gas':'explicit ART specimen only; official FIO2 conversion; no predicted arterial classification','privacy':'B index-distance metric composition, not standard whole-patient epsilon-DP','git_source_ref':'303d26c623dcc9c49cc0f204468d4acc2f063797'},indent=2))
print('completed',len(results),'cells',flush=True)
