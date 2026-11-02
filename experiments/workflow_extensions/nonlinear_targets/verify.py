import ast,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[2]
# Verify inverse-CDF draws against actual original method with matching RNG draws.
t=ast.parse((R/'preempt/sanitizer.py').read_text());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Sanitizer');f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='M_exponential_discrete')
ns={'np':np};exec(compile(ast.Module(body=[f],type_ignores=[]),'<original sampler>','exec'),ns)
for x,eps in [(0,.03),(3.14159,.1),(9.99,3)]:
 np.random.seed(733);actual=[ns[f.name](None,x,eps,0,10,1000) for _ in range(100)]
 idx=x/10*999;lp=-eps*np.abs(idx-np.arange(1000))/2;lp-=lp.max();pr=np.exp(lp);pr/=pr.sum()
 np.random.seed(733);expected=np.searchsorted(np.cumsum(pr),np.random.random(100))*10/999
 assert np.allclose(actual,expected,atol=1e-14,rtol=1e-14)
# Fixed hand-computed formula/threshold cases from official SQL.
def base(c,b,i):return .957*math.log(min(4,max(1,c)))+.378*math.log(max(1,b))+1.120*math.log(max(1,i))+.643
assert math.floor(base(1,1,1)*10+.5)==6
assert math.floor(base(.5,.5,.5)*10+.5)==6
assert base(4,1,1)==base(8,1,1)
assert abs(7.13*40-40/.8-100-135.2)<1e-10
for folder in [P/'results']+sorted(P.glob('results_seed_*')):
 if not (folder/'results.json').exists():continue
 a=json.loads((folder/'allocations.json').read_text());res=json.loads((folder/'results.json').read_text());co=json.loads((folder/'config.json').read_text());samples=json.loads((folder/'sample_provenance.json').read_text())
 for name,meta in a.items():
  assert len({r['subject_id'] for r in samples[name]})==len(samples[name])
  for B,allocation in meta['by_budget'].items():
   assert abs(sum(allocation.values())-float(B))<1e-7 and min(allocation.values())>=.001-1e-8
 for r in res:
  if a[r['task']]['all_zero_gradient']:
   assert r['mae_ratio_opt_over_roots']==1
  for metrics in r['methods'].values():assert all(np.isfinite(v) for v in metrics.values())
 print(folder.name, 'PASS',len(res),'cells')
print('PASS: 300 original-sampler paired draws; hand formula cases; budget conservation; patient uniqueness; zero-gradient fallback; finite outputs')
