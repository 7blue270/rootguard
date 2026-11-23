"""Isolate the task readout from joint filtering on exactly M9's frozen route."""
import json,sys
import numpy as np
from .study import COHORTS,OUT,PN12,setup,plans,pick,sim,estimate,BUDGETS,write
def run(name):
    ctx,(_,_,test),_=setup(name);dest=OUT/name;cfg=json.loads((dest/'frozen.json').read_text())
    pool=plans(ctx,cfg['package']);rows=[]
    archive=json.loads((PN12/(ctx.work.template+'_frozen.json')).read_text()) if name.startswith('2012') else None
    for r in test:
        for B in BUDGETS:
            arm=pick(cfg['route'],B,r['_public']);ps=sim(ctx,r,B,pool[arm],4,True)
            candidates={'joint_only':None}
            if archive:
                phase='static' if arm=='static' else 'period'
                candidates['root_loss']={bool(int(k[0])):np.asarray(v) for k,v in archive['models'][f'{B}/{phase}'].items() if k.endswith('/root_loss')}
            for label,co in candidates.items():
                c=estimate(ctx,ps,co,np.asarray(cfg['median']))
                mae=abs(ctx.work.from_cache(c)[ctx.work.target]-r['_y']).mean()
                rows.append(dict(subject=r['subject_id'],B=B,method=label,mae=mae))
    write(dest/'estimator.csv',rows)
if __name__=='__main__':
    for name in sys.argv[1:] or COHORTS:run(name)
