"""Threshold sensitivity for meaningful attribute changes, offline evaluator."""
import numpy as np
from adapter import original
from dataset import make,observed,split


def details(cfg):
    rows=[]
    data=[]
    for name in cfg['templates']:
        for H in cfg['horizons']:
            for scenario in cfg['scenarios']:
                data.append((f'syn_{name}_{scenario}_H{H}',make(name,H,scenario,cfg['amplitude'])))
    data.append(('real_ANEMIA_H8',observed()))
    for did,d in data:
        _,test=split(d);x=test['x'];p=np.array(original.EXPONENTS[d['name']])
        dx=np.diff(np.log(x),axis=1);attr=np.max(abs(dx),axis=-1);task=abs((dx*p).sum(axis=-1))
        for at in [.01,.02,.05]:
            for tt in [.01,.02,.05]:
                mask=attr>np.log1p(at);den=int(mask.sum())
                rows.append(dict(dataset=did,attribute_tolerance=at,task_tolerance=tt,attribute_events=den,
                    small_task_pct=None if not den else float(100*(task[mask]<=np.log1p(tt)).mean())))
    return rows
