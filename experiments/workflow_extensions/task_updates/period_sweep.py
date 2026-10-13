"""Predefined period tradeoffs on test data; not test-set parameter selection."""
import argparse,json,time
from pathlib import Path
import numpy as np
from dataset import make,observed,split
from release import run,metrics
from adapter import original


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();out=args.run
    target=out/'period_tradeoffs.json'
    if target.exists():raise ValueError('period sweep output already exists')
    cfg=json.loads((out/'config.json').read_text());meta=json.loads((out/'metadata.json').read_text());rows=[];start=time.time()
    tracks=[('metric',b,b) for b in cfg['budgets']]+[('standard',e,e/999) for e in cfg['standard_eps']]
    for di,did in enumerate(meta['datasets']):
        if did.startswith('real'):d=observed()
        else:
            name,rest=did[4:].split('_',1)
            scenario,h=rest.rsplit('_',1)
            d=make(name,int(h[1:]),scenario,cfg['amplitude'])
        _,test=split(d);N,H,k=test['x'].shape;R=cfg['test_repeats']
        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],di,1]))
        u=rng.random((N,R,H,k));q=rng.laplace(size=(N,R,H));t=rng.laplace(size=(N,R,4))
        for track,label,B in tracks:
            for P in cfg['periods']:
                if P>H:continue
                z,l=run(test,B,dict(method='period',period=P),u,q,t);a,_=metrics(test,z,l)
                rows.append(dict(dataset=did,track=track,budget=label,period=P,**a))
        print(did,'sweep complete',flush=True)
    target.write_text(json.dumps(rows,indent=2,allow_nan=False))
    (out/'period_sweep_metadata.json').write_text(json.dumps(dict(cells=len(rows),seconds=time.time()-start,
          source_sha256=original.sha(Path(__file__)),test_selection=False),indent=2))


if __name__=='__main__':main()
