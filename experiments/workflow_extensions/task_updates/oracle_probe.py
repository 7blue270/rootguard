"""Offline counterfactual: same Oracle schedule, remove only refresh noise.

Both arms are non-DP diagnostics. Not a deployable release mechanism.
"""
import argparse,json
from pathlib import Path
import numpy as np
from dataset import observed,split
from release import run,metrics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();out=args.run
    cfg=json.loads((out/'config.json').read_text());meta=json.loads((out/'metadata.json').read_text())
    d=split(observed())[1];N,H,k=d['x'].shape;R=cfg['test_repeats']
    di=list(meta['datasets']).index('real_ANEMIA_H8')
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],di,1]))
    u=rng.random((N,R,H,k));q=rng.laplace(size=(N,R,H));t=rng.laplace(size=(N,R,4));rows=[]
    for B in cfg['budgets']:
        for C in [2,4]:
            z,l=run(d,B,dict(method='oracle',C=C),u,q,t)
            clean=z.copy()  # keep initial private release identical
            for j in range(1,H):
                clean[:,:,j]=np.where(l['refresh_flags'][:,:,j,None],d['x'][:,None,j],clean[:,:,j-1])
            for label,cache in [('noisy_refresh',z),('exact_refresh_non_DP',clean)]:
                a,_=metrics(d,cache,l);rows.append(dict(B=B,C=C,arm=label,**a))
    (out/'oracle_noise_ablation.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    print(json.dumps(rows))


if __name__=='__main__':main()
