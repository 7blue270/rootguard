"""Known-offset scalar reconstruction diagnostic; not a DP proof or real attack."""
import argparse
import csv
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261018)
    m, n, repeats = 1000, 100, 30
    candidates = np.arange(200, 801, 5)
    latent = rng.choice(candidates, n)
    grid = np.arange(m)
    rows = []
    for H in [4, 8, 16, 32]:
        offsets = np.rint(40*np.sin(np.linspace(0, 2*np.pi, H))).astype(int)
        for B in [.1, .3, .7]:
            for method in ['cache', 'fixed_total', 'fixed_per_version']:
                eps = B/H if method == 'fixed_total' else B
                centers = candidates[:, None] + offsets[None, :]
                # Candidate x time x output likelihood under the actual PMF.
                weights = np.exp(-eps*np.abs(centers[:, :, None]-grid)/2)
                probs = weights/weights.sum(axis=-1, keepdims=True)
                lookup = {int(c): i for i,c in enumerate(candidates)}
                draw_probs = probs[[lookup[int(c)] for c in latent]]
                if method == 'cache': draw_probs = draw_probs[:, :1]
                draws = (np.cumsum(draw_probs,axis=-1)[:,None] < rng.random((n,repeats,draw_probs.shape[1],1))).sum(axis=-1)
                loglik = np.zeros((n,repeats,len(candidates)))
                for t in range(draws.shape[-1]):
                    loglik += np.log(probs[:,t,draws[:,:,t]].transpose(1,2,0))
                estimate = candidates[np.argmax(loglik,axis=-1)]
                rows.append(dict(H=H,B=B,method=method,
                    charged_budget=B*H if method=='fixed_per_version' else B,
                    mae_index=float(np.abs(estimate-latent[:,None]).mean()),
                    within_5_index_pct=float(100*(np.abs(estimate-latent[:,None])<=5).mean()),
                    unique_releases=1 if method=='cache' else H))
    with (args.out/'reconstruction.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.out/'PROTOCOL.md').write_text(
        '# Reconstruction diagnostic\n\n100 latent scalar values, 30 repetitions, '
        'public known offsets, uniform candidate prior, exact likelihood MAP. '
        'Cache means only the first sanitized value is released repeatedly; '
        'it does not release current private changes. Fresh mechanisms emit '
        'changing values; attacker knows offsets. Separate nonpaired random draws. '
        'This is an attacker-favorable scalar diagnostic, not evidence of full '
        'medical-trajectory reconstruction or a privacy proof.\n')
    print(f'{len(rows)} reconstruction cells complete')


if __name__ == '__main__': main()
