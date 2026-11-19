"""Plots from summary.csv only: measured test results, no simulated plot data.

Contract: compare six mechanisms on the full budget grid, preserve paired
inference in paired_mae.csv, and display primary-pair pointwise two-layer CIs.
Four independent single-panel comparison figures; report exports, no journal
submission target. n = patients (4 or 5); R=20 mechanism repetitions.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from .core import METHODS


def figures(out):
    out=Path(out)
    cfg=json.loads((out/'config.json').read_text())
    rows=list(csv.DictReader((out/'summary.csv').open()))
    target=out/'figures'
    target.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial'],
                         'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    colors=['#777777','#0072B2','#56B4E9','#D55E00','#009E73','#CC79A7']
    notes={}
    for template in cfg['templates']:
        selected=[r for r in rows if r['template']==template]
        if not selected:
            continue
        fig,ax=plt.subplots(figsize=(7.2,4.8),layout='constrained')
        for method,color in zip(METHODS,colors):
            rr=sorted((r for r in selected if r['method']==method),key=lambda r:float(r['B']))
            x=np.array([float(r['B']) for r in rr])
            y=np.array([float(r['mae']) for r in rr])
            assert np.all(x>0) and np.all(np.diff(x)>0)
            ax.plot(x,y,label=method,color=color,lw=2 if method=='M4-carry' else 1.4,
                    ls='--' if method in ('M4','B2-pub') else '-')
            if method in ('M4-carry','B2-val'):
                lo=np.array([float(r['mae_two_layer_lo']) for r in rr])
                hi=np.array([float(r['mae_two_layer_hi']) for r in rr])
                assert np.all(lo<=hi)
                ax.fill_between(x,lo,hi,color=color,alpha=.12,lw=0)
        n=int(selected[0]['n_patients'])
        ax.set(xlabel='Session privacy budget B',ylabel='Final-task MAE (all 40 turns)',
               title=f'{template}: {n} test patients, R={cfg["repeats"]}')
        ax.set_xscale('log')
        ax.set_xticks([80,160,320,640,960],labels=['80','160','320','640','960'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.18)
        ax.legend(fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False)
        fig.savefig(target/f'{template}_mae.png',dpi=600)
        fig.savefig(target/f'{template}_mae.pdf')
        fig.savefig(target/f'{template}_mae.svg')
        plt.close(fig)
        notes[template]=dict(archetype='single quantitative comparison panel',n=n,R=cfg['repeats'],
                             center='patient-weighted mean all-turn final-task MAE',
                             uncertainty='pointwise two-layer patient+mechanism 95% bootstrap CI; primary pair only',
                             omitted_bands='B0/B2-pub/M4/replan bands in summary.csv; omitted from picture for legibility',
                             inference='paired differences and intervals in paired_mae.csv; no multiple-comparison correction',
                             source='summary.csv; no rows or budget points omitted',
                             alignment='NOT APPLICABLE: one axes per independent figure',
                             export='7.2 by 4.8 inches, PDF/SVG editable text, PNG600dpi for internal report')
    (target/'figure_notes.json').write_text(json.dumps(notes,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    figures(parser.parse_args().output)
