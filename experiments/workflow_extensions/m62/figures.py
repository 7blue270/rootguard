"""Python quantitative figures: predictive calibration and net utility gains.

Figure contract: ask whether pre-session information identifies the better side
and achieves net benefit. Use all 24 cells and all actual observations via
patient-equal summaries; no practical eta invented. Internal-report export,
editable PDF/SVG plus PNG; single axes, global-patient paired nested 95% CI.
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':9,'axes.titlesize':11,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
SKILL=Path('/root/.codex/skills/nature-figure/scripts')
COLORS={'Observed':'#242424','L0':'#2271B2','L1':'#D55E00','L2':'#009E73','Local-L0':'#888888','TrainLookup':'#AA4499','B2-val':'#CCBB44'}


def values(rows,key):return np.array([float(r[key]) if r.get(key) not in (None,'') else np.nan for r in rows])


def save(fig,ax,path):
    fig.canvas.draw()
    sys.path.insert(0,str(SKILL))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig,json_out=str(path)+'.alignment.json',overlay_svg=str(path)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(str(path)+'.pdf');fig.savefig(str(path)+'.svg');fig.savefig(str(path)+'.png',dpi=300)
    for script,extra in [('audit_pdf_text.py',['--json']),
                         ('audit_figure_collisions.py',['--json-out',str(path)+'.collision-qa.json','--overlay-pdf',str(path)+'.collision-overlay.pdf'])]:
        cmd=[sys.executable,str(SKILL/script),str(path)+'.pdf']+extra
        env=os.environ.copy()
        # Reuse the already installed local PDF-analysis dependency without
        # altering the scientific Python environment or simulation imports.
        if Path('/tmp/rootguard-analysis-deps/pymupdf').exists():
            env['PYTHONPATH']='/tmp/rootguard-analysis-deps'+os.pathsep+env.get('PYTHONPATH','')
        result=subprocess.run(cmd,capture_output=True,text=True,env=env)
        if script=='audit_pdf_text.py':Path(str(path)+'.text-qa.json').write_text(result.stdout)
        (Path(str(path)+'.'+script+'.log')).write_text(result.stdout+'\n'+result.stderr)
        if result.returncode:raise RuntimeError('Figure QA failed: '+str(path)+' '+script+'\n'+result.stdout+'\n'+result.stderr)
    plt.close(fig)


def axes(title):
    fig,ax=plt.subplots(figsize=(7.1,4.6))
    fig.subplots_adjust(left=.15,right=.97,bottom=.19,top=.73)
    fig.suptitle(title,y=.98,fontsize=11)
    ax.set_xlabel('Session privacy budget B');ax.axhline(0,color='#555555',linewidth=.7,zorder=0)
    return fig,ax


def plot(out):
    out=Path(out);cfg=json.loads((out/'config.json').read_text());folder=out/'figures';folder.mkdir(exist_ok=True)
    cells=list(csv.DictReader((out/'table1_cell_decisions.csv').open()));effects=list(csv.DictReader((out/'table2_baseline_gains.csv').open()))
    sources=[]
    for t in cfg['templates']:
        fig,ax=axes(t+': can the public model predict S versus D?')
        base=sorted([r for r in cells if r['template']==t and r['selector']=='L0'],key=lambda r:float(r['B']));x=values(base,'B');g=values(base,'G_true')
        ax.fill_between(x,values(base,'G_true_two_layer_lo'),values(base,'G_true_two_layer_hi'),color=COLORS['Observed'],alpha=.12)
        ax.plot(x,g,'o-',color=COLORS['Observed'],label='Observed G',linewidth=2,zorder=5)
        maxima=[np.max(abs(g))]
        for method in ['L0','L1','L2','Local-L0']:
            rr=sorted([r for r in cells if r['template']==t and r['selector']==method],key=lambda r:float(r['B']));y=values(rr,'predicted_gain');maxima.append(np.max(abs(y)))
            ax.plot(x,y,marker='s' if method=='Local-L0' else 'o',linestyle=':' if method=='Local-L0' else '--',color=COLORS[method],label=method,linewidth=1.2,markersize=3)
        symlog=max(maxima)>10*max(maxima[0],1e-12)
        if symlog:ax.set_yscale('symlog',linthresh=max(maxima[0]/5,1e-3))
        ax.set_ylabel('S minus D MAE'+(' (symlog scale)' if symlog else ''))
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.14,.92),ncol=3,frameon=False)
        save(fig,ax,folder/(t+'_prediction'))
        fig,ax=axes(t+': net utility gain relative to trained default')
        for method in ['L0','L1','L2','TrainLookup','B2-val']:
            if method=='B2-val':
                # Baseline-vs-L0 reversed: gain(B2 vs default) is directly
                # derived from the same paired contrast, not a new noise draw.
                aa=sorted([r for r in effects if r['template']==t and r['selector']=='L0' and r['reference']=='Default'],key=lambda r:float(r['B']))
                bb=sorted([r for r in effects if r['template']==t and r['selector']=='L0' and r['reference']=='B2-val'],key=lambda r:float(r['B']))
                x=values(aa,'B');y=values(aa,'utility_gain')-values(bb,'utility_gain')
                ax.plot(x,y,'s--',color=COLORS[method],label=method,linewidth=1.2,markersize=3);continue
            rr=sorted([r for r in effects if r['template']==t and r['selector']==method and r['reference']=='Default'],key=lambda r:float(r['B']))
            x=values(rr,'B');y=values(rr,'utility_gain')
            if method=='L0':ax.fill_between(x,values(rr,'utility_gain_two_layer_lo'),values(rr,'utility_gain_two_layer_hi'),color=COLORS[method],alpha=.13)
            ax.plot(x,y,'o-',color=COLORS[method],label=method,linewidth=2 if method=='L0' else 1.1,markersize=3)
        ax.set_ylabel('Default MAE minus selector MAE')
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.14,.92),ncol=3,frameon=False)
        save(fig,ax,folder/(t+'_net_gain'))
        sources.append(dict(template=t,source_cells='table1_cell_decisions.csv',source_gains='table2_baseline_gains.csv',prediction_symlog=bool(symlog),
            uncertainty='patient-plus-mechanism paired percentile 95%; observed G and primary L0 net gain shaded; other lines point estimates',
            qa='single-axis alignment not applicable; PDF font and collision audits performed; inspect full PNGs'))
    (folder/'figure_contract.json').write_text(json.dumps(sources,indent=2)+'\n')
