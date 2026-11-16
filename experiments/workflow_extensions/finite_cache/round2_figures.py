"""Python-only internal quantitative figures; retain all frontier/censor records."""
import json
import os
from pathlib import Path
import sys
import numpy as np


def render(output,cross,info,periods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output=Path(output)
    skill=Path(os.environ.get('NATURE_FIGURE_SKILL_ROOT',Path.home()/'.codex/skills/nature-figure'))
    sys.path.insert(0,str(skill/'scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial','Helvetica'],'font.size':9,'axes.titlesize':10,
                         'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    contract={'question':'Does frozen-training drift accompany M1/B2 budget-frontier differences across three templates?',
              'claim':'controlled internal comparison; descriptive three-template relation, no regression/causal claim',
              'archetype':'quantitative grid; single-panel scatter and separate sensitivity diagnostics',
              'unit':'patients; 20 mechanism repeats; no seed-as-patient inference',
              'source':'cross_template.csv; period_sensitivity.csv','interval':'paired hierarchical percentile95, conditional both reached',
              'exports':['png','pdf','svg'],'dimensions_inches':[7.2,4.8],
              'censoring':'all rows retained; open markers show scan-window LOWER LIMITS only; no finite ratio imputed'}
    (output/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    def save(fig,stem):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig,json_out=str(output/(stem+'.alignment.json')),
                                          overlay_svg=str(output/(stem+'.alignment.svg')),strict=True)
        fig.savefig(output/(stem+'.png'),dpi=600,bbox_inches='tight')
        fig.savefig(output/(stem+'.pdf'),bbox_inches='tight')
        fig.savefig(output/(stem+'.svg'),bbox_inches='tight')
        plt.close(fig)
    colors={'ANEMIA':'#2878B5','FIB4':'#D97A2B','BUN_CR':'#2A9D8F'}
    markers={320:'o',640:'s',960:'^'}
    fig,ax=plt.subplots(figsize=(7.2,4.8));fig.subplots_adjust(bottom=.28,top=.80)
    finite=0;censored=[];limits=[]
    for r in cross:
        x=r['drift'];ci=r['hierarchical_ratio_ci_conditional'];anchor=(r['tau_reference_budgets'] or [320])[0]
        if r['ratio'] is None:
            bound=r.get('scan_window_ratio_lower_limit')
            if bound is not None:
                y=100*(bound-1)
                assert x > 0
                ax.scatter(x,y,s=48,marker=markers[anchor],facecolor='none',edgecolor=colors[r['template']],linewidth=1.2,zorder=3)
                ax.annotate(f'>{y:.1f}%',(x,y),xytext=(7,4),textcoords='offset points',fontsize=8,color=colors[r['template']])
                limits.append({'template':r['template'],'anchor':anchor,'relative_gap_lower_limit_percent':y})
            else:
                censored.append(f"{r['template']}/{anchor}")
            continue
        assert x > 0, 'Log drift axis requires positive train drift'
        y=100*(r['ratio']-1);finite+=1
        if ci:
            lo,hi=100*(ci[0]-1),100*(ci[1]-1)
            ax.errorbar(x,(lo+hi)/2,yerr=(hi-lo)/2,fmt='none',ecolor=colors[r['template']],alpha=.55,capsize=3,linewidth=1)
        ax.scatter(x,y,s=42,marker=markers[anchor],color=colors[r['template']],edgecolor='white',linewidth=.6,zorder=3)
    ax.axhline(0,color='.35',linewidth=.8,linestyle='--')
    ax.set_xscale('log')
    ordered=sorted(info,key=lambda t:info[t]['train_normalized_drift'])
    ax.set_xticks([info[t]['train_normalized_drift'] for t in ordered],
                 [f"{t}\n{info[t]['train_normalized_drift']:.4f}" for t in ordered])
    from matplotlib.ticker import NullLocator, NullFormatter
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel='Train mean normalized root drift per measurement batch',ylabel='Relative B_min gap vs B2 (%)',
           title='Drift vs M1-a / B2 budget gap (internal pilot)')
    ax.legend(handles=[Line2D([],[],marker=m,linestyle='none',color='.2',label=f'Validation anchor {b}')
                       for b,m in markers.items()],loc='lower center',bbox_to_anchor=(.5,1.12),ncol=3)
    if limits:
        fig.text(.12,.075,'Open markers: scan-window lower limits; M1 not reached by B=960.',fontsize=7,ha='left')
    if censored:
        import textwrap
        fig.text(.12,.015,'\n'.join(textwrap.wrap('Both frontiers unidentified: '+', '.join(censored),90)),fontsize=7,ha='left')
    save(fig,'drift_vs_gap')
    qa={'frontier_rows':len(cross),'finite_ratios_plotted':finite,'scan_window_lower_limits_plotted':limits,'unidentified_not_plotted':censored,
        'reason':'at least one B_min not reached in <=960; no finite ratio identified',
        'sensitivity_panels':{}}
    for template in info:
        fig,ax=plt.subplots(figsize=(7.2,4.8));fig.subplots_adjust(right=.78,bottom=.17,top=.88)
        for P in range(2,13):
            rr=sorted([r for r in periods if r['template']==template and r['split']=='test_period_sensitivity' and r['P']==P],key=lambda r:r['B'])
            x=np.array([r['B'] for r in rr]);y=np.array([r['mae'] for r in rr])
            low=np.array([r['ci_low'] for r in rr]);high=np.array([r['ci_high'] for r in rr])
            assert len(x) and np.all(x > 0) and np.all(low > 0), 'Log sensitivity axes require positive data and interval bounds'
            color=plt.get_cmap('viridis')((P-2)/10)
            ax.plot(x,y,label=f'P={P}',color=color,linewidth=1.2)
            ax.fill_between(x,low,high,color=color,alpha=.08,linewidth=0)
        ax.set(xlabel='Session budget B',ylabel='Final-node MAE',title=f'{template}: B2 period sensitivity (test diagnostic only)')
        ax.set_xscale('log');ax.set_yscale('log')
        ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),title='Period')
        ax.grid(alpha=.15,axis='y')
        save(fig,'period_sensitivity_'+template)
        qa['sensitivity_panels'][template]={'periods':list(range(2,13)),'interval':'all curves hierarchical95',
                                             'selection':'P=7/9/11 diagnostic only'}
    (output/'figure_qa_notes.json').write_text(json.dumps(qa,indent=2)+'\n')
