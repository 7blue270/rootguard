"""Follow-up: evaluate one conditional appointment, retaining the frozen M7 fallback."""
from dataclasses import asdict,replace
import numpy as np
from .appointment_study import (OUT,OLD,DAT,BUDGETS,SEED,STATIC,Branch,Context,
                                setup,read,write,risk,dump,confirm,route_choice)
from .appointment_analysis import main as analyze

def run():
    out=OUT.parent/'appointment_gate_20261119'
    out.mkdir(exist_ok=True)
    parts,_,_=setup(DAT)
    plans=read(OUT/'frozen.json')
    old=read(OLD/'frozen.json')
    logs=[]
    for t,p in parts.items():
        ctx=Context(t,p['train'],SEED)
        for B in BUDGETS:
            slot=plans[t][str(B)]
            forced=Branch(**slot['win'])
            gate=replace(forced,gate='workload')
            incumbent=Branch(**old[t][str(B)]['branches']['R-new'])
            controls=[incumbent,STATIC] if incumbent!=STATIC else [STATIC]
            proposal=risk(ctx,p['validation'],B,gate,'m7-refresh-validation')
            scores=[risk(ctx,p['validation'],B,b,'m7-refresh-validation') for b in controls]
            accept=all(route_choice(ref,proposal) for ref in scores)
            slot.update(win=asdict(gate),pack=asdict(forced),full=asdict(replace(gate,fixed_roots=ctx.roots)),
                        dynamic=accept,gain=float((scores[0]['mae']-proposal['mae']).mean()))
            logs.append(dict(template=t,B=B,accepted=accept,val_gain_vs_old=slot['gain'],
                             val_gain_vs_static=float((scores[-1]['mae']-proposal['mae']).mean())))
    dump(out/'frozen.json',plans)
    write(out/'iterations.csv',logs)
    print('validation accepted',sum(x['accepted'] for x in logs),flush=True)
    confirm(out,fallback=True)
    analyze(out)

if __name__=='__main__':
    run()
