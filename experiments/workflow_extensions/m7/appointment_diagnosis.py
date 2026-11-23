"""Ordered offline counterfactuals with identical anchors, roots and calendars."""
import numpy as np
from .appointment_study import OUT,STATIC,DAT,BUDGETS,SEED,Branch,Context,setup,read,write,execute
from ..finite_cache import data

def main():
    parts,_,_=setup(DAT)
    plans=read(OUT/'frozen.json')
    out=[]
    for t,p in parts.items():
        ctx=Context(t,p['train'],SEED)
        for B in BUDGETS:
            branch=Branch(**plans[t][str(B)]['win'])
            rows=[]
            for rec in p['test']:
                u=execute(rec,B,branch,ctx,'test',20,True)
                s=execute(rec,B,STATIC,ctx,'test',20)
                raw=data.replay(rec)
                truth=ctx.work.evaluate(raw)[ctx.work.target]
                norm=ctx.work.normalize(raw)
                def mae(stream):return np.mean(abs(ctx.work.from_cache(stream)[ctx.work.target]-truth))
                old=np.broadcast_to(u['anchor_service'][:,None],u['caches'].shape)
                ideal=old.copy()
                h=u['histories'][0];turn=h['turn']-1
                for i in branch.fixed_roots:ideal[:,turn:,i]=norm[turn,i]
                a,b,c,d=[mae(v) for v in (s['caches'],old,ideal,u['caches'])]
                diff=np.abs(np.diff(norm,axis=0))>.05
                events=[e['turn']-1 for e in rec['events'] if e['turn']>1]
                last=np.zeros(len(ctx.roots))
                aoi=[]
                for j in range(len(norm)):
                    if j==turn:last[list(branch.fixed_roots)]=j
                    aoi.append(np.mean(j-last))
                # Same-round capture on a true >5%-domain root jump; no future grace window.
                missed=diff.sum()-diff[turn-1,list(branch.fixed_roots)].sum()
                rows.append([b-a,b-c,d-c,d-a,missed,diff.sum(),np.mean(aoi),np.mean((raw<ctx.work.lo)|(raw>ctx.work.hi)),turn+1])
                assert abs((b-a)-(b-c)+(d-c)-(d-a))<1e-9
            a,f,n,d,miss,events,age,outside,at=np.mean(rows,axis=0)
            out.append(dict(template=t,B=B,anchor_cost=a,perfect_selected_freshness=f,refresh_measurement_cost=n,
                            delta_static=d,miss_rate=miss/events if events else None,significant_root_events=int(round(events*len(rows))),
                            mean_root_age_turns=age,outside_domain_fraction=outside,avg_appointment_turn=at))
    write(OUT/'diagnosis.csv',out)

if __name__=='__main__':
    main()
