"""Diagnostic hindsight bound; future truth is never used by service estimators."""
import numpy as np
from .window_study import COHORTS, DEST, load, branches, BUDGETS, sim, windows, write

def main():
    rows=[]
    for label,t in COHORTS:
        large=label=='PhysioNet'
        ctx,_,patients,_,_=load(t,large)
        for r in patients:
            y=r['_y']
            for B in BUDGETS:
                plan,_=branches(r,t,B,large)
                ps=sim(ctx,r,B,plan,4 if large else 20,True)
                segments=list(windows(ctx,ps))
                first=segments[0][0] if segments else len(y)
                anchor=ctx.work.from_cache(ps['anchor_service'])[ctx.work.target]
                fixed=abs(anchor[:,None]-y[:first]).mean(axis=0).sum()/len(y)
                floor=joint=fixed
                for start,end,x,_ in segments:
                    truth=y[start:end]
                    floor+=abs(truth-np.median(truth)).sum()/len(y)
                    pred=ctx.work.from_cache(x)[ctx.work.target]
                    joint+=abs(pred[:,None]-truth).mean(axis=0).sum()/len(y)
                rows.append(dict(dataset=label,subject=r['subject_id'],B=B,fixed=fixed,joint=joint,
                                 ideal_window=floor,anchor_fraction=first/len(y)))
    write(DEST/'headroom.csv',rows)
if __name__=='__main__':main()
