"""Run the frozen BUN/Cr policy on public measurement opportunities."""
import math
from .engine import Branch,run_live
from .routing import Route

def choose(frozen,budget,public):
    if budget<=0:
        raise ValueError('A positive session budget is required')
    density=int(len(public['events'])>=8)
    anchors=sorted({float(k.split('/')[0]) for k in frozen})
    near=min(anchors,key=lambda b:abs(math.log(b/budget)))
    return Branch(**frozen[f'{near}/{density}']['route'])

def run(ctx,frozen,budget,public,read_roots):
    branch=choose(frozen,budget,public)
    route=Route({budget:(branch,branch)},{budget:False},'source-A frozen density rule')
    return run_live(ctx,route,budget,public,read_roots)
