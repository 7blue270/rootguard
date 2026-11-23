"""Prospective transfer of the ceiling-budget rule to disjoint three-batch stays."""
import csv
from pathlib import Path
from zipfile import ZipFile
import json
import numpy as np
from data_mine.build_physionet2019 import record,stream,window
from .study import (OUT,HERE,ORIGINAL,OLDPLANS,SEED,STATIC,BUDGETS,Branch,replace,asdict,order,
                    dump,prep,sim,losses,data,Context,write,ledger,utility,cumulative,attack_result)

def main():
    src=Path('data_mine/finite_cache_physionet2019')
    oldids=set(json.loads((src/'split.json').read_text()))
    limit=960.
    original=json.loads((OUT/'PhysioNet-frozen.json').read_text())
    frozen={k:{**p,'route':ORIGINAL[k]['dynamic'] if float(k.split('/')[0])==limit else p['m8']} for k,p in original.items()}
    dump(OUT/'ceiling-frozen.json',frozen)
    dump(OUT/'transfer_protocol.json',dict(eligibility='exactly three later measurement batches, both roots remeasured; previously excluded; all B stays; no outcome-based selection',
        prior='original 48 A refs',threshold='original training median 5.6061',primary='at B=960 high vs periodic and base M8; full and low MAE safety; no refit of candidates',forced='at top anchor always use original fixed-count dynamic; transfer triggered by previous 1173-patient result'))
    archive=Path('data_mine/training.zip')
    panel={'A':[],'B':[]}
    with ZipFile(archive) as z:
        for member in sorted(z.namelist()):
            p=Path(member)
            if len(p.parts)!=3 or p.parts[0]!='training' or p.parts[1] not in ('training_setA','training_setB') or p.suffix!='.psv' or not p.stem[1:].isdigit():continue
            site=p.parts[1][-1];sid=f'physionet2019-{site}-{p.stem}'
            if sid in oldids:continue
            reader=list(csv.DictReader(z.read(member).decode().splitlines(),delimiter='|'))
            w=window(stream(reader))
            if not w:continue
            start,latest,rest=w
            if len(rest)!=3 or len(set().union(*(set(v) for _,v in rest)))!=2:continue
            r=record(sid,site,member,start,latest,rest,int(float(reader[0]['ICULOS'])))
            r['selection']='exactly three later batches, both roots remeasured'
            data.validate(r);panel[site].append(r)
    train=sorted(panel['A'],key=order)[:160];test=sorted(panel['B'],key=order)
    assert not ({r['subject_id'] for r in train+test}&oldids)
    cfg=json.loads((HERE/'original_protocol.json').read_text())
    by={r['subject_id']:r for r in data.load(src,['BUN_CR'])}
    ctx=Context('BUN_CR',[by[s] for s in cfg['references']],SEED);prep(ctx,train+test)
    methods={}
    for B in BUDGETS:
        periodic=min((replace(STATIC,anchor=.5,period=p) for p in (1,2,3)),key=lambda b:losses(ctx,train,B,b).mean())
        p=original[f'{B}/0']; d=Branch(**ORIGINAL[f'{B}/0']['dynamic'])
        methods[B]=dict(static=STATIC,periodic=periodic,m7=Branch(**OLDPLANS['BUN_CR'][str(B)]['branches']['R-new']),
            m8_original=Branch(**p['m8_original']),m8_mean=Branch(**p['m8_mean']),m8=Branch(**p['m8']),forced=d,
            ceiling=Branch(**frozen[f'{B}/0']['route']),native_static=Branch(1.),native_periodic=replace(periodic,calibration='none',shrink=0.))
    dump(OUT/'three_batches-frozen.json',{str(b):{m:asdict(v) for m,v in plans.items()} for b,plans in methods.items()})
    scores=[];privacy=[]
    for i,r in enumerate(test):
        for B,arms in methods.items():
            cache={}
            for m,b in arms.items():
                if b not in cache:
                    simout=sim(ctx,r,B,b,4,True);ledger(simout,B,b)
                    u={k:float(v.mean()) for k,v in utility(ctx,r,simout).items()}
                    pr={}
                    if i<32 and b.calibration=='marginal':
                        pr={k:v.mean() for k,v in cumulative(ctx,r,simout).items() if k.endswith('all_hit')}
                        for z in attack_result(ctx,r,simout,b):
                            key=z['attack']+'_'+z['prior'];pr[key]=pr.get(key,0)+25*z['success_005']
                    cache[b]=u,pr
                u,pr=cache[b];meta=dict(dataset='three_batches',template='BUN_CR',B=B,subject=r['subject_id'],method=m,high=int(r['_span']>cfg['threshold']))
                scores.append({**meta,**u})
                if pr:privacy.append({**meta,**pr})
    write(OUT/'three_batches-paths.csv',scores);write(OUT/'three_batches-privacy.csv',privacy)
    print('three batches',len(panel['A']),len(test),flush=True)
if __name__=='__main__':main()
