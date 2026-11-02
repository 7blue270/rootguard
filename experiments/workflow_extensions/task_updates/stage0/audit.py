"""Stage 0 ONLY: true category-event audit and paper metric checks.

No sampling, detector, Oracle, or A/B run is called. Existing files are read-only.
"""
import argparse,csv,json,hashlib,datetime,subprocess,sys
from pathlib import Path
from collections import Counter
from decimal import Decimal
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from adapter import old_data,original
from dataset import make,split
from metrics import classification_metrics

THRESHOLDS={'ANEMIA':[32.,36.],'FIB4':[1.30,2.67],'HOMA':[1.,2.5]}


def write_csv(path,rows,fields=None):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)


def describe(values):
    a=np.array(values,dtype=float)
    return None if not len(a) else dict(n=len(a),min=float(a.min()),median=float(np.median(a)),
                    q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),max=float(a.max()))


def subset(d,indices):
    indices=np.array(indices,int)
    return dict(d,x=d['x'][indices],ids=[d['ids'][i] for i in indices],
                audit=dict(d['audit'],timestamps=[d['audit']['timestamps'][i] for i in indices]))


def audit_cohort(cohort,d,source):
    name=d['name'];x=d['x'];N,T,k=x.shape;y=original.compute(name,x);c=original.classes(name,y)
    boundary=np.array(THRESHOLDS[name]);events=[];patients=[];gap_rows=[];panels=[]
    stamps=d['audit'].get('timestamps');counts=[]
    for i,pid in enumerate(d['ids']):
        event_times=np.flatnonzero(c[i,1:]!=c[i,:-1])+1
        counts.append(len(event_times))
        patients.append(dict(cohort=cohort,source=source,template=name,patient=pid,T=T,event_count=len(event_times),
                  initial_class=int(c[i,0]),final_class=int(c[i,-1]),event_rounds=[int(t+1) for t in event_times]))
        for t in range(T):
            nearest=int(np.argmin(abs(y[i,t]-boundary)/boundary))
            panels.append(dict(cohort=cohort,patient=pid,round=t+1,target=float(y[i,t]),category=int(c[i,t]),
                nearest_threshold=float(boundary[nearest]),distance_abs=float(abs(y[i,t]-boundary[nearest])),
                distance_relative=float(abs(y[i,t]-boundary[nearest])/boundary[nearest])))
        for t in event_times:
            a=int(c[i,t-1]);b=int(c[i,t]);crossed=boundary[min(a,b):max(a,b)]
            prev=float(y[i,t-1]);now=float(y[i,t])
            before_dist=float(np.min(abs(prev-crossed)/crossed));after_dist=float(np.min(abs(now-crossed)/crossed))
            previous_time=stamps[i][t-1] if stamps else None;current_time=stamps[i][t] if stamps else None
            observation_gap=None if not stamps else (datetime.datetime.fromisoformat(current_time)-datetime.datetime.fromisoformat(previous_time)).total_seconds()/3600
            events.append(dict(cohort=cohort,source=source,template=name,patient=pid,round=int(t+1),from_class=a,to_class=b,
                direction=f'{a}->{b}',class_distance=abs(b-a),previous_target=prev,current_target=now,
                crossed_thresholds=crossed.tolist(),previous_near_relative=before_dist,current_near_relative=after_dist,
                previous_charttime=previous_time,current_charttime=current_time,observation_gap_hours=observation_gap))
        for a,b in zip(event_times[:-1],event_times[1:]):
            hours=None if not stamps else (datetime.datetime.fromisoformat(stamps[i][b])-datetime.datetime.fromisoformat(stamps[i][a])).total_seconds()/3600
            gap_rows.append(dict(cohort=cohort,patient=pid,from_round=int(a+1),to_round=int(b+1),round_gap=int(b-a),hours_gap=hours))
    active=np.array(counts);active=active[active>0]
    threshold_rows=[]
    for radius in [.01,.02,.05]:
        for label,values in [('all_panels',[r['distance_relative'] for r in panels]),
                ('event_before',[r['previous_near_relative'] for r in events]),('event_after',[r['current_near_relative'] for r in events])]:
            near=sum(v<=radius for v in values)
            threshold_rows.append(dict(cohort=cohort,scope=label,relative_radius=radius,total=len(values),near_count=near,
                    near_pct=None if not values else float(100*near/len(values))))
    decimal_mismatch=None
    if source=='observed':
        exact=np.empty_like(c)
        for i in range(N):
            for t in range(T):
                hb,hct=(Decimal(str(v)) for v in x[i,t,:2]);scaled=100*hb
                exact[i,t]=0 if scaled<32*hct else 2 if scaled>36*hct else 1
        decimal_mismatch=int((exact!=c).sum())
    summary=dict(cohort=cohort,source=source,template=name,T=T,patients=N,events=len(events),event_patients=len(active),
        pass_event_gate=len(events)>=50 and len(active)>=20,
        reason='pass' if len(events)>=50 and len(active)>=20 else 'events<50 and event_patients<20' if len(events)<50 and len(active)<20 else 'events<50' if len(events)<50 else 'event_patients<20',
        directions=dict(Counter(r['direction'] for r in events)),round_distribution=dict(Counter(r['round'] for r in events)),
        active_event_counts=describe(active),event_round_gaps=describe([r['round_gap'] for r in gap_rows]),
        event_hour_gaps=describe([r['hours_gap'] for r in gap_rows if r['hours_gap'] is not None]),
        observation_gap_hours=describe([r['observation_gap_hours'] for r in events if r['observation_gap_hours'] is not None]),
        adjacent_event_pairs=sum(r['round_gap']==1 for r in gap_rows),
        top3_event_share_pct=None if not len(events) else float(100*sum(sorted(counts,reverse=True)[:3])/len(events)),
        decimal_class_mismatches=decimal_mismatch,
        auxiliary_continuous_events=int((abs(np.diff(np.log(y),axis=1))>np.log1p(.02)).sum()))
    return summary,patients,events,gap_rows,threshold_rows,panels


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=False)
    repo=original.REPO
    # Snapshot existing experimental sources; newly created stage0 is excluded.
    before={str(p):original.sha(p) for p in (repo/'experiments/workflow_extensions').rglob('*')
            if p.is_file() and p.suffix in ['.py','.md','.json'] and 'outputs' not in p.parts
            and '__pycache__' not in p.parts and 'stage0' not in p.parts}
    primary_path=HERE.parent.parent/'outputs/task_updates/exploration_20261013/metadata.json'
    previous=json.loads(primary_path.read_text());prior_ids=previous['datasets']['real_ANEMIA_H8']
    real8=old_data.real(8,100);real16=old_data.real(16,100)
    index8={pid:i for i,pid in enumerate(real8['ids'])}
    cohorts=[('real_H8_existing_test',subset(real8,[index8[p] for p in prior_ids['test_ids']]),'observed'),
        ('real_H8_existing_validation',subset(real8,[index8[p] for p in prior_ids['validation_ids']]),'observed'),
        ('real_H8_all_available',real8,'observed'),('real_H16_all_available_unsplit',real16,'observed')]
    matched=[index8[p] for p in real16['ids']]
    cohorts.append(('real_H8_long_cohort_patient_overlap',subset(real8,matched),'observed'))
    cfg=json.loads((HERE.parent/'configs/exploration.json').read_text())
    for name in cfg['templates']:
        for T in cfg['horizons']:
            scenarios=cfg['scenarios']+(['irrelevant'] if name=='ANEMIA' else [])
            for scenario in scenarios:
                d=make(name,T,scenario,cfg['amplitude']);v,t=split(d)
                for part,sd in [('validation',v),('test',t)]:cohorts.append((f'synthetic_{name}_{scenario}_H{T}_{part}',sd,'synthetic'))
    summaries=[];patients=[];events=[];gaps=[];thresholds=[];panels=[]
    for label,d,source in cohorts:
        a,p,e,g,th,pa=audit_cohort(label,d,source)
        summaries.append(a);patients+=p;events+=e;gaps+=g;thresholds+=th;panels+=pa
    for file,value in [('summary',summaries),('real_loader_audit',dict(H8=real8['audit'],H16=real16['audit']))]:
        (out/(file+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False))
    for file,rows in [('patients',patients),('events',events),('event_gaps',gaps),('threshold_proximity',thresholds),('panels',panels)]:write_csv(out/(file+'.csv'),rows)
    static=json.loads((repo/'data/nhanes_benchmark_200.json').read_text())
    availability=[dict(template=name,static_nhanes_records=len(records),observed_longitudinal='CBC H8/H16' if name=='ANEMIA' else 'not assembled in current experiment',
                  synthetic_audited=name in cfg['templates']) for name,records in static.items()]
    write_csv(out/'template_availability.csv',availability)
    fixtures=[dict(name='three_class_mixed',truth=[0,1,2],pred=[1,1,0],C=3),
              dict(name='adjacent',truth=[0],pred=[1],C=3),dict(name='two_step',truth=[0],pred=[2],C=3),
              dict(name='binary',truth=[0,1],pred=[1,1],C=2)]
    for f in fixtures:f.update(classification_metrics(f['truth'],f['pred'],f['C']))
    assert fixtures[0]['rce']==50 and np.isclose(fixtures[0]['cmr'],100*2/3)
    assert fixtures[1]['rce']==50 and fixtures[1]['cmr']==100 and fixtures[2]['rce']==100
    (out/'metric_formula_checks.json').write_text(json.dumps(fixtures,indent=2))
    changed=[p for p,h in before.items() if original.sha(Path(p))!=h]
    assert not changed
    proof=[repo/'utils/risk_class.py',repo/'data/nhanes_benchmark_200.json',repo/'data/holdout_population_means.json',primary_path,
       Path('/home/rootguard_research_audit/paper-extracted.txt'),Path('/home/Anshumaan 等 - 2026 - Dependency-Aware Privacy for Multi-turn Agents.pdf'),old_data.MIMIC_DIR/'labevents.csv.gz']
    provenance=dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        existing_files_checked=len(before),existing_files_changed=changed,
        existing_sources_sha256=before,
        sources_sha256={str(p):original.sha(p) for p in proof+list(HERE.rglob('*.py'))},
        no_AB_run=True,no_private_release_run=True,gate=dict(events=50,event_patients=20),
        H8_H16_patient_overlap=len(set(real8['ids'])&set(real16['ids'])),
        H8_H16_same_patient_different_initial_times=sum(real8['audit']['timestamps'][index8[p]][0]!=real16['audit']['timestamps'][i][0] for i,p in enumerate(real16['ids'])))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    real=[r for r in summaries if r['source']=='observed']
    for r in real:print(json.dumps(r,ensure_ascii=False))
    from report import build
    build(out,summaries,thresholds,provenance)


if __name__=='__main__':main()
