"""One global L2 scale, deduplicating original lab rows across templates."""
from collections import defaultdict
import numpy as np
from experiments.workflow_extensions.finite_cache.workload import Workload


def estimate_delta(records,excluded_subject=None):
    measurements={};excluded=set();seen=0;domains={}
    for rec in records:
        sid=str(rec['subject_id'])
        if sid==str(excluded_subject):excluded.add(sid);continue
        w=Workload(rec['template'])
        domains.update({r:(w.lo[i],w.width[i]) for i,r in enumerate(w.roots)})
        sources=[rec['initial_source_rows']]+[e['source_rows'] for e in rec['events']]
        for source in sources:
            for root,row in source.items():
                seen+=1;key=(sid,str(row['labevent_id']))
                value=dict(subject=sid,admission=str(row['hadm_id']),root=root,
                           time=row['charttime'],available=row.get('storetime') or row['charttime'],
                           value=float(row['valuenum']),id=str(row['labevent_id']))
                if key in measurements:assert measurements[key]==value
                measurements[key]=value
    # Resolve same-root/same-charttime measurements deterministically using
    # the latest recorded result, consistently with source data preparation.
    bytime={}
    for m in measurements.values():
        key=m['subject'],m['admission'],m['root'],m['time']
        if key not in bytime or (m['available'],int(m['id']))>(bytime[key]['available'],int(bytime[key]['id'])):bytime[key]=m
    opportunity=defaultdict(set);series=defaultdict(list)
    for m in bytime.values():
        opportunity[m['subject'],m['admission']].add(m['time'])
        series[m['subject'],m['admission'],m['root']].append(m)
    positions={key:{t:i for i,t in enumerate(sorted(times))} for key,times in opportunity.items()}
    edges=[];patient_root=defaultdict(list)
    for (sid,admission,root),ss in series.items():
        ss.sort(key=lambda m:m['time']);lo,width=domains[root]
        for a,b in zip(ss,ss[1:]):
            gap=positions[sid,admission][b['time']]-positions[sid,admission][a['time']]
            assert gap>0
            change=abs(np.clip((b['value']-lo)/width,0,1)-np.clip((a['value']-lo)/width,0,1))/gap
            patient_root[sid,root].append(change)
            edges.append(dict(subject=sid,admission=admission,root=root,from_id=a['id'],to_id=b['id'],opportunity_gap=gap,normalized_change_per_opportunity=float(change)))
    # First equal-weight roots within each patient, then equal-weight patients.
    # The same root across multiple admissions contributes one combined series
    # mean, rather than receiving extra weight for appearing in many templates.
    patients=defaultdict(list)
    for (sid,root),values in patient_root.items():patients[sid].append(float(np.mean(values)))
    per_patient={sid:float(np.mean(values)) for sid,values in patients.items()}
    assert per_patient and str(excluded_subject) not in per_patient
    return dict(delta=float(np.mean(list(per_patient.values()))),n_patients=len(per_patient),
                subjects=sorted(per_patient),raw_row_occurrences=seen,unique_source_rows=len(measurements),
                unique_root_measurements=len(bytime),valid_differences=len(edges),
                patient_root_series=len(patient_root),excluded_subject=str(excluded_subject),
                patient_means=per_patient,age='not estimated: no longitudinal age measurement; fixed admission root',
                interval='unique observed lab charttime opportunities across retained templates within admission',
                aggregation='differences within patient/root, roots within patient, then patients equally'),edges
