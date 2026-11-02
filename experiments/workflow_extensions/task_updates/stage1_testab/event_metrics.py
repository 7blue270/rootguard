"""Frozen strict event protocol: observe -> refresh -> answer in round t."""
import numpy as np
from common import original,classes,events,THRESHOLDS


def measure(d,cache,ledger):
    truth,gt,e=events(d);pred_value=original.compute(d['name'],cache);pred=classes(d['name'],pred_value)
    N,R,T=pred.shape;reports=np.zeros((N,R,T),bool);reports[:,:,1:]=pred[:,:,1:]!=pred[:,:,:-1]
    hits=np.zeros((3,N));recover=np.zeros((3,N));den=e.sum(axis=1)*R
    group_names=['adjacent','threshold_near','persistent'];gden={g:np.zeros(N) for g in group_names};ghit={g:np.zeros(N) for g in group_names}
    delay_sum=np.zeros(N)
    for t in range(1,T):
        if not e[:,t].any():continue
        later=np.zeros(N,int)+T
        for j in range(t+1,T):later=np.where((later==T)&e[:,j],j,later)
        old=gt[:,t-1,None];new=gt[:,t,None];pre=pred[:,:,t-1]==old
        success=np.zeros((N,R),bool);recovered=np.zeros((N,R),bool)
        for lag in range(3):
            j=t+lag
            if j<T:
                available=e[:,t,None]&(j<later[:,None])
                now=available&pre&(pred[:,:,j-1]==old)&(pred[:,:,j]==new)&reports[:,:,j]
                first=now&~success;delay_sum+=lag*first.sum(axis=1);success|=now
                recovered|=available&(pred[:,:,j]==new)
            hits[lag]+=success.sum(axis=1);recover[lag]+=recovered.sum(axis=1)
        previous=np.zeros(N,int)-T
        for j in range(1,t):previous=np.where(e[:,j],j,previous)
        b=np.array(THRESHOLDS[d['name']]);margin=np.minimum(np.min(abs(truth[:,t-1,None]-b)/b,axis=1),np.min(abs(truth[:,t,None]-b)/b,axis=1))
        masks={'adjacent':e[:,t]&((t-previous==1)|(later-t==1)),
               'threshold_near':e[:,t]&(margin<=.02),'persistent':e[:,t]&(later-t>=3)}
        for group,mask in masks.items():gden[group]+=mask*R;ghit[group]+=(success&mask[:,None]).sum(axis=1)
    report_count=reports.sum(axis=(1,2));stable=~e;stable[:,0]=False
    raw_stable_changes=(reports&stable[:,None]).sum(axis=(1,2));stable_den=stable.sum(axis=1)*R
    # Accepted delayed responses lie in a true-stable round because matching
    # stops before the next true event. They are TP, not false alarms.
    false_stable=raw_stable_changes-(hits[2]-hits[0])
    assert np.all(false_stable>=-1e-8)
    distance=abs(pred-gt[:,None,:])/2;wrong=pred!=gt[:,None,:]
    age=np.zeros((N,R));aoii=np.zeros((N,R))
    for t in range(T):age=np.where(wrong[:,:,t],age+1,0);aoii+=age*distance[:,:,t]
    patient=dict(error=abs(pred_value-truth[:,None]).mean(axis=(1,2)),truth=abs(truth).mean(axis=1),
        rce=distance.mean(axis=(1,2)),cmr=wrong.mean(axis=(1,2)),event_den=den,report_count=report_count,
        stable_false=false_stable,stable_raw=raw_stable_changes,stable_den=stable_den,delay_sum=delay_sum,aoii=(aoii/T).mean(axis=1),
        refreshes=ledger['flags'].sum(axis=2).mean(axis=1),calibration_cost=ledger['spent'].mean(axis=1))
    for lag in range(3):patient['hit'+str(lag)]=hits[lag];patient['recovery'+str(lag)]=recover[lag]
    for g in group_names:patient[g+'_den']=gden[g];patient[g+'_hit2']=ghit[g]
    total=float(den.sum());reported=float(report_count.sum());h2=float(hits[2].sum())
    row=dict(wmape=float(100*patient['error'].sum()/patient['truth'].sum()),rce=float(100*patient['rce'].mean()),cmr=float(100*patient['cmr'].mean()),
        events=int(e.sum()),event_patients=int((e.sum(axis=1)>0).sum()),
        precision2=None if not reported else 100*h2/reported,
        f1_2=None if not total else 200*h2/(total+reported),
        delay_success2=None if not h2 else float(delay_sum.sum()/h2),
        failure2=None if not total else 100*(1-h2/total),
        false_change=None if not stable_den.sum() else float(100*false_stable.sum()/stable_den.sum()),
        stable_output_change_rate=None if not stable_den.sum() else float(100*raw_stable_changes.sum()/stable_den.sum()),
        false_events=float((report_count-hits[2]).sum()/R),report_events=float(report_count.sum()/R),
        refreshes=float(patient['refreshes'].mean()),spent_max=float(ledger['spent'].max()) if ledger['dp'] else None,spent_mean=float(ledger['spent'].mean()) if ledger['dp'] else None,
        nominal_calibration_cost=float(ledger['spent'].mean()),
        budget_bound=ledger['bound'],dp=ledger['dp'],aoii=float(patient['aoii'].mean()))
    for lag in range(3):
        row['recall'+str(lag)]=None if not total else float(100*hits[lag].sum()/total)
        row['recovery'+str(lag)]=None if not total else float(100*recover[lag].sum()/total)
    for g in group_names:row[g+'_recall2']=None if not gden[g].sum() else float(100*ghit[g].sum()/gden[g].sum())
    assert h2<=total+1e-8 and h2<=reported+1e-8
    return row,patient
