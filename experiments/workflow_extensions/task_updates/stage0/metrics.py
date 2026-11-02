"""Paper Eq.(4) versus ordinary class mismatch; no release mechanism."""
import numpy as np


def classification_metrics(truth,pred,C):
    truth=np.asarray(truth);pred=np.asarray(pred)
    if C<2 or truth.shape!=pred.shape or not truth.size:raise ValueError('Invalid class metric inputs')
    for a in [truth,pred]:
        if not np.all(np.isfinite(a)) or np.any(a!=np.floor(a)) or np.any(a<0) or np.any(a>=C):
            raise ValueError('Classes must be integers in [0,C)')
    return dict(rce=float(100*np.mean(np.abs(pred-truth)/(C-1))),
                cmr=float(100*np.mean(pred!=truth)))
