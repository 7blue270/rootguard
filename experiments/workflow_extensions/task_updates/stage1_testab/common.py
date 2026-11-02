import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from adapter import old,old_data,original
import dataset as previous_data
import release as previous_release
import numpy as np
THRESHOLDS={'ANEMIA':[32.,36.],'FIB4':[1.3,2.67],'HOMA':[1.,2.5]}


def classes(name,y):
    a,b=THRESHOLDS[name]
    return np.where(y<a,0,np.where(y>=b if name=='HOMA' else y>b,2,1))


def events(d):
    y=original.compute(d['name'],d['x']);c=classes(d['name'],y)
    e=np.zeros(c.shape,bool);e[:,1:]=c[:,1:]!=c[:,:-1]
    return y,c,e
