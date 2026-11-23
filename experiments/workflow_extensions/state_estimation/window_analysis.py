"""Check baseline parity against v1 and summarize the locked replay."""
import csv
import numpy as np
from .analyze import main as analyze
from .window_study import COHORTS,DEST,OUT,dump

def archived_rows(path):
    with path.open() as source:
        return {(r['subject'],r['B']):float(r['mae']) for r in csv.DictReader(source) if r['method']=='joint'}

def main(dest=DEST):
    differences={}
    for name,_ in COHORTS:
        old=archived_rows(OUT/(name+'.csv'))
        new=archived_rows(dest/(name+'.csv'))
        assert new.keys()==old.keys()
        error=max(abs(new[k]-v) for k,v in old.items())
        assert error<1e-9
        differences[name]=error
    dump(dest/'joint_parity.json',differences)
    analyze(dest,[('PhysioNet_A_check','BUN_CR'),*COHORTS],'task_window',[('root_window','joint'),('task_now','joint')])
if __name__=='__main__':
    import sys
    if sys.argv[1:]==['continuous']:
        from .calibrated_window import DEST as continuous
        main(continuous)
    else:
        main()
