"""Load a fitted M9 from its frozen decisions and designated prior records."""
import json
from .study import COHORTS,OUT,Context,SEED,references
from .policy import run

def load(cohort):
    cfg=json.loads((OUT/cohort/'frozen.json').read_text())
    _,t=COHORTS[cohort]
    prior=references(t,t=='BUN_CR' and not cohort.startswith('old'))
    if [r['subject_id'] for r in prior]!=cfg['references']:
        raise ValueError('Frozen prior IDs differ')
    return Context(t,prior,SEED),cfg
