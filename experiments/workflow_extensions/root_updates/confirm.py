"""Independent noise replication and matched-patient horizon control."""
import json
import sys
from pathlib import Path
import run_experiments as experiment
from data import real, synthetic


def confirmation_datasets(cfg):
    # Select once at H=16, then truncate the same admission for each horizon.
    longest = real(16, 24)
    for H in cfg['horizons']:
        d = dict(longest)
        d['x'] = longest['x'][:, :H].copy()
        d['plan'] = longest['plan'][:H].copy()
        d['audit'] = dict(longest['audit'], matched_cohort=True,
                          timestamps=[ts[:H] for ts in longest['audit']['timestamps']])
        yield f'real_matched_ANEMIA_H{H}', d
    for name in cfg['templates']:
        for scenario in ['constant', 'drift', 'jump', 'sparse']:
            amp = 0. if scenario == 'constant' else .05
            yield f'syn_confirm_{name}_{scenario}_H8_a{amp}', synthetic(
                name, cfg['synthetic_patients'], 8, scenario, amp,
                481516 + ['FIB4', 'HOMA', 'ANEMIA'].index(name))


if __name__ == '__main__':
    experiment.datasets = confirmation_datasets
    experiment.main()
