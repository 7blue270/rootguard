"""Controlled three-template run; all policy, budget and noise code unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from . import data
from .analyze import write_json
from .run import ROOT, run


def prepare(config, base, data_directory):
    changes = {k: {'base': base.get(k), 'run': config.get(k)} for k in set(base) | set(config) if base.get(k) != config.get(k)}
    assert set(changes) == {'templates', 'refine'}
    assert config['templates'] == ['ANEMIA', 'FIB4', 'BUN_CR'] and config['refine'] is True
    records = data.load(data_directory, config['templates'])
    sizes = {t: len({r['subject_id'] for r in records if r['template'] == t}) for t in config['templates']}
    eligible = [t for t in config['templates'] if sizes[t] >= 10]
    if eligible != config['templates']:
        config = dict(config, templates=eligible)
    assert eligible
    mechanisms = Path(__file__).with_name('mechanisms.py')
    tracked = subprocess.check_output(['git', 'show', 'HEAD:' + str(mechanisms.relative_to(ROOT))], cwd=ROOT)
    digest = lambda content: hashlib.sha256(content).hexdigest()
    assert digest(mechanisms.read_bytes()) == digest(tracked), 'DP policy/ledger modified'
    protocol = {'status': 'internal_go_no_go_only', 'config_changes_from_round1': changes,
                'refinement_exception': 'explicit 1.05 local refinement protocol takes priority over base refine=false',
                'refinement_selection': 'one validation-only step around B0/B2/B2p/M1-a/M1-g crossings, union across templates',
                'test_never_used_for_refinement_or_selection': True,
                'sensitivity_periods': list(range(2, 13)),
                'sensitivity_exclusion': '7/9/11 diagnostic only; main P selection uses original eight periods',
                'split': 'single group split on union of unique subjects; same subject never crosses train/val/test',
                'template_subject_counts': sizes, 'skipped_requested_templates': [t for t in sizes if t not in eligible],
                'not_run_small_templates': {'NLR': 2, 'TYG': 6},
                'drift_axis': 'train-only patient-equal mean max_i abs(normalized_root_update), public domain scaling',
                'B0_strict_floor': 'patient-equal temporal-median absolute residual; bound for ANY constant final output',
                'B0_noiseless_limit': 'frozen protected initial roots at infinite budget; not a finite-budget lower bound',
                'BUN_CR': {'formula': 'ratio=bun/cr', 'class': None, 'public_domains': {'bun': [1, 200], 'cr': [.2, 15]},
                           'ordering': 'charttime, prefix-max storetime availability; >=16 numerical changed batches'},
                'significance': 'paired patient AND hierarchical Bmin ratio CIs exclude 1, each with >=95% both frontiers reached',
                'P_stability_rule': 'report validation bootstrap selection frequency; >=80% called stable',
                'bootstrap_selection_scope': 'conditional on frozen validation P/q/tau; selection stability reported separately',
                'mechanisms_sha256': digest(mechanisms.read_bytes()),
                'base_config_sha256': digest(json.dumps(base, sort_keys=True).encode())}
    return config, protocol


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('configs') / 'round2_multi.json')
    p.add_argument('--data', type=Path, default=ROOT / 'data_mine/finite_cache')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    base = json.loads((Path(__file__).with_name('configs') / 'round1_meld.json').read_text())
    config, protocol = prepare(config, base, a.data)
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
    metadata = run(config, a.output, a.data, refinement_methods=['B0', 'B2', 'B2p', 'M1-a', 'M1-g'],
                   sensitivity_periods=list(range(2, 13)), protocol=protocol)
    write_json(a.output / 'input_config.json', config)
    from .round2_report import analyze
    analyze(a.output, a.data)
    print(json.dumps({'complete': True, 'output': str(a.output), 'counts': metadata['patient_counts']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
