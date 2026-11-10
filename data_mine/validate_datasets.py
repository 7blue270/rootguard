"""Validate compact JSON or finite JSONL; only --report writes a new report."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from data_mine.finite_trajectories import ROOTS, DEFAULT_SOURCE, SCHEMA, ITEMS, UNITS


def validate_compact(path):
    from data_mine.build_datasets import calc, category, LABELS, TARGETS
    counts = {}
    for template, records in json.loads(path.read_text()).items():
        seen = set()
        for rec in records:
            assert rec['id'] not in seen
            seen.add(rec['id'])
            state = {}
            for turn, row in enumerate(rec['turns'], 1):
                assert row['turn'] == turn
                pv = row['private_value']
                roots = {k: v for k, v in pv.items() if k in ROOTS[template]}
                if roots:
                    assert len(roots) == len(pv)
                    assert all(math.isfinite(v) and v > 0 for v in roots.values())
                    state.update(roots)
                else:
                    values = calc(template, state)
                    for node, value in pv.items():
                        if node in values:
                            assert math.isclose(values[node], value, rel_tol=1e-10)
                        else:
                            assert node == TARGETS[template][1]
                            assert LABELS[template][category(template, values[TARGETS[template][0]])] == value
                assert set(state) == set(ROOTS[template])
                assert all(f'{k} {v}' in row['user_response'] for k, v in pv.items())
        counts[template] = {'patients': len(records), 'turns': sum(len(r['turns']) for r in records)}
    return counts


def validate_finite(path, raw, demographics):
    from experiments.workflow_extensions.finite_cache.data import validate, replay
    from experiments.workflow_extensions.finite_cache.workload import Workload
    records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    seen, refs, unchanged = set(), 0, 0
    for rec in records:
        validate(rec)
        assert rec['subject_id'] not in seen
        seen.add(rec['subject_id'])
        prior_chart = {}
        state = dict(rec['initial_state'])
        batches = [(rec['initial_source_rows'], rec['initial_available_time'], rec['initial_measurement_time'])]
        batches += [(e['source_rows'], e['available_time'], e['measurement_time']) for e in rec['events']]
        for bi, (rows, available, measured) in enumerate(batches):
            values = rec['initial_state'] if bi == 0 else rec['events'][bi - 1]['values']
            assert set(rows) == set(values) - {'age'}
            for key, source in rows.items():
                actual = raw[source['labevent_id']]
                assert all(actual[k] == v for k, v in source.items())
                assert actual['subject_id'] == rec['subject_id'] and actual['hadm_id'] == rec['admission_id']
                assert ITEMS[actual['itemid']] == key and actual['valueuom'] in UNITS[key]
                assert actual['charttime'] == measured
                assert (actual['storetime'] or actual['charttime']) <= available
                assert key not in prior_chart or actual['charttime'] > prior_chart[key]
                prior_chart[key] = actual['charttime']
                assert float(actual['valuenum']) == values[key]
                if bi and state[key] == values[key]:
                    unchanged += 1
                state[key] = values[key]
                refs += 1
        if rec['template'] == 'FIB4':
            demo = demographics[rec['subject_id']]
            age = int(demo['anchor_age']) + int(rec['initial_measurement_time'][:4]) - int(demo['anchor_year'])
            assert rec['initial_state']['age'] == age
        w = Workload(rec['template'])
        assert all(math.isfinite(float(v)) for v in w.evaluate(replay(rec))[w.target])
    return {'trajectories': len(records), 'source_references_checked': refs,
            'measured_but_unchanged_root_references': unchanged, 'schema': SCHEMA}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', type=Path, default=BASE)
    p.add_argument('--format', choices=['compact', 'finite'], default='compact')
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    raw, demo = {}, {}
    if a.format == 'finite':
        with gzip.open(a.source / 'labevents.csv.gz', 'rt') as f:
            raw = {r['labevent_id']: r for r in csv.DictReader(f)}
        with gzip.open(BASE / 'sources/patients.csv.gz', 'rt') as f:
            demo = {r['subject_id']: r for r in csv.DictReader(f)}
    paths = [a.input] if a.input.is_file() else sorted(a.input.glob('*.jsonl' if a.format == 'finite' else '*.json'))
    if a.format == 'compact':
        paths = [p for p in paths if p.stem in ROOTS]
    if not paths:
        raise ValueError('No matching input datasets')
    report = {'passed': True, 'format': a.format, 'datasets': {}}
    for path in paths:
        checks = validate_finite(path, raw, demo) if a.format == 'finite' else validate_compact(path)
        report['datasets'][path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'checks': checks}
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
