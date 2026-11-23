"""Convert the observed BUN/creatinine stream of PhysioNet Challenge 2019."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from data_mine.finite_trajectories import SCHEMA, ROOTS, UNITS, sha
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload

COLUMNS = {'bun': 'BUN', 'cr': 'Creatinine'}
TEMPLATE = 'BUN_CR'
ROOT = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def stream(rows):
    result = []
    for row in rows:
        hour = int(float(row['ICULOS']))
        observed = {}
        for root in ROOTS[TEMPLATE]:
            value = float(row[COLUMNS[root]])
            if np.isfinite(value) and value > 0:
                observed[root] = value
        if observed:
            result.append((hour, observed))
    return result


def window(batches):
    # Start at the first hour at which both roots are known.
    latest = {}
    for index, (hour, values) in enumerate(batches):
        latest.update({root: (hour, value) for root, value in values.items()})
        if len(latest) == 2:
            return hour, latest, batches[index + 1:]
    return None


def record(sid, site, member, start, latest, following, first_hour):
    following = following[:24]
    positions = [2 + j * 38 // (len(following) - 1) for j in range(len(following))]

    def clock(hour):
        return f'ICU-hour-{hour:06d}'

    def source(root, hour, value):
        return dict(file=member, column=COLUMNS[root], icu_hour=hour, row_1based=hour - first_hour + 2,
                    valuenum=value, valueuom=UNITS[root][0])

    events = [
        dict(turn=turn, batch_id=f'{sid}:{clock(hour)}', measurement_time=clock(hour),
             available_time=clock(hour), measured_roots=sorted(values), values=values,
             source_rows={root: source(root, hour, value) for root, value in values.items()})
        for turn, (hour, values) in zip(positions, following)
    ]
    occupied = {1, *positions}
    return dict(schema_version=SCHEMA, trajectory_id=f'{TEMPLATE}:{sid}', subject_id=sid,
                admission_id=sid, template=TEMPLATE, T=40, source_group=site,
                initial_state={root: value for root, (_, value) in latest.items()},
                initial_measurement_time=clock(start), initial_available_time=clock(start),
                initial_source_rows={root: source(root, *item) for root, item in latest.items()},
                events=events, queries=[dict(turn=t, node='ratio') for t in range(2, 41) if t not in occupied],
                source_kind='real_measurements_synthetic_query_schedule',
                state_semantics='latest observed hourly measurement; distinct roots may have different measurement hours',
                selection='earliest complete state; >=4 later observation batches and both roots remeasured in first 24',
                units={root: UNITS[root] for root in ROOTS[TEMPLATE]})


def summary(values):
    values = np.asarray(values, float)
    if not len(values):
        return {}
    return dict(n=len(values), p10=float(np.quantile(values,.1)), median=float(np.median(values)),
                p90=float(np.quantile(values,.9)), max=float(values.max()))


def drift(records):
    w = Workload(TEMPLATE)
    metrics = defaultdict(list)
    for rec in records:
        x = data.replay(rec)
        norm = w.normalize(x)
        raw_root_events = np.any(np.diff(x, axis=0) != 0, axis=1)
        y = w.evaluate(x)[w.target]
        bounded = w.from_cache(norm)[w.target]
        span = np.ptp(y)
        metrics['task_range'].append(span)
        metrics['task_range_over_anchor_pct'].append(100 * span / y[0])
        metrics['static_staleness_mae'].append(abs(y-y[0]).mean())
        metrics['clipped_task_range'].append(np.ptp(bounded))
        metrics['clipped_staleness_mae'].append(abs(bounded-bounded[0]).mean())
        metrics['max_domain_normalized_root_jump'].append(abs(np.diff(norm,axis=0)).max())
        metrics['root_changes'].append(raw_root_events.sum())
        metrics['batches'].append(len(rec['events']))
        metrics['outside_existing_domain_frac'].append(((x < w.lo) | (x > w.hi)).mean())
        if rec['initial_available_time'].startswith('ICU-hour-'):
            metrics['actual_endpoint_hours'].append(int(rec['events'][-1]['available_time'].split('-')[-1]) -
                                                  int(rec['initial_available_time'].split('-')[-1]))
            metrics['anchor_root_gap_hours'].append(np.ptp([z['icu_hour'] for z in rec['initial_source_rows'].values()]))
    result = {key: summary(vals) for key, vals in metrics.items()}
    if records:
        result['fractions'] = dict(any_actual_root_change=float(np.mean(np.array(metrics['root_changes']) > 0)),
            nonzero_task_range=float(np.mean(np.array(metrics['task_range']) > 1e-12)),
            task_range_ge_25pct_anchor=float(np.mean(np.array(metrics['task_range_over_anchor_pct']) >= 25)),
            root_jump_gt_5pct_domain=float(np.mean(np.array(metrics['max_domain_normalized_root_jump']) > .05+1e-12)),
            any_outside_domain=float(np.mean(np.array(metrics['outside_existing_domain_frac']) > 0)))
    return result


def build(archive, output):
    if output.exists():
        raise FileExistsError(output)
    counts = defaultdict(Counter)
    hours = defaultdict(list)
    coverage = defaultdict(Counter)
    rows = []
    seen = set()
    expected_header = None
    with ZipFile(archive) as z:
        for member in sorted(z.namelist()):
            path = Path(member)
            if (len(path.parts) != 3 or path.parts[0] != 'training' or
                path.parts[1] not in ('training_setA','training_setB') or path.suffix != '.psv'
                or not path.stem.startswith('p') or not path.stem[1:].isdigit()):
                continue
            site = path.parts[1][-1]
            if path.stem in seen:
                raise ValueError(f'Duplicate source patient ID {path.stem}')
            seen.add(path.stem)
            sid = f'physionet2019-{site}-{path.stem}'
            reader = csv.DictReader(z.read(member).decode().splitlines(), delimiter='|')
            if expected_header is None:
                expected_header = reader.fieldnames
            if reader.fieldnames != expected_header:
                raise ValueError(f'Unexpected header in {member}')
            patient = list(reader)
            counts[site]['source_patients'] += 1
            counts[site]['hour_rows'] += len(patient)
            hours[site].append(len(patient))
            values = np.array([list(row.values()) for row in patient], dtype=float)
            coverage[site].update(c for c,present in zip(expected_header,np.isfinite(values).any(axis=0)) if present)
            first_hour = int(float(patient[0]['ICULOS']))
            if any(float(row['ICULOS']) != i for i, row in enumerate(patient,first_hour)):
                raise ValueError(f'Nonconsecutive ICULOS in {member}')
            win = window(stream(patient))
            if win is None:
                continue
            counts[site]['has_complete_anchor'] += 1
            start, latest, following = win
            for minimum in (1, 4, 8, 16, 24):
                counts[site][f'at_least_{minimum}_later_batches'] += len(following) >= minimum
            if len(following) < 4 or set().union(*(obs for _, obs in following[:24])) != set(ROOTS[TEMPLATE]):
                continue
            counts[site]['eligible'] += 1
            rows.append(record(sid,site,member,start,latest,following,first_hour))
    output.mkdir(parents=True, exist_ok=False)
    path = output/f'{TEMPLATE}.jsonl'
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    loaded = data.load(output, [TEMPLATE])
    assert loaded == rows
    split = {r['subject_id']: (
        'test' if r['source_group'] == 'B' else
        'validation' if int(hashlib.sha256(r['subject_id'].encode()).hexdigest(),16) % 5 == 0
        else 'train') for r in rows}
    dump(output/'split.json', split)
    dense = [r for r in rows if len(r['events']) >= 16]
    write_panels(output, dense, split)
    old = data.load(ROOT/'finite_cache', [TEMPLATE])
    dump(output/'manifest.json',dict(source=str(archive), source_sha256=sha(archive),
        builder_sha256=sha(__file__), schema_version=SCHEMA, source_groups=counts, columns=expected_header,
        hour_summary={site:summary(values) for site,values in hours.items()},
        finite_column_patient_counts=coverage, files={path.name:sha(path)},
        splits=Counter(split.values()), dense_splits=Counter(split[r['subject_id']] for r in dense),
        drift_all=drift(rows), drift_by_split={part:drift([r for r in rows if split[r['subject_id']]==part])
                                             for part in ('train','validation','test')},
        drift_dense=drift(dense), drift_old_BUN_CR=drift(old)))


def write_panels(output, dense, split):
    panels = {
        'split_dense.json': dense,
        'split_dense_value_change.json': [
            r for r in dense if int(np.any(np.diff(data.replay(r), axis=0) != 0, axis=1).sum()) >= 16
        ],
    }
    for name, rows in panels.items():
        dump(output/name, {r['subject_id']: split[r['subject_id']] for r in rows})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=ROOT/'training.zip')
    parser.add_argument('--output',type=Path,default=ROOT/'finite_cache_physionet2019')
    args = parser.parse_args()
    build(args.archive,args.output)
