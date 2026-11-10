"""Real, admission-local measurement streams for finite-cache experiments.

Unlike the legacy dialogue builder, retain measured-but-unchanged values and
choose the earliest eligible admission/window, without inspecting task events.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path('/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2')
DEFAULT_OUT = ROOT / 'data_mine/finite_cache'
SCHEMA = 'finite-cache-v1'
ROOTS = {'ANEMIA': ['hb', 'hct', 'rbc'], 'FIB4': ['age', 'ast', 'alt', 'plt'],
         'NLR': ['neu', 'lym'], 'TYG': ['tg', 'glu'],
         'MELD_NA': ['bil', 'inr', 'cr', 'na']}
ITEMS = {'51222': 'hb', '51221': 'hct', '51279': 'rbc', '50861': 'alt',
         '50878': 'ast', '51265': 'plt', '51133': 'lym', '52075': 'neu',
         '51000': 'tg', '50931': 'glu', '50885': 'bil', '53089': 'bil',
         '51237': 'inr', '51675': 'inr', '50912': 'cr', '52546': 'cr',
         '50983': 'na', '52623': 'na'}
UNITS = {'hb': ['g/dL'], 'hct': ['%'], 'rbc': ['m/uL'], 'ast': ['IU/L'],
         'alt': ['IU/L'], 'plt': ['K/uL'], 'lym': ['K/uL'], 'neu': ['K/uL'],
         'tg': ['mg/dL'], 'glu': ['mg/dL'], 'bil': ['mg/dL'], 'cr': ['mg/dL'],
         'inr': [''], 'na': ['mEq/L', 'mmol/L']}
NODES = {'ANEMIA': ['mcv', 'mch', 'mchc'],
         'FIB4': ['fib4_prod', 'fib4_denom', 'fib4'],
         'NLR': ['nlr_sum', 'nlr_diff', 'nlr'], 'TYG': ['tyg_prod', 'tyg_half', 'tyg'],
         'MELD_NA': ['bil_term', 'inr_term', 'cr_term', 'meld', 'meld_na']}
TARGETS = {k: v[-1] for k, v in NODES.items()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_rows(source=DEFAULT_SOURCE):
    """All units/duplicate decisions precede admission-local chronological replay."""
    grouped = defaultdict(lambda: defaultdict(list))
    with gzip.open(Path(source) / 'labevents.csv.gz', 'rt') as f:
        for row in csv.DictReader(f):
            key = ITEMS.get(row['itemid'])
            if key is None or not row['hadm_id'] or not row['valuenum']:
                continue
            try:
                value = float(row['valuenum'])
            except ValueError:
                continue
            if not math.isfinite(value) or value <= 0 or row['valueuom'] not in UNITS[key]:
                continue
            grouped[(row['subject_id'], row['hadm_id'], row['charttime'])][key].append(row)
    stays = defaultdict(list)
    for (sid, admission, time), values in grouped.items():
        accepted = {key: max(rows, key=lambda r: (r['storetime'] or r['charttime'], int(r['labevent_id'])))
                    for key, rows in values.items() if len({float(r['valuenum']) for r in rows}) == 1}
        if accepted:
            stays[(sid, admission)].append((time, accepted))
    return stays


def evidence(row):
    return {k: row[k] for k in ['labevent_id', 'subject_id', 'hadm_id', 'itemid',
                              'charttime', 'storetime', 'valuenum', 'valueuom']}


def admission_stream(template, sid, batches, demographics):
    labs = set(ROOTS[template]) - {'age'}
    events = []
    for charttime, batch in batches:
        rows = {k: row for k, row in batch.items() if k in labs}
        if rows:
            events.append({'measurement_time': charttime,
                           'available_time': max(r['storetime'] or charttime for r in rows.values()),
                           'rows': rows})
    events.sort(key=lambda e: (e['available_time'], e['measurement_time']))
    complete = [e for e in events if set(e['rows']) == labs]
    if not complete:
        return []
    first = complete[0]
    times = {k: r['charttime'] for k, r in first['rows'].items()}
    state = {k: float(r['valuenum']) for k, r in first['rows'].items()}
    if template == 'FIB4':
        demo = demographics.get(sid)
        if not demo:
            return []
        state['age'] = int(demo['anchor_age']) + int(first['measurement_time'][:4]) - int(demo['anchor_year'])
        if state['age'] <= 0:
            return []
    result = [{**first, 'state': dict(state)}]
    for event in events:
        if event['available_time'] < first['available_time']:
            continue
        rows = {k: r for k, r in event['rows'].items() if r['charttime'] > times[k]}
        if not rows:
            continue
        state.update({k: float(r['valuenum']) for k, r in rows.items()})
        times.update({k: r['charttime'] for k, r in rows.items()})
        # Deliberately retain a measurement even if every numerical value is unchanged.
        result.append({**event, 'rows': rows, 'state': dict(state)})
    return result


def record(template, sid, admission, stream, T, updates, demographics):
    if not 1 <= updates <= T - 1:
        raise ValueError('Need 1 <= updates <= T-1')
    stream = stream[:updates + 1]
    positions = [2 + j * (T - 2) // max(updates - 1, 1) for j in range(updates)]
    assert len(set(positions)) == updates
    first = stream[0]
    events = []
    for pos, event in zip(positions, stream[1:]):
        rows = event['rows']
        events.append({'turn': pos, 'batch_id': f'{admission}:{event["measurement_time"]}',
                       'measurement_time': event['measurement_time'], 'available_time': event['available_time'],
                       'measured_roots': sorted(rows), 'values': {k: float(r['valuenum']) for k, r in rows.items()},
                       'source_rows': {k: evidence(r) for k, r in rows.items()}})
    nodes = NODES[template]
    queries = []
    for pos in range(1, T + 1):
        if pos not in positions and pos != 1:
            queries.append({'turn': pos, 'node': nodes[len(queries) % len(nodes)]})
    rec = {'schema_version': SCHEMA, 'trajectory_id': f'{template}:{sid}:{admission}',
           'subject_id': sid, 'admission_id': admission, 'template': template, 'T': T,
           'initial_state': first['state'], 'initial_available_time': first['available_time'],
           'initial_measurement_time': first['measurement_time'],
           'initial_source_rows': {k: evidence(r) for k, r in first['rows'].items()},
           'events': events, 'queries': queries,
           'source_kind': 'real_measurements_synthetic_query_schedule',
           'state_semantics': 'latest available measurement; not a simultaneous clinical panel',
           'selection': 'earliest eligible admission and earliest window; no task-event selection',
           'units': {k: UNITS[k] for k in ROOTS[template] if k != 'age'}}
    if template == 'FIB4':
        rec['age_provenance'] = {**demographics[sid], 'reference_time': first['measurement_time'],
                                 'rule': 'anchor_age + initial_measurement_year - anchor_year; fixed within admission'}
    return rec


def build(output=DEFAULT_OUT, source=DEFAULT_SOURCE, templates=None, T=40,
          min_updates=16, max_updates=24, patients=20, overwrite=False):
    output, source = Path(output), Path(source)
    templates = list(templates or ROOTS)
    if min_updates < 1 or max_updates < min_updates or max_updates >= T:
        raise ValueError('Require 1 <= min_updates <= max_updates < T')
    destinations = [output / f'{t}.jsonl' for t in templates] + [output / 'manifest.json']
    if not overwrite and any(p.exists() for p in destinations):
        raise FileExistsError('Output exists; select a new --output or explicitly use --overwrite')
    with gzip.open(ROOT / 'data_mine/sources/patients.csv.gz', 'rt') as f:
        demo = {r['subject_id']: r for r in csv.DictReader(f)}
    stays = source_rows(source)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': SCHEMA, 'T': T, 'min_updates': min_updates, 'max_updates': max_updates,
                'templates': {}, 'source_sha256': {str(p): sha(p) for p in
                    [source / 'labevents.csv.gz', ROOT / 'data_mine/sources/patients.csv.gz']},
                'builder_sha256': sha(__file__), 'selection_uses_task_values': False,
                'retain_measured_unchanged_values': True,
                'exclude_superseded_or_conflicting_results': True}
    for template in templates:
        candidates = defaultdict(list)
        for (sid, admission), batches in sorted(stays.items()):
            stream = admission_stream(template, sid, batches, demo)
            if len(stream) > min_updates:
                candidates[sid].append((admission, stream))
        records = []
        for idx, sid in enumerate(sorted(candidates)[:patients]):
            admission, stream = min(candidates[sid], key=lambda p: (p[1][0]['available_time'], p[0]))
            n = min(min_updates + idx % (max_updates - min_updates + 1), len(stream) - 1)
            records.append(record(template, sid, admission, stream, T, n, demo))
        path = output / f'{template}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records))
        manifest['templates'][template] = {'records': len(records), 'eligible_patients': len(candidates),
                                            'sha256': sha(path), 'file': path.name,
                                            'batches': sum(len(r['events']) for r in records)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return manifest
