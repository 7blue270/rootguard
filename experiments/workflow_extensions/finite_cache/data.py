"""Offline evaluator inputs and causal public event views are distinct objects."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from data_mine.finite_trajectories import ROOTS, SCHEMA


@dataclass(frozen=True)
class PublicBatch:
    turn: int
    measured_roots: tuple[str, ...]
    batch_id: str = ''
    available_time: str = ''


def load(directory, templates):
    result = []
    for template in templates:
        path = Path(directory) / f'{template}.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for rec in records:
            validate(rec)
        result.extend(records)
    if len({r['trajectory_id'] for r in result}) != len(result):
        raise ValueError('Duplicate trajectory IDs')
    return result


def validate(rec):
    assert rec['schema_version'] == SCHEMA
    roots = set(ROOTS[rec['template']])
    assert set(rec['initial_state']) == roots and rec['T'] >= 1
    assert all(np.isfinite(v) and v > 0 for v in rec['initial_state'].values())
    previous_turn, previous_time = 1, rec['initial_available_time']
    for event in rec['events']:
        assert previous_turn < event['turn'] <= rec['T']
        assert previous_time <= event['available_time']
        assert set(event['measured_roots']) == set(event['values']) <= roots
        assert event['measured_roots'] and len(event['measured_roots']) == len(set(event['measured_roots']))
        assert all(np.isfinite(v) and v > 0 for v in event['values'].values())
        previous_turn, previous_time = event['turn'], event['available_time']
    assert all(1 <= q['turn'] <= rec['T'] for q in rec['queries'])


def replay(rec):
    """Trusted evaluator only: latest known roots at every dialogue round."""
    roots = ROOTS[rec['template']]
    state = dict(rec['initial_state'])
    events = {e['turn']: e for e in rec['events']}
    result = []
    for turn in range(1, rec['T'] + 1):
        if turn in events:
            state.update(events[turn]['values'])
        result.append([state[k] for k in roots])
    return np.asarray(result, dtype=float)


def public_batch(event):
    return PublicBatch(event['turn'], tuple(event['measured_roots']), event['batch_id'], event['available_time'])


def significant_roots(previous, current, h, roots, threshold, tolerance=1e-12):
    """Strict mathematical > threshold, excluding floating-point equality noise."""
    delta = np.abs(np.asarray(current) - np.asarray(previous))
    return {root for i, root in enumerate(roots) if h[i] > 0 and delta[i] > threshold + tolerance}


def split(records, seed=20261110, train_fraction=.5, validation_fraction=.25):
    ids = sorted({r['subject_id'] for r in records}, key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).digest())
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError('Invalid patient split fractions')
    a = int(len(ids) * train_fraction)
    b = a + int(len(ids) * validation_fraction)
    if not 0 < a < b < len(ids):
        raise ValueError('Not enough independent patients for a three-way split')
    assignment = {sid: ('train' if idx < a else 'validation' if idx < b else 'test')
                  for idx, sid in enumerate(ids)}
    return assignment


def partition(records, assignment, part, template):
    return [r for r in records if r['template'] == template and assignment[r['subject_id']] == part]


def input_hashes(directory, templates):
    return {f'{t}.jsonl': hashlib.sha256((Path(directory) / f'{t}.jsonl').read_bytes()).hexdigest() for t in templates}
