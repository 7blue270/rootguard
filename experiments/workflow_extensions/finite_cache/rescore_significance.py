"""Correct only strict-threshold event/Miss counts from saved public actions.

No DP sampling, parameter selection, or utility resimulation is performed.
Original metrics/summary/config are retained for a complete evaluation audit.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from . import data
from .workload import Workload
from .analyze import write_json, csv_rows, frontiers


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rescore(directory, data_directory, tolerance=1e-12):
    directory, data_directory = Path(directory), Path(data_directory)
    if (directory / 'metrics_fp_original.jsonl').exists():
        raise FileExistsError('This output was already rescored')
    meta = json.loads((directory / 'metadata.json').read_text())
    statistics = json.loads((directory / 'public_stats.json').read_text())
    assignment = json.loads((directory / 'splits.json').read_text())
    records = data.load(data_directory, meta['config']['templates'])
    expected = {}
    cohort = defaultdict(lambda: {'before': 0, 'after': 0, 'boundary': 0})
    threshold = meta['config']['significant_delta']
    for rec in records:
        w = Workload(rec['template'], meta['config'].get('domains', {}).get(rec['template']))
        h = statistics[rec['template']]['h']
        z = w.normalize(data.replay(rec))
        previous = z[0]
        sig = {}
        for event in rec['events']:
            current = z[event['turn'] - 1]
            before = data.significant_roots(previous, current, h, w.roots, threshold, 0.)
            after = data.significant_roots(previous, current, h, w.roots, threshold, tolerance)
            sig[event['turn']] = after
            c = cohort[assignment[rec['subject_id']]]
            c['before'] += bool(before)
            c['after'] += bool(after)
            c['boundary'] += bool(before != after)
            previous = current
        expected[rec['trajectory_id']] = sig
    counts = {}
    with (directory / 'ledger.jsonl').open() as f:
        for line in f:
            log = json.loads(line)
            if log['repeat'] != 0:
                continue
            releases = {r['turn']: set(r['roots']) for r in log['transcript']}
            sig = expected[log['trajectory_id']]
            den = sum(bool(roots) for roots in sig.values())
            miss = sum(bool(roots) and not roots <= releases.get(turn, set()) for turn, roots in sig.items())
            key = (log['split'], log['trajectory_id'], log['method'], log['parameter'], log['B'])
            counts[key] = (den, miss)
    preserved = {p: sha(directory / p) for p in ['selections.json', 'targets.json']}
    old_hash = sha(directory / 'metrics.jsonl')
    aggregate = defaultdict(lambda: [0, 0])
    original_digest, corrected_digest = hashlib.sha256(), hashlib.sha256()
    n = 0
    with (directory / 'metrics.jsonl').open() as src, (directory / 'metrics_corrected.jsonl').open('w') as dst:
        for line in src:
            row = json.loads(line)
            stable = {k: v for k, v in row.items() if k not in ('significant_events', 'missed_events')}
            original_digest.update(json.dumps(stable, sort_keys=True).encode())
            key = (row['split'], row['trajectory_id'], row['method'], row['parameter'], row['B'])
            den, miss = counts[key]
            row['significant_events'], row['missed_events'] = den, miss
            corrected_digest.update(json.dumps({k: v for k, v in row.items() if k not in ('significant_events', 'missed_events')}, sort_keys=True).encode())
            dst.write(json.dumps(row, allow_nan=False) + '\n')
            g = (row['split'], row['template'], row['method'], row['parameter'], row['B'])
            aggregate[g][0] += den
            aggregate[g][1] += miss
            n += 1
    assert original_digest.digest() == corrected_digest.digest()
    (directory / 'metrics.jsonl').rename(directory / 'metrics_fp_original.jsonl')
    (directory / 'metrics_corrected.jsonl').rename(directory / 'metrics.jsonl')
    summary = json.loads((directory / 'summary.json').read_text())
    for row in summary:
        key = tuple(row[k] for k in ['split', 'template', 'method', 'parameter', 'B'])
        den, miss = aggregate[key]
        row['miss_rate'] = miss / den if den else None
    (directory / 'summary.json').rename(directory / 'summary_fp_original.json')
    write_json(directory / 'summary.json', summary)
    (directory / 'config.json').rename(directory / 'config_execution.json')
    meta['config']['significant_tolerance'] = tolerance
    write_json(directory / 'config.json', meta['config'])
    root = Path(__file__).resolve().parents[3]
    for name in ['data.py', 'run.py', 'batch.py', 'rescore_significance.py']:
        p = Path(__file__).parent / name
        meta.setdefault('postprocessing_code_sha256', {})[str(p.relative_to(root))] = sha(p)
    test_path = Path(__file__).parent / 'tests/test_protocol.py'
    meta.setdefault('verification_code_sha256', {})[str(test_path.relative_to(root))] = sha(test_path)
    revision = {'reason': 'exclude floating-point equality noise from strict >0.05 event counts',
                'threshold_unchanged': threshold, 'tolerance': tolerance, 'runs_rescored': n,
                'cohort_event_counts': dict(cohort), 'all_other_metric_fields_unchanged': True,
                'metrics_execution_sha256': old_hash, 'metrics_rescored_sha256': sha(directory / 'metrics.jsonl'),
                'no_new_sampling_or_tuning': True}
    meta['evaluation_revision'] = revision
    for p, h in preserved.items():
        assert sha(directory / p) == h
    write_json(directory / 'metadata.json', meta)
    write_json(directory / 'evaluation_revision.json', revision)
    csv_rows(directory / 'frontiers.csv', frontiers(summary, meta['selections'], meta['targets']))
    print(json.dumps(revision, indent=2))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[3] / 'data_mine/finite_cache')
    a = p.parse_args()
    rescore(a.directory, a.data)


if __name__ == '__main__':
    main()
