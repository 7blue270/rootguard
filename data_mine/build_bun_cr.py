"""Append BUN/Cr real changed-state trajectories without rewriting other inputs."""
import argparse
import json
from pathlib import Path
import shutil

from .finite_trajectories import DEFAULT_OUT, DEFAULT_SOURCE, source_rows, record, sha


def chart_stream(batches):
    """Chart-time ordered, one admission; at least one numerical root change/event.

    Availability is a prefix maximum of actual store times: a later chart batch
    is never declared available before its predecessor has been processed.
    Unchanged batches update latest observations but are not counted as changes.
    """
    initial = None
    state = {}
    result = []
    available = ''
    for charttime, batch in sorted(batches):
        rows = {k: r for k, r in batch.items() if k in ('bun', 'cr')}
        if not rows:
            continue
        if initial is None:
            if set(rows) != {'bun', 'cr'}:
                continue
            initial = charttime
        previous = dict(state)
        state.update({k: float(r['valuenum']) for k, r in rows.items()})
        available = max(available, *(r['storetime'] or charttime for r in rows.values()))
        event = {'measurement_time': charttime, 'available_time': available, 'rows': rows, 'state': dict(state)}
        if not result or state != previous:
            result.append(event)
    return result


def build(output=DEFAULT_OUT, source=DEFAULT_SOURCE, patients=20, T=40, min_updates=16, max_updates=24):
    output, source = Path(output), Path(source)
    path = output / 'BUN_CR.jsonl'
    if path.exists():
        raise FileExistsError('BUN_CR input already exists; preserve it and use another directory')
    assert 1 <= min_updates <= max_updates < T
    old_manifest = output / 'manifest.json'
    manifest = json.loads(old_manifest.read_text()) if old_manifest.exists() else {'templates': {}}
    old_hashes = {p.name: sha(p) for p in output.glob('*.jsonl')}
    stays = source_rows(source)
    candidates = {}
    for (sid, hadm), batches in sorted(stays.items()):
        stream = chart_stream(batches)
        if len(stream) > min_updates:
            candidates.setdefault(sid, []).append((hadm, stream))
    records = []
    for idx, sid in enumerate(sorted(candidates)[:patients]):
        hadm, stream = min(candidates[sid], key=lambda v: (v[1][0]['measurement_time'], v[0]))
        n = min(min_updates + idx % (max_updates - min_updates + 1), len(stream) - 1)
        rec = record('BUN_CR', sid, hadm, stream, T, n, {})
        rec.update(selection='earliest chart-time eligible admission/window; >=16 numerically changed real lab batches',
                   ordering='charttime; available_time=prefix_max_actual_storetime',
                   count_semantics='at least one numerical root change per retained post-init batch',
                   state_semantics='latest observed roots in chart order; wait for current and previous store times')
        assert len(rec['events']) >= min_updates
        assert all(a['measurement_time'] < b['measurement_time'] for a, b in zip(rec['events'], rec['events'][1:]))
        records.append(rec)
    output.mkdir(parents=True, exist_ok=True)
    if old_manifest.exists():
        shutil.copyfile(old_manifest, output / 'manifest_before_bun_cr_20261112.json')
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records))
    entry = {'records': len(records), 'eligible_patients': len(candidates), 'file': path.name, 'sha256': sha(path),
             'batches': sum(len(r['events']) for r in records), 'T': T, 'min_changed_batches': min_updates,
             'max_changed_batches': max_updates, 'charttime_ordered': True, 'one_admission_per_subject': True,
             'source_itemids': {'bun': ['51006'], 'cr': ['50912', '52546']},
             'source_sha256': sha(source / 'labevents.csv.gz'), 'builder_sha256': sha(__file__)}
    manifest['templates']['BUN_CR'] = entry
    manifest['bun_cr_extension'] = {'builder_sha256': sha(__file__), 'other_input_sha256_preserved': old_hashes,
                                     'ordering_and_update_definition': rec['ordering'] if records else None}
    old_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    assert old_hashes == {name: sha(output / name) for name in old_hashes}
    return entry


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--patients', type=int, default=20)
    a = p.parse_args()
    print(json.dumps(build(a.output, a.source, a.patients), indent=2))


if __name__ == '__main__':
    main()
