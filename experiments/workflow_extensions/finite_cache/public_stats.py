"""Only training patients contribute public population and drift statistics."""
from collections import Counter
import hashlib
import json
import numpy as np
from .data import replay


def fit(records, workload, quantiles, scales=None):
    if not records:
        raise ValueError(f'{workload.template}: no training trajectories')
    states, drift = [], [[] for _ in workload.roots]
    count = np.zeros(len(workload.roots), int)
    total_events = 0
    queries = Counter()
    outside = np.zeros(len(workload.roots), int)
    for rec in records:
        raw = replay(rec)
        outside += ((raw < workload.lo) | (raw > workload.hi)).sum(axis=0)
        z = workload.normalize(raw)
        states.extend(workload.denormalize(z))
        prev = workload.normalize([rec['initial_state'][r] for r in workload.roots])
        for event in rec['events']:
            current = z[event['turn'] - 1]
            total_events += 1
            for i, root in enumerate(workload.roots):
                if root in event['measured_roots']:
                    count[i] += 1
                    drift[i].append(abs(current[i] - prev[i]))
            prev = current
        queries.update(q['node'] for q in rec['queries'] if q['node'] in workload.nodes)
    mu = np.mean(states, axis=0)
    weights = np.array([queries[n] for n in workload.nodes], float)
    if weights.sum() == 0:
        weights[workload.nodes.index(workload.target)] = 1
    weights /= weights.sum()
    bounds = workload.node_bounds()
    # Node units differ; scales are public-domain ranges, configurable before runs.
    scale = np.array([(scales or {}).get(n, max(bounds[n][1] - bounds[n][0], 1e-12)) for n in workload.nodes])
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError('Workload scales must be finite and positive')
    Abar = workload.derivative_bounds() / scale[:, None]
    A = workload.jacobian(mu) / scale[:, None]
    h = weights @ Abar
    measured_mean = np.array([np.mean(d) if d else 0. for d in drift])
    f = count / max(total_events, 1)
    Cs = {str(q): [float(np.quantile(d, q)) if d else 0. for d in drift] for q in quantiles}
    ids = sorted(r['subject_id'] for r in records)
    return {'template': workload.template, 'roots': workload.roots, 'nodes': workload.nodes,
            'mu': mu.tolist(), 'weights': weights.tolist(), 'scales': scale.tolist(),
            'Abar': Abar.tolist(), 'A': A.tolist(), 'M': (A.T @ np.diag(weights) @ A).tolist(),
            'h': h.tolist(), 'f': f.tolist(), 'measured_mean_drift': measured_mean.tolist(),
            'd': (f * measured_mean).tolist(), 'C': Cs, 'drift_samples': count.tolist(),
            'train_subject_ids': ids, 'train_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'train_out_of_domain_root_rounds': outside.tolist(), 'C_has_coverage_guarantee': False}
