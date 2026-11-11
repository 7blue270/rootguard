"""Vectorize noise repeats for the public, noise-independent mainline scheduler.

The same Cache controller determines actions/epsilon once. Independent paired
noise streams and numerical caches are maintained for EVERY repeat. Reference
simulate() equivalence is tested; this path must not be used for private or
DP-output-dependent feedback policies.
"""
from collections import Counter
from dataclasses import asdict
import numpy as np
from . import data
from .mechanisms import Cache, paired_uniforms, laplace_standard


def simulate_repeats(rec, workload, stats, method, parameter, B, repeats, config):
    if method not in ('B0', 'B2', 'B2p', 'M1-a', 'M1-g'):
        raise ValueError('Batch repeats only supports the noise-independent mainline policies')
    h = np.array(stats['h'])
    q = float(parameter.split('=')[1]) if parameter.startswith('q=') else config['quantiles'][0]
    period = int(parameter.split('=')[1]) if parameter.startswith('P=') else config.get('fixed_gate_period', 4)
    controller = Cache(workload.roots, h, stats['C'][str(q)], B, method,
                       rec['T'] - 1, period=period, **config['mechanism'])
    T, k = rec['T'], len(h)
    raw = data.replay(rec)
    z = workload.normalize(raw)
    truth = workload.evaluate(raw)
    protected = workload.evaluate(workload.denormalize(z))
    def uniforms(turn):
        return np.array([paired_uniforms(config['seed'], rec['subject_id'], rec['trajectory_id'],
                                        B, r, turn, k) for r in range(repeats)])
    controller.initialize(np.zeros(k), np.full(k, .5))
    ii = np.flatnonzero(controller.active)
    cache = np.zeros((repeats, k))
    cache[:, ii] = z[0, ii] + laplace_standard(uniforms(1)[:, ii]) / controller.last_eps[ii]
    histories = np.zeros((repeats, T, k))
    n_logs = min(repeats, config.get('log_repeats', repeats)) if config.get('save_logs', True) else 0
    transcripts = [[] for _ in range(n_logs)]
    projected_publications = np.sum((cache[:, ii] < 0) | (cache[:, ii] > 1), axis=1)
    publications = len(ii)

    def record_release(turn, indices, eps):
        for r in range(n_logs):
            transcripts[r].append({'turn': turn, 'roots': [workload.roots[i] for i in indices],
                                   'eps': np.asarray(eps).tolist(), 'values': cache[r, indices].tolist()})
    record_release(1, ii, controller.last_eps[ii])
    events = {e['turn']: e for e in rec['events']}
    reasons, decisions, public_rounds = Counter(), [], []
    missed = significant = age_refreshes = 0
    last_release_turn = np.ones(k, int)
    aois = []
    previous = z[0]
    for turn in range(1, T + 1):
        refreshed = set()
        if turn in events:
            event = events[turn]
            decision = controller.decide(data.public_batch(event))
            reasons[decision.trigger] += 1
            before = controller.snapshot() if n_logs else None
            if decision.trigger not in ('none', 'empty', 'frozen'):
                ii = np.array(decision.indices)
                cache[:, ii] = z[turn - 1, ii] + laplace_standard(uniforms(turn)[:, ii]) / np.array(decision.eps)
                projected_publications += np.sum((cache[:, ii] < 0) | (cache[:, ii] > 1), axis=1)
                publications += len(ii)
                # Controller cache values are irrelevant to all supported policies.
                controller.publish(decision, np.zeros(k), np.full(k, .5))
                record_release(turn, ii, decision.eps)
                refreshed = {workload.roots[i] for i in ii}
                last_release_turn[ii] = turn
                age_refreshes += int(decision.trigger in ('age_force', 'both'))
            changed = data.significant_roots(previous, z[turn - 1], h, workload.roots,
                                              config['significant_delta'], config.get('significant_tolerance', 1e-12))
            if changed:
                significant += 1
                missed += int(not changed <= refreshed)
            previous = z[turn - 1]
            if n_logs:
                decisions.append({'turn': turn, 'decision': asdict(decision), 'before': before,
                                  'after': controller.snapshot()})
        histories[:, turn - 1] = cache
        aois.append(float(np.mean((turn - last_release_turn)[controller.active])))
        if n_logs:
            public_rounds.append({'turn': turn, **controller.snapshot()})
    estimates = workload.from_cache(histories)
    errors = np.abs(estimates[workload.target] - truth[workload.target])
    protected_errors = np.abs(estimates[workload.target] - protected[workload.target])
    scale, weights = np.array(stats['scales']), np.array(stats['weights'])
    per_node = np.stack([np.abs(estimates[n] - truth[n]) for n in workload.nodes], axis=-1) / scale
    workload_errors = per_node @ weights
    query_errors = [per_node[:, q_['turn'] - 1, workload.nodes.index(q_['node'])]
                    for q_ in rec['queries'] if q_['node'] in workload.nodes]
    query_mean = np.mean(query_errors, axis=0) if query_errors else [None] * repeats
    e = np.clip(histories, 0, 1) - z
    J = np.einsum('rti,ij,rtj->rt', e, np.array(stats['M']), e).mean(axis=1)
    project = np.mean(((histories < 0) | (histories > 1))[:, :, controller.active], axis=(1, 2))
    classes = workload.classes(estimates[workload.target])
    true_classes = workload.classes(truth[workload.target])
    class_error = np.mean(classes != true_classes, axis=1) if classes is not None else [None] * repeats
    stop = controller.stop_turn
    clipping_mae = float(np.mean(np.abs(truth[workload.target] - protected[workload.target])))
    outside = (raw < workload.lo) | (raw > workload.hi)
    results = []
    for r in range(repeats):
        metric = {'subject_id': rec['subject_id'], 'trajectory_id': rec['trajectory_id'], 'template': rec['template'],
                  'method': method, 'parameter': parameter, 'B': float(B), 'repeat': r,
                  'mae': float(errors[r].mean()), 'mae_protected': float(protected_errors[r].mean()),
                  'p95_abs_error': float(np.quantile(errors[r], .95)), 'input_clipping_mae': clipping_mae,
                  'input_outside_root_rounds': int(outside.sum()),
                  'query_loss': float(query_mean[r]) if query_mean[r] is not None else None,
                  'workload_loss': float(workload_errors[r].mean()), 'J': float(J[r]),
                  'class_error': float(class_error[r]) if class_error[r] is not None else None,
                  'spent': controller.spent, 'remaining_fraction': controller.remaining / B,
                  'refresh_count': controller.refresh_count, 'age_refreshes': age_refreshes,
                  'significant_events': significant, 'missed_events': missed, 'stop_turn': stop,
                  'post_stop_mae': float(errors[r, stop:].mean()) if stop is not None and stop < T else None,
                  'aoi_mean': float(np.mean(aois)), 'cache_projection_fraction': float(project[r]),
                  'publication_projection_fraction': float(projected_publications[r] / publications),
                  'input_clipping_fraction': float(outside.mean()),
                  'early_budget_freeze': int(method != 'B0' and stop is not None and stop < T),
                  'root_release_count': publications}
        for reason in ['gate', 'age_force', 'both', 'periodic', 'none', 'empty', 'frozen']:
            metric['reason_' + reason] = reasons[reason]
        assert all(np.isfinite(v) for v in metric.values() if isinstance(v, float))
        log = {'decisions': decisions, 'transcript': transcripts[r],
               'rounds': [dict(s, target_estimate=float(estimates[workload.target][r, i]))
                          for i, s in enumerate(public_rounds)]} if r < n_logs else {}
        results.append((metric, log))
    return results
