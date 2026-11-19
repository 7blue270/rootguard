"""Train-only public population and bounded, deterministic plan search.

Public ranking minimizes mixture mean MAE * (1 + lambda * spend/B), not
minimax risk. Validation chooses only P or (K,rho); times/root sets are public.
"""
from __future__ import annotations
from itertools import combinations, product
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from .core import Plan, simulate, task, key_seed


def public_population(train, w, cfg, delta):
    rng = np.random.default_rng(key_seed(cfg['population_seed'], w.template))
    out = []
    for j in range(cfg['public_trajectories_per_delta']):
        rec = train[int(rng.integers(len(train)))]
        original = w.normalize(data.replay(rec))
        state = original[0].copy()
        events = []
        for e in rec['events']:
            t = e['turn']
            roots = [w.roots.index(r) for r in e['measured_roots']]
            direction = original[t-1] - original[t-2]
            scale = np.max(abs(direction))
            if scale < 1e-12:
                direction = np.zeros(len(w.roots))
                direction[roots] = rng.choice([-1., 1.], len(roots))
                scale = 1.
            # Delta means pre-projection expected max normalized change per batch.
            # Shared directions preserve joint changes; zero-update roots (age)
            # remain constant because they never occur in measured_roots.
            direction = direction / scale * rng.choice([-1., 1.])
            state = np.clip(state + direction * delta * rng.exponential(), 0, 1)
            raw = w.denormalize(state)
            events.append({**e, 'values': {w.roots[i]: float(raw[i]) for i in roots}})
        initial = w.denormalize(original[0])
        out.append({**rec, 'subject_id': f'public-{j}', 'trajectory_id': f'public-{w.template}-{delta}-{j}',
                    'initial_state': dict(zip(w.roots, initial.tolist())), 'events': events,
                    'source_kind': 'train-only-public-synthetic'})
    return out


def candidates(k, h, cfg):
    roots = tuple(range(k))
    top = tuple(sorted(np.argsort(-np.asarray(h))[:max(1, k//2)].tolist()))
    subsets = sorted(set([roots, top] + [(i,) for i in range(k)]))
    out = [Plan(1.)]
    for K in cfg['K_grid']:
        if K == 0:
            continue
        for rho in cfg['rho_grid']:
            for times in combinations(cfg['time_fractions'], K):
                for sets in product(subsets, repeat=K):
                    out.append(Plan(rho, times, sets))
    return out


def proxy_rank(plans, B, stats, cfg):
    """Vectorized public upper-bound proxy, ignoring veto only for shortlisting."""
    N, T, k = len(plans), 40, len(stats['h'])
    h, c = np.array(stats['h']), np.array(stats['cbar'])
    freq = np.array(stats['measured_frequency'])
    s = np.sqrt(h)
    anchor = np.array([p.rho for p in plans])[:, None] * B * s / s.sum()
    precision = anchor.copy()
    age = np.zeros((N, k))
    costs = np.zeros(N)
    for t in range(1, T+1):
        age += (cfg['proxy_batches'] / T) * freq
        for j in range(max(cfg['K_grid'])):
            idx = [n for n, p in enumerate(plans) if p.K > j and round(p.fractions[j]*T) == t]
            if not idx:
                continue
            for n in idx:
                p = plans[n]
                roots = list(p.sets[j])
                precision[n, roots] = (1-p.rho)*B/p.K * s[roots] / s[roots].sum()
                age[n, roots] = 0
        costs += np.sum(h * (age*c + 1/precision), axis=1) / T
    return costs


def shortlist(plans, B, stats, cfg):
    scores = proxy_rank(plans, B, stats, cfg)
    groups = {}
    for i, p in enumerate(plans):
        shape = 'full' if all(len(s) == len(stats['h']) for s in p.sets) else 'subset'
        groups.setdefault((p.K, p.rho, shape), []).append(i)
    selected = []
    for key in sorted(groups):
        ids = sorted(groups[key], key=lambda i: (scores[i], i))[:cfg['proxy_top_per_group']]
        selected.extend(ids)
    return [(plans[i], float(scores[i])) for i in selected]


def score_population(population, B, plan, mode, policy, cfg, phase, repeats):
    vals, spent = [], []
    for rec in population:
        result = simulate(rec, B, plan, mode, policy, cfg, phase, repeats)
        truth = policy.w.evaluate(data.replay(rec))[policy.w.target]
        vals.append(np.mean(abs(task(policy.w, result['caches']) - truth[None, :])))
        spent.append(np.mean(result['spent'])/B)
    return float(np.mean(vals)), float(np.mean(spent))


def public_select(short, population, B, mode, policy, cfg):
    groups, scores = {}, []
    for plan, proxy in short:
        loss, spend = score_population(population, B, plan, mode, policy, cfg, 'public_plan', cfg['public_repeats'])
        J = loss * (1 + cfg['lambda_cost'] * spend)
        entry = dict(plan=plan, proxy=proxy, mae=loss, spend=spend, J=J)
        scores.append(entry)
        key = (plan.K, plan.rho)
        if key not in groups or J < groups[key]['J']:
            groups[key] = entry
    return min(scores, key=lambda x: x['J']), list(groups.values()), scores
