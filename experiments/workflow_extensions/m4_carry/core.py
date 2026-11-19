"""Policy never receives private truth; release and offline evaluation are separate.

Normalized identity releases, pathwise pure-metric-DP ledger. Fixed random streams
are an offline pairing device, not a prescription for deployed secret noise.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import numpy as np
from experiments.workflow_extensions.finite_cache import data
from experiments.workflow_extensions.finite_cache.workload import Workload

METHODS = ['B0', 'B2-val', 'B2-pub', 'M4', 'M4-carry', 'M4-carry-replan']


def key_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).digest()[:8], 'little')


@lru_cache(maxsize=12000)
def paired_noise(seed, phase, subject, trajectory, B, repeats, T, roots):
    # Every coordinate has the exact requested semantic key, excluding method.
    out = np.empty((T + 1, repeats, len(roots)))
    for t in range(1, T + 1):
        for r in range(repeats):
            for i, root in enumerate(roots):
                v = key_seed(seed, phase, subject, trajectory, float(B), r, t, root)
                u = ((v >> 11) + .5) / 2**53 - .5
                out[t, r, i] = -np.sign(u) * np.log1p(-2 * abs(u))
    out.setflags(write=False)
    return out


def task(w, z):
    """Final target only; same clinical formula and public domains as finite_cache."""
    x = w.lo + np.clip(z, 0., 1.) * w.width
    if w.template == 'ANEMIA':
        return 100 * x[..., 0] / x[..., 1]
    if w.template == 'FIB4':
        return x[..., 0] * x[..., 1] / (np.sqrt(x[..., 2]) * x[..., 3])
    if w.template == 'BUN_CR':
        return x[..., 0] / x[..., 1]
    b, i, c, n = np.moveaxis(x, -1, 0)
    meld = 3.78 * np.log(np.maximum(b, 1)) + 11.2 * np.log(np.maximum(i, 1)) + 9.57 * np.log(np.clip(c, 1, 4)) + 6.43
    d = 137 - np.clip(n, 125, 137)
    return np.clip(meld + 1.32 * d - .033 * meld * d, 6, 40)


def allocate(total, mask, h):
    a = np.asarray(mask, float) * np.sqrt(h)
    denom = a.sum(axis=-1, keepdims=True)
    return np.asarray(total)[..., None] * a / np.where(denom > 0, denom, 1)


def fit_stats(records, w, cfg):
    states = [w.normalize(data.replay(r)) for r in records]
    mu = np.mean(np.concatenate(states), axis=0)
    changes = [[] for _ in w.roots]
    batches = np.zeros(len(w.roots))
    count = 0
    for rec in records:
        z = w.normalize(data.replay(rec))
        for e in rec['events']:
            count += 1
            for root in e['measured_roots']:
                i = w.roots.index(root)
                changes[i].append(abs(z[e['turn'] - 1, i] - z[e['turn'] - 2, i]))
                batches[i] += 1
    cbar = np.array([np.quantile(x, cfg['cbar_quantile']) if x else 0. for x in changes])
    raw_h = w.derivative_bounds()[w.nodes.index(w.target)]
    # Original alloc_eps also floors h at 1e-9. Preserve all roots, including
    # a root irrelevant to the primary task (e.g. ANEMIA rbc).
    h = np.maximum(raw_h, cfg['h_floor'])
    return dict(mu=mu.tolist(), cbar=cbar.tolist(), h=h.tolist(), raw_h=raw_h.tolist(),
                measured_frequency=(batches / max(count, 1)).tolist(),
                node_weights={n: float(n == w.target) for n in w.nodes},
                h_source='analytic final-target normalized-coordinate derivative bound; fixed positive floor',
                train_subjects=sorted({r['subject_id'] for r in records}),
                train_trajectories=[r['trajectory_id'] for r in records])


@dataclass(frozen=True)
class Plan:
    rho: float
    fractions: tuple[float, ...] = ()
    sets: tuple[tuple[int, ...], ...] = ()
    period: int | None = None

    @property
    def K(self):
        return len(self.fractions)

    def to_dict(self):
        return dict(rho=self.rho, K=self.K, fractions=self.fractions, sets=self.sets, period=self.period)


def realize(plan, opportunities, T):
    """Deterministic minimum-distance increasing alignment; public metadata only."""
    if plan.period is not None:
        ts = opportunities[plan.period - 1::plan.period]
        # If P exceeds the public count, a last-opportunity refresh keeps B2 dynamic.
        return [(t, plan.sets[0]) for t in (ts or opportunities[-1:])]
    K = plan.K
    if not K:
        return []
    if len(opportunities) < K:
        raise ValueError('Insufficient public opportunities for plan')
    # K <= 2 in this protocol; exact deterministic matching, no value input.
    candidates = itertools.combinations(opportunities, K)
    ts = min(candidates, key=lambda q: (sum((t / T - f)**2 for t, f in zip(q, plan.fractions)), q))
    return list(zip(ts, plan.sets))


class Policy:
    """Only cache, last release precision, public age and public MC are visible."""
    def __init__(self, w, stats, cfg):
        self.w, self.cfg = w, cfg
        self.h, self.cbar = np.array(stats['h']), np.array(stats['cbar'])
        rng = np.random.default_rng(cfg['public_mc_seed'])
        self.eta = rng.laplace(size=cfg['n_mc'])
        self.drift = rng.laplace(size=cfg['n_mc'])

    def decide(self, cache, eps_last, ages, candidate_mask, candidate_epsilon):
        cand = allocate(candidate_epsilon, candidate_mask, self.h)
        selected = np.zeros_like(candidate_mask, dtype=bool)
        base = task(self.w, cache)[:, None]
        for i in range(len(self.h)):
            active = candidate_mask[:, i]
            if not np.any(active):
                continue
            c = cache[active]
            z = np.broadcast_to(c[:, None, :], (len(c), self.cfg['n_mc'], len(self.h))).copy()
            z[..., i] += self.eta[None, :] / eps_last[active, i, None]
            old = np.mean(abs(task(self.w, z) - base[active]), axis=1)
            z[..., i] = c[:, i, None] + np.sign(self.drift) * np.minimum(abs(self.drift) * .5, 2) * ages[active, i, None] * self.cbar[i] * .5
            old += np.mean(abs(task(self.w, z) - base[active]), axis=1)
            z[..., i] = c[:, i, None] + self.eta / cand[active, i, None]
            new = np.mean(abs(task(self.w, z) - base[active]), axis=1)
            selected[active, i] = old - (1 + self.cfg['c_op']) * new > 0
        return selected, cand


class PrivateSource:
    """Trusted local owner. Engine can obtain private roots only inside release()."""
    def __init__(self, w, rec):
        self.__states = w.normalize(data.replay(rec))
        self.T = rec['T']
        self.release_reads = []

    def release(self, turn, mask, eps, noise, cache):
        assert 1 <= turn <= self.T
        assert np.all(eps[mask] > 0) and np.all(np.isfinite(eps[mask]))
        # replay(t) is the latest known state, never the next measurement.
        raw = self.__states[turn - 1] + np.divide(noise, eps, out=np.zeros_like(noise), where=mask)
        projection = ((raw < 0) | (raw > 1)) & mask
        out = cache.copy()
        out[mask] = np.clip(raw, 0, 1)[mask]
        self.release_reads.append(turn)
        return out, projection


def simulate(rec, B, plan, mode, policy, cfg, phase, repeats, detailed=False):
    """Common release/cache/ledger executor. Returns transcript; no truth metric."""
    w, k, R, T = policy.w, len(policy.h), repeats, rec['T']
    public_events = {e['turn']: tuple(w.roots.index(r) for r in e['measured_roots']) for e in rec['events']}
    schedule = realize(plan, sorted(public_events), T)
    assert len({t for t, _ in schedule}) == len(schedule)
    assert all(a[0] < b[0] for a, b in zip(schedule, schedule[1:]))
    assert 0 < plan.rho <= 1
    if not schedule:
        assert plan.rho == 1
    noise = paired_noise(cfg['noise_seed'], phase, str(rec['subject_id']), str(rec['trajectory_id']), float(B), R, T, tuple(w.roots))
    source = PrivateSource(w, rec)
    mask = np.ones((R, k), bool)
    cache = np.tile(np.array(policy.cfg.get('initial_public_cache', [.5] * k)), (R, 1))
    eps_last = allocate(np.full(R, plan.rho * B), mask, policy.h)
    cache, proj = source.release(1, mask, eps_last, noise[1], cache)
    spent = eps_last.sum(axis=1)
    left = B - spent
    g = np.zeros((R, k))
    K, n_left = len(schedule), len(schedule)
    n_pub = np.full(R, k)
    n_proj = proj.sum(axis=1)
    m, empty, root_veto = np.zeros(R), np.zeros(R), np.zeros(R)
    root_count = np.zeros(R)
    caches = np.empty((R, T, k))
    histories = []
    step = {t: (j, roots) for j, (t, roots) in enumerate(schedule)}
    for t in range(1, T + 1):
        if t in public_events:
            g[:, public_events[t]] += 1
        if t in step:
            j, roots = step[t]
            assert n_left > 0
            total = left / n_left if mode == 'carry' else np.full(R, (1 - plan.rho) * B / K)
            n_before = n_left
            n_left -= 1
            cm = np.zeros((R, k), bool)
            cm[:, list(roots)] = True
            assert np.all(total > 0)
            if mode == 'periodic':
                selected, cand = cm, allocate(total, cm, policy.h)
            else:
                selected, cand = policy.decide(cache, eps_last, g, cm, total)
            actual = allocate(total, selected, policy.h)
            publishing = np.any(selected, axis=1)
            before, left_before = cache.copy(), left.copy()
            if np.any(publishing):
                cache, projected = source.release(t, selected, actual, noise[t], cache)
                eps_last[selected] = actual[selected]
                g[selected] = 0
                charge = actual.sum(axis=1)
                assert np.allclose(charge[publishing], total[publishing], rtol=0, atol=1e-10)
                spent += charge
                left -= charge
                n_proj += projected.sum(axis=1)
                n_pub += selected.sum(axis=1)
            else:
                charge = np.zeros(R)
            m += publishing
            empty += ~publishing
            root_veto += (cm & ~selected).sum(axis=1)
            root_count += cm.sum(axis=1)
            assert np.all(spent <= B + 1e-10) and np.all(left >= -1e-10)
            assert np.all(np.isfinite(cache))
            if detailed:
                histories.append(dict(turn=t, action_index=j + 1, planned_roots=roots, selected=selected,
                                      candidate_epsilon=cand, actual_epsilon=actual, eps_can=total,
                                      left_before=left_before, left_after=left.copy(), n_before=n_before,
                                      n_after=n_left, cache_before=before, cache_after=cache.copy()))
        caches[:, t - 1] = cache
    stop = schedule[-1][0] if schedule else 1
    assert n_left == 0
    assert np.allclose(spent + left, B, rtol=0, atol=1e-10)
    return dict(caches=caches, spent=spent, left=left, m=m, n_pub=n_pub, n_proj=n_proj, empty=empty,
                root_veto=root_veto, root_count=root_count, K=K, stop=stop, histories=histories,
                anchor_epsilon=allocate(np.full(R, plan.rho * B), mask, policy.h),
                anchor_cache=caches[:, 0], release_reads=source.release_reads,
                stop_reason='static' if not schedule else 'opportunities_exhausted')


def evaluate(rec, result, w, stats, cfg, detailed=False):
    """Offline evaluator only; nothing returned here is passed to Policy.decide."""
    truth_raw = data.replay(rec)
    truth_z = w.normalize(truth_raw)
    truth = w.evaluate(truth_raw)[w.target]
    pred = task(w, result['caches'])
    err = abs(pred - truth[None, :])
    out = dict(mae=err.mean(axis=1), spend_fraction=result['spent'] / cfg.get('_B', 1))
    if not detailed:
        return out
    R, T = err.shape
    stop = result['stop']
    out.update(p95=np.quantile(err, .95, axis=1), post_stop_mae=err[:, stop:].sum(axis=1) / max(T - stop, 1),
               projection_rate=result['n_proj'] / result['n_pub'], m=result['m'],
               frac_veto=np.divide(result['empty'], result['K']) if result['K'] else np.full(R, np.nan),
               frac_root_veto=np.divide(result['root_veto'], result['root_count'], out=np.full(R, np.nan), where=result['root_count'] > 0),
               n_pub=result['n_pub'], stop_turn=np.full(R, stop), n_post=np.full(R, T-stop),
               static_plan=np.full(R, result['K'] == 0), used=result['spent'])
    masks_by_t = {h['turn']: h['selected'] for h in result['histories']}
    miss, significant, harm, refresh = np.zeros(R), np.zeros(R), np.zeros(R), np.zeros(R)
    for e in rec['events']:
        t = e['turn']
        roots = [w.roots.index(r) for r in e['measured_roots']]
        sig = np.zeros(len(w.roots), bool)
        sig[roots] = (abs(truth_z[t-1, roots] - truth_z[t-2, roots]) > cfg['significant_delta'] + 1e-12) & (np.array(stats['raw_h'])[roots] > 0)
        if sig.any():
            significant += sig.sum()
            miss += (sig[None, :] & ~masks_by_t.get(t, np.zeros((R, len(sig)), bool))).sum(axis=1)
    for h in result['histories']:
        pub = h['selected'].any(axis=1)
        refresh += pub
        before = abs(task(w, h['cache_before']) - truth[h['turn']-1])
        after = abs(task(w, h['cache_after']) - truth[h['turn']-1])
        harm += pub & (after > before + 1e-12)
        h['immediate_error_drop'] = before - after
    out['MissRate'] = np.divide(miss, significant, out=np.full(R, np.nan), where=significant > 0)
    out['harmful_refresh'] = np.divide(harm, refresh, out=np.full(R, np.nan), where=refresh > 0)
    # Numerical danger: fixed finite-step downward perturbation, compared with
    # the same perturbation at train mu. It is not a clinical event label.
    mu = np.array(stats['mu'])
    base = task(w, result['caches'])
    amp = np.zeros((R, T))
    amp_mu = 0.
    for i in range(len(w.roots)):
        z = result['caches'].copy()
        z[..., i] -= cfg['danger_step']
        amp = np.maximum(amp, abs(task(w, z) - base))
        z0 = mu.copy()
        z0[i] -= cfg['danger_step']
        amp_mu = max(amp_mu, float(abs(task(w, z0) - task(w, mu))))
    danger = amp > cfg['danger_multiple'] * amp_mu + 1e-12
    out['danger_rate'] = danger.mean(axis=1)
    out['danger_mae'] = np.divide((err * danger).sum(axis=1), danger.sum(axis=1), out=np.full(R, np.nan), where=danger.sum(axis=1) > 0)
    out['input_clip_rate'] = np.full(R, np.mean((truth_raw < w.lo) | (truth_raw > w.hi)))
    return out
