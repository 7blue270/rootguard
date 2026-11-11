"""Standard-Laplace, pure metric-DP root cache with a pathwise total ledger.

decide() cannot see true values or future events. publish() alone accesses the
current roots. All experiments share this publication implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .data import PublicBatch

METHODS = ('B0', 'B1', 'B2', 'B2p', 'M1-a', 'M1-g', 'M1-b', 'M1-a_fixed')


def paired_uniforms(seed, subject, trajectory, B, repeat, turn, k):
    # Method is intentionally absent. Each repeat/root/turn is distinct.
    key = json.dumps([seed, subject, trajectory, float(B), repeat, turn], separators=(',', ':'))
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], 'little')
    u = np.random.default_rng(value).random(k)
    return np.clip(u, np.finfo(float).eps, 1 - np.finfo(float).eps)


def laplace_standard(u):
    v = np.asarray(u) - .5
    return -np.sign(v) * np.log1p(-2 * np.abs(v))


@dataclass(frozen=True)
class Decision:
    turn: int
    indices: tuple[int, ...]
    eps: tuple[float, ...]
    trigger: str
    gate: bool
    age_force: bool
    L_keep: float | None
    L_refresh: float | None
    ages_before: tuple[int, ...]


class Cache:
    def __init__(self, roots, h, C, B, method, N_max, *, period=4, mu_a=.5,
                 c=.125, eps_max=40., eps_min=.01, q_min=.05, gamma=1., age_mode='per_root'):
        self.roots = list(roots)
        self.h, self.C = np.array(h, float), np.array(C, float)
        if method not in METHODS or B <= 0 or not np.isfinite(B):
            raise ValueError('Invalid method or finite positive B')
        if self.h.shape != self.C.shape or len(roots) != len(h) or np.any(self.h < 0) or np.any(self.C < 0) or not np.all(np.isfinite(self.h + self.C)):
            raise ValueError('Invalid public importance/drift vectors')
        if not self.h.any() or not 0 < mu_a <= 1 or not 0 < c <= 1 or not 0 < q_min <= 1 or eps_min <= 0 or eps_max <= 0 or gamma < 0 or period < 1 or N_max < 0:
            raise ValueError('Invalid public mechanism parameters')
        self.B, self.method, self.period, self.N_max = float(B), method, period, N_max
        self.mu_a = 1. if method == 'B0' or N_max == 0 else mu_a
        self.c, self.eps_max, self.eps_min, self.q_min = c, eps_max, eps_min, q_min
        # B2 preserves the unweighted-quota M-Opt baseline; B2p shares M1 allocation.
        self.gamma = 0. if method in ('B1', 'B2', 'M1-b') else gamma
        planning_h = np.ones_like(self.h) if method == 'B1' else self.h
        self.active = self.h > 0
        w = np.where(self.active, np.sqrt(planning_h), 0.)
        self.quota = self.B * w / w.sum()
        self.root_remaining = self.quota.copy()
        self.last_eps = np.zeros_like(self.h)
        self.ages = np.zeros(len(roots), int)
        self.cache = np.zeros_like(self.h)  # Inactive roots use a fixed public constant.
        self.remaining, self.spent = self.B, 0.
        self.batch_count, self.refresh_count = 0, 0
        self.frozen, self.stop_turn = False, None
        self.charges, self.transcript = [], []
        self.initialized = False
        self.fixed_eps = (1 - self.mu_a) * B / max(N_max // period, 1)
        self.G = np.full_like(self.C, np.inf)
        np.divide(1., self.C, out=self.G, where=self.C > 0)
        self.G = np.maximum(np.floor(self.G), 1.)
        if method == 'M1-g':
            self.G[:] = np.inf
        elif age_mode == 'global':
            self.G[:] = np.min(self.G[self.active])
        elif age_mode == 'none':
            self.G[:] = np.inf
        elif age_mode != 'per_root':
            raise ValueError('Unknown age mode')
        self._pending = None

    def _publish(self, turn, indices, eps, z, uniforms):
        idx = np.array(indices, int)
        eps = np.asarray(eps, float)
        z = np.asarray(z, float)
        if z.shape != self.h.shape or not np.all(np.isfinite(z)) or np.any((z < 0) | (z > 1)):
            raise ValueError('Publication requires normalized, clipped current roots')
        if not np.all(eps > 0) or not np.all(np.isfinite(eps)):
            raise ValueError('Every published root needs positive finite epsilon')
        cost = float(eps.sum())
        if cost > self.remaining + 1e-9:
            raise RuntimeError('Global privacy budget exceeded')
        noise = laplace_standard(np.asarray(uniforms)[idx]) / eps
        self.cache[idx] = z[idx] + noise
        self.last_eps[idx] = eps
        self.root_remaining[idx] -= eps
        self.spent += cost
        self.remaining = max(0., self.B - self.spent)
        self.ages[idx] = 0
        self.charges.extend(float(v) for v in eps)
        assert np.isclose(sum(self.charges), self.spent) and self.spent <= self.B + 1e-8
        release = {'turn': turn, 'roots': [self.roots[i] for i in idx],
                   'eps': eps.tolist(), 'values': self.cache[idx].tolist()}
        self.transcript.append(release)

    def initialize(self, z, uniforms):
        if self.initialized:
            raise RuntimeError('Cache already initialized')
        idx = np.flatnonzero(self.active)
        self._publish(1, idx, self.mu_a * self.quota[idx], z, uniforms)
        self.initialized = True
        self._check_stop(1)

    def _check_stop(self, turn):
        if not self.frozen and (self.remaining < self.eps_min or self.method == 'B0'):
            self.frozen, self.stop_turn = True, turn

    def allocate(self, indices, total):
        """Common root allocator; total is supplied by the scheduling policy."""
        ii = np.asarray(indices, int)
        if not len(ii) or total <= 0 or total > self.remaining + 1e-9:
            raise ValueError('Invalid publication indices or total epsilon')
        ratio = np.maximum(self.root_remaining[ii] / self.quota[ii], self.q_min)
        base = np.ones(len(ii)) if self.method == 'B1' else np.sqrt(self.h[ii])
        weights = base * ratio ** self.gamma
        eps = total * weights / weights.sum()
        assert np.all(eps > 0) and np.isclose(eps.sum(), total)
        return eps

    def decide_scheduled(self, batch: PublicBatch, total=None):
        """Offline matched-timing diagnostic, using the ordinary ledger/publish.

        None means skip. A positive total reuses the ordinary root selection and
        allocator at this opportunity. This API never accepts root values.
        """
        if not isinstance(batch, PublicBatch):
            raise TypeError('Only the current public batch is accepted')
        if not self.initialized or self._pending is not None:
            raise RuntimeError('Initialize first and finish the preceding decision')
        self.batch_count += 1
        self.ages += 1
        if self.batch_count > self.N_max:
            raise ValueError('Public N_max was violated')
        self._check_stop(batch.turn)
        idx = tuple(i for i, r in enumerate(self.roots) if r in batch.measured_roots and self.active[i])
        ages = tuple(int(v) for v in self.ages)
        if total is None:
            return Decision(batch.turn, (), (), 'none', False, False, None, None, ages)
        if self.frozen or not idx:
            raise RuntimeError('A matched release must have budget and currently measured active roots')
        eps = self.allocate(idx, total)
        decision = Decision(batch.turn, idx, tuple(float(v) for v in eps),
                            'matched_periodic', False, False, None, None, ages)
        self._pending = decision
        return decision

    def decide(self, batch: PublicBatch):
        if not isinstance(batch, PublicBatch):
            raise TypeError('Only the current public batch is accepted')
        if not self.initialized or self._pending is not None:
            raise RuntimeError('Initialize first and finish the preceding decision')
        self.batch_count += 1
        self.ages += 1
        if self.batch_count > self.N_max:
            raise ValueError('Public N_max was violated')
        self._check_stop(batch.turn)
        idx = tuple(i for i, r in enumerate(self.roots) if r in batch.measured_roots and self.active[i])
        ages = tuple(int(v) for v in self.ages)
        if self.frozen or not idx:
            return Decision(batch.turn, (), (), 'frozen' if self.frozen else 'empty', False, False, None, None, ages)
        fixed = self.method in ('B1', 'B2', 'M1-a_fixed')
        total = self.fixed_eps if fixed else min(self.c * self.remaining, self.eps_max)
        if total <= 0 or total > self.remaining + 1e-9:
            self.frozen, self.stop_turn = True, batch.turn
            return Decision(batch.turn, (), (), 'frozen', False, False, None, None, ages)
        ii = np.array(idx)
        eps = self.allocate(idx, total)
        keep = float(np.sum(self.h[ii] * (self.ages[ii] * self.C[ii] + 1 / self.last_eps[ii])))
        refresh = float(np.sum(self.h[ii] / eps))
        gate = keep > refresh
        age = bool(np.any(self.ages[ii] >= self.G[ii]))
        if self.method in ('B1', 'B2', 'B2p'):
            trigger = 'periodic' if self.batch_count % self.period == 0 else 'none'
        else:
            trigger = 'both' if gate and age else 'gate' if gate else 'age_force' if age else 'none'
        decision = Decision(batch.turn, idx, tuple(float(v) for v in eps), trigger, gate, age, keep, refresh, ages)
        if trigger != 'none':
            self._pending = decision
        return decision

    def publish(self, decision, z, uniforms):
        if decision is not self._pending:
            raise RuntimeError('Only a fresh committed decision can publish')
        self._publish(decision.turn, decision.indices, decision.eps, z, uniforms)
        self.refresh_count += 1
        self._pending = None
        self._check_stop(decision.turn)

    def snapshot(self):
        return {'spent': self.spent, 'remaining': self.remaining, 'ages': self.ages.tolist(),
                'last_eps': self.last_eps.tolist(), 'root_remaining': self.root_remaining.tolist(),
                'refresh_count': self.refresh_count, 'frozen': self.frozen, 'stop_turn': self.stop_turn}
