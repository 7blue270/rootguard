"""Smoke tests for WP1 (mechanisms + RootGuard) and WP2 (templates)."""

import math
import sys
import pathlib

# Add repo root to path so tests run with `python tests/test_smoke.py`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from sanitizers.mechanisms import (
    sanitize, value_to_index, index_to_value,
    EXPONENTIAL, BLAPLACE, STAIRCASE, VALID_MECHANISMS,
)
from sanitizers.rootguard import RootGuard
from sanitizers.allocations import load_allocation, load_per_call_eps
from templates.fib4 import FIB4
from templates.homa import HOMA
from templates.anemia import ANEMIA


# Path to the example allocations file used in tests below.
EXAMPLE_ALLOC = pathlib.Path(__file__).resolve().parent.parent / "data" / "allocations.example.csv"


# ---------------------------------------------------------------------------
# Sanitizer tests
# ---------------------------------------------------------------------------

def test_index_roundtrip():
    """value -> index -> value is identity (up to grid quantization) for grid points."""
    lo, hi, m = 0.0, 100.0, 1000
    delta = (hi - lo) / (m - 1)
    for s_int in [0, 1, 500, 999]:
        v = index_to_value(s_int, lo, hi, m)
        t_back = value_to_index(v, lo, hi, m)
        assert abs(t_back - s_int) < 1e-9, f"roundtrip failed at s={s_int}: t_back={t_back}"


def test_sanitize_within_domain():
    """All mechanisms produce outputs in [lo, hi]."""
    rng = np.random.default_rng(seed=42)
    for mech in VALID_MECHANISMS:
        for _ in range(20):
            v = rng.uniform(0, 100)
            out = sanitize(v, lo=0.0, hi=100.0, eps=0.5, mechanism=mech, rng=rng)
            assert 0.0 <= out <= 100.0, f"{mech} produced out-of-range output {out}"


def test_sanitize_high_eps_low_noise():
    """At very high eps, sanitized output should be very close to input."""
    rng = np.random.default_rng(seed=42)
    v = 50.0
    delta = (100.0 - 0.0) / (1000 - 1)
    for mech in VALID_MECHANISMS:
        # Average noise magnitude across many samples at high eps
        diffs = []
        for _ in range(50):
            out = sanitize(v, lo=0.0, hi=100.0, eps=10.0, mechanism=mech, rng=rng)
            diffs.append(abs(out - v))
        mean_diff = sum(diffs) / len(diffs)
        # At eps=10, expected noise should be much less than the domain width
        assert mean_diff < 5.0, f"{mech} noise at eps=10 too high: {mean_diff}"


def test_sanitize_reproducible():
    """Same RNG seed produces same output."""
    for mech in VALID_MECHANISMS:
        out1 = sanitize(50.0, 0.0, 100.0, 0.5, mech, rng=np.random.default_rng(seed=7))
        out2 = sanitize(50.0, 0.0, 100.0, 0.5, mech, rng=np.random.default_rng(seed=7))
        assert out1 == out2, f"{mech} not reproducible: {out1} vs {out2}"


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------

def test_fib4_compute():
    """FIB-4 = 66 * 12 / (170 * sqrt(10)) ≈ 1.473 (matches paper example)."""
    t = FIB4()
    out = t.compute({"age": 66, "AST": 12.0, "PLT": 170.0, "ALT": 10.0})
    expected = (66 * 12.0) / (170.0 * math.sqrt(10.0))
    assert abs(out - expected) < 1e-9
    # Should be in indeterminate class (1)
    assert t.risk_class(out) == 1


def test_fib4_risk_classes():
    t = FIB4()
    assert t.risk_class(0.5) == 0
    assert t.risk_class(2.0) == 1
    assert t.risk_class(5.0) == 2


def test_homa_compute():
    t = HOMA()
    # Glu=100, Ins=8.1 -> 810/405 = 2.0 (borderline class 1)
    assert abs(t.compute({"Glu": 100.0, "Ins": 8.1}) - 2.0) < 1e-9
    assert t.risk_class(2.0) == 1


def test_anemia_compute():
    t = ANEMIA()
    # Hb=14, Hct=42 -> 14/42*100 ≈ 33.33 -> normochromic (1)
    out = t.compute({"Hb": 14.0, "Hct": 42.0, "RBC": 5.0})
    assert abs(out - 100.0 * 14.0 / 42.0) < 1e-9
    assert t.risk_class(out) == 1


# ---------------------------------------------------------------------------
# RootGuard tests
# ---------------------------------------------------------------------------

def test_rootguard_caches():
    """get(name) returns the same value on repeated calls."""
    template = FIB4()
    patient = {"age": 50.0, "AST": 25.0, "PLT": 200.0, "ALT": 30.0}
    eps = {r: 0.025 for r in template.roots}  # B = 0.1
    rg = RootGuard.initialize(
        patient, template, eps,
        rng=np.random.default_rng(seed=42),
    )
    v1 = rg.get("AST")
    v2 = rg.get("AST")
    v3 = rg.get("AST")
    assert v1 == v2 == v3, f"cache failure: {v1}, {v2}, {v3}"


def test_rootguard_target_deterministic():
    """compute_target() is deterministic given a fixed cached vector."""
    template = HOMA()
    patient = {"Glu": 100.0, "Ins": 10.0}
    eps = {r: 0.05 for r in template.roots}  # B = 0.1
    rg = RootGuard.initialize(
        patient, template, eps,
        rng=np.random.default_rng(seed=42),
    )
    t1 = rg.compute_target()
    t2 = rg.compute_target()
    assert t1 == t2, f"target not deterministic: {t1}, {t2}"


def test_rootguard_different_seeds_differ():
    """Different RNG seeds produce different cached values (sanity)."""
    template = ANEMIA()
    patient = {"Hb": 14.0, "Hct": 42.0, "RBC": 5.0}
    eps = {r: 0.033 for r in template.roots}
    rg1 = RootGuard.initialize(patient, template, eps, rng=np.random.default_rng(seed=1))
    rg2 = RootGuard.initialize(patient, template, eps, rng=np.random.default_rng(seed=2))
    # At least one root should differ (probabilistically very likely)
    same = all(rg1.get(r) == rg2.get(r) for r in template.roots)
    assert not same, "different seeds gave identical cached vectors"


# ---------------------------------------------------------------------------
# Allocation loader tests
# ---------------------------------------------------------------------------

def test_load_allocation_opt():
    """M-Opt allocation for FIB4 sums to total budget B=(2k+1)*eps=0.9 at eps=0.1."""
    alloc = load_allocation("FIB4", "exp", "2k+1", "opt", csv_path=EXAMPLE_ALLOC)
    assert set(alloc.keys()) == {"age", "AST", "PLT", "ALT"}
    total = sum(alloc.values())
    assert abs(total - 0.9) < 1e-6, f"expected B=0.9, got {total}"
    # PLT should get the most budget (it's in the denominator with the largest sensitivity)
    assert alloc["PLT"] > alloc["age"]


def test_load_allocation_roots_uniform():
    """M-Roots is uniform — all roots get the same eps."""
    alloc = load_allocation("FIB4", "exp", "2k+1", "roots", csv_path=EXAMPLE_ALLOC)
    values = list(alloc.values())
    assert len(set(values)) == 1, f"M-Roots should be uniform, got {alloc}"


def test_load_allocation_anemia_zero_sensitivity():
    """ANEMIA M-Opt clamps RBC at eps_min (zero-sensitivity root)."""
    alloc = load_allocation("ANEMIA", "exp", "2k+1", "opt", csv_path=EXAMPLE_ALLOC)
    assert alloc["RBC"] < 0.01, f"RBC should be clamped at eps_min, got {alloc['RBC']}"
    assert alloc["Hb"] > alloc["RBC"] * 100  # active roots get vastly more budget


def test_load_per_call_eps():
    """M-All returns a single per-call epsilon scalar."""
    eps = load_per_call_eps("FIB4", "exp", "2k+1", csv_path=EXAMPLE_ALLOC)
    assert eps == 0.1


def test_load_allocation_missing_cell_raises():
    """Asking for a non-existent (template, mechanism, B_setting, config) raises."""
    try:
        load_allocation("FIB4", "blap", "2k+1", "opt", csv_path=EXAMPLE_ALLOC)
    except KeyError:
        return
    raise AssertionError("expected KeyError for missing cell")


def test_load_allocation_threads_through_rootguard():
    """End-to-end: load M-Opt allocation, hand to RootGuard.initialize(), no errors."""
    alloc = load_allocation("HOMA", "exp", "2k+1", "opt", csv_path=EXAMPLE_ALLOC)
    template = HOMA()
    patient = {"Glu": 100.0, "Ins": 10.0}
    rg = RootGuard.initialize(
        patient, template, alloc,
        mechanism="exp",
        rng=np.random.default_rng(seed=42),
    )
    # All roots should have a cached value
    for root in template.roots:
        v = rg.get(root)
        lo, hi = template.bounds[root]
        assert lo <= v <= hi, f"{root}={v} outside [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_index_roundtrip,
        test_sanitize_within_domain,
        test_sanitize_high_eps_low_noise,
        test_sanitize_reproducible,
        test_fib4_compute,
        test_fib4_risk_classes,
        test_homa_compute,
        test_anemia_compute,
        test_rootguard_caches,
        test_rootguard_target_deterministic,
        test_rootguard_different_seeds_differ,
        test_load_allocation_opt,
        test_load_allocation_roots_uniform,
        test_load_allocation_anemia_zero_sensitivity,
        test_load_per_call_eps,
        test_load_allocation_missing_cell_raises,
        test_load_allocation_threads_through_rootguard,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed")
