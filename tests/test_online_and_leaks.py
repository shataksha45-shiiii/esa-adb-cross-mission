"""Online (causal) rules and leakage guards. No ESA data needed, except the last test (skipped without the cache).

ESA-ADB's online rule (paper p.30): a decision at time t may use only samples up to t.
1. causal event builder: the output up to t never changes when the future changes, and never back-dates
2. rolling normalisation: past outputs never depend on future samples
3. normaliser: statistics come from the training slice only
4. data grid: each 5-min value is labelled by the END of its bin (needs the preprocessed cache)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from normaliser import fit_matrix, transform_matrix          # noqa: E402
from rolling import DAY, rolling_center, rolling_robust_z    # noqa: E402
from tune import build_events_causal                          # noqa: E402


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("gap,min_len", [(0, 1), (12, 1), (12, 3), (36, 6)])
def test_event_builder_is_causal(seed, gap, min_len):
    rng = np.random.default_rng(seed)
    a = rng.random(3000) < 0.05
    full = build_events_causal(a, gap, min_len)
    for k in rng.integers(1, len(a), 40):
        assert np.array_equal(build_events_causal(a[:k], gap, min_len), full[:k])


@pytest.mark.parametrize("min_len", [1, 3, 6])
def test_event_builder_never_backdates(min_len):
    a = np.zeros(100, bool)
    a[40:40 + min_len + 2] = True
    out = build_events_causal(a, gap=12, min_len=min_len)
    assert not out[:40 + min_len - 1].any()          # the alarm cannot start before min_len exceedances are seen
    assert out[40 + min_len - 1]


def test_event_builder_needs_min_len_consecutive():
    a = np.zeros(50, bool)
    a[[10, 12, 14]] = True                           # never 3 in a row
    assert not build_events_causal(a, gap=12, min_len=3).any()


@pytest.mark.parametrize("fn", [rolling_center, rolling_robust_z])
def test_rolling_normalisation_is_causal(fn):
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((40 * DAY, 3)).astype(np.float32)
    cut = 25 * DAY + 17
    Z2 = Z.copy()
    Z2[cut:] += 50.0                                 # change only the future
    assert np.array_equal(fn(Z)[:cut], fn(Z2)[:cut])


def test_normaliser_uses_training_slice_only():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 4)).astype(np.float32)
    i = 1000
    n1 = fit_matrix(X[:i])
    X2 = X.copy()
    X2[i:] *= 100.0                                  # wildly different test period
    n2 = fit_matrix(X2[:i])
    assert np.array_equal(transform_matrix(X[:i], n1), transform_matrix(X2[:i], n2))


CACHE = ROOT / "cache" / "M2"


@pytest.mark.skipif(not CACHE.exists(), reason="preprocessed cache not built (python src/preprocess.py)")
def test_grid_labels_bins_by_end():
    import pandas as pd
    from data import load_mission
    t, X, chans = load_mission(2)
    z = np.load(CACHE / f"channel_{chans[0]}.npz")
    start = pd.Timestamp(int(z["start"]))
    assert t[0] >= start + pd.Timedelta("5min")     # first value is known only at the END of its bin
