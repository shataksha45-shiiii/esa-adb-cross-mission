"""The metric is the most important code in the project: every number depends on it.

Checks, on random synthetic labels and alarms (no ESA data needed):
1. our readable evaluator (evaluator.ew_scores) == the official ESA-ADB code (skipped if ESA-ADB/ is not cloned)
2. the fast evaluator used for tuning (tune.fast_ew) == evaluator.ew_scores
3. the per-month decomposition used for bootstrap CIs (step13.parts) == evaluator.ew_scores
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluator import ew_scores, official_scores            # noqa: E402
from step13 import f_from, month_edges, parts
from step13 import test_gt as gt_for_test               # noqa: E402
from tune import fast_ew, make_gt                           # noqa: E402

HAVE_OFFICIAL = (ROOT / "ESA-ADB" / "timeeval" / "metrics" / "ESA_ADB_metrics.py").exists()


def synthetic(seed, n=6000, n_events=25):
    """5-min grid over ~3 weeks, random labelled events (2 categories + an ignored one), random alarm runs."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2005-01-01", periods=n, freq="5min", unit="ns")   # pipeline works in ns (as data.py)
    rows = []
    for k in range(n_events):
        a = int(rng.integers(0, n - 200))
        L = int(rng.integers(1, 120))
        cat = rng.choice(["Anomaly", "Rare Event", "Communication Gap"], p=[0.45, 0.45, 0.10])
        for ch in rng.choice(10, size=int(rng.integers(1, 3)), replace=False):
            rows.append(dict(ID=f"id_{k}", Channel=f"channel_{ch}", StartTime=times[a],
                             EndTime=times[min(a + L, n - 1)], Category=cat))
    lab = pd.DataFrame(rows)
    pred = np.zeros(n, int)
    for _ in range(int(rng.integers(5, 60))):
        a = int(rng.integers(0, n - 50))
        pred[a:a + int(rng.integers(1, 40))] = 1
    return times, lab, pred


@pytest.mark.parametrize("seed", range(8))
def test_fast_ew_matches_readable_evaluator(seed):
    times, lab, pred = synthetic(seed)
    slow = ew_scores(lab, pred, times, (times[0], times[-1]))
    gt = make_gt(lab, times[0], times[-1] + pd.Timedelta("5min"))
    gt["nominal"] = (times[-1] - times[0]).value - (slow_gt_ns(lab, times))
    fast = fast_ew(gt, pred.astype(bool), times.values.astype(np.int64))
    assert fast["TP"] == slow["TP"] and fast["FP"] == slow["FP"] and fast["FN"] == slow["FN"]
    assert fast["F"] == pytest.approx(slow["F0.5"], abs=1e-12)


def slow_gt_ns(lab, times):
    """GT time inside the full range, as evaluator.ew_scores computes it (merged intervals, clipped)."""
    iv = sorted(zip(lab.StartTime, lab.EndTime))
    merged = []
    for s, e in iv:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    t0, t1 = times[0], times[-1]
    return sum((min(e, t1) - max(s, t0)).value for s, e in merged if e >= t0 and s <= t1)


@pytest.mark.parametrize("seed", range(8))
def test_month_parts_match_readable_evaluator(seed):
    times, lab, pred = synthetic(seed)
    slow = ew_scores(lab, pred, times, (times[0], times[-1]))
    pa = parts(gt_for_test(lab, times), pred.astype(bool), times.values.astype(np.int64), month_edges(times))
    assert float(f_from(pa.sum(1))) == pytest.approx(slow["F0.5"], abs=1e-12)
    assert int(pa[0].sum()) == slow["TP"] and int(pa[2].sum()) == slow["FP"]


@pytest.mark.skipif(not HAVE_OFFICIAL, reason="official ESA-ADB code not cloned into ESA-ADB/")
@pytest.mark.parametrize("seed", range(5))
def test_readable_evaluator_matches_official_code(seed):
    times, lab, pred = synthetic(seed)
    ours = ew_scores(lab, pred, times, (times[0], times[-1]))
    off = official_scores(lab, pred, times, (times[0], times[-1]))
    assert ours["F0.5"] == pytest.approx(off["EW_F_0.50"], abs=1e-12)
    for cats in (("Anomaly",),):
        o = ew_scores(lab, pred, times, (times[0], times[-1]), categories=cats)
        f = official_scores(lab, pred, times, (times[0], times[-1]), categories=cats)
        assert o["F0.5"] == pytest.approx(f["EW_F_0.50"], abs=1e-12)
