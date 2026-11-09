"""Evaluator: corrected event-wise F-beta (ESA-ADB). Own implementation + wrapper around the official one.

Inputs
  labels : DataFrame with columns ID, StartTime, EndTime, Category   (labels.csv merged with anomaly_types.csv,
           timestamps tz-naive UTC). One row per (event, channel) is fine: we group by ID.
  pred   : binary array (1 = alarm) on a uniform time grid `times` (DatetimeIndex, tz-naive).
  full_range : (start, end) of the evaluated period.
"""
import io
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd


def pred_to_events(pred: np.ndarray, times: pd.DatetimeIndex):
    """Runs of 1s -> list of (start, end) half-open [start, end). Ends at the next timestamp (as official code)."""
    p = np.asarray(pred).astype(np.int8)
    d = np.diff(np.concatenate([[0], p, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)            # index of first 0 after the run (may be len(p))
    out = []
    for a, b in zip(starts, ends):
        t0 = times[a]
        t1 = times[b] if b < len(times) else times[-1]   # official: last run closes at last timestamp
        out.append((t0, t1))
    return out


def _overlap(a0, a1, b0, b1):
    return a0 <= b1 and b0 <= a1


def merge_intervals(ivs):
    ivs = sorted(ivs)
    out = []
    for s, e in ivs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def ew_scores(labels: pd.DataFrame, pred, times, full_range, beta: float = 0.5, categories=("Rare Event", "Anomaly")):
    """Corrected event-wise precision/recall/F-beta.  Returns dict."""
    events_pred = pred_to_events(pred, times)
    sel = labels[labels["Category"].isin(categories)]
    all_gt = merge_intervals(list(zip(labels["StartTime"], labels["EndTime"])))   # ALL events (also unselected)

    tp = fn = 0
    matched = [False] * len(events_pred)
    for _, g in sel.groupby("ID"):
        ivs = list(zip(g["StartTime"], g["EndTime"]))
        hit = False
        for k, (p0, p1) in enumerate(events_pred):
            if any(_overlap(p0, p1, s, e) for s, e in ivs):
                matched[k] = True
                hit = True
        tp += hit
        fn += not hit

    # a predicted event that matched nothing counts as FP only if it also touches no GT event at all
    fp = sum((not m) and not any(_overlap(p0, p1, s, e) for s, e in all_gt)
             for m, (p0, p1) in zip(matched, events_pred))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    tnr = 1.0
    if precision > 0:                                  # Sehili et al. correction: time-based true negative rate
        total = (full_range[1] - full_range[0]).total_seconds()
        gt_secs = sum((min(e, full_range[1]) - max(s, full_range[0])).total_seconds()
                      for s, e in all_gt if e >= full_range[0] and s <= full_range[1])
        nominal = total - gt_secs
        fp_secs = 0.0
        for p0, p1 in events_pred:                     # predicted time that is NOT inside any GT event
            cur = (p1 - p0).total_seconds()
            for s, e in all_gt:
                o = (min(p1, e) - max(p0, s)).total_seconds()
                if o > 0:
                    cur -= o
            fp_secs += max(cur, 0.0)
        tnr = 1 - fp_secs / nominal
        precision *= tnr

    f = 0.0 if (beta ** 2 * precision + recall) == 0 else (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    return dict(TP=tp, FP=fp, FN=fn, TNR=tnr, precision=precision, recall=recall, **{f"F{beta}": f})


def official_scores(labels, pred, times, full_range, beta=0.5, categories=("Rare Event", "Anomaly")):
    """Run the official ESA-ADB ESAScores (only for cross-checking our implementation)."""
    root = Path(__file__).resolve().parent.parent / "ESA-ADB" / "timeeval"
    for name, p in [("timeeval", root), ("timeeval.metrics", root / "metrics")]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(p)]
            sys.modules[name] = m
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from timeeval.metrics.ESA_ADB_metrics import ESAScores
    y_true = labels[["ID", "StartTime", "EndTime", "Category"]].drop_duplicates()
    y_true = y_true.copy()
    # official code iterates events per ID; channel column must be removed
    y_pred = np.column_stack([np.asarray(times, dtype=object), np.asarray(pred, dtype=int)])
    m = ESAScores(betas=beta, select_labels={"Category": list(categories)}, full_range=tuple(full_range))
    with redirect_stdout(io.StringIO()):          # official code prints debug lists
        return m.score(y_true, y_pred)
