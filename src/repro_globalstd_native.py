"""Step 12 (roadmap item 2): GlobalSTD on Mission 2 at the NATIVE ESA-ADB grid (18 s) vs coarser grids.

Question: is our 5-min grid the reason GlobalSTD5 scores 0.400 for us but 0.064 in ESA-ADB (v2, Table 2)?
Preprocessing copied from ESA-ADB/notebooks/data-prep/Mission2_semiunsupervised_prep_from_raw.py:
  zero-order hold onto a regular grid (value at grid time t = last raw sample at or before t),
  train = samples <= 2001-10-01, test = samples > 2001-10-01, M2 target channels need no differencing.
GlobalSTD copied from ESA-ADB/TimeEval-algorithms/std/algorithm.py:
  mean/std per channel over TRAINING grid points not inside a labelled event of that channel (std 0 -> 1),
  alarm = value outside mean +- tol*std, OR over target channels, no post-processing.
Not copied (flagged): their "restore annotated samples" step (re-inserts an annotated raw sample hidden inside a
bin); grid start = first raw sample floored to the grid (theirs) vs a common grid for all channels (ours).
Scored with the OFFICIAL ESA-ADB metric (Rare Event + Anomaly) on the test period. Baseline only: no setting of ours
is chosen from these test labels.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SPLIT, load_labels                       # noqa: E402
from evaluator import ew_scores, official_scores          # noqa: E402
from loader import load_channel                           # noqa: E402
from preprocess import target_channels                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GRIDS = ["18s", "60s", "300s"]
TOLS = (3, 5)
M = 2


def zoh(raw_t, raw_v, grid):
    idx = np.searchsorted(raw_t, grid, side="right") - 1
    return raw_v[np.clip(idx, 0, None)]


def main():
    lab = load_labels(M)
    chans = target_channels(M)
    split = SPLIT[M].value
    raws = {}
    t0 = time.time()
    for c in chans:
        s = load_channel(M, c)
        raws[c] = (s.index.values.astype("datetime64[ns]").astype(np.int64), s.to_numpy(np.float64))
    lo = min(v[0][0] for v in raws.values())
    hi = max(v[0][-1] for v in raws.values())
    print(f"loaded {len(chans)} channels in {time.time() - t0:.0f}s", flush=True)
    rows = []
    for g in GRIDS:
        step = pd.Timedelta(g).value
        grid = np.arange(lo - lo % step, hi + step, step, dtype=np.int64)
        tr, te = grid <= split, grid > split
        tt = pd.DatetimeIndex(grid[te])
        alarm = {tol: np.zeros(te.sum(), bool) for tol in TOLS}
        for c in chans:
            rt, rv = raws[c]
            v = zoh(rt, rv, grid)
            labelled = np.zeros(len(grid), bool)
            for _, r in lab[lab.Channel == f"channel_{c}"].iterrows():
                a = np.searchsorted(grid, r.StartTime.value, "left")
                b = np.searchsorted(grid, r.EndTime.value, "right")
                labelled[a:b] = True
            keep = tr & ~labelled
            mu, sd = v[keep].mean(), v[keep].std()
            sd = sd if sd > 0 else 1.0
            for tol in TOLS:
                alarm[tol] |= (v[te] > mu + tol * sd) | (v[te] < mu - tol * sd)
        tl = lab[lab["EndTime"] > SPLIT[M]].copy()
        tl["StartTime"] = tl["StartTime"].clip(lower=SPLIT[M])
        for tol in TOLS:
            p = alarm[tol].astype(int)
            t1 = time.time()
            off = official_scores(tl, p, tt, (tt[0], tt[-1])) if p.any() else {"EW_F_0.50": 0.0}
            r = ew_scores(tl, p, tt, (tt[0], tt[-1]))
            n_ev = int(np.sum(np.diff(np.concatenate([[0], p])) == 1))
            rows.append(dict(grid=g, tol=tol, official_F=round(off["EW_F_0.50"], 4), P=round(r["precision"], 4),
                             R=round(r["recall"], 4), TP=r["TP"], FP=r["FP"], alarm_events=n_ev,
                             alarm_frac=round(p.mean(), 4), test_points=len(p)))
            print(rows[-1], f"[{time.time() - t1:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results" / "repro_globalstd_native_M2.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
