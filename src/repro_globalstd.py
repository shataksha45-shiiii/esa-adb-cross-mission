"""Roadmap item 2: reproduce an official ESA-ADB baseline (GlobalSTD) inside OUR pipeline.

Official algorithm (ESA-ADB/TimeEval-algorithms/std/algorithm.py), copied rule for rule:
  train  : per target channel, mean and std over TRAINING points that are NOT labelled (any category)
           (std == 0 -> 1)
  execute: alarm on channel j at time t if x_j(t) > mean_j + tol*std_j or x_j(t) < mean_j - tol*std_j
  no post-processing; mission alarm = OR over target channels (the ESA metric takes the union)
Differences from the paper's run (flagged): our 5-min grid (last value per bin, labelled by bin END) instead of
their preprocessed native-rate data; raw values (no differencing). Published targets (ESA-ADB v2, Table 2, all
target channels, corrected event-wise F0.5, all events): M1 STD3 < 0.001, STD5 0.003; M2 STD3 0.008, STD5 0.064.
Uses test labels only to score this BASELINE (no setting of ours is chosen from it).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SPLIT, load_labels, load_mission, split_index      # noqa: E402
from evaluator import ew_scores, official_scores                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = {(1, 3): "< 0.001", (1, 5): "0.003", (2, 3): "0.008", (2, 5): "0.064"}


def labelled_mask(t, lab, chans):
    """[T, C] bool: point lies inside any labelled event (any category) on that channel."""
    m = np.zeros((len(t), len(chans)), bool)
    col = {f"channel_{c}": j for j, c in enumerate(chans)}
    for _, r in lab.iterrows():
        j = col.get(r.Channel)
        if j is None:
            continue
        a, b = t.searchsorted(r.StartTime), t.searchsorted(r.EndTime, side="right")
        m[a:b, j] = True
    return m


def main():
    rows = []
    for m in (1, 2):
        t, X, chans = load_mission(m)
        i = split_index(m, t)
        lab = load_labels(m)
        L = labelled_mask(t[:i], lab[lab.StartTime < SPLIT[m]], chans)
        Xtr = X[:i].astype(np.float64)
        mu = np.array([Xtr[~L[:, j], j].mean() for j in range(X.shape[1])])
        sd = np.array([Xtr[~L[:, j], j].std() for j in range(X.shape[1])])
        sd = np.where(sd == 0, 1, sd)
        tt = t[i:]
        tl = lab[lab["EndTime"] > SPLIT[m]].copy()
        tl["StartTime"] = tl["StartTime"].clip(lower=SPLIT[m])
        for tol in (3, 5):
            Xte = X[i:].astype(np.float64)
            pred = ((Xte > mu + tol * sd) | (Xte < mu - tol * sd)).any(1).astype(int)
            r = ew_scores(tl, pred, tt, (tt[0], tt[-1]))
            off = official_scores(tl, pred, tt, (tt[0], tt[-1])) if pred.any() else {"EW_F_0.50": 0.0}
            rows.append(dict(mission=m, tol=tol, published_F=PUBLISHED[(m, tol)], ours_F=round(off["EW_F_0.50"], 4),
                             P=round(r["precision"], 4), R=round(r["recall"], 4), TP=r["TP"], FP=r["FP"],
                             alarm_frac=round(pred.mean(), 4)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "repro_globalstd.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
