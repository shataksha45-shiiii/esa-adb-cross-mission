"""Full pipeline + cross-mission transfer table.

For each (train mission A, test mission B):
  1. normaliser: fit on B's TRAIN period (B's own statistics, no labels)
  2. model: trained on A's TRAIN period only (all A target channels, pooled, no labels)  [A==B -> in-mission]
  3. scores on all of B, per channel
  4. EVT threshold per channel, calibrated on B's TRAIN-period scores (no labels, no test data)
  5. alarms on B's TEST period -> event builder -> mission-level alarm vector
  6. ESA-ADB corrected event-wise F0.5 on B's test labels (labels used ONLY here)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SPLIT, load_labels, load_mission, split_index          # noqa: E402
from detector import W, baseline_score_channel, score_channel, train_model  # noqa: E402
from evaluator import ew_scores                                          # noqa: E402
from events import mission_alarm                                         # noqa: E402
from normaliser import fit_matrix, transform_matrix                      # noqa: E402
from threshold import evt_threshold                                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HEADLINE_Q = 1e-5       # fixed BEFORE looking at any test result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qs", type=float, nargs="+", default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    ap.add_argument("--out", default=str(ROOT / "results" / "transfer.csv"))
    args = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)

    data = {}
    for m in (1, 2):
        t, X, ch = load_mission(m)
        i = split_index(m, t)
        norms = fit_matrix(X[:i])                           # fit on TRAIN only
        data[m] = dict(t=t, i=i, Z=transform_matrix(X, norms), ch=ch, lab=load_labels(m))

    models = {}
    for a in (1, 2):
        t0 = time.time()
        print(f"[train] model on mission {a} train period", flush=True)
        models[a] = train_model(data[a]["Z"][:data[a]["i"]], steps=args.steps, seed=args.seed)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for b in (1, 2):
        D = data[b]
        t, i, Z, lab = D["t"], D["i"], D["Z"], D["lab"]
        tt = t[i:]
        lab_test = lab[lab["StartTime"] >= SPLIT[b]]
        full_range = (tt[0], tt[-1])
        detectors = {f"AE(trained on M{a})": (lambda z, a=a: score_channel(models[a], z)) for a in (1, 2)}
        detectors["baseline max|z|"] = baseline_score_channel
        for name, fn in detectors.items():
            t0 = time.time()
            S = np.stack([fn(Z[:, j]) for j in range(Z.shape[1])], axis=1)       # [T, C]
            for q in args.qs:
                thr = np.array([evt_threshold(S[W:i, j], q=q) for j in range(S.shape[1])])
                alarms = S[i:] > thr[None, :]
                pred = mission_alarm(alarms)
                for cats in (("Rare Event", "Anomaly"), ("Anomaly",)):
                    r = ew_scores(lab_test, pred, tt, full_range, beta=0.5, categories=cats)
                    rows.append(dict(test_mission=b, detector=name, q=q, headline=(q == HEADLINE_Q),
                                     categories="+".join(cats), alarm_frac=pred.mean(), **r))
            print(f"[eval] test M{b} {name}: {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df[df.headline].to_string())


if __name__ == "__main__":
    main()
