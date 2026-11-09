"""Step 8a: for every post-processing setting, measure UNLABELLED alarm statistics and (for diagnosis) CV F0.5.

Rows: (target b, model in {own, transferred}, fold, setting) -> alarm_frac, events_per_month, F.
alarm_frac / events_per_month need no labels; F uses the target's TRAINING labels (diagnosis / rule design only).
Seed 0 only. Output: results/adapt_stats.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_labels, load_mission, split_index                  # noqa: E402
from diag_transfer import pred_from                                      # noqa: E402
from normaliser import fit_matrix, transform_matrix                      # noqa: E402
from run_final import ABS_W, AE, STEPS                                   # noqa: E402
from tune import POST, Q2, evt_fit, fast_ew, make_gt, score_ae, score_maxabs, train_ae  # noqa: E402

GRID = [dict(q=q, q2=q2, gap=g, min_len=ml, kofn=k) for q, q2, g, ml, k in
        itertools.product(POST["q"] + [0.0], Q2, POST["gap"], POST["min_len"], POST["kofn"]) if q > 0 or q2 > 0]
MONTH_NS = 30.44 * 86400e9


def n_events(pred):
    return int(np.sum(np.diff(np.concatenate([[0], pred.astype(np.int8)])) == 1))


def main(seed=0):
    full = {}
    for m in (1, 2):
        t, X, _ = load_mission(m)
        i = split_index(m, t)
        full[m] = dict(t=t, X=X, i=i, Zsrc=transform_matrix(X[:i], fit_matrix(X[:i])), lab=load_labels(m))
    transferred = {a: train_ae(full[a]["Zsrc"], AE["w"], AE["hidden"], AE["code"], STEPS, seed, center=True) for a in (1, 2)}
    rows = []
    for b in (1, 2):
        d = full[b]
        mid = d["i"] // 2
        for k, (fit, ev) in enumerate([(slice(0, mid), slice(mid, d["i"])), (slice(mid, d["i"]), slice(0, mid))]):
            Z = transform_matrix(d["X"][:d["i"]], fit_matrix(d["X"][fit]))
            te = d["t"][ev]
            gt, tns = make_gt(d["lab"], te[0], te[-1]), te.values.astype(np.int64)
            months = (tns[-1] - tns[0]) / MONTH_NS
            Sb = np.stack([score_maxabs(Z[:, j], ABS_W) for j in range(Z.shape[1])], 1)
            fitsB = [evt_fit(Sb[fit][ABS_W:][:, j]) for j in range(Z.shape[1])]
            own = train_ae(Z[fit], AE["w"], AE["hidden"], AE["code"], STEPS, seed, center=True)
            for mname, model in (("own", own), ("transferred", transferred[3 - b])):
                Sa = np.stack([score_ae([model], Z[:, j], AE["w"], AE["reduce"], True) for j in range(Z.shape[1])], 1)
                fitsA = [evt_fit(Sa[fit][AE["w"]:][:, j]) for j in range(Z.shape[1])]
                for gi, c in enumerate(GRID):
                    p = pred_from(fitsA, fitsB, Sa[ev], Sb[ev], c)
                    r = fast_ew(gt, p, tns)
                    rows.append(dict(target=b, model=mname, fold=k, gi=gi, **c, alarm_frac=p.mean(),
                                     ev_month=n_events(p) / months, F=r["F"], P=r["P"], R=r["R"]))
            print(f"target M{b} fold {k} done", flush=True)
    pd.DataFrame(rows).to_csv(Path(__file__).resolve().parent.parent / "results" / "adapt_stats.csv", index=False)


if __name__ == "__main__":
    main()
