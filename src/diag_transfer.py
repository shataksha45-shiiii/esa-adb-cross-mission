"""Step 7b diagnosis: does transfer fail because of the MODEL or the SETTINGS?  (Training periods only, no test.)

For target mission B, on B's 2-fold blocked CV (same folds as tune.py):
  model    : trained on B's fold-fit half (own)   OR  on A's full training period (transferred)
  settings : CFG[B] (tuned on B)                   OR  CFG[A] (tuned on A, what transfer actually uses)
  + "best"  : best of the full grid for that model (how much signal the model has if settings were perfect)
NOTE: this uses B's TRAINING labels to evaluate. Diagnosis only; never used to pick the reported config.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_labels, load_mission, split_index                       # noqa: E402
from tune import build_events_causal as build_events                          # noqa: E402  (Step 9: online builder)
from normaliser import fit_matrix, transform_matrix                           # noqa: E402
from run_final import ABS_W, AE, CFG, STEPS                                   # noqa: E402
from tune import POST, Q2, evt_fit, evt_thr, fast_ew, make_gt, score_ae, score_maxabs, train_ae  # noqa: E402

SEEDS = [0, 1]


def pred_from(fitsA, fitsB, SEa, SEb, c):
    al = np.zeros(SEa.shape, bool)
    if c["q"] > 0:
        al |= SEa > np.array([evt_thr(f, c["q"]) for f in fitsA])[None, :]
    if c["q2"] > 0:
        al |= SEb > np.array([evt_thr(f, c["q2"]) for f in fitsB])[None, :]
    cnt = np.zeros(len(al), dtype=np.int16)
    for j in range(al.shape[1]):
        if al[:, j].any():
            cnt += build_events(al[:, j], c["gap"], c["min_len"])
    return cnt >= c["kofn"]


def main():
    full = {}
    for m in (1, 2):
        t, X, _ = load_mission(m)
        i = split_index(m, t)
        full[m] = dict(t=t, X=X, i=i, Zsrc=transform_matrix(X[:i], fit_matrix(X[:i])), lab=load_labels(m))
    grid = [dict(q=q, q2=q2, gap=g, min_len=ml, kofn=k) for q, q2, g, ml, k in
            itertools.product(POST["q"], Q2, POST["gap"], POST["min_len"], POST["kofn"])]
    rows = []
    for seed in SEEDS:
        transferred = {a: train_ae(full[a]["Zsrc"], AE["w"], AE["hidden"], AE["code"], STEPS, seed, center=True)
                       for a in (1, 2)}
        for b in (1, 2):
            a = 3 - b
            d = full[b]
            mid = d["i"] // 2
            for k, (fit, ev) in enumerate([(slice(0, mid), slice(mid, d["i"])), (slice(mid, d["i"]), slice(0, mid))]):
                Z = transform_matrix(d["X"][:d["i"]], fit_matrix(d["X"][fit]))
                te = d["t"][ev]
                gt, tns = make_gt(d["lab"], te[0], te[-1]), te.values.astype(np.int64)
                Sb = np.stack([score_maxabs(Z[:, j], ABS_W) for j in range(Z.shape[1])], 1)
                fitsB = [evt_fit(Sb[fit][ABS_W:][:, j]) for j in range(Z.shape[1])]
                own = train_ae(Z[fit], AE["w"], AE["hidden"], AE["code"], STEPS, seed, center=True)
                for mname, model in (("own", own), ("transferred", transferred[a])):
                    Sa = np.stack([score_ae([model], Z[:, j], AE["w"], AE["reduce"], True) for j in range(Z.shape[1])], 1)
                    fitsA = [evt_fit(Sa[fit][AE["w"]:][:, j]) for j in range(Z.shape[1])]
                    SEa, SEb = Sa[ev], Sb[ev]
                    for sname, c in ((f"tuned on M{b} (own)", CFG[b]), (f"tuned on M{a} (transfer)", CFG[a])):
                        r = fast_ew(gt, pred_from(fitsA, fitsB, SEa, SEb, c), tns)
                        rows.append(dict(target=b, seed=seed, fold=k, model=mname, settings=sname, **r))
                    # AE-only grid (q2 = 0): how good is this model's score by itself with ideal settings?
                    for gi, c in enumerate(grid):
                        if c["q2"] == 0:
                            r = fast_ew(gt, pred_from(fitsA, fitsB, SEa, SEb, c), tns)
                            rows.append(dict(target=b, seed=seed, fold=k, model=mname, settings=f"grid{gi}", **r))
                print(f"seed {seed} target M{b} fold {k} done", flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "results" / "diag_transfer.csv"
    df.to_csv(out, index=False)
    named = df[~df.settings.str.startswith("grid")]
    print(named.groupby(["target", "model", "settings"])[["F", "P", "R", "TP", "FP"]].mean().round(3).to_string())
    g = df[df.settings.str.startswith("grid")].groupby(["target", "model", "settings"]).F.mean().reset_index()
    best = g.loc[g.groupby(["target", "model"]).F.idxmax()]
    print("\nAE-only, best grid setting (optimistic):")
    print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
