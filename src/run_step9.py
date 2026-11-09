"""Step 9 evaluation of injection-calibrated settings (design frozen in notes/step9-design-frozen.md).

--dev  : diagnosis on the TARGET's TRAINING labels (2-fold CV, same folds as tune.py). Seed 0.
         Compares 3 settings for each (source, target): injection-chosen | source-tuned (current transfer) | target-tuned (upper bound).
--test : ONE final run on test, 5 seeds, official metric. Settings = v3 (replayed real events + v2 rule). -> results/final_step9.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SPLIT, load_labels, load_mission, split_index                  # noqa: E402
from diag_transfer import pred_from                                             # noqa: E402 (online builder since Step 9)
from evaluator import ew_scores, official_scores                                # noqa: E402
from normaliser import fit_matrix, transform_matrix                             # noqa: E402
from run_final import ABS_W, AE, CFG, SEEDS, STEPS, alarm_vector                # noqa: E402
from tune import evt_fit, fast_ew, make_gt, score_ae, score_maxabs, train_ae    # noqa: E402
from inject_v2 import cfg_v2                                                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["q", "q2", "gap", "min_len", "kofn"]


def inj_cfg(s, t):
    """Injection-chosen setting: same aggregation as inject.calibrate (mean F over folds x repeats, first max)."""
    df = pd.read_csv(ROOT / "results" / "inject" / f"S{s}_T{t}.csv")
    agg = df.groupby(KEYS, sort=False).F.mean().reset_index()
    r = agg.loc[agg.F.idxmax()]
    return dict(q=float(r.q), q2=float(r.q2), gap=int(r.gap), min_len=int(r.min_len), kofn=int(r.kofn))


def load_all():
    D = {}
    for m in (1, 2):
        t, X, _ = load_mission(m)
        i = split_index(m, t)
        D[m] = dict(t=t, X=X, i=i, lab=load_labels(m))
    return D


def dev(D, seed=0):
    rows = []
    for s in (1, 2):
        d = D[s]
        model = train_ae(transform_matrix(d["X"][:d["i"]], fit_matrix(d["X"][:d["i"]])), AE["w"], AE["hidden"],
                         AE["code"], STEPS, seed, center=True)
        for t in (1, 2):
            d = D[t]
            mid = d["i"] // 2
            cfgs = {"injection v1 (label-free)": inj_cfg(s, t), "injection v2 (label-free)": cfg_v2(s, t)[0],
                    "injection v3 (label-free)": cfg_v2(s, t, "inject_v3")[0],
                    "injection v4 (label-free)": cfg_v2(s, t, "inject_v4")[0], f"tuned on M{s} labels": CFG[s],
                    f"tuned on M{t} labels (upper bound)": CFG[t]}
            for k, (fit, ev) in enumerate([(slice(0, mid), slice(mid, d["i"])), (slice(mid, d["i"]), slice(0, mid))]):
                Z = transform_matrix(d["X"][:d["i"]], fit_matrix(d["X"][fit]))
                te = d["t"][ev]
                gt, tns = make_gt(d["lab"], te[0], te[-1]), te.values.astype(np.int64)
                Sa = np.stack([score_ae([model], Z[:, j], AE["w"], AE["reduce"], True) for j in range(Z.shape[1])], 1)
                Sb = np.stack([score_maxabs(Z[:, j], ABS_W) for j in range(Z.shape[1])], 1)
                fa = [evt_fit(Sa[fit][AE["w"]:][:, j]) for j in range(Z.shape[1])]
                fb = [evt_fit(Sb[fit][ABS_W:][:, j]) for j in range(Z.shape[1])]
                for name, c in cfgs.items():
                    r = fast_ew(gt, pred_from(fa, fb, Sa[ev], Sb[ev], c), tns)
                    rows.append(dict(source=s, target=t, fold=k, settings=name, **c, **r))
            print(f"dev M{s}->M{t} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "dev_step9_v4.csv", index=False)
    print(df.groupby(["source", "target", "settings"], sort=False)[["F", "P", "R", "TP", "FP"]].mean().round(3).to_string())


def test(D):
    for m in (1, 2):
        D[m]["Z"] = transform_matrix(D[m]["X"], fit_matrix(D[m]["X"][:D[m]["i"]]))
    rows = []
    for seed in SEEDS:
        for s in (1, 2):
            model = train_ae(D[s]["Z"][:D[s]["i"]], AE["w"], AE["hidden"], AE["code"], STEPS, seed, center=True)
            for t in (1, 2):
                d, c = D[t], cfg_v2(s, t, "inject_v3")[0]           # FINAL method = v3 (frozen rule, notes/step9f)
                Z, i = d["Z"], d["i"]
                Sa = np.stack([score_ae([model], Z[:, j], AE["w"], AE["reduce"], True) for j in range(Z.shape[1])], 1)
                Sb = np.stack([score_maxabs(Z[:, j], ABS_W) for j in range(Z.shape[1])], 1)
                pred = alarm_vector([(Sa[AE["w"]:i], Sa[i:], c["q"]), (Sb[ABS_W:i], Sb[i:], c["q2"])], c)
                tt = d["t"][i:]
                lab = d["lab"][d["lab"]["EndTime"] > SPLIT[t]].copy()            # official: events overlapping test
                lab["StartTime"] = lab["StartTime"].clip(lower=SPLIT[t])
                r = ew_scores(lab, pred, tt, (tt[0], tt[-1]))
                off = official_scores(lab, pred, tt, (tt[0], tt[-1]))        # every seed: all official metrics
                rows.append(dict(source=s, target=t, seed=seed, **c, alarm_frac=pred.mean(),
                                 official_F=off.get("EW_F_0.50", np.nan), **r,
                                 **{f"off_{k}": v for k, v in off.items()}))
                print(f"  M{s}->M{t} seed {seed}: F0.5={r['F0.5']:.3f} official={rows[-1]['official_F']} "
                      f"TP={r['TP']} FP={r['FP']}", flush=True)
            pd.DataFrame(rows).to_csv(ROOT / "results" / "final_step9.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["source", "target"]).agg(F=("F0.5", "mean"), Fstd=("F0.5", "std"), P=("precision", "mean"),
                                               R=("recall", "mean"), TP=("TP", "mean"), FP=("FP", "mean")).round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    D = load_all()
    if a.dev:
        dev(D)
    if a.test:
        test(D)
