"""Step 7 final run: source-tuned configs, evaluated ONCE on test. 5 seeds.

Every setting below was chosen by tune.py (2-fold CV inside the SOURCE mission's training period,
source training labels only). The target mission's labels are used only in the final ew_scores call.
Transfer A->B: model trained on A's train, post-processing settings tuned on A, applied to B.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SPLIT, load_labels, load_mission, split_index      # noqa: E402
from evaluator import ew_scores, official_scores                     # noqa: E402
from normaliser import fit_matrix, transform_matrix                  # noqa: E402
from tune import build_events_causal, evt_fit, evt_thr, score_ae, score_maxabs, train_ae  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AE = dict(w=24, hidden=64, code=8, reduce="max", center=True)
ABS_W = 12
# Step 8 (CAUSAL event builder, bins labelled by their end): picked from results/tune/round_causal.jsonl,
# best mean CV F0.5 per source mission (source training labels only). Baseline window also chosen there.
CFG = {1: dict(q=1e-5, q2=0.0, gap=12, min_len=1, kofn=2),
       2: dict(q=1e-4, q2=1e-4, gap=36, min_len=3, kofn=2)}
BASE = {1: dict(w=48, q=1e-7, gap=12, min_len=1, kofn=3),
        2: dict(w=12, q=1e-4, gap=36, min_len=3, kofn=2)}
STEPS = 4000
SEEDS = [0, 1, 2, 3, 4]


def alarm_vector(parts, cfg):
    """parts: list of (S_train, S_test, q). OR the parts per channel, clean events, k-of-n across channels."""
    al = None
    for s_tr, s_te, q in parts:
        if q <= 0:
            continue
        thr = np.array([evt_thr(evt_fit(s_tr[:, j]), q) for j in range(s_tr.shape[1])])
        a = s_te > thr[None, :]
        al = a if al is None else al | a
    cnt = np.zeros(len(al), dtype=np.int16)
    for j in range(al.shape[1]):
        if al[:, j].any():
            cnt += build_events_causal(al[:, j], cfg["gap"], cfg["min_len"])
    return (cnt >= cfg["kofn"]).astype(int)


def main():
    D = {}
    for m in (1, 2):
        t, X, _ = load_mission(m)
        i = split_index(m, t)
        D[m] = dict(t=t, i=i, Z=transform_matrix(X, fit_matrix(X[:i])), lab=load_labels(m))
    rows = []

    def score(name, a, b, pred, seed):
        d = D[b]
        tt = d["t"][d["i"]:]
        # official test definition: every event OVERLAPPING the test period, clipped at the split
        lab = d["lab"][d["lab"]["EndTime"] > SPLIT[b]].copy()
        lab["StartTime"] = lab["StartTime"].clip(lower=SPLIT[b])
        r = ew_scores(lab, pred, tt, (tt[0], tt[-1]))
        off = official_scores(lab, pred, tt, (tt[0], tt[-1])) if seed in (-1, 0) else {}
        offF = off.get("EW_F_0.50", off.get("EW_F0.5", np.nan)) if off else np.nan
        rows.append(dict(detector=name, source=a, target=b, seed=seed, alarm_frac=pred.mean(), official_F=offF,
                         official_keys=",".join(off.keys()) if off else "", **r))
        print(f"  {name:10s} M{a}->M{b} seed {seed}: F0.5={r['F0.5']:.3f} official={offF} TP={r['TP']} FP={r['FP']}", flush=True)

    # baseline: no learning, only its source-tuned settings
    for a in (1, 2):
        for b in (1, 2):
            c, Z, i = BASE[a], D[b]["Z"], D[b]["i"]
            S = np.stack([score_maxabs(Z[:, j], c["w"]) for j in range(Z.shape[1])], 1)
            score("baseline", a, b, alarm_vector([(S[c["w"]:i], S[i:], c["q"])], c), -1)

    for seed in SEEDS:
        for a in (1, 2):
            t0 = time.time()
            model = train_ae(D[a]["Z"][:D[a]["i"]], AE["w"], AE["hidden"], AE["code"], STEPS, seed, center=True)
            for b in (1, 2):
                Z, i, c = D[b]["Z"], D[b]["i"], CFG[a]
                Sa = np.stack([score_ae([model], Z[:, j], AE["w"], AE["reduce"], True) for j in range(Z.shape[1])], 1)
                Sb = np.stack([score_maxabs(Z[:, j], ABS_W) for j in range(Z.shape[1])], 1)
                pred = alarm_vector([(Sa[AE["w"]:i], Sa[i:], c["q"]), (Sb[ABS_W:i], Sb[i:], c["q2"])], c)
                score("fusedAE", a, b, pred, seed)
            print(f"  [{time.time() - t0:.0f}s]", flush=True)
            pd.DataFrame(rows).to_csv(ROOT / "results" / "final_step8.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "final_step8.csv", index=False)
    s = df.groupby(["detector", "source", "target"]).agg(F=("F0.5", "mean"), Fstd=("F0.5", "std"),
                                                         P=("precision", "mean"), R=("recall", "mean"),
                                                         TP=("TP", "mean"), FP=("FP", "mean"))
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
