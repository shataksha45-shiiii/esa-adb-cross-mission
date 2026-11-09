"""Step 9 v2: contamination-aware selection (design frozen in notes/step9-design-frozen.md, section "v2").

Re-uses the v1 calibration runs (results/inject/S{s}_T{t}.csv). No new injections, no labels of the target.
Per setting θ (means over 2 folds x 3 repeats):
    R   = recall on fake (injected) events
    B   = background alarm events (alarms touching no fake event)     = column FP in the v1 CSV
    TNR = 1 - background alarm time / background time                 = column TNR in the v1 CSV
N_hat = min over θ with R >= 0.2 and B > 0 of B / R      (estimated number of REAL events hiding in the background)
P_hat = clip(R * N_hat / B, 0, 1)        (1 when B = 0)
TNR_hat = 1 - (1 - P_hat) * (1 - TNR)
F_hat = F0.5(P_hat * TNR_hat, R)   -> select argmax
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
KEYS = ["q", "q2", "gap", "min_len", "kofn"]
R_MIN = 0.2


def f_beta(p, r, beta=0.5):
    d = beta ** 2 * p + r
    return np.where(d > 0, (1 + beta ** 2) * p * r / np.where(d > 0, d, 1), 0.0)


def table_v2(s, t, sub="inject"):
    """sub: "inject" = v1 injections (-> v2 rule), "inject_v3" = replayed real source events (-> v3)."""
    df = pd.read_csv(ROOT / "results" / sub / f"S{s}_T{t}.csv")
    a = df.groupby(KEYS, sort=False).agg(R=("R", "mean"), B=("FP", "mean"), TNR=("TNR", "mean"),
                                         F_v1=("F", "mean")).reset_index()
    ok = (a.R >= R_MIN) & (a.B > 0)
    n_hat = float((a.B[ok] / a.R[ok]).min())
    p_hat = np.where(a.B > 0, np.clip(a.R * n_hat / a.B.where(a.B > 0, 1), 0, 1), 1.0)
    tnr_hat = 1 - (1 - p_hat) * (1 - a.TNR)
    a["P_hat"], a["TNR_hat"] = p_hat, tnr_hat
    a["F_hat"] = f_beta(p_hat * tnr_hat, a.R.values)
    return a, n_hat


def cfg_v2(s, t, sub="inject"):
    a, n_hat = table_v2(s, t, sub)
    r = a.loc[a.F_hat.idxmax()]
    return dict(q=float(r.q), q2=float(r.q2), gap=int(r.gap), min_len=int(r.min_len), kofn=int(r.kofn)), n_hat


if __name__ == "__main__":
    for s in (1, 2):
        for t in (1, 2):
            a, n_hat = table_v2(s, t)
            c, _ = cfg_v2(s, t)
            print(f"M{s}->M{t}: N_hat = {n_hat:.1f} real events per eval half; chosen {c}")
            print(a.sort_values("F_hat", ascending=False).head(3)[KEYS + ["R", "B", "P_hat", "F_hat"]]
                  .round(4).to_string(index=False))
