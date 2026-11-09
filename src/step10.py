"""Step 10: baselines + improvement round (protocol FROZEN in notes/step10-protocol-frozen.md).

--dev  : 2-fold blocked CV inside the TRAINING periods, seed 0. Target training labels used ONLY to score (diagnosis).
--test : ONE run, 5 seeds, official metric + anomalies-only. 4th use of test labels -> every number is POST-HOC.

Settings sources (none uses target labels, except B0 = upper bound):
  B0 CFG[target]  B1 CFG[source]  B2 fixed default  B3/B4 Heydari-style  B5 |z|-max baseline (source-tuned)
  C1 v3   C2 v3 AE-only   C3 fake-recall matching (v3 runs)   C4 v3 rule on v5 runs (rolling |z|)   C5 C3 rule on v5 runs
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SPLIT, load_labels, load_mission, split_index                  # noqa: E402
from diag_transfer import pred_from                                             # noqa: E402
from evaluator import ew_scores, official_scores                                # noqa: E402
from inject import source_prior                                                 # noqa: E402
from inject_v2 import KEYS, table_v2                                            # noqa: E402
from normaliser import fit_matrix, transform_matrix                             # noqa: E402
from rolling import rolling_center, rolling_robust_z                            # noqa: E402
from run_final import ABS_W, AE, BASE, CFG, SEEDS, STEPS                        # noqa: E402
from tune import build_events_causal, evt_fit, evt_thr, fast_ew, make_gt, score_ae, score_maxabs, train_ae  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MONTH_NS = 30.44 * 86400e9
DEFAULT = dict(q=1e-4, q2=1e-4, gap=12, min_len=3, kofn=1)                      # B2
HEY = dict(gap=12, min_len=1, topk=3)                                           # B3/B4


# ---------------------------------------------------------------- label-free setting choosers
def _cfg(r):
    return dict(q=float(r.q), q2=float(r.q2), gap=int(r.gap), min_len=int(r.min_len), kofn=int(r.kofn))


def cfg_v3_ae_only(s, t, sub="inject_v3"):
    a, _ = table_v2(s, t, sub)                    # N_hat from the full grid, selection restricted to q2 = 0
    a = a[a.q2 == 0]
    return _cfg(a.loc[a.F_hat.idxmax()])


def _match(a, c):
    m = np.ones(len(a), bool)
    for k in KEYS:
        m &= np.isclose(a[k].astype(float), float(c[k]))
    return m


def cfg_fake_recall(s, t, sub="inject_v3"):
    """C3/C5: R* = fake recall of CFG[s] on the S->S run; on S->T pick fewest background alarms with R >= R*."""
    src = pd.read_csv(ROOT / "results" / sub / f"S{s}_T{s}.csv").groupby(KEYS, sort=False).R.mean().reset_index()
    r_star = float(src.R[_match(src, CFG[s])].iloc[0])
    a = pd.read_csv(ROOT / "results" / sub / f"S{s}_T{t}.csv").groupby(KEYS, sort=False).agg(
        R=("R", "mean"), B=("FP", "mean")).reset_index()
    ok = a[a.R >= r_star - 1e-9]
    r = ok.sort_values(["B", "R"], ascending=[True, False]).iloc[0] if len(ok) else a.loc[a.R.idxmax()]
    return _cfg(r), r_star


def choose(s, t):
    """All settings-based configs for pair (s, t). Returns {name: (cfg, roll)}."""
    from inject_v2 import cfg_v2
    out = {"B0 target-label-tuned (upper bound)": (CFG[t], False), "B1 source-label-tuned": (CFG[s], False),
           "B2 fixed default": (DEFAULT, False), "C1 v3 (Step 9)": (cfg_v2(s, t, "inject_v3")[0], False),
           "C2 v3 AE-only": (cfg_v3_ae_only(s, t), False), "C3 fake-recall match": (cfg_fake_recall(s, t)[0], False)}
    if (ROOT / "results" / "inject_v5" / f"S{s}_T{t}.csv").exists():
        out["C4 v3 rule + rolling |z|"] = (cfg_v2(s, t, "inject_v5")[0], True)
        out["C5 fake-recall + rolling |z|"] = (cfg_fake_recall(s, t, "inject_v5")[0], True)
    return out


# ---------------------------------------------------------------- Heydari-style scalar alarm (B3 / B4)
def topk_score(Z, k=HEY["topk"]):
    return np.sort(np.abs(Z), axis=1)[:, -k:].mean(1)


def n_events(p):
    return int(np.sum(np.diff(np.concatenate([[0], p.astype(np.int8)])) == 1))


def hey_threshold(s_fit, tns_fit, budget):
    """Threshold on the fit period whose alarm-event rate (causal builder) is closest (log) to budget events/month."""
    months = (tns_fit[-1] - tns_fit[0]) / MONTH_NS
    cands = np.quantile(s_fit, 1 - np.logspace(-1, -6, 41))
    best, bd = cands[-1], np.inf
    for c in cands:
        rate = n_events(build_events_causal(s_fit > c, HEY["gap"], HEY["min_len"])) / months
        d = abs(np.log((rate + 1e-3) / budget))
        if d < bd:
            best, bd = c, d
    return best


def hey_pred(Zfit, Zev, tns_fit, budget):
    thr = hey_threshold(topk_score(Zfit), tns_fit, budget)
    return build_events_causal(topk_score(Zev) > thr, HEY["gap"], HEY["min_len"])


def base_pred(Zfit, Zev, c):
    """B5: |z|-max baseline with source-tuned settings (Step 8 BASE)."""
    w = c["w"]
    Sf = np.stack([score_maxabs(Zfit[:, j], w) for j in range(Zfit.shape[1])], 1)[w:]
    Se = np.stack([score_maxabs(Zev[:, j], w) for j in range(Zev.shape[1])], 1)
    al = Se > np.array([evt_thr(evt_fit(Sf[:, j]), c["q"]) for j in range(Sf.shape[1])])[None, :]
    cnt = np.zeros(len(al), np.int16)
    for j in range(al.shape[1]):
        if al[:, j].any():
            cnt += build_events_causal(al[:, j], c["gap"], c["min_len"])
    return cnt >= c["kofn"]


# ---------------------------------------------------------------- shared
def load_all():
    D = {}
    for m in (1, 2):
        t, X, _ = load_mission(m)
        D[m] = dict(t=t, X=X, i=split_index(m, t), lab=load_labels(m), rate=source_prior(m)["rate"])
    return D


def scores_for(model, Z):
    Sa = np.stack([score_ae([model], Z[:, j], AE["w"], AE["reduce"], True) for j in range(Z.shape[1])], 1)
    Sb = np.stack([score_maxabs(Z[:, j], ABS_W) for j in range(Z.shape[1])], 1)
    Zr = rolling_center(Z)
    Sbr = np.stack([score_maxabs(Zr[:, j], ABS_W) for j in range(Z.shape[1])], 1)
    return Sa, Sb, Sbr


def all_preds(s, t, model, Z, fit, ev, tns_fit, D):
    """Predictions on Z[ev] for every config. EVT/thresholds fitted on Z[fit] only."""
    Sa, Sb, Sbr = scores_for(model, Z)
    fa = [evt_fit(Sa[fit][AE["w"]:][:, j]) for j in range(Z.shape[1])]
    fb = [evt_fit(Sb[fit][ABS_W:][:, j]) for j in range(Z.shape[1])]
    fbr = [evt_fit(Sbr[fit][ABS_W:][:, j]) for j in range(Z.shape[1])]
    P = {}
    for name, (c, roll) in choose(s, t).items():
        P[name] = pred_from(fa, fbr if roll else fb, Sa[ev], (Sbr if roll else Sb)[ev], c)
    budget = D[s]["rate"]
    P["B3 Heydari static (source-rate budget)"] = hey_pred(Z[fit], Z[ev], tns_fit, budget)
    Zrz = rolling_robust_z(Z)
    P["B4 Heydari rolling (source-rate budget)"] = hey_pred(Zrz[fit], Zrz[ev], tns_fit, budget)
    P["B5 |z|-max baseline, source-tuned"] = base_pred(Z[fit], Z[ev], BASE[s])
    return P


# ---------------------------------------------------------------- dev
def dev(D, seed=0):
    rows = []
    for s in (1, 2):
        d = D[s]
        model = train_ae(transform_matrix(d["X"][:d["i"]], fit_matrix(d["X"][:d["i"]])), AE["w"], AE["hidden"],
                         AE["code"], STEPS, seed, center=True)
        for t in (1, 2):
            d = D[t]
            mid = d["i"] // 2
            for k, (fit, ev) in enumerate([(slice(0, mid), slice(mid, d["i"])), (slice(mid, d["i"]), slice(0, mid))]):
                Z = transform_matrix(d["X"][:d["i"]], fit_matrix(d["X"][fit]))
                te = d["t"][ev]
                gt, tns = make_gt(d["lab"], te[0], te[-1]), te.values.astype(np.int64)
                tf = d["t"][fit].values.astype(np.int64)
                for name, p in all_preds(s, t, model, Z, fit, ev, tf, D).items():
                    r = fast_ew(gt, p, tns)
                    rows.append(dict(source=s, target=t, fold=k, method=name, alarm_frac=float(p.mean()), **r))
            print(f"dev M{s}->M{t} done", flush=True)
            pd.DataFrame(rows).to_csv(ROOT / "results" / "dev_step10.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "dev_step10.csv", index=False)
    summary(df, "F")


def summary(df, col):
    g = df.groupby(["method", "source", "target"])[col].mean().unstack(["source", "target"])
    g.columns = [f"M{a}->M{b}" for a, b in g.columns]
    g["min transfer"] = g[["M1->M2", "M2->M1"]].min(1)
    print(g.sort_index().round(3).to_string())
    return g


# ---------------------------------------------------------------- test (run ONCE)
def test(D):
    for m in (1, 2):
        D[m]["Z"] = transform_matrix(D[m]["X"], fit_matrix(D[m]["X"][:D[m]["i"]]))
    rows = []
    for seed in SEEDS:
        for s in (1, 2):
            model = train_ae(D[s]["Z"][:D[s]["i"]], AE["w"], AE["hidden"], AE["code"], STEPS, seed, center=True)
            for t in (1, 2):
                d = D[t]
                i = d["i"]
                fit, ev = slice(0, i), slice(i, len(d["Z"]))
                P = all_preds(s, t, model, d["Z"], fit, ev, d["t"][:i].values.astype(np.int64), D)
                tt = d["t"][i:]
                lab = d["lab"][d["lab"]["EndTime"] > SPLIT[t]].copy()            # official: events overlapping test
                lab["StartTime"] = lab["StartTime"].clip(lower=SPLIT[t])
                for name, p in P.items():
                    if seed > 0 and name.startswith(("B3", "B4", "B5")):
                        continue                                                 # B3-B5 have no AE: seed-free
                    p = p.astype(int)
                    r = ew_scores(lab, p, tt, (tt[0], tt[-1]))
                    ra = ew_scores(lab, p, tt, (tt[0], tt[-1]), categories=("Anomaly",))
                    # official code crashes on an empty prediction; F = 0 by definition (no TP) -> use ours
                    off = official_scores(lab, p, tt, (tt[0], tt[-1])) if p.any() else {"EW_F_0.50": r["F0.5"]}
                    rows.append(dict(source=s, target=t, seed=seed, method=name, alarm_frac=float(p.mean()),
                                     official_F=off.get("EW_F_0.50", np.nan), F=r["F0.5"], P=r["precision"],
                                     R=r["recall"], TP=r["TP"], FP=r["FP"], F_anom=ra["F0.5"], R_anom=ra["recall"]))
                print(f"test seed {seed} M{s}->M{t} done", flush=True)
                pd.DataFrame(rows).to_csv(ROOT / "results" / "final_step10.csv", index=False)
    df = pd.DataFrame(rows)
    print("All events (official F0.5), mean over seeds:")
    summary(df, "official_F")
    print("\nAnomalies only (F0.5), mean over seeds:")
    summary(df, "F_anom")


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
