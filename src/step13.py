"""Step 13: ablations, block-bootstrap confidence intervals, sensitivity (protocol: notes/step13-protocol-frozen.md).

ANALYSIS ONLY, POST-HOC (5th use of test labels). Nothing here changes the selected method (C3).

python src/step13.py --check      : decomposition check (per-month parts reproduce fast_ew and ew_scores), M1->M2 seed 0
python src/step13.py --run        : all methods / ablations / sensitivity variants, 4 pairs x 5 seeds -> results/step13_*.csv
python src/step13.py --bootstrap  : month-block bootstrap CIs + paired differences -> results/step13_ci.csv
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SPLIT                                                          # noqa: E402
from diag_transfer import build_events                                          # noqa: E402 (causal builder)
from evaluator import ew_scores                                                 # noqa: E402
from inject_v2 import KEYS, R_MIN, f_beta                                       # noqa: E402
from normaliser import fit_matrix, transform_matrix                            # noqa: E402
from rolling import rolling_center                                              # noqa: E402
from run_final import ABS_W, AE, CFG, SEEDS, STEPS                              # noqa: E402
from run_step9 import inj_cfg                                                   # noqa: E402
from step10 import DEFAULT, hey_pred, load_all                                  # noqa: E402
from tune import _merge, evt_fit, evt_thr, score_ae, score_maxabs, train_ae     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DAY = 288


# ---------------------------------------------------------------- setting choosers with a repeats filter
def _agg(sub, s, t, reps):
    df = pd.read_csv(RES / sub / f"S{s}_T{t}.csv")
    df = df[df.rep.isin(reps)]
    return df.groupby(KEYS, sort=False).agg(R=("R", "mean"), B=("FP", "mean"), TNR=("TNR", "mean")).reset_index()


def _cfg(r):
    return dict(q=float(r.q), q2=float(r.q2), gap=int(r.gap), min_len=int(r.min_len), kofn=int(r.kofn))


def cfg_nhat(s, t, sub, reps=(0, 1, 2)):
    """v2/v3 rule (inject_v2.table_v2), optionally on a subset of injection repeats."""
    a = _agg(sub, s, t, reps)
    ok = (a.R >= R_MIN) & (a.B > 0)
    n_hat = float((a.B[ok] / a.R[ok]).min())
    p_hat = np.where(a.B > 0, np.clip(a.R * n_hat / a.B.where(a.B > 0, 1), 0, 1), 1.0)
    f_hat = f_beta(p_hat * (1 - (1 - p_hat) * (1 - a.TNR)), a.R.values)
    return _cfg(a.loc[int(np.argmax(f_hat))])


def cfg_frm(s, t, sub, reps=(0, 1, 2)):
    """C3/C5 rule (step10.cfg_fake_recall), optionally on a subset of repeats."""
    src = _agg(sub, s, s, reps)
    m = np.ones(len(src), bool)
    for k in KEYS:
        m &= np.isclose(src[k].astype(float), float(CFG[s][k]))
    r_star = float(src.R[m].iloc[0])
    a = _agg(sub, s, t, reps)
    ok = a[a.R >= r_star - 1e-9]
    r = ok.sort_values(["B", "R"], ascending=[True, False]).iloc[0] if len(ok) else a.loc[a.R.idxmax()]
    return _cfg(r)


def _complete(sub):
    return all((RES / sub / f"S{a}_T{b}.csv").exists() for a in (1, 2) for b in (1, 2))


def configs(s, t):
    """name -> (cfg, |z| source, threshold mode). |z| source: 'static' or a rolling window in days."""
    C3, C5 = cfg_frm(s, t, "inject_v3"), cfg_frm(s, t, "inject_v5")
    out = {
        "B0 target-label-tuned": (CFG[t], "static", "evt"),
        "B1 source-label-tuned": (CFG[s], "static", "evt"),
        "B2 fixed default": (DEFAULT, "static", "evt"),
        "L1 v1 random fakes + best injected F": (inj_cfg(s, t), "static", "evt"),
        "L2 v2 random fakes + N-hat": (cfg_nhat(s, t, "inject"), "static", "evt"),
        "C1 v3 replayed fakes + N-hat": (cfg_nhat(s, t, "inject_v3"), "static", "evt"),
        "C3 fake-recall match": (C3, "static", "evt"),
        "C5 fake-recall + rolling |z|": (C5, 30, "evt"),
    }
    for base, c, zs in (("C3", C3, "static"), ("C5", C5, 30)):
        out[f"{base} ab-AE (|z| off)"] = (dict(c, q2=0.0), zs, "evt")
        out[f"{base} ab-Z (AE off)"] = (dict(c, q=0.0), zs, "evt")
        out[f"{base} ab-EB (no event builder)"] = (dict(c, gap=0, min_len=1), zs, "evt")
        out[f"{base} ab-K (k-of-n = 1)"] = (dict(c, kofn=1), zs, "evt")
        out[f"{base} ab-Q (quantile, not EVT)"] = (c, zs, "quantile")
    for reps in ((0,), (0, 1)):                                          # sensitivity: injection repeats
        tag = f"reps={len(reps)}"
        out[f"C1 sens {tag}"] = (cfg_nhat(s, t, "inject_v3", reps), "static", "evt")
        out[f"C3 sens {tag}"] = (cfg_frm(s, t, "inject_v3", reps), "static", "evt")
        out[f"C5 sens {tag}"] = (cfg_frm(s, t, "inject_v5", reps), 30, "evt")
    for w in (15, 60):                                                   # sensitivity: rolling window
        if _complete(f"inject_v5_win{w}"):
            out[f"C5 sens window={w}d"] = (cfg_frm(s, t, f"inject_v5_win{w}"), w, "evt")
    if _complete("inject_v3_floor02"):                                   # sensitivity: amplitude floor
        out["C1 sens floor=0.2"] = (cfg_nhat(s, t, "inject_v3_floor02"), "static", "evt")
        out["C3 sens floor=0.2"] = (cfg_frm(s, t, "inject_v3_floor02"), "static", "evt")
    return out


# ---------------------------------------------------------------- predictions
class Scorer:
    """Score matrices for one (model, target), with cached EVT fits and training quantiles."""

    def __init__(self, model, Z, i):
        self.i = i
        self.S = {"ae": np.stack([score_ae([model], Z[:, j], AE["w"], AE["reduce"], True) for j in range(Z.shape[1])], 1),
                  "static": np.stack([score_maxabs(Z[:, j], ABS_W) for j in range(Z.shape[1])], 1)}
        self.Z, self.fits, self.quant = Z, {}, {}

    def mat(self, key):
        if key not in self.S:                                             # rolling |z| with window key (days)
            Zr = rolling_center(self.Z, key * DAY)
            self.S[key] = np.stack([score_maxabs(Zr[:, j], ABS_W) for j in range(Zr.shape[1])], 1)
        return self.S[key]

    def thr(self, key, q, mode):
        S = self.mat(key)
        w = AE["w"] if key == "ae" else ABS_W
        tr = S[w:self.i]
        if mode == "evt":
            if key not in self.fits:
                self.fits[key] = [evt_fit(tr[:, j]) for j in range(tr.shape[1])]
            return np.array([evt_thr(f, q) for f in self.fits[key]])
        k = (key, q)
        if k not in self.quant:
            self.quant[k] = np.quantile(tr, 1 - q, axis=0)
        return self.quant[k]

    def pred(self, c, zsrc, mode):
        al = None
        for key, q in (("ae", c["q"]), (zsrc, c["q2"])):
            if q > 0:
                a = self.mat(key)[self.i:] > self.thr(key, q, mode)[None, :]
                al = a if al is None else al | a
        if al is None:
            return np.zeros(len(self.Z) - self.i, bool)
        cnt = np.zeros(len(al), np.int16)
        for j in range(al.shape[1]):
            if al[:, j].any():
                cnt += build_events(al[:, j], c["gap"], c["min_len"])
        return cnt >= c["kofn"]


# ---------------------------------------------------------------- per-month decomposition of corrected event-wise F0.5
def test_gt(lab, tt, cats=("Rare Event", "Anomaly")):
    """Same definition as evaluator.ew_scores on the clipped test labels, full_range = (tt[0], tt[-1])."""
    ns = lambda c: c.values.astype("datetime64[ns]").astype(np.int64)
    sel = lab[lab.Category.isin(cats)]
    ev = [(ns(g.StartTime), ns(g.EndTime)) for _, g in sel.groupby("ID")]
    gs, ge = _merge(ns(lab.StartTime), ns(lab.EndTime))
    return dict(sel=ev, gs=gs, ge=ge, t0=tt[0].value, t1=tt[-1].value)


def _cover(gs, ge, t):
    """Total GT time before each t (gs, ge disjoint, sorted)."""
    cum = np.concatenate([[0], np.cumsum(ge - gs)])
    k = np.searchsorted(gs, t, "right")
    prev = np.maximum(k - 1, 0)
    part = np.clip(t - gs[prev], 0, (ge - gs)[prev])
    return np.where(k > 0, cum[prev] + part, 0)


def parts(gt, pred, tns, edges):
    """Per-month totals [6, n_months]: TP, FN, FP, fp_ns, nominal_ns, (unused). Months by event start."""
    nm = len(edges) - 1
    out = np.zeros((5, nm))
    p = pred.astype(np.int8)
    d = np.diff(np.concatenate([[0], p, [0]]))
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    ps = tns[a]
    pe = np.where(b < len(tns), tns[np.minimum(b, len(tns) - 1)], tns[-1])
    matched = np.zeros(len(ps) + 1, np.int64)
    for s_arr, e_arr in gt["sel"]:
        lo = np.searchsorted(pe, s_arr, "left")
        hi = np.searchsorted(ps, e_arr, "right")
        hit = hi > lo
        mth = np.searchsorted(edges, s_arr.min(), "right") - 1
        if hit.any():
            out[0, mth] += 1
            np.add.at(matched, lo[hit], 1)
            np.add.at(matched, hi[hit], -1)
        else:
            out[1, mth] += 1
    matched = np.cumsum(matched)[:-1] > 0
    gs, ge = gt["gs"], gt["ge"]
    touches = np.searchsorted(gs, pe, "right") > np.searchsorted(ge, ps, "left")
    pm = np.searchsorted(edges, ps, "right") - 1
    np.add.at(out[2], pm, (~matched & ~touches).astype(float))
    fp_ns = (pe - ps) - (_cover(gs, ge, pe) - _cover(gs, ge, ps))
    np.add.at(out[3], pm, fp_ns.astype(float))
    lo_e, hi_e = np.clip(edges[:-1], gt["t0"], gt["t1"]), np.clip(edges[1:], gt["t0"], gt["t1"])
    out[4] = (hi_e - lo_e) - (_cover(gs, ge, hi_e) - _cover(gs, ge, lo_e))
    return out


def f_from(tot):
    """tot: [..., 5] summed parts -> F0.5 (same formula as fast_ew / ew_scores)."""
    tp, fn, fp, fpns, nom = (tot[..., k] for k in range(5))
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    prec = np.where(prec > 0, prec * (1 - fpns / nom), prec)
    return f_beta(prec, rec)


def month_edges(tt):
    s = pd.Timestamp(tt[0]).to_period("M").to_timestamp()
    e = pd.Timestamp(tt[-1]).to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
    return pd.date_range(s, e, freq="MS").values.astype("datetime64[ns]").astype(np.int64)


# ---------------------------------------------------------------- run
def run(check=False):
    D = load_all()
    for m in (1, 2):
        D[m]["Z"] = transform_matrix(D[m]["X"], fit_matrix(D[m]["X"][:D[m]["i"]]))
    rows, P = [], {}
    seeds = [0] if check else SEEDS
    pairs = [(1, 2)] if check else [(a, b) for a in (1, 2) for b in (1, 2)]
    for seed in seeds:
        models = {}
        for s, t in pairs:
            if s not in models:
                models[s] = train_ae(D[s]["Z"][:D[s]["i"]], AE["w"], AE["hidden"], AE["code"], STEPS, seed, center=True)
            d = D[t]
            i = d["i"]
            tt = d["t"][i:]
            tns = tt.values.astype(np.int64)
            lab = d["lab"][d["lab"]["EndTime"] > SPLIT[t]].copy()
            lab["StartTime"] = lab["StartTime"].clip(lower=SPLIT[t])
            gt, gta, edges = test_gt(lab, tt), test_gt(lab, tt, ("Anomaly",)), month_edges(tt)
            sc = Scorer(models[s], d["Z"], i)
            preds = {n: sc.pred(c, zs, mode) for n, (c, zs, mode) in configs(s, t).items()}
            if seed == SEEDS[0]:
                preds["B3 Heydari static"] = hey_pred(d["Z"][:i], d["Z"][i:], d["t"][:i].values.astype(np.int64),
                                                      D[s]["rate"])
            for n, p in preds.items():
                pa = parts(gt, p, tns, edges)
                F = float(f_from(pa.sum(1)))
                Fa = float(f_from(parts(gta, p, tns, edges).sum(1)))
                P[(n, s, t, seed)] = pa
                rows.append(dict(method=n, source=s, target=t, seed=seed, F=F, F_anom=Fa, TP=int(pa[0].sum()),
                                 FN=int(pa[1].sum()), FP=int(pa[2].sum()), alarm_frac=float(p.mean())))
                if check and n in ("C3 fake-recall match", "C1 v3 replayed fakes + N-hat", "C3 ab-EB (no event builder)"):
                    r = ew_scores(lab, p.astype(int), tt, (tt[0], tt[-1]))
                    print(f"CHECK {n}: parts F={F:.12f}  ew_scores F={r['F0.5']:.12f}  diff={abs(F - r['F0.5']):.1e}")
            print(f"seed {seed} M{s}->M{t}: {len(preds)} methods", flush=True)
            if not check:
                pd.DataFrame(rows).to_csv(RES / "step13_points.csv", index=False)
                pickle.dump(P, open(RES / "step13_parts.pkl", "wb"))
    if check:
        print(pd.DataFrame(rows).groupby("method").F.mean().round(3).to_string())


# ---------------------------------------------------------------- bootstrap
PAIRS_DIFF = [("C3 fake-recall match", "C1 v3 replayed fakes + N-hat"),
              ("C5 fake-recall + rolling |z|", "C1 v3 replayed fakes + N-hat"),
              ("C3 fake-recall match", "B1 source-label-tuned"),
              ("C5 fake-recall + rolling |z|", "B3 Heydari static")]


def bootstrap(B=2000):
    P = pickle.load(open(RES / "step13_parts.pkl", "rb"))
    methods = sorted({k[0] for k in P})
    rows = []
    rng = np.random.default_rng(13)
    for s in (1, 2):
        for t in (1, 2):
            nm = next(v.shape[1] for k, v in P.items() if k[1] == s and k[2] == t)
            W = np.stack([np.bincount(rng.integers(0, nm, nm), minlength=nm) for _ in range(B)])   # [B, months]

            def dist(n):
                fs = [f_from(np.einsum("bm,km->bk", W, P[k])) for k in P if k[0] == n and k[1] == s and k[2] == t]
                pt = [float(f_from(P[k].sum(1))) for k in P if k[0] == n and k[1] == s and k[2] == t]
                return np.mean(fs, 0), float(np.mean(pt))

            cache = {}
            for n in methods:
                if any(k[0] == n and k[1] == s and k[2] == t for k in P):
                    cache[n] = dist(n)
                    lo, hi = np.percentile(cache[n][0], [2.5, 97.5])
                    rows.append(dict(kind="F", what=n, source=s, target=t, point=cache[n][1], lo=lo, hi=hi))
            for a, b in PAIRS_DIFF:
                if a in cache and b in cache:
                    dd = cache[a][0] - cache[b][0]
                    lo, hi = np.percentile(dd, [2.5, 97.5])
                    rows.append(dict(kind="diff", what=f"{a} minus {b}", source=s, target=t,
                                     point=cache[a][1] - cache[b][1], lo=lo, hi=hi, clear=bool(lo > 0 or hi < 0)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "step13_ci.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    a = ap.parse_args()
    if a.check:
        run(check=True)
    if a.run:
        run()
    if a.bootstrap:
        bootstrap()
