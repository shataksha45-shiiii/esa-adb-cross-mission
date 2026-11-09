"""Tuning bench (Step 7). Uses ONLY the source mission's TRAINING period and its labels. Never test.

Protocol: 2-fold blocked cross-validation inside the training period.
  fold 0: fit (normaliser, model, EVT) on 1st half of train, evaluate on 2nd half
  fold 1: fit on 2nd half, evaluate on 1st half
Score of a config = mean corrected event-wise F0.5 (Rare Event + Anomaly) over the two folds.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import genpareto

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_labels, load_mission, split_index          # noqa: E402
from events import build_events as _build_events_offline          # noqa: E402,F401
from normaliser import fit_matrix, transform_matrix              # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ROOT = Path(__file__).resolve().parent.parent
CATS = ("Rare Event", "Anomaly")


# ---------------------------------------------------------------- fast evaluator (same maths as evaluator.ew_scores)
def _merge(s, e):
    if len(s) == 0:
        return s, e
    o = np.argsort(s, kind="stable")
    s, e = s[o], e[o]
    cm = np.maximum.accumulate(e)
    new = np.concatenate([[True], s[1:] > cm[:-1]])
    g = np.cumsum(new) - 1
    return s[new], np.maximum.reduceat(e, np.flatnonzero(new))


def fast_ew(gt, pred, tns, beta=0.5):
    """gt: dict(sel_ids=list of (s_arr, e_arr) per selected event, all_s, all_e merged); pred bool [T]; tns int64 ns."""
    p = pred.astype(np.int8)
    d = np.diff(np.concatenate([[0], p, [0]]))
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    ps = tns[a]
    pe = np.where(b < len(tns), tns[np.minimum(b, len(tns) - 1)], tns[-1])
    matched = np.zeros(len(ps) + 1, dtype=np.int64)
    tp = fn = 0
    for s_arr, e_arr in gt["sel"]:
        lo = np.searchsorted(pe, s_arr, "left")
        hi = np.searchsorted(ps, e_arr, "right")
        hit = hi > lo
        if hit.any():
            tp += 1
            np.add.at(matched, lo[hit], 1)
            np.add.at(matched, hi[hit], -1)
        else:
            fn += 1
    matched = np.cumsum(matched)[:-1] > 0
    gs, ge = gt["all_s"], gt["all_e"]
    lo = np.searchsorted(ge, ps, "left")
    hi = np.searchsorted(gs, pe, "right")
    touches = hi > lo
    fp = int(np.sum(~matched & ~touches))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    tnr = 1.0
    if prec > 0:
        # predicted time outside GT = |pred| - |pred ∩ GT| ;  |A∩B| = |A| + |B| - |A∪B|
        plen = (pe - ps).sum()
        us, ue = _merge(np.concatenate([ps, gs]), np.concatenate([pe, ge]))
        inter = plen + (ge - gs).sum() - (ue - us).sum()
        fp_ns = plen - inter
        tnr = 1 - fp_ns / gt["nominal"]
        prec *= tnr
    f = 0.0 if beta ** 2 * prec + rec == 0 else (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec)
    return dict(TP=tp, FP=fp, FN=fn, TNR=tnr, P=prec, R=rec, F=f)


def build_events_causal(alarm, gap=12, min_len=3):
    """ONLINE event builder (no future samples).
    on  : only once min_len consecutive steps exceed (the event starts at that step, never back-dated)
    hold: stays on for up to `gap` steps after the last exceeding step (gap-closing done forward in time)
    """
    a = np.asarray(alarm, bool)
    T = len(a)
    if not a.any():
        return np.zeros(T, bool)
    idx = np.arange(T)
    last = np.maximum.accumulate(np.where(a, idx, -10 ** 9))            # index of most recent exceedance
    run_start = np.maximum.accumulate(np.where(~a, idx, -1)) + 1        # start of current run of 1s
    confirmed = a & (idx - run_start + 1 >= min_len)
    in_block = (idx - last) <= gap                                     # within hold distance of an exceedance
    newblk = in_block & ~np.concatenate([[False], in_block[:-1]])
    blk = np.cumsum(newblk)
    acc = np.maximum.accumulate(np.where(confirmed, blk, 0))
    return in_block & (acc == blk) & (blk > 0)


def make_gt(lab, t0, t1):
    """Events (selected categories) that START in [t0, t1); GT context = all labels (as evaluator.ew_scores)."""
    sel = lab[(lab.StartTime >= t0) & (lab.StartTime < t1) & lab.Category.isin(CATS)]
    as_ = lambda c: c.values.astype("datetime64[ns]").astype(np.int64)
    out = [(as_(g.StartTime), as_(g.EndTime)) for _, g in sel.groupby("ID")]
    gs, ge = _merge(as_(lab.StartTime), as_(lab.EndTime))
    total = (t1 - t0).value
    gt_in = np.clip(np.minimum(ge, t1.value) - np.maximum(gs, t0.value), 0, None).sum()
    return dict(sel=out, all_s=gs, all_e=ge, nominal=total - gt_in)


# ---------------------------------------------------------------- detectors
class AE(nn.Module):
    def __init__(self, w, hidden, code):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(w, hidden), nn.ReLU(), nn.Linear(hidden, code))
        self.dec = nn.Sequential(nn.Linear(code, hidden), nn.ReLU(), nn.Linear(hidden, w))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(Z, w, hidden, code, steps, seed, lr=1e-3, batch=2048, center=False):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    Zt = torch.from_numpy(np.ascontiguousarray(Z)).to(DEVICE)
    T, C = Zt.shape
    m = AE(w, hidden, code).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ar = torch.arange(w, device=DEVICE)
    for _ in range(steps):
        ch = torch.randint(0, C, (batch,), generator=g).to(DEVICE)
        st = torch.randint(0, T - w, (batch,), generator=g).to(DEVICE)
        x = Zt[st[:, None] + ar[None, :], ch[:, None]]
        if center:
            x = x - x[:, :1]
        loss = ((m(x) - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return m.eval()


@torch.no_grad()
def score_ae(models, z, w, reduce, center=False, chunk=200_000):
    zt = torch.from_numpy(z).to(DEVICE)
    out = torch.zeros(len(z), device=DEVICE)
    wins = zt.unfold(0, w, 1)
    for a in range(0, wins.shape[0], chunk):
        x = wins[a:a + chunk]
        if center:
            x = x - x[:, :1]
        err = torch.stack([(m(x) - x) ** 2 for m in models]).mean(0)      # ensemble = mean error over seeds
        if reduce == "mean":
            s = err.mean(1)
        elif reduce == "max":
            s = err.max(1).values
        else:                                                           # "lastK": mean of the K most recent steps
            s = err[:, -int(reduce[4:]):].mean(1)
        out[w - 1 + a: w - 1 + a + len(x)] = s
    return out.cpu().numpy()


@torch.no_grad()
def score_maxabs(z, w):
    zt = torch.from_numpy(np.abs(z)).to(DEVICE)
    out = torch.zeros(len(z), device=DEVICE)
    out[w - 1:] = zt.unfold(0, w, 1).max(1).values
    return out.cpu().numpy()


# ---------------------------------------------------------------- EVT: fit once, threshold for any q
def evt_fit(s, p0=0.98):
    s = s[np.isfinite(s)]
    u = np.quantile(s, p0)
    y = s[s > u] - u
    if len(y) < 30 or y.max() <= 0:
        return dict(deg=True, mx=float(s.max()))
    xi, _, sig = genpareto.fit(y, floc=0)
    return dict(deg=False, u=u, xi=xi, sig=sig, n=len(s), nu=len(y))


def evt_thr(f, q):
    if f["deg"]:
        return f["mx"]
    r = q * f["n"] / f["nu"]
    if abs(f["xi"]) < 1e-6:
        return f["u"] - f["sig"] * np.log(r)
    return f["u"] + f["sig"] / f["xi"] * (r ** (-f["xi"]) - 1)


# ---------------------------------------------------------------- one detector config, both folds
def _scorer(det, Zfit, seeds, steps):
    w = det.get("w", 48)
    if det["kind"] == "ae":
        models = [train_ae(Zfit, w, det["hidden"], det["code"], steps, s, center=det.get("center", False))
                  for s in seeds]
        return w, lambda z: score_ae(models, z, w, det["reduce"], det.get("center", False))
    return w, lambda z: score_maxabs(z, w)


def fold_scores(m, det, seeds, steps):
    """Yield (fold, [S_fit per part], [S_eval per part], eval_times_ns, gt). det kind "fuse" has det["parts"]."""
    t, X, ch = load_mission(m)
    i = split_index(m, t)
    lab = load_labels(m)
    mid = i // 2
    parts = det["parts"] if det["kind"] == "fuse" else [det]
    for k, (fit, ev) in enumerate([(slice(0, mid), slice(mid, i)), (slice(mid, i), slice(0, mid))]):
        Z = transform_matrix(X[:i], fit_matrix(X[fit]))
        SF, SE = [], []
        for p in parts:
            w, fn = _scorer(p, Z[fit], seeds, steps)
            S = np.stack([fn(Z[:, j]) for j in range(Z.shape[1])], axis=1)
            SF.append(S[fit][w:])
            SE.append(S[ev])
        te = t[ev]
        yield k, SF, SE, te.values.astype(np.int64), make_gt(lab, te[0], te[-1])


POST = dict(q=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], gap=[0, 12, 36], min_len=[1, 3, 6], kofn=[1, 2, 3])
Q2 = [0.0, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]          # q of the 2nd fused part; 0.0 = part switched off


def evaluate(m, det, seeds=(0,), steps=4000, post=POST):
    rows = []
    for k, SF, SE, tns, gt in fold_scores(m, det, seeds, steps):
        fits = [[evt_fit(sf[:, j]) for j in range(sf.shape[1])] for sf in SF]
        q2s = Q2 if len(SF) == 2 else [0.0]
        for q, q2 in itertools.product(post["q"], q2s):
            al = SE[0] > np.array([evt_thr(f, q) for f in fits[0]])[None, :]
            if q2 > 0:
                al |= SE[1] > np.array([evt_thr(f, q2) for f in fits[1]])[None, :]
            for gap, ml in itertools.product(post["gap"], post["min_len"]):
                cnt = np.zeros(len(al), dtype=np.int16)
                for j in range(al.shape[1]):
                    if al[:, j].any():
                        cnt += build_events_causal(al[:, j], gap, ml)
                for kn in post["kofn"]:
                    r = fast_ew(gt, cnt >= kn, tns)
                    rows.append(dict(fold=k, q=q, q2=q2, gap=gap, min_len=ml, kofn=kn, **r))
    df = pd.DataFrame(rows)
    keys = ["q", "q2", "gap", "min_len", "kofn"]
    agg = df.groupby(keys).agg(F=("F", "mean"), Fmin=("F", "min"), P=("P", "mean"), R=("R", "mean"),
                               TP=("TP", "sum"), FP=("FP", "sum")).reset_index()
    return df, agg.sort_values("F", ascending=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", type=int, required=True)
    ap.add_argument("--det", required=True, help='JSON, e.g. {"kind":"ae","w":48,"hidden":64,"code":8,"reduce":"mean"}')
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    det = json.loads(a.det)
    t0 = time.time()
    df, agg = evaluate(a.mission, det, tuple(a.seeds), a.steps)
    out = ROOT / "results" / "tune"
    out.mkdir(parents=True, exist_ok=True)
    name = f"M{a.mission}_{a.tag or 'run'}"
    df.to_csv(out / f"{name}.csv", index=False)
    best = agg.iloc[0].to_dict()
    print(json.dumps(dict(mission=a.mission, det=det, seeds=a.seeds, secs=round(time.time() - t0), best=best)))
    print(agg.head(5).to_string(index=False))
