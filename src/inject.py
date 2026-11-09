"""Step 9: label-free calibration of post-processing settings by synthetic anomaly injection.

Design frozen in notes/step9-design-frozen.md BEFORE any result was seen.
Uses: source TRAINING labels (prior: durations, channel counts, rate), target UNLABELLED training data.
Never uses target labels or any test data.

Run: python src/inject.py --source 1 --target 2      -> results/inject/S1_T2.csv (+ best setting printed)
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SPLIT, load_labels, load_mission, split_index                     # noqa: E402
from normaliser import fit_matrix, transform_matrix                                # noqa: E402
from run_final import ABS_W, AE, STEPS                                             # noqa: E402
from tune import (CATS, POST, Q2, _merge, build_events_causal, evt_fit, evt_thr,   # noqa: E402
                  fast_ew, score_ae, score_maxabs, train_ae)

ROOT = Path(__file__).resolve().parent.parent
STEP_MIN = 5
DAY = 24 * 60 // STEP_MIN                      # steps per day (288)
MAX_DUR = 3 * DAY                              # duration cap: 3 days
MONTH_NS = 30.44 * 86400e9
TYPES = ("shift", "spike", "noise", "flat", "ramp")
REPEATS = (0, 1, 2)


def source_prior(m):
    """From the SOURCE's training labels: (duration_steps, n_channels) pairs and events per month."""
    lab = load_labels(m)
    tr = lab[(lab.StartTime < SPLIT[m]) & lab.Category.isin(CATS)]
    g = tr.groupby("ID").agg(s=("StartTime", "min"), e=("EndTime", "max"), nch=("Channel", "nunique"))
    dur = np.ceil((g.e - g.s).dt.total_seconds() / 60 / STEP_MIN).clip(1, MAX_DUR).astype(int).values
    t, _, _ = load_mission(m)
    months = (SPLIT[m] - t[0]).value / MONTH_NS
    return dict(pairs=np.stack([dur, g.nch.values], 1), rate=len(g) / months)


def source_prior_v3(m):
    """v3: one entry per real SOURCE training event: duration (steps) + measured amplitude on each source target channel."""
    lab = load_labels(m)
    tr = lab[(lab.StartTime < SPLIT[m]) & lab.Category.isin(CATS)]
    t, X, ch = load_mission(m)
    i = split_index(m, t)
    Z = transform_matrix(X[:i], fit_matrix(X[:i]))                       # source normaliser, source train only
    col = {f"channel_{c}": j for j, c in enumerate(ch)}
    events = []
    for _, g in tr.groupby("ID"):
        s0, e0 = g.StartTime.min(), g.EndTime.max()
        dur = int(np.clip(np.ceil((e0 - s0).total_seconds() / 60 / STEP_MIN), 1, MAX_DUR))
        amps = []
        for _, r in g.iterrows():
            if r.Channel not in col:
                continue
            j, a = col[r.Channel], t.searchsorted(r.StartTime)
            b = max(t.searchsorted(r.EndTime), a + 1)
            if a < 1 or b > i:
                continue
            amps.append(max(float(np.abs(Z[a:b, j] - Z[a - 1, j]).max()), 0.05))   # floor 0.05
        if amps:
            events.append((dur, np.array(amps, dtype=np.float32)))
    months = (SPLIT[m] - t[0]).value / MONTH_NS
    return dict(events=events, rate=len(tr.ID.unique()) / months)


def inject(Z, lo, hi, tns, prior, rng):
    """Paste fake events into Z[lo:hi] (in place on a copy). Returns (Zinj, list of (start_idx, end_idx))."""
    Z = Z.copy()
    T, C = Z.shape
    n = int(round(prior["rate"] * (tns[hi - 1] - tns[lo]) / MONTH_NS))
    placed = []
    for _ in range(n):
        for _try in range(200):                                        # rejection sampling: no overlap, >= 1 day apart
            if "events" in prior:                                       # v3: replay a real source event
                L, amps = prior["events"][rng.integers(len(prior["events"]))]
                k = len(amps)
            else:
                L, k = prior["pairs"][rng.integers(len(prior["pairs"]))]
                amps = None
            L = int(min(L, (hi - lo) // 4))
            s = int(rng.integers(lo + 1, hi - L))
            e = s + L
            if all(e + DAY <= a or s >= b + DAY for a, b in placed):
                break
        else:
            continue
        placed.append((s, e))
        chans = rng.choice(C, size=int(min(k, C)), replace=False)
        typ = TYPES[rng.integers(len(TYPES))]
        A0 = float(np.exp(rng.uniform(np.log(2), np.log(10))))          # v1/v2: log-uniform [2, 10]
        for n_j, j in enumerate(chans):
            A = float(amps[n_j]) if amps is not None else A0           # v3: that real event's amplitude on its n-th channel
            sg = rng.choice([-1.0, 1.0])
            if typ == "shift":
                Z[s:e, j] += sg * A
            elif typ == "spike":
                idx = rng.choice(np.arange(s, e), size=max(1, L // 6), replace=False)
                Z[idx, j] += sg * A
            elif typ == "noise":
                Z[s:e, j] += 0.5 * A * rng.standard_normal(L).astype(np.float32)
            elif typ == "flat":
                Z[s:e, j] = Z[s - 1, j]
            else:                                                       # ramp
                Z[s:e, j] += sg * A * np.linspace(0, 1, L, dtype=np.float32)
    np.clip(Z, -20, 20, out=Z)
    return Z, sorted(placed)


# ---------------------------------------------------------------- v4: transplant real shapes, rescaled to normal variability
GRID_L = 2 ** np.arange(11)                    # 1, 2, 4, ..., 1024 steps


def variability(Z, rng, n=2000):
    """D[c, g] = median over n random windows of max_u |z[a+u] - z[a-1]|, window length GRID_L[g]. Floor 0.05."""
    T, C = Z.shape
    D = np.empty((C, len(GRID_L)), dtype=np.float32)
    for g, L in enumerate(GRID_L):
        a = rng.integers(1, T - L, size=n)
        idx = a[:, None] + np.arange(L)[None, :]
        for c in range(C):
            z = Z[:, c]
            D[c, g] = np.median(np.abs(z[idx] - z[a - 1][:, None]).max(1))
    return np.maximum(D, 0.05)


def grid_idx(L):
    return int(np.argmin(np.abs(np.log2(GRID_L) - np.log2(max(L, 1)))))


def source_prior_v4(m):
    """One entry per real SOURCE training event: its deviation curves d_t = z_t - z_{s-1} on each source target
    channel it touched (event-level window), the source channel ids, and the source variability table D_src."""
    lab = load_labels(m)
    tr = lab[(lab.StartTime < SPLIT[m]) & lab.Category.isin(CATS)]
    t, X, ch = load_mission(m)
    i = split_index(m, t)
    Z = transform_matrix(X[:i], fit_matrix(X[:i]))
    col = {f"channel_{c}": j for j, c in enumerate(ch)}
    events = []
    for _, g in tr.groupby("ID"):
        a = t.searchsorted(g.StartTime.min())
        b = max(t.searchsorted(g.EndTime.max()), a + 1)
        b = min(b, a + MAX_DUR)
        if a < 1 or b > i:
            continue
        js = sorted({col[c] for c in g.Channel if c in col})
        if js:
            curves = Z[a:b, js] - Z[a - 1, js][None, :]               # [L, n_ch] real shape + sign
            events.append((b - a, np.array(js), curves.astype(np.float32)))
    months = (SPLIT[m] - t[0]).value / MONTH_NS
    return dict(events=events, rate=len(tr.ID.unique()) / months,
                D=variability(Z, np.random.default_rng(12345)))


def inject_v4(Z, lo, hi, tns, prior, D_tgt, rng):
    """Paste real source shapes into Z[lo:hi], each scaled by D_tgt[k, L] / D_src[j, L]."""
    Z = Z.copy()
    T, C = Z.shape
    n = int(round(prior["rate"] * (tns[hi - 1] - tns[lo]) / MONTH_NS))
    placed = []
    for _ in range(n):
        for _try in range(200):
            L0, js, curves = prior["events"][rng.integers(len(prior["events"]))]
            L = int(min(L0, (hi - lo) // 4))
            s = int(rng.integers(lo + 1, hi - L))
            e = s + L
            if all(e + DAY <= a or s >= b + DAY for a, b in placed):
                break
        else:
            continue
        placed.append((s, e))
        g = grid_idx(L)
        ks = rng.choice(C, size=min(len(js), C), replace=False)       # random distinct target channels
        for n_j, kc in enumerate(ks):
            j = js[n_j]
            Z[s:e, kc] += curves[:L, n_j] * (D_tgt[kc, g] / prior["D"][j, g])
    np.clip(Z, -20, 20, out=Z)
    return Z, sorted(placed)


def gt_from(placed, tns_eval, offset):
    """Ground-truth dict for fast_ew from injected (start_idx, end_idx) in full-array indices."""
    s = np.array([tns_eval[a - offset] for a, _ in placed], dtype=np.int64)
    e = np.array([tns_eval[b - 1 - offset] for _, b in placed], dtype=np.int64)
    gs, ge = _merge(s, e)
    total = tns_eval[-1] - tns_eval[0]
    return dict(sel=[(np.array([a]), np.array([b])) for a, b in zip(s, e)], all_s=gs, all_e=ge,
                nominal=total - (ge - gs).sum())


def scores(model, Z):
    Sa = np.stack([score_ae([model], Z[:, j], AE["w"], AE["reduce"], True) for j in range(Z.shape[1])], 1)
    Sb = np.stack([score_maxabs(Z[:, j], ABS_W) for j in range(Z.shape[1])], 1)
    return Sa, Sb


def calibrate(src, tgt, seed=0, log=print, version=1):
    ts, Xs, _ = load_mission(src)
    i_s = split_index(src, ts)
    model = train_ae(transform_matrix(Xs[:i_s], fit_matrix(Xs[:i_s])), AE["w"], AE["hidden"], AE["code"],
                     STEPS, seed, center=True)
    prior = {1: source_prior, 3: source_prior_v3, 4: source_prior_v4}[version](src)
    n_ev = len(prior["events"]) if version in (3, 4) else len(prior["pairs"])
    log(f"source M{src} prior (v{version}): {prior['rate']:.2f} events/month, {n_ev} source events", flush=True)
    t, X, _ = load_mission(tgt)
    i = split_index(tgt, t)
    mid = i // 2
    tall = t.values.astype(np.int64)
    rows = []
    for k, (fit, ev) in enumerate([(slice(0, mid), slice(mid, i)), (slice(mid, i), slice(0, mid))]):
        Z = transform_matrix(X[:i], fit_matrix(X[fit]))               # normaliser: clean fit half only
        Sa, Sb = scores(model, Z)                                     # clean scores -> EVT fits on fit half
        fa = [evt_fit(Sa[fit][AE["w"]:][:, j]) for j in range(Z.shape[1])]
        fb = [evt_fit(Sb[fit][ABS_W:][:, j]) for j in range(Z.shape[1])]
        lo = ev.start + AE["w"]                                       # keep injections away from the fold edge
        hi = ev.stop
        D_tgt = variability(Z[fit], np.random.default_rng(54321 + k)) if version == 4 else None   # clean fit half
        for rep in REPEATS:
            rng = np.random.default_rng(1000 * k + rep)
            if version == 4:
                Zi, placed = inject_v4(Z, lo, hi, tall, prior, D_tgt, rng)
            else:
                Zi, placed = inject(Z, lo, hi, tall, prior, rng)
            Sai, Sbi = scores(model, Zi)
            SEa, SEb, tns = Sai[ev], Sbi[ev], tall[ev]
            gt = gt_from(placed, tns, ev.start)
            for q, q2 in itertools.product(POST["q"], Q2):
                al = SEa > np.array([evt_thr(f, q) for f in fa])[None, :]
                if q2 > 0:
                    al |= SEb > np.array([evt_thr(f, q2) for f in fb])[None, :]
                for gap, ml in itertools.product(POST["gap"], POST["min_len"]):
                    cnt = np.zeros(len(al), dtype=np.int16)
                    for j in range(al.shape[1]):
                        if al[:, j].any():
                            cnt += build_events_causal(al[:, j], gap, ml)
                    for kn in POST["kofn"]:
                        r = fast_ew(gt, cnt >= kn, tns)
                        rows.append(dict(fold=k, rep=rep, n_inj=len(placed), q=q, q2=q2, gap=gap, min_len=ml,
                                         kofn=kn, **r))
            log(f"  fold {k} rep {rep}: {len(placed)} injected events", flush=True)
    df = pd.DataFrame(rows)
    keys = ["q", "q2", "gap", "min_len", "kofn"]
    agg = df.groupby(keys, sort=False).agg(F=("F", "mean"), P=("P", "mean"), R=("R", "mean"),
                                           TP=("TP", "mean"), FP=("FP", "mean")).reset_index()
    best = agg.loc[agg.F.idxmax()].to_dict()                          # first max in grid order on ties
    return df, agg, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--version", type=int, default=1, choices=[1, 3, 4])
    a = ap.parse_args()
    t0 = time.time()
    df, agg, best = calibrate(a.source, a.target, version=a.version)
    out = ROOT / "results" / ("inject" if a.version == 1 else f"inject_v{a.version}")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"S{a.source}_T{a.target}.csv", index=False)
    print(json.dumps(dict(source=a.source, target=a.target, secs=round(time.time() - t0), best=best)), flush=True)
    print(agg.sort_values("F", ascending=False).head(8).to_string(index=False))
