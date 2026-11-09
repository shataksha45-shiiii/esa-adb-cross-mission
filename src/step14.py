"""Step 14 (roadmap items 6 + 10): all official ESA-ADB metrics + an operator view, for the main methods.

POST-HOC (6th use of test labels), reporting only: no method or setting is chosen here.
Official metrics (ESA-ADB code): corrected event-wise P/R/F0.5 (all events and anomalies only), alarming precision,
affiliation-based P/R/F0.5, ADTQC (timing quality of detections). The mission-level alarm is given to every channel
for ADTQC (we raise one alarm for the whole mission, as ESA-ADB's unsupervised baselines do).
Operator view (from the same predictions): alarm events per month, false-alarm events per month, share of events
caught, time in alarm, median detection delay (first overlapping alarm start minus event start).
"""
import io
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import SPLIT                                                       # noqa: E402
from evaluator import official_scores                                       # noqa: E402
from normaliser import fit_matrix, transform_matrix                         # noqa: E402
from run_final import AE, SEEDS, STEPS                                      # noqa: E402
from step10 import hey_pred, load_all                                       # noqa: E402
from step13 import Scorer, configs, f_from, parts, month_edges, test_gt                                 # noqa: E402
from tune import train_ae                                                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAIN = ["B0 target-label-tuned", "B1 source-label-tuned", "B2 fixed default", "C1 v3 replayed fakes + N-hat",
        "C3 fake-recall match", "C5 fake-recall + rolling |z|"]
MONTH_NS = 30.44 * 86400e9


def adtqc(lab, pred, tt):
    from timeeval.metrics.latency_metrics import ADTQC                     # path set up by official_scores()
    p = pred.astype(int)
    chg = np.flatnonzero(np.diff(np.concatenate([[-1], p])) != 0)          # change points only (same events)
    pairs = np.column_stack([np.asarray(tt[chg], dtype=object), p[chg]])
    if chg[-1] != len(p) - 1:
        pairs = np.vstack([pairs, [tt[-1], p[-1]]])
    chans = lab.Channel.unique()
    m = ADTQC(full_range=(tt[0], tt[-1]), select_labels={"Category": ["Rare Event", "Anomaly"]})
    with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        r = m.score(lab, {c: pairs.copy() for c in chans})
    return r["Total"], r["AfterRate"]


def operator(gt, pred, tns):
    p = pred.astype(np.int8)
    d = np.diff(np.concatenate([[0], p, [0]]))
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    ps = tns[a]
    pe = np.where(b < len(tns), tns[np.minimum(b, len(tns) - 1)], tns[-1])
    delays, caught = [], 0
    for s_arr, e_arr in gt["sel"]:
        lo = np.searchsorted(pe, s_arr, "left")
        hi = np.searchsorted(ps, e_arr, "right")
        hit = hi > lo
        if hit.any():
            caught += 1
            delays.append((ps[lo[hit].min()] - s_arr.min()) / 60e9)
    months = (tns[-1] - tns[0]) / MONTH_NS
    return dict(alarms_per_month=len(ps) / months, caught=caught, n_events=len(gt["sel"]),
                caught_share=caught / max(len(gt["sel"]), 1), time_in_alarm=float(p.mean()),
                median_delay_min=float(np.median(delays)) if delays else np.nan)


def main():
    D = load_all()
    for m in (1, 2):
        D[m]["Z"] = transform_matrix(D[m]["X"], fit_matrix(D[m]["X"][:D[m]["i"]]))
    rows = []
    for seed in SEEDS:
        models = {}
        for s in (1, 2):
            for t in (1, 2):
                if s not in models:
                    models[s] = train_ae(D[s]["Z"][:D[s]["i"]], AE["w"], AE["hidden"], AE["code"], STEPS, seed,
                                         center=True)
                d = D[t]
                i = d["i"]
                tt = d["t"][i:]
                tns = tt.values.astype(np.int64)
                lab = d["lab"][d["lab"]["EndTime"] > SPLIT[t]].copy()
                lab["StartTime"] = lab["StartTime"].clip(lower=SPLIT[t])
                gt = test_gt(lab, tt)
                sc = Scorer(models[s], d["Z"], i)
                cf = configs(s, t)
                preds = {n: sc.pred(*cf[n]) for n in MAIN}
                if seed == SEEDS[0]:
                    preds["B3 Heydari static"] = hey_pred(d["Z"][:i], d["Z"][i:], d["t"][:i].values.astype(np.int64),
                                                          D[s]["rate"])
                for n, p in preds.items():
                    row = dict(method=n, source=s, target=t, seed=seed)
                    if p.any() and seed == SEEDS[0]:          # official extras (slow on M1): seed 0 only
                        off = official_scores(lab, p.astype(int), tt, (tt[0], tt[-1]))
                        offa = official_scores(lab, p.astype(int), tt, (tt[0], tt[-1]), categories=("Anomaly",))
                        tq, after = adtqc(lab, p, tt)
                    else:
                        off, offa, tq, after = {}, {}, np.nan, np.nan
                    row.update({f"all_{k}": v for k, v in off.items()})
                    row.update({f"anom_{k}": v for k, v in offa.items()})
                    pa = parts(gt, p, tns, month_edges(tt))
                    fp = pa[2].sum()
                    row.update(F_all=float(f_from(pa.sum(1))),
                               F_anom=float(f_from(parts(test_gt(lab, tt, ("Anomaly",)), p, tns,
                                                         month_edges(tt)).sum(1))))
                    row.update(ADTQC=tq, ADTQC_after_rate=after,
                               false_alarms_per_month=fp / ((tns[-1] - tns[0]) / MONTH_NS), **operator(gt, p, tns))
                    rows.append(row)
                print(f"seed {seed} M{s}->M{t} done", flush=True)
                pd.DataFrame(rows).to_csv(ROOT / "results" / "step14_metrics.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "step14_metrics.csv", index=False)


if __name__ == "__main__":
    main()
