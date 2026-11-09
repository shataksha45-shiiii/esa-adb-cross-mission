"""Step 14 (roadmap item 8): paper figures for the failure analysis. IEEE sizes (3.5 in / 7.16 in), PNG + PDF.

fig5_forest      : test F0.5 with 95% month-block bootstrap CIs, transfer pairs (results/step13_ci.csv)
fig6_ablation    : remove-one-part ablations for C3 and C5 (results/step13_points.csv)
fig7_timeline    : where alarms fire on Mission 1's test years, M2->M1, seed 0 (recomputed here), with |z| drift
fig8_fake_vs_real: recall on injected fakes vs recall on real events (dev, target training labels)
Palette: validated categorical slots 1-3 (blue, orange, aqua; dataviz reference palette, light mode); baselines gray.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RES, FIG = ROOT / "results", ROOT / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "font.family": "DejaVu Sans", "savefig.dpi": 300, "savefig.bbox": "tight"})
PAIRS = [(1, 2), (2, 1)]
SHORT = {"B0 target-label-tuned": "Target-label-tuned (oracle)", "B1 source-label-tuned": "Source-label-tuned",
         "B2 fixed default": "Fixed default", "B3 Heydari static": "Heydari et al. (static)",
         "C1 v3 replayed fakes + N-hat": "C1: replay + N-hat (Step 9)", "C3 fake-recall match": "C3: fake-recall (selected)",
         "C5 fake-recall + rolling |z|": "C5: + rolling |z| (post-hoc)"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def forest():
    ci = pd.read_csv(RES / "step13_ci.csv")
    ci = ci[(ci.kind == "F") & ci.what.isin(SHORT)]
    order = list(SHORT)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4), sharey=True)
    for ax, (s, t) in zip(axes, PAIRS):
        d = ci[(ci.source == s) & (ci.target == t)].set_index("what").reindex(order)
        y = np.arange(len(order))
        ours = np.array([w.startswith("C") for w in order])
        for yy, (w, r) in zip(y, d.iterrows()):
            c = BLUE if w.startswith("C") else GRAY
            ax.plot([r.lo, r.hi], [yy, yy], color=c, lw=2, solid_capstyle="round")
            ax.plot(r.point, yy, "o", color=c, ms=5, mec="white", mew=1)
            ax.text(r.hi + 0.025, yy, f"{r.point:.2f}", va="center", fontsize=7, color=INK2)
        ax.set_xlim(0, 1.1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(f"M{s} → M{t}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Test F0.5 (95% CI, month-block bootstrap)")
        ax.grid(axis="y", visible=False)
        _ = ours
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([SHORT[w] for w in order])
    fig.text(0.99, 1.0, "blue = label-free (ours)    gray = baselines", ha="right", va="bottom", fontsize=7,
             color=INK2)
    save(fig, "fig5_forest")


def ablation():
    pts = pd.read_csv(RES / "step13_points.csv")
    m = pts.groupby(["method", "source", "target"]).F.mean()
    variants = [("full", ""), ("|z| off (AE only)", " ab-AE (|z| off)"), ("AE off (|z| only)", " ab-Z (AE off)"),
                ("no event builder", " ab-EB (no event builder)"), ("k-of-n = 1", " ab-K (k-of-n = 1)"),
                ("quantile, not EVT", " ab-Q (quantile, not EVT)")]
    full = {"C3": "C3 fake-recall match", "C5": "C5 fake-recall + rolling |z|"}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2), sharey=True)
    y = np.arange(len(variants))[::-1]
    for ax, (s, t) in zip(axes, PAIRS):
        for base, col, dy in (("C3", BLUE, 0.13), ("C5", ORANGE, -0.13)):
            vals = [m[(full[base] if suf == "" else base + suf, s, t)] for _, suf in variants]
            ax.scatter(vals, y + dy, s=26, color=col, edgecolor="white", linewidth=1, zorder=3, label=base)
            for v, yy in zip(vals, y + dy):
                ax.text(v + 0.015, yy, f"{v:.2f}", va="center", fontsize=6.5, color=INK2)
        ax.set_xlim(-0.02, 1)
        ax.set_title(f"M{s} → M{t}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Test F0.5 (mean of 5 seeds)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([v for v, _ in variants])
    h, _ = axes[1].get_legend_handles_labels()
    fig.legend(h, ["C3: fake-recall (selected)", "C5: + rolling |z| (post-hoc)"], frameon=False, fontsize=7,
               loc="lower right", bbox_to_anchor=(0.99, 0.98), ncol=2)
    save(fig, "fig6_ablation")


def timeline():
    from data import SPLIT
    from normaliser import fit_matrix, transform_matrix
    from run_final import AE, STEPS
    from step10 import hey_pred, load_all
    from step13 import Scorer, configs
    from tune import train_ae
    D = load_all()
    for k in (1, 2):
        D[k]["Z"] = transform_matrix(D[k]["X"], fit_matrix(D[k]["X"][:D[k]["i"]]))
    s, t = 2, 1
    model = train_ae(D[s]["Z"][:D[s]["i"]], AE["w"], AE["hidden"], AE["code"], STEPS, 0, center=True)
    d = D[t]
    i = d["i"]
    tt = d["t"][i:]
    sc = Scorer(model, d["Z"], i)
    cf = configs(s, t)
    rows = [("C3: fake-recall", sc.pred(*cf["C3 fake-recall match"]), BLUE),
            ("C5: + rolling |z|", sc.pred(*cf["C5 fake-recall + rolling |z|"]), ORANGE),
            ("Heydari (static)", hey_pred(d["Z"][:i], d["Z"][i:], d["t"][:i].values.astype(np.int64), D[s]["rate"]),
             AQUA)]
    lab = d["lab"][(d["lab"].EndTime > SPLIT[t]) & d["lab"].Category.isin(["Anomaly", "Rare Event"])]
    ev = lab.groupby("ID").agg(s=("StartTime", "min"), e=("EndTime", "max"), c=("Category", "first"))
    zshare = pd.Series((np.abs(d["Z"][i:]) > 5).mean(1), index=tt).resample("MS").mean()
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(7.16, 2.9), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2]))
    ax0.fill_between(zshare.index, zshare.values * 100, step="post", color=GRAY, alpha=0.5, lw=0)
    ax0.set_ylabel("|z| > 5\n(% of values)", fontsize=7)
    ax0.set_title("Mission 1 test years, transfer M2 → M1 (seed 0)", fontsize=9, color=INK, loc="left")
    ax0.grid(axis="x", visible=False)
    labels = []
    for k, (name, p, col) in enumerate(rows):
        yk = len(rows) - k + 1                                    # methods at 4, 3, 2; labelled events at 1
        pp = p.astype(np.int8)
        dd = np.diff(np.concatenate([[0], pp, [0]]))
        a, b = np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)
        ax.broken_barh([(tt[x], max(tt[min(y, len(tt) - 1)] - tt[x], pd.Timedelta(hours=12))) for x, y in zip(a, b)],
                       (yk - 0.32, 0.64), color=col, lw=0)
        labels.append((yk, f"{name}\n{len(a)} alarms, {p.mean() * 100:.1f}% of time"))
    for cat, col in (("Anomaly", INK), ("Rare Event", INK2)):
        e = ev[ev.c == cat]
        ax.broken_barh([(r.s, max(r.e - r.s, pd.Timedelta(hours=12))) for _, r in e.iterrows()],
                       (1.02 if cat == "Anomaly" else 0.66, 0.32), color=col, lw=0)
    labels.append((1.0, f"Labelled events (upper = anomaly, {int((ev.c == 'Anomaly').sum())};\n"
                        f"lower = rare event, {int((ev.c == 'Rare Event').sum())})"))
    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([l for _, l in labels], fontsize=6.5)
    ax.set_ylim(0.5, len(rows) + 1.5)
    ax.grid(axis="y", visible=False)
    save(fig, "fig7_timeline")


def fake_vs_real():
    from step13 import cfg_frm, cfg_nhat
    from inject_v2 import KEYS
    dev = pd.read_csv(RES / "dev_step10.csv")
    names = {"C1": "C1 v3 (Step 9)", "C3": "C3 fake-recall match"}
    pts = []
    for s in (1, 2):
        for t in (1, 2):
            inj = pd.read_csv(RES / "inject_v3" / f"S{s}_T{t}.csv").groupby(KEYS).R.mean()
            for k, fn in (("C1", cfg_nhat), ("C3", cfg_frm)):
                c = fn(s, t, "inject_v3")
                fake = float(inj.loc[tuple(c[x] for x in KEYS)])
                real = float(dev[(dev.source == s) & (dev.target == t) & (dev.method == names[k])].R.mean())
                pts.append((k, s, t, fake, real))
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", zorder=1)
    ax.text(0.97, 0.80, "y = x", fontsize=6.5, color=INK2, ha="right")
    offs = {("C3", 1, 1): (4, -11), ("C1", 1, 1): (6, -2)}
    for k, col in (("C1", BLUE), ("C3", ORANGE)):
        p = [x for x in pts if x[0] == k]
        ax.scatter([x[3] for x in p], [x[4] for x in p], s=30, color=col, edgecolor="white", lw=1, zorder=3,
                   label={"C1": "C1: replay + N-hat", "C3": "C3: fake-recall"}[k])
        for _, s, t, fx, ry in p:
            twin = [x for x in pts if x[0] != k and x[1:3] == (s, t) and abs(x[3] - fx) < 1e-9
                    and abs(x[4] - ry) < 1e-9]
            if twin and k == "C1":
                continue                                    # same setting chosen by both: one point, one label
            txt = f"M{s}→M{t}" + (" (C1 = C3)" if twin else "")
            ax.annotate(txt, (fx, ry), xytext=offs.get((k, s, t), (5, 5 if k == "C3" else -11)),
                        textcoords="offset points",
                        fontsize=6.5, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall on injected fake events")
    ax.set_ylabel("Recall on real events (dev)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Fakes are easier to catch than real events", fontsize=9, color=INK, loc="left")
    save(fig, "fig8_fake_vs_real")


if __name__ == "__main__":
    which = sys.argv[1:] or ["forest", "ablation", "fake_vs_real", "timeline"]
    for w in which:
        globals()[w]()
