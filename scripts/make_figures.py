"""Make the README figures (figures/*.png) from cached data and saved results. No test-label tuning happens here.

Run: python scripts/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import SPLIT, load_labels, load_mission, split_index      # noqa: E402
from inject import inject, source_prior_v3                            # noqa: E402
from normaliser import fit_matrix, transform_matrix                   # noqa: E402

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
# validated palette (dataviz reference, light mode): slots 1-3 + neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TXT, TXT2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 11,
    "axes.edgecolor": AXIS, "axes.labelcolor": TXT2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlecolor": TXT, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "legend.labelcolor": TXT2,
})


def shade(ax, s, e, color, alpha=0.18):
    ax.axvspan(s, e, color=color, alpha=alpha, lw=0)


# ---------------------------------------------------------------- 1. one channel, whole mission, with labelled events
def fig_channel(m=1):
    t, X, ch = load_mission(m)
    lab = load_labels(m)
    lab = lab[lab.Category.isin(["Anomaly", "Rare Event"])]
    counts = lab.Channel.value_counts()
    name = next(f"channel_{c}" for c in ch if f"channel_{c}" in counts.index)
    j = ch.index(int(name.split("_")[1]))
    step = 288                                                          # 1 point per day for the overview
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(t[::step], X[::step, j], color=BLUE, lw=1.2)
    for _, r in lab[lab.Channel == name].iterrows():
        shade(ax, r.StartTime, max(r.EndTime, r.StartTime + pd.Timedelta("2D")), ORANGE, 0.35)
    ax.axvline(SPLIT[m], color=TXT2, lw=1.5, ls="--")
    ax.text(SPLIT[m], ax.get_ylim()[1], "  train | test", va="top", color=TXT2, fontsize=10)
    ax.set_title(f"ESA Mission {m}, {name}: 14 years of telemetry (daily samples), labelled events in orange")
    ax.set_ylabel("raw value")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_channel.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 2. a fake event pasted into real data (v3 = replayed real event)
def fig_injection():
    t, X, ch = load_mission(2)
    i = split_index(2, t)
    mid = i // 2
    Z = transform_matrix(X[:i], fit_matrix(X[:mid]))
    tall = t.values.astype(np.int64)
    rng = np.random.default_rng(7)
    Zi, placed = inject(Z, mid + 24, i, tall, source_prior_v3(1), rng)
    best = None                                      # clearest example: calm original, visible fake change
    for a, b in placed:
        if not 6 <= b - a <= 200:
            continue
        calm = np.abs(np.clip(Z[a - 288:b + 288], -20, 20)).max(0) < 3
        diff = np.abs(Zi[a:b] - np.clip(Z[a:b], -20, 20)).max(0) * calm
        j = int(np.argmax(diff))
        if best is None or diff[j] > best[0]:
            best = (diff[j], a, b, j)
    _, s, e, j = best
    lo, hi = s - 288, e + 288
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(t[lo:hi], np.clip(Z[lo:hi, j], -20, 20), color=BLUE, lw=1.6, label="original (unlabelled target data)")
    ax.plot(t[lo:hi], Zi[lo:hi, j], color=ORANGE, lw=1.6, ls="--", label="with a pasted real-shaped fake event")
    shade(ax, t[s], t[e - 1], ORANGE)
    ax.set_title("Synthetic injection: a fake event at a known place = a free label")
    ax.set_ylabel("normalised value (z)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_injection.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 3. test results
def fig_results():
    old = pd.read_csv(ROOT / "results" / "final_step8.csv")
    old = old[old.detector == "fusedAE"]
    new = pd.read_csv(ROOT / "results" / "final_step9.csv")
    pairs = [(1, 2), (2, 1)]
    labels = ["M1 → M2", "M2 → M1"]
    o = [old[(old.source == a) & (old.target == b)]["F0.5"] for a, b in pairs]
    n = [new[(new.source == a) & (new.target == b)]["F0.5"] for a, b in pairs]
    x = np.arange(len(pairs))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k, (vals, col, name) in enumerate([(o, MUTED, "source-tuned settings (before)"),
                                           (n, BLUE, "label-free calibration (ours, v3)")]):
        m = [v.mean() for v in vals]
        sd = [v.std() for v in vals]
        xs = x + (k - 0.5) * (w + 0.02)
        ax.bar(xs, m, w, color=col, label=name, yerr=sd, capsize=4, error_kw=dict(ecolor=TXT2, lw=1))
        for xi, mi in zip(xs, m):
            ax.text(xi, mi + 0.012, f"{mi:.3f}", ha="center", va="bottom", color=TXT, fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylabel("event-wise F0.5 (official ESA-ADB)")
    ax.set_ylim(0, 0.25)
    ax.grid(axis="x", visible=False)
    ax.set_title("Cross-mission transfer on the test set (5 seeds, mean ± std)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_results.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 4. drift: share of extreme |z| per month on M1
def fig_drift():
    t, X, _ = load_mission(1)
    i = split_index(1, t)
    Z = transform_matrix(X, fit_matrix(X[:i]))
    ext = pd.Series((np.abs(Z) > 5).mean(1), index=t).resample("30D").mean() * 100
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(ext.index, ext.values, color=BLUE, lw=2)
    ax.axvline(SPLIT[1], color=TXT2, lw=1.5, ls="--")
    ax.text(SPLIT[1], ax.get_ylim()[1], "  train | test", va="top", color=TXT2, fontsize=10)
    tr, te = (np.abs(Z[:i]) > 5).mean() * 100, (np.abs(Z[i:]) > 5).mean() * 100
    ax.set_title(f"Mission 1 test years contain level shifts: readings beyond |z| > 5 rise from {tr:.2f}% to {te:.2f}%")
    ax.set_ylabel("% of channel-samples, monthly")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_drift.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_channel, fig_injection, fig_results, fig_drift):
        f()
        print("done", f.__name__, flush=True)
