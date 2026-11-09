"""Step 10: rolling (online) re-normalisation. Uses ONLY past samples (causal).

rolling_center : subtract each channel's trailing-window median (drift guard for the |z| detector, our C4/C5).
rolling_robust_z: (z - trailing median) / (1.4826 * trailing MAD), as in Heydari et al. 2026 (their rolling
                  label-free adaptation; window length and update cadence are our choice, flagged in
                  notes/step10-protocol-frozen.md).
The centre/scale used for day d is computed from the `win` steps BEFORE day d starts, then held for the whole day.
Day 0 has no history: centre 0 / scale 1 (i.e. the static normaliser).
"""
import numpy as np

DAY = 288                      # 5-min steps per day
WIN = 30 * DAY                 # 30 days
CLIP = 20.0


def _daily_stats(Z, win, scale):
    T, C = Z.shape
    starts = np.arange(0, T, DAY)
    med = np.zeros((len(starts), C), np.float32)
    sc = np.ones((len(starts), C), np.float32)
    for n, a in enumerate(starts):
        if a == 0:
            continue
        w = Z[max(0, a - win):a]
        m = np.median(w, axis=0)
        med[n] = m
        if scale:
            sc[n] = np.maximum(1.4826 * np.median(np.abs(w - m), axis=0), 1e-3)
    return starts, med, sc


def _apply(Z, starts, med, sc):
    out = np.empty_like(Z)
    for n, a in enumerate(starts):
        out[a:a + DAY] = (Z[a:a + DAY] - med[n]) / sc[n]
    np.clip(out, -CLIP, CLIP, out=out)
    return out


def rolling_center(Z, win=WIN):
    s, m, c = _daily_stats(Z, win, scale=False)
    return _apply(Z, s, m, c)


def rolling_robust_z(Z, win=WIN):
    s, m, c = _daily_stats(Z, win, scale=True)
    return _apply(Z, s, m, c)
