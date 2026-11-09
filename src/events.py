"""Event builder: per-step alarms -> clean alarm events -> one mission-level alarm vector.

Per channel:  alarm = score > threshold
              close gaps <= GAP steps (alarms split by a short dip become one event)
              drop events shorter than MIN_LEN steps (single-step blips)
Mission:      alarm_mission(t) = 1 if ANY target channel alarms at t.
"""
import numpy as np


def _runs(b: np.ndarray):
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def build_events(alarm: np.ndarray, gap: int = 12, min_len: int = 3) -> np.ndarray:
    a = alarm.astype(bool)
    s, e = _runs(a)
    out = np.zeros(len(a), dtype=bool)
    if len(s) == 0:
        return out
    ms, me = [s[0]], [e[0]]
    for si, ei in zip(s[1:], e[1:]):
        if si - me[-1] <= gap:
            me[-1] = ei
        else:
            ms.append(si)
            me.append(ei)
    for si, ei in zip(ms, me):
        if ei - si >= min_len:
            out[si:ei] = True
    return out


def mission_alarm(alarms: np.ndarray, gap: int = 12, min_len: int = 3) -> np.ndarray:
    """alarms: [T, C] bool -> [T] int (OR over channels after per-channel cleaning)."""
    out = np.zeros(alarms.shape[0], dtype=bool)
    for j in range(alarms.shape[1]):
        out |= build_events(alarms[:, j], gap, min_len)
    return out.astype(int)
