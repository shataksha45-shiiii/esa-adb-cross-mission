"""Data access: cached 5-min channels as one matrix per mission, labels, and the train/test split."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from loader import DATA
from preprocess import CACHE, STEP, target_channels

# Paper: first half = train (84 months M1, 21 months M2), second half = test. Mission starts 2000-01-01.
# Boundary dates below are 2000-01-01 + 84 / 21 months (derived by us, not printed in the paper).
SPLIT = {1: pd.Timestamp("2007-01-01"), 2: pd.Timestamp("2001-10-01")}


@lru_cache(maxsize=None)
def load_mission(mission: int):
    """Return (times: DatetimeIndex, X: float32 [T, n_channels], channel_ids: list[int]) for TARGET channels."""
    chans = target_channels(mission)
    zs = [np.load(CACHE / f"M{mission}" / f"channel_{c}.npz") for c in chans]
    step_ns = pd.Timedelta(STEP).value
    start = min(int(z["start"]) for z in zs)
    ends = [int(z["start"]) + (len(z["values"]) - 1) * step_ns for z in zs]
    T = (max(ends) - start) // step_ns + 1
    X = np.empty((T, len(chans)), dtype=np.float32)
    for j, z in enumerate(zs):                         # channels differ by at most 1 bin at head/tail:
        v, off = z["values"], (int(z["start"]) - start) // step_ns
        X[:off, j] = v[0]                              # repeat first value at the head
        X[off:off + len(v), j] = v
        X[off + len(v):, j] = v[-1]                    # repeat last value at the tail
    # Each cached value is the LAST sample inside bin [b, b+5min). Label it by the bin's END (b+5min):
    # the value is only known then. Labelling by the start would use up to 5 min of future (online rule).
    times = pd.date_range(pd.Timestamp(start) + pd.Timedelta(STEP), periods=T, freq=STEP)
    return times, X, chans


def load_labels(mission: int) -> pd.DataFrame:
    """labels.csv joined with anomaly_types.csv; timestamps tz-naive UTC."""
    d = DATA / f"ESA-Mission{mission}"
    lab = pd.read_csv(d / "labels.csv")
    typ = pd.read_csv(d / "anomaly_types.csv")
    lab = lab.merge(typ, on="ID", how="left")
    for col in ("StartTime", "EndTime"):
        lab[col] = pd.to_datetime(lab[col], utc=True).dt.tz_localize(None)
    return lab


def split_index(mission: int, times: pd.DatetimeIndex) -> int:
    """First index of the test period."""
    return int(times.searchsorted(SPLIT[mission]))
