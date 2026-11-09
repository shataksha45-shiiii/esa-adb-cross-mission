"""Preprocess: raw channel (10M irregular rows) -> uniform 5-minute grid, cached on disk.

Zero-order hold (ZOH): each 5-min bin gets the LAST known value, empty bins repeat the
previous value. This is the ESA-ADB paper's recommended interpolation (no future info).
Output: cache/M{mission}/channel_{c}.npz with  start (ns since epoch)  and  values (float32).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader import DATA, load_channel  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "cache"
STEP = "5min"  # OUR choice (paper uses 30 s for M1, ~18 s for M2); coarser = faster, may smear short anomalies


def target_channels(mission: int) -> list[int]:
    ch = pd.read_csv(DATA / f"ESA-Mission{mission}" / "channels.csv")
    ch = ch[ch["Target"] == "YES"]
    return [int(c.split("_")[1]) for c in ch["Channel"]]


def preprocess_one(args):
    mission, channel = args
    out = CACHE / f"M{mission}" / f"channel_{channel}.npz"
    if out.exists():
        return mission, channel, "cached"
    s = load_channel(mission, channel)
    r = s.resample(STEP).last().ffill().bfill()   # ZOH, then back-fill any leading gap
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, start=np.int64(r.index[0].value), values=r.to_numpy(np.float32))
    return mission, channel, len(r)


if __name__ == "__main__":
    jobs = [(m, c) for m in (1, 2) for c in target_channels(m)]
    print(len(jobs), "target channels to preprocess", flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        for res in ex.map(preprocess_one, jobs):
            print(res, flush=True)
