# Step 1: Load, look at, and resample a channel

## What we did
1. Checked the environment (Python 3.13, pandas 3.1, numpy 2.5, matplotlib 3.11, scipy, jupyter all present).
2. Loaded `channel_1` of Mission 1 with `pd.read_pickle`. **It worked under pandas 3.1** (this was an open risk).
3. Wrote `src/loader.py` (`load_channel(mission, channel)`).
4. Looked at the channel: 10,513,336 rows, 1 column (`float32`), index = timestamps, 0 missing values, strictly increasing time.
5. Plotted it (`notebooks/step1_channel1.png`).
6. Wrote `src/preprocess.py`: resample every **target** channel to a uniform 5-minute grid and cache it in `cache/` (so later steps load in seconds, not minutes).

## Why
- A model needs evenly spaced samples. The raw channel is **not** evenly spaced: about 5.1M gaps are 3 s, about 4.7M are 90 s, the rest are jitter around those. (Measured by us with `diff().value_counts()`.)
- Loading a 126 MB pickle 105 times for every experiment is slow, so we cache a small version once.

## Key terms (plain words)
- **Channel**: one sensor's readings over time.
- **Target channel**: a channel the benchmark checks for anomalies. Mission 1 has 58, Mission 2 has 47 (paper, and confirmed by counting `Target == YES` in `channels.csv`). The other channels are only supporting information.
- **Pickle**: Python's "freeze this object to a file" format. `read_pickle` thaws it. Only load pickles from trusted sources, because thawing can run code.
- **DataFrame / Series**: a table / one column of a table. The row labels (index) are timestamps.
- **Resampling**: re-reading the series on a regular time grid.
- **Zero-order hold (ZOH)**: for each grid time, use the last value we actually saw. It never uses future data and never invents values in between. This is what the ESA-ADB paper recommends.

## Code, line by line

`src/loader.py`
```python
DATA = Path(__file__).resolve().parent.parent / "ESA-ADB" / "data"   # project root / ESA-ADB/data
path = DATA / f"ESA-Mission{mission}" / "channels" / f"channel_{channel}.zip"   # build the file path
df = pd.read_pickle(path)           # open the zip, thaw the table
return df[f"channel_{channel}"]     # the table has one column; return it as a Series
```

`src/preprocess.py` (core line)
```python
r = s.resample("5min").last().ffill().bfill()
```
- `resample("5min")` groups the rows into 5-minute bins.
- `.last()` takes the last reading in each bin (ZOH).
- `.ffill()` fills empty bins with the previous value (ZOH again).
- `.bfill()` fills any empty bins at the very start with the first known value (the paper does the same and notes it uses a little future information, only at the very start).

The result is saved as `cache/M{mission}/channel_{c}.npz` (start time + float32 values). One mission month has 8,640 bins.

## What the data looks like
- Mission 1: 2000-01-01 to 2013-12-31, 14 years, 1,472,832 five-minute bins.
- Mission 2: 2000-01-01 to 2003-06-30, 3.5 years, 367,776 bins.
- `channel_1` is almost flat (about 0.138) with rare sharp spikes up to about 0.43.
- Plotting every 1000th point can hide short spikes, so always also look at a zoomed window.

## Choices and flags
- **5-minute grid is OUR choice.** The paper uses 0.033 Hz (30 s) for Mission 1 and 0.056 Hz for Mission 2. A coarser grid is about 10x less data, but it can lose events shorter than 5 minutes. (We measured the cost: see Step 2.)
- `channels.csv` Target column identifies target channels (verified count: 58 and 47, matches paper).
- Channels start/end within one 5-minute bin of each other. `src/data.py` aligns them by repeating the first/last value.

## What broke and how we fixed it
- First alignment attempt asserted all channels start at the same time. A few channels start 5 minutes later, so the assert failed. Fix: build one common grid and pad the edge by one bin.
- `sed -i` on macOS needs a different syntax than Linux; we used the editor tool instead.

## Self-test
1. Why can't we feed the raw channel directly into a model?
2. What does zero-order hold do, and why is it a good fit for satellite telemetry (which has many step-like signals)?
3. If an anomaly lasts 2 minutes, what can go wrong with a 5-minute grid?
