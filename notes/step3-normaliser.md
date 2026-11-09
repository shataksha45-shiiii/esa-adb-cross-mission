# Step 3: Per-channel normaliser (no leakage)

## What we did
Wrote `src/normaliser.py`: fit scaling constants for each channel on the **training period only**, then apply them to any period (train and test).

## Why
- Channels live on completely different scales (volts, degrees, flags). A single shared model needs every channel on the same footing.
- **Leakage guard**: if the scaling constants were computed from the test period, test information would flow into the model. We fit on train only.

## Key terms
- **Normalisation / standardisation**: shift and scale values so a typical value is near 0 and spread is near 1.
- **Median**: the middle value. **MAD (median absolute deviation)**: the median of how far values are from the median. Both barely move when a few extreme values are present.
- **Clipping**: capping values at a limit so one giant spike cannot dominate.
- **Leakage**: any way test information reaches training.

## The maths

Robust standardisation of channel values x:

$$
z = \frac{x - \mathrm{median}(x_{train})}{1.4826 \cdot \mathrm{MAD}(x_{train})},
\qquad
\mathrm{MAD} = \mathrm{median}\big(|x_{train} - \mathrm{median}(x_{train})|\big)
$$

The factor 1.4826 makes the MAD comparable to a standard deviation when data are Gaussian. After that we clip: z is limited to [-20, 20].

## Rules (from the paper's standardisation section, with one substitution)
| Channel kind | Detected how (on train) | Treatment |
|---|---|---|
| binary | 2 or fewer unique values | min-max to [0, 1] |
| constant | MAD = 0 and std = 0 | subtract median |
| monotonic | all differences >= 0 or all <= 0 | first difference, then robust standardise |
| normal | everything else | robust standardise (if MAD = 0, use std instead) |

- **Substitution (flagged):** the paper standardises with mean/std over **nominal** training points, which needs labels to know which points are nominal. We are label-free, so we use median/MAD over **all** training points.
- Clip value 20 is **our choice**, not from the paper.

## Code, line by line (`fit_channel`)
```python
uniq = np.unique(x)
if len(uniq) <= 2: ...                    # binary -> min-max
d = np.diff(x)
if (d >= 0).all() or (d <= 0).all(): ...  # monotonic -> work on the differences
med = np.median(x); mad = 1.4826 * np.median(np.abs(x - med))
if mad > 0: return ChannelNorm(kind, med, mad)       # normal case
std = x.std()                                         # MAD can be 0 for mostly-flat channels
if std > 0: return ChannelNorm(kind, med, std)
return ChannelNorm("constant", med, 1.0)
```
`transform` then does `(x - center) / scale` and clips to +-20.

## What we found
- After the 5-minute resample every target channel in both missions was classified **normal** (no binary, constant or monotonic channels among targets). The other kinds are implemented but unused on these target channels.
- Fraction of train values clipped: M1 mean 0.026%, worst channel 0.32%; M2 mean 0.096%, worst channel 1.1%. Test: M1 mean 0.18%, M2 mean 0.13%. So clipping touches few points.
- Train z standard deviation per channel is not exactly 1 (median 0.84 for M1, 1.22 for M2), expected with robust scaling when tails are heavy.

## Cross-mission use
For "model trained on mission A, tested on mission B", the normaliser is fitted on **B's own training period** (no labels). This is the only B information the model sees, and it is how a new mission would be adapted label-free.

## What broke
Nothing; the first run worked.

## Self-test
1. Why use median/MAD instead of mean/std when anomalies are not labelled?
2. What would go wrong if we fitted the normaliser on the whole mission including test?
3. Which channels would be treated specially, and why do first differences help on a counter?
