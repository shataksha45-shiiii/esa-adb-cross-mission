# Step 2: The metric, corrected event-wise F0.5 (evaluator)

## What we did
1. Read the official metric code (`ESA-ADB/timeeval/metrics/ESA_ADB_metrics.py`) and the paper's metric section.
2. Wrote our own version in `src/evaluator.py` (`ew_scores`) and a wrapper that can call the official code (`official_scores`).
3. Cross-checked ours against the official one on 4 hand-made predictions per mission. **They match to 4 decimals in every case.**
4. Defined the train/test split in `src/data.py`, and joined `labels.csv` with `anomaly_types.csv`.

## Why this comes before any model
The score decides what "good" means. A model that looks great on a point-wise metric can be terrible on this one.

## Key terms (plain words)
- **Event**: one anomaly, identified by an `ID` in `labels.csv`. One event can touch several channels (several rows with the same ID). The metric works per event, not per row.
- **Alarm event**: a run of consecutive time steps where we predicted "anomaly".
- **Detected (TP)**: a ground-truth event that overlaps at least one of our alarm events.
- **Missed (FN)**: a ground-truth event with no overlapping alarm.
- **False alarm (FP)**: one of our alarm events that overlaps no ground-truth event at all.
- **TNR (true negative rate, time-based)**: the fraction of *normal time* in which we correctly stayed quiet.
- **F-beta**: one number mixing precision and recall. beta = 0.5 means precision matters more than recall.

## The maths (equations on their own lines)

Event-wise precision and recall (counts of events):

$$
P_{ev} = \frac{TP}{TP + FP}
\qquad
R_{ev} = \frac{TP}{TP + FN}
$$

Correction from Sehili et al. 2023 (used by ESA-ADB): multiply precision by how well we stay quiet in normal time:

$$
\mathrm{TNR} = 1 - \frac{\text{seconds of alarm that lie in normal time}}{\text{total seconds of normal time}}
\qquad
P = P_{ev} \cdot \mathrm{TNR}
$$

F-beta (we use beta = 0.5):

$$
F_{\beta} = \frac{(1+\beta^{2})\, P\, R}{\beta^{2} P + R}
$$

Symbols: TP/FP/FN as above; P, R are the corrected precision and recall; beta < 1 weights precision more.

Why the correction? Without it, one huge alarm covering the whole mission "detects" everything and has few false-alarm *events*. TNR punishes alarms that cover lots of normal time.

## Code (our `ew_scores`), line by line
```python
events_pred = pred_to_events(pred, times)         # runs of 1s -> (start, end) intervals
sel = labels[labels.Category.isin(categories)]    # which GT events count (e.g. Rare Event + Anomaly)
all_gt = merge_intervals(all labels)              # ALL events, selected or not, for the FP/TNR rules
for each ID in sel:                               # per event
    hit = any alarm overlaps any interval of that ID
    tp += hit; fn += not hit
fp = alarms that matched nothing AND touch no GT interval at all
precision = tp/(tp+fp); recall = tp/(tp+fn)
tnr = 1 - (alarm seconds outside all GT intervals) / (normal seconds)
precision *= tnr
F = (1+b^2) P R / (b^2 P + R)
```

## Validation against the official code (4 test predictions per mission)
Test period, categories Rare Event + Anomaly, beta 0.5:

| Prediction | M1 ours | M1 official | M2 ours | M2 official |
|---|---|---|---|---|
| oracle (alarm exactly on every GT event) | 0.9860 | 0.9860 | 0.9975 | 0.9975 |
| alarm on half the events | 0.9860 | 0.9860 | 0.9973 | 0.9973 |
| random alarms (0.05% of steps) | 0.0586 | 0.0586 | 0.0192 | 0.0192 |
| oracle + random alarms | 0.2256 | 0.2256 | 0.8126 | 0.8126 |

- Empty prediction: ours returns 0 (recall 0). **The official code crashes** (`max()` of an empty array). Known edge case, we handle it.
- The oracle gets recall 0.934 on M1, not 1.0. Likely cause (inferred, partly checked): 9 of 91 M1 test events last under 5 minutes and can fall between 5-minute grid points. Checked: 9 such events exist; we did not trace every missed one.
- Practical meaning: with this grid the best possible F0.5 is about 0.986 on M1 and 0.9975 on M2.

## Choices and flags
- **Train/test split**: paper says first half train, second half test: 84 months (M1), 21 months (M2), last 3 months of train = validation. Boundary dates `2007-01-01` (M1) and `2001-10-01` (M2) are **our calculation** (2000-01-01 + 84 / 21 months), not printed dates.
- **Which events count in test**: we keep events whose `StartTime` is on or after the split date (M1: 91 events of 200; M2: 362 of 644). An event straddling the boundary is dropped. Our simplification.
- Category sets: "Rare Event + Anomaly" and "Anomaly only" (both used in the official experiment scripts).
- M2 test has only 6 "Anomaly" events and 356 "Rare Event" events, so "Anomaly only" on M2 is extremely noisy.
- We did not reproduce the affiliation-based metric (the official code also reports it); only event-wise F0.5.

## What broke and how we fixed it
- Importing `timeeval` fails (needs docker and many packages). Fix: register stub package modules so only `timeeval/metrics/ESA_ADB_metrics.py` is imported.
- Official code prints debug lists; we silence stdout while calling it.

## Self-test
1. A detector marks the entire test period as one alarm. What are its TP, FP, FN, and why does TNR matter here?
2. Why does beta = 0.5 favour precision?
3. Why must the ground truth be grouped by `ID` and not by row?
