# Step 9 design (FROZEN 9 Nov 2026, written before any result of this method was seen)

## Problem in one line
The model transfers between missions, but the post-processing settings (q, q2, gap, min_len, k-of-n) don't.
M1 settings are too strict on M2 (7 TP, recall 2%). M2 settings are too loose on M1 (about 430 false alarms).

## Idea: synthetic anomaly injection (label-free calibration)
On the TARGET mission we have no labels, but we do have its unlabelled training data. So:
1. Paste fake anomalies ("injections") into the target's unlabelled training data. We know where they are.
2. Try every post-processing setting. Score each one on how well it finds the fake anomalies (event-wise F0.5).
   Any alarm that does not touch a fake anomaly counts as a false alarm.
3. Keep the best setting and use it on the target's test period.

The model is still the source model, trained on the source's training data.
What the source gives us: the model, plus a description of what anomalies look like (durations, how many channels each one touches, how often they happen).
That description comes from the source's training labels, which we are allowed to use.

## Frozen choices (flagged guesses; none tuned on any label)
| Choice | Value | Why |
|---|---|---|
| Where injected | target TRAINING period only, normalised z-space, clipped to ±20 afterwards | never touch test |
| Folds | same 2-fold blocked split as tune.py: normaliser + EVT fitted on the clean fit half, injections only in the eval half | same protocol as Step 7 |
| Event rate | source training rate (M1: 105/84 = 1.25 per month, M2: 282/21 = 13.4 per month) | source prior |
| Duration + channel count | resampled as pairs from the source's training events (Rare Event + Anomaly), duration capped at 3 days, channel count capped at the target's channel count | source prior |
| Placement | random, non-overlapping, at least 1 day apart, inside the eval half | |
| Type (uniform) | level shift, spike burst, noise burst, flatline, ramp | common types in the literature (guess, not taken from ESA-ADB) |
| Amplitude A | log-uniform in [2, 10] robust-z units, random sign per channel | guess |
| Repeats | 3 injection draws per fold (seeds 0,1,2), score = mean F0.5 over 2 folds x 3 draws | reduce luck |
| Grid | same as tune.py (POST x Q2) | |
| Selection | argmax of mean injected F0.5 | |
| Model for selection | source AE seed 0 only (compute budget); chosen settings then used for all 5 test seeds | |

Type definitions, for one channel z over steps [s, e), L = e - s:
- shift: z += sign * A
- spike: max(1, L // 6) random steps get z += sign * A
- noise: z += 0.5 * A * N(0, 1)
- flat: z = z[s - 1] (sensor stuck)
- ramp: z += sign * A * linspace(0, 1, L)

## Evaluation plan (fixed in advance)
1. Dev check (diagnosis only, after selection): CV F0.5 on the target's TRAINING labels, comparing injection-chosen settings with the source settings (current transfer) and the target's own label-tuned settings (upper bound).
2. Test: one run, 5 seeds, official metric, all 4 pairs (M1->M1 and M2->M2 then become fully label-free).
3. If it fails, it is reported as failed. Any change after this point is "v2" and gets flagged as designed after seeing target-training-label diagnostics.

Known weakness, stated in advance: real (unlabelled) events in the target background get counted as false alarms during selection.
This pushes towards stricter settings, which matters most on M2, where real events are frequent.

---

# v2 design (FROZEN 9 Nov 2026, written AFTER seeing v1 dev results, BEFORE any v2 number)

**Flag:** v1's dev check used the target training labels, and v2 was designed after seeing it. So v2 is not a blind design. This must be said in the paper.

## What v1 got wrong
v1 counted **every** alarm in the unlabelled background as a false alarm. On M2 most of those alarms are real events (about 13 per month), so v1 went too strict.
v1 also copied the source's event rate, the same mistake that made alarm-rate matching fail in Step 8a.

## Idea: estimate how many real events hide in the background (positive-unlabelled reasoning)
For a setting θ, measured on the target with no labels:
- R(θ) = recall on injected fake events (assumed to equal recall on real events: "fake events are representative")
- B(θ) = number of alarm events in the background (alarms that touch no fake event) = real-event hits + false alarms

If the target has N real events in the eval half, a setting catches about R(θ)·N of them, so

$$
B(\theta) = R(\theta)\,N + FP(\theta), \qquad FP(\theta) \ge 0 \;\Rightarrow\; N \le \frac{B(\theta)}{R(\theta)} \text{ for every } \theta
$$

Estimate N by the tightest bound:

$$
\hat N = \min_{\theta:\; R(\theta) \ge 0.2} \frac{B(\theta)}{R(\theta)}
$$

(R ≥ 0.2 avoids dividing by tiny, noisy recalls.) Then for every setting:

$$
\hat P(\theta) = \frac{R(\theta)\,\hat N}{B(\theta)}, \qquad
\widehat{TNR}(\theta) = 1 - \big(1 - \hat P(\theta)\big)\big(1 - TNR_{bg}(\theta)\big)
$$

$$
\hat F(\theta) = F_{0.5}\big(\hat P \cdot \widehat{TNR},\; R\big)
$$

- TNR_bg = 1 − (background alarm time / background time), which inject.py already measures.
- The alarm time that isn't real-event time is assumed to be the same fraction (1 − P̂) of background alarm time.

Select argmax F̂. B, R and TNR_bg are means over 2 folds × 3 repeats per setting, taken before the ratio.

## What else is unchanged
Same injections, same grid, same CSVs from v1 (no new runs needed), same source model.

## Allowed validation, per direction (source labels only)
For direction S→T, check N̂ from the S→S calibration against the true number of S events in the eval halves.
That only uses S's labels. Report it, but change nothing because of it.

---

# v3 design (FROZEN 9 Nov 2026, after v2 dev results and the amplitude diagnosis, BEFORE any v3 number)

**Flag:** designed after seeing v1/v2 dev diagnostics on target training labels.

## Change: replay real source events instead of inventing amplitudes
v1/v2 drew the amplitude from log-uniform [2, 10]. Real source events are much smaller (M1 median 0.79) or bimodal (M2).
v3 builds each fake event from one **real source training event**:
- duration = that event's duration (cap 3 days, as before)
- channel count = how many of the source's target channels it touched
- amplitude of channel k = that source event's measured amplitude on its k-th channel: max |z_t − z_{s−1}| inside the event (source normaliser fitted on source train)
- amplitude floor 0.05 (keeps tiny events from being exactly 0)
- type, sign, rate, placement, folds, repeats, grid: unchanged from v1
Selection: v2 rule (N̂ = min B/R with R ≥ 0.2, then argmax F̂). No other change.

## Allowed validation (per direction, source labels only)
For S→T, run S→S and report: N̂ vs true N, and fake recall vs real recall at the S→S chosen setting.
Report only; change nothing because of it.

## Then
Dev check on target training labels (diagnosis). If v3 holds up on all four pairs, run the test ONCE with v3 (5 seeds, official metric).

---

# v4 design (FROZEN 9 Nov 2026, after v3 dev results, BEFORE any v4 number). LAST version: after v4 we freeze and test.

**Flag:** designed after seeing v1–v3 dev diagnostics on target training labels.

## Problem v4 fixes
v3 copied raw sizes (|z − z_before|) and used random shapes and signs. A size that is normal for a busy source channel became a glaring anomaly on a quiet target channel. Result: fakes were caught far more often than real events (M2→M2: 0.66 vs 0.09).

## v4: transplant the real shape, rescaled to the new channel's normal variability
For a real source event on source channel j, from step s to e (length L):
1. Deviation curve (the real shape and sign): d_t = z_t − z_{s−1}, for t = s … e−1.
2. Normal variability of a channel c over length L:

$$
D_c(L) = \mathrm{median}_{\text{windows}} \; \max_{0 \le u < L} \big| z_{a+u} - z_{a-1} \big|
$$

   This is the median over 2000 random windows of that length taken from training data: how far the channel normally wanders in L steps.
   It is computed for L on a grid of powers of 2 (1 … 1024), using the nearest grid value. Floor 0.05.
3. Paste it on target channel k (random) at start s′:

$$
z'_{s'+u} = z_{s'+u} + d_{s+u} \cdot \frac{D_k(L)}{D_j(L)}
$$

- D_j: computed on the source train (source normaliser). D_k: computed on the target's clean fit half (target normaliser).
- Channels: the real event's channels, each mapped to a random distinct target channel.
- Everything else (rate, placement, folds, repeats, grid, v2 selection rule, clip ±20) unchanged.

## Then
1. Validation on source labels only (S→S): N̂ vs true N, and fake vs real recall.
2. Dev check on target training labels.
3. Test ONCE with v4 (5 seeds, official metric), whatever the dev result, unless v4 is clearly worse than v3 on the source-only validation. In that case test v3. That rule uses no target labels.
