# Step 5: EVT threshold and the event builder

## What we did
- `src/threshold.py`: per-channel threshold from Extreme Value Theory (peaks over threshold).
- `src/events.py`: turn per-step alarms into clean alarm events and combine channels into one mission-level alarm vector.

## Why
A score is just a number. To raise alarms we need a cutoff that is **not hand-tuned on labels**. EVT chooses it from the tail of normal-ish scores. The metric counts *events*, so jittery step-by-step alarms must be merged first.

## Key terms
- **Threshold**: score above this = alarm.
- **Peaks over threshold (POT)**: look only at the biggest scores (those above a high cutoff u), and model by how much they exceed u.
- **Generalised Pareto distribution (GPD)**: the distribution theory says those excesses roughly follow (shape xi, scale sigma).
- **Risk q**: the probability we accept that a normal score exceeds the threshold.
- **Calibration data**: the scores used to fit the GPD. Ours: the same channel's **training-period** scores. Never test scores.

## The maths

Let n be the number of calibration scores, u the 98th percentile of them, and y_i = s_i - u the excesses over u, N_u of them. Fit GPD(xi, sigma) to the y_i. The threshold with tail probability q:

$$
z_q = u + \frac{\sigma}{\xi}\left(\left(\frac{q\,n}{N_u}\right)^{-\xi} - 1\right)
\qquad (\xi \neq 0)
$$

$$
z_q = u - \sigma \ln\!\left(\frac{q\,n}{N_u}\right)
\qquad (\xi = 0)
$$

Symbols: z_q = threshold; xi = tail shape (heavier tail when positive); sigma = scale; n/N_u converts q (a probability over all scores) into a probability among the excesses.

Reference: Siffer et al. 2017 (SPOT). The paper's own pipeline is the benchmark's, our version is a simplified static (non-streaming) POT.

## Event builder
Per channel: alarm = score > threshold, then
1. **Close gaps**: alarms separated by 12 steps (1 hour) or fewer become one event.
2. **Drop blips**: events shorter than 3 steps (15 minutes) are removed.

Then the mission-level alarm is **1 if any target channel alarms** (OR). Both numbers (12, 3) and the OR rule are **our choices, not tuned**.

## Code, line by line (`evt_threshold`)
```python
u = np.quantile(s, p0)                       # p0 = 0.98 cutoff
y = s[s > u] - u                             # excesses
xi, _, sigma = genpareto.fit(y, floc=0)      # fit GPD with location fixed at 0
r = q * n / len(y)
return u + sigma/xi * (r ** (-xi) - 1)       # formula above (log version when xi ~ 0)
```
Fallback: if fewer than 30 excesses (degenerate scores) we return the maximum score.

## Leakage guard
Thresholds use only training-period scores of the target mission. Test scores and labels never touch them.

## The q we used (honest protocol)
- **Headline q = 1e-5**, fixed before looking at any test result.
- We also ran q from 1e-3 to 1e-7. Those numbers use test labels to *look at* sensitivity. They must **not** be used to pick the headline. Anything chosen from them is an oracle-tuned number and has to be labelled as such.

## What broke
- The baseline detector on M1 at q = 1e-3 raised alarms across essentially all normal time (TNR about 0), giving an F0.5 of about -3e-16, which is just floating-point rounding of zero. Harmless.

## Flags
- An OR across 58 channels means every channel's false alarms add up. That is why precision drops quickly as q grows.
- Static POT assumes the score distribution of the test period matches the train period. Drift breaks it (relevant when test is years later).

## Self-test
1. Why do we fit only the extreme tail instead of using "mean + 3 std"?
2. What does a smaller q do to precision and recall?
3. Why is it wrong to pick q by checking which value gives the best test F0.5?
