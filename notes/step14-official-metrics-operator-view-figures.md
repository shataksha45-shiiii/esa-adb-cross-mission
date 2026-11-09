# Step 14: all official metrics, the operator view, failure-analysis figures (roadmap items 6, 8, 10) (9–10 Nov 2026)

Code: `src/step14.py`, `scripts/make_figures_step14.py`. Results: `results/step14_metrics.csv`. Figures: `figures/fig5_forest`, `fig6_ablation`, `fig7_timeline`, `fig8_fake_vs_real` (.png and .pdf).
**Post-hoc (6th use of the test labels). Reporting only: no method or setting chosen here.**

## What we did
1. Computed every official ESA-ADB metric for the main methods:
   - corrected event-wise F0.5, all events and anomalies only;
   - alarming precision;
   - affiliation-based F0.5;
   - ADTQC (timing quality).

   The official code is slow on Mission 1 (about 15 min per pair), so these extras are for **seed 0 only**. The event-wise F0.5 is for all 5 seeds. Seed-0 official F matches our own evaluator: C3 M1→M2 is 0.379 in both.
2. Built an **operator view** (5-seed mean): alarms per month, false alarms per month, share of events caught, time spent in alarm, and median detection delay.
3. Drew 4 paper figures for the failure analysis.

## Item 6: official metrics (seed 0)
| Method | Pair | EW F0.5 (all) | EW F0.5 (anomalies) | Alarming precision | Affiliation F0.5 | ADTQC |
|---|---|---|---|---|---|---|
| Target-label-tuned (oracle) | M1→M2 | 0.848 | 0.270 | 0.947 | 0.703 | 0.774 |
| Heydari static | M1→M2 | **0.934** | **0.498** | 0.896 | **0.843** | **0.953** |
| Fixed default | M1→M2 | 0.892 | 0.073 | 0.884 | 0.832 | 0.794 |
| C3 (selected) | M1→M2 | 0.379 | 0.000 | 0.918 | 0.418 | 0.685 |
| C5 | M1→M2 | 0.379 | 0.000 | 0.918 | 0.418 | 0.685 |
| Target-label-tuned (oracle) | M2→M1 | 0.245 | 0.211 | 0.389 | 0.330 | 0.995 |
| Heydari static | M2→M1 | 0.007 | 0.004 | 0.074 | 0.558 | 0.520 |
| Fixed default | M2→M1 | 0.040 | 0.025 | 0.361 | 0.537 | 0.466 |
| C3 (selected) | M2→M1 | 0.083 | 0.056 | 0.731 | 0.409 | 0.316 |
| C5 | M2→M1 | **0.245** | **0.121** | 0.643 | 0.314 | **0.593** |

(Full table, all 4 pairs: `results/step14_metrics.csv`.)

What it means:
- **Affiliation F0.5 rewards different things.** It measures how close alarms are to events, not how many separate false alarms there are. So the flood-prone Heydari (0.558) and fixed default (0.537) score *higher* than C5 (0.314) on M2→M1, even though their event-wise F0.5 is near zero. The paper must say which metric it optimises: the official primary one, corrected event-wise F0.5.
- **Anomalies only:** on M1→M2 our methods catch **none of the real anomalies** (0.000). Heydari catches some (0.498). Most of our M2 hits are rare (commanded) events.
- **Timing (ADTQC):** C5 is clearly better than C3 when M1 is the target (0.593 vs 0.316): its alarms start nearer the event start.

## Item 10: the operator view (mean of 5 seeds)
| Method | Pair | Alarms per month | False alarms per month | Events caught | Time in alarm |
|---|---|---|---|---|---|
| Oracle | M1→M2 | 10.1 | 0.10 | 59% | 5.7% |
| Heydari static | M1→M2 | 14.1 | 0.00 | 76% | 3.6% |
| C3 / C5 | M1→M2 | 3.9 | 1.47 | 15% | 2.6% |
| Old transfer (source-tuned) | M1→M2 | 0.35 | 0.00 | 2% | 0.1% |
| Oracle | M2→M1 | 0.34 | 0.10 | 10% | 0.1% |
| Heydari static | M2→M1 | **88.8** | **79.5** | 66% | **42%** |
| Old transfer | M2→M1 | 6.0 | 5.15 | 35% | 22% |
| C3 | M2→M1 | 2.7 | 2.56 | 20% | 12% |
| **C5** | M2→M1 | **0.31** | **0.17** | 11% | 4.5% |

In plain words:
- **M2→M1:** an operator using Heydari's method or the old transfer gets **80 or 5 false alarms a month**, with the alarm on 42% or 22% of the time. Nobody would keep that system switched on. C5 gives **one false alarm every six months**, close to the oracle's workload, but catches only about 1 event in 9.
- **M1→M2:** Heydari is excellent: 76% of events caught, no false alarms. Ours is cautious: 15% caught, about 1.5 false alarms a month.
- **Median detection delay** is not reported in the table: when an alarm is already on before an event starts (an alarm flood), the "delay" is a large negative number that means nothing. ADTQC above handles timing properly. The values are in the CSV.

## Item 8: failure-analysis figures
1. **fig5_forest:** test F0.5 with 95% month-block bootstrap CIs. Shows at a glance that no method wins both directions, and how wide the M1-target CIs are.
2. **fig6_ablation:** what removing each part does, for C3 and C5. The event builder and the AE matter most.
3. **fig7_timeline (key failure figure):** M2→M1 alarms across the M1 test years, under the monthly share of |z| > 5.
   - **New detail:** each level-shift episode **starts with a labelled event** (an anomaly in mid-2009, rare events in mid-2013), but the shifted level **stays long after the event ends**.
   - A static level detector keeps alarming for months: Heydari is on from mid-2010 to the end (42% of time); C3 stays on from mid-2013 to the end.
   - The rolling version (C5) switches off after a few weeks.
   - C3 also raises a cluster of alarms in early 2012 with no labelled event and no level shift: false alarms from the AE part.
4. **fig8_fake_vs_real:** recall on injected fakes vs recall on real events (dev). Every point is on or below the diagonal: **fakes are always at least as easy to catch as real events**, often much easier (M2→M1: 0.78 vs 0.24). That is why label-free calibration leans too strict when M2 is the target.

Palette: the validated reference categorical slots 1–3 (blue, orange, aqua; all-pairs CVD ΔE ≥ 9.2), baselines in gray. Aqua is below 3:1 contrast, so every series is direct-labelled.

## What changes for the paper
- Lead with corrected event-wise F0.5 (official primary metric). Report affiliation F0.5 and ADTQC, and explain why they rank methods differently.
- Use the operator table: it is the clearest way to show why Heydari-style and source-tuned transfer are unusable on M2→M1 (80 or 5 false alarms per month).
- The timeline figure is the centrepiece of the failure analysis.
- New limitation: on M1→M2 our methods detect no real **anomalies** (only rare events).

## Self-test
1. Why can a method with near-zero event-wise F0.5 have a decent affiliation F0.5?
2. From the operator table: why would an operator switch Heydari's method off on M2→M1 but keep it on for M1→M2?
3. In the timeline, why does a static level detector keep alarming after the 2013 event ends, while the rolling one stops?
4. What does it mean that every point in fig8 is below the diagonal?
