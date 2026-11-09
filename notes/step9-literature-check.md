# Literature check (9 Nov 2026)

What we did: we checked the two unverified numbers in our project log by reading the papers that cite them.
Why: a paper must never quote a number we have not traced to a source.

## Results

| Claim | Source found | Status |
|---|---|---|
| PitIA (Tejedor et al. 2025, BiDS'25): M1 0.424, M2 0.882 | arXiv 2609.10017, Sec. 3 "Comparison with existing baselines", described as "under static train/test configurations" | **Confirmed second-hand only.** The PitIA paper itself (BiDS'25 proceedings) was not found. Cite it as "reported in [2609.10017]" until we read the original. |
| ESA-ADB best unsupervised, all channels: M1 0.061, M2 0.241 | arXiv 2609.10017, Sec. 3 | 2609.10017 says 0.241. Our earlier reading of the ESA-ADB v2 PDF gave 0.079. **Still open:** re-read ESA-ADB arXiv 2406.17826 (the latest version) and find the exact table row. Possibly different table rows (anomalies only vs. anomalies + rare events, or channel subset). |
| ESA-ADB supervised with channel-subset selection: M1 0.786, M2 0.949 | arXiv 2609.10017 | New to us. Supervised (uses target labels), so not our setting. |

## Closest competitors (none does cross-mission)

1. **arXiv 2609.10017** (Alcarria et al., Sep 2026), "Structure-Aware Unsupervised Anomaly Detection ... Adaptive EVT Thresholding".
   Unsupervised + EVT, like ours. Reports F0.5 M1 0.700, M2 0.698.
   **Not comparable to our numbers:** monthly retraining on a sliding window (train on the previous month, test on the next), not the official 50/50 split.
   No cross-mission experiment. Shares an author (Auñón) with PitIA.
2. **arXiv 2607.07335** (Nguyen et al., Jul 2026), benchmark study.
   Uses plain event F0.5 (no TNR correction) and offline detection, so not comparable either. No cross-mission experiment.
   Possible copy error in their Table IV: the GCN row is identical to Table III.
3. **arXiv 2605.06681** (Allegrini & Pompei, May 2026), hierarchical ensemble. Not read yet (the PDF did not extract).

## What this means for us
- **Our novelty holds:** no paper we found evaluates train-on-one-mission, test-on-the-other.
- In-mission, we beat the ESA-ADB unsupervised baselines on both missions. Against PitIA's reported numbers: M1 ours 0.481 > 0.424, M2 ours 0.846 < 0.882.

## Self-test
1. Why can't we compare our 0.481 with 2609.10017's 0.700?
2. What is the difference between "corrected" event-wise F0.5 and plain event F0.5?
3. Why do we write "reported in [2609.10017]" instead of citing PitIA directly?

Sources: https://arxiv.org/html/2609.10017 , https://arxiv.org/html/2607.07335v1 , https://arxiv.org/pdf/2605.06681

---

## Addendum (9 Nov 2026, later): novelty claim is broken — a cross-mission ESA-ADB paper exists

**Heydari, Mostafavi, Nyarko (2026). "Workload-Constrained Cross-Mission Deployment of Spacecraft Telemetry Anomaly Alerting: Policy-Layer Feasibility Diagnostics and Limits." IEEE Access 14, 113378–113402. DOI 10.1109/ACCESS.2026.3714708. Published 17 Jul 2026, open access.** Read in full-text HTML on IEEE Xplore (Sections I–V and the Conclusion).

What they do:
- They study M1→M2 and M2→M1 threshold transfer on ESA-ADB, using the 77 channels both missions share. Mission3 is used only as a small extra check.
- The scorer is fixed and has no learned model: the mean of the top-3 robust |z| values (median/MAD). The study is about the settings layer, not the detector.
- They compare 4 strategies:
  1. Reuse the source mission's threshold as-is.
  2. Calibrate on a fixed early window of the target mission, without labels.
  3. **Rolling label-free target-side adaptation**: re-estimate the normalisation using only past target data.
  4. Oracle versions that use target labels.
- Their metrics are about operator workload: alerts per hour, fraction of time spent alarming (duty), event recall/precision, and time to detect. **No ESA-ADB corrected event-wise F0.5.**
- What they found:
  - Reusing the source threshold blows past the alert budget.
  - Label-free rolling adaptation helps M1→M2.
  - **M2→M1 stays infeasible.** This matches our Step 9g finding that M1 as a target is the hard direction.

Consequence for us:
- **"First cross-mission study on ESA-ADB" is FALSE.** We must cite this paper and position against it.
- Still unclaimed, as far as we know:
  - A learned shared model (AE) transferred across missions.
  - Results on the official ESA-ADB metric.
  - The model-vs-settings diagnosis (M1 model on M2: 0.860 vs 0.858).
  - Choosing settings by synthetic injection (v1–v3) and estimating the number of hidden events (N̂).
  - The level-shift explanation of the M1 collapse.
- Prior art for injection-based label-free selection: **Goswami et al., ICLR 2023, "Unsupervised Model Selection for Time-series Anomaly Detection"** (arXiv 2210.01078). It uses performance on injected synthetic anomalies as one of its proxies. We must cite it. Our v1 is an application of that idea.

Also found:
- **github.com/JSempereH/ESA-ADB-anomaly-detection** (not peer-reviewed). No cross-mission transfer. Reports simple baselines very high on M2 with label-tuned thresholds (GlobalSTD 0.984 on the paper split, from a summary, so unverified). It also reports **anomalies-only** scores, which are much lower than all-events scores. Reviewers will ask for anomalies-only numbers.

**More prior art (9 Nov):**
- Perini, Bürkner, Klami, ICML 2023, "Estimating the Contamination Factor's Distribution in Unsupervised Anomaly Detection" (arXiv 2210.10487). It estimates the share of anomalies in unlabelled data and sets the threshold from that estimate. This is prior art for our N̂ (hidden-event count). Cite it, and use it as a baseline for N̂.
- Jung et al., "Time Series Anomaly Detection with label-free Model Selection" (arXiv 2106.07473).
- Mission3 is on Zenodo: ESA-Mission3.zip, 3.73 GB, open access, in record 15237121. The ESA-ADB paper excluded it because its anomalies are few and trivial, with many gaps. Heydari used it only as a limited robustness check.
- IEEE SPACE 2027 (ieeespace.org): IEEE AESS, Singapore, 5–7 Jul 2027.
  - 3–6 pages, including references. Deadline 31 Jan 2027. Submissions cannot be updated once submitted.
  - Our fit: the special session "AI/ML for SPACE".
  - Accepted papers are considered for a special section of IEEE TAES.
