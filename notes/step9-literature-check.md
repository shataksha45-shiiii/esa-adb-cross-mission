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
