"""EVT threshold (Peaks-Over-Threshold, as in SPOT, Siffer et al. 2017).

Idea: the extreme upper tail of NORMAL-ish scores follows a Generalised Pareto Distribution (GPD).
  1. u  = high quantile (level p0) of the calibration scores  (the "peak" cutoff)
  2. excesses  y_i = s_i - u  for s_i > u   -> fit GPD(shape xi, scale sigma)
  3. threshold for tail probability q:

        z_q = u + (sigma / xi) * ( (q * n / N_u)^(-xi) - 1 )          (xi != 0)
        z_q = u - sigma * ln( q * n / N_u )                          (xi == 0)

     n = number of calibration scores, N_u = number of excesses.  z_q is the score level that a
     calibration-like score exceeds with probability q.
Calibration scores = scores on the TRAIN period of the SAME channel/mission. Never test scores (leakage guard).
"""
import numpy as np
from scipy.stats import genpareto


def evt_threshold(scores: np.ndarray, q: float = 1e-4, p0: float = 0.98) -> float:
    s = scores[np.isfinite(scores)]
    n = len(s)
    u = np.quantile(s, p0)
    y = s[s > u] - u
    if len(y) < 30 or y.max() <= 0:                       # degenerate (e.g. constant scores): fall back to max
        return float(s.max())
    xi, _, sigma = genpareto.fit(y, floc=0)
    r = q * n / len(y)
    if abs(xi) < 1e-6:
        return float(u - sigma * np.log(r))
    return float(u + sigma / xi * (r ** (-xi) - 1))
