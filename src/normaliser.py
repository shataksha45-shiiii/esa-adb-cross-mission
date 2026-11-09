"""Per-channel normaliser. FIT on the training period only (leakage guard), APPLY to any period.

Rules follow the ESA-ADB paper's standardisation section, with ONE substitution (flagged):
  paper: mean/std over NOMINAL training points (needs labels to know which points are nominal)
  ours : median / (1.4826*MAD) over ALL training points (robust to unlabelled anomalies; label-free)

kinds:
  binary    (<=2 unique train values)  -> min-max to [0, 1]
  constant  (std == 0)                 -> subtract median
  monotonic (non-decreasing or non-increasing) -> first difference, then robust standardise
  normal                               -> robust standardise
After transforming, values are clipped to +-CLIP so one huge spike cannot dominate training (our choice).
"""
from dataclasses import dataclass

import numpy as np

CLIP = 20.0


@dataclass
class ChannelNorm:
    kind: str
    center: float
    scale: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        if self.kind == "monotonic":
            x = np.concatenate([[0.0], np.diff(x)])
        z = (x - self.center) / self.scale
        return np.clip(z, -CLIP, CLIP).astype(np.float32)


def fit_channel(train: np.ndarray) -> ChannelNorm:
    x = train.astype(np.float64)
    uniq = np.unique(x)
    if len(uniq) <= 2:
        lo, hi = x.min(), x.max()
        return ChannelNorm("binary", lo, (hi - lo) if hi > lo else 1.0)
    d = np.diff(x)
    kind = "normal"
    if (d >= 0).all() or (d <= 0).all():
        kind, x = "monotonic", d
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    if mad > 0:
        return ChannelNorm(kind, med, mad)
    std = x.std()
    if std > 0:
        return ChannelNorm(kind, med, std)
    return ChannelNorm("constant", med, 1.0)


def fit_matrix(X_train: np.ndarray) -> list[ChannelNorm]:
    return [fit_channel(X_train[:, j]) for j in range(X_train.shape[1])]


def transform_matrix(X: np.ndarray, norms: list[ChannelNorm]) -> np.ndarray:
    return np.stack([n.transform(X[:, j]) for j, n in enumerate(norms)], axis=1)
