"""Train-only feature preprocessing.

The processor is fitted on labeled training rows only (guide 8.4, 10.2) and applied unchanged
to every other row. time_step is never a feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def between_step_variance_share(x: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Eta squared per column: share of variance explained by the time step alone.

    A value near 1 means the column almost encodes the time step, which is a forbidden input
    because test steps lie outside the training range.
    """
    total = ((x - x.mean(axis=0)) ** 2).sum(axis=0)
    grand = x.mean(axis=0)
    between = np.zeros(x.shape[1])
    for step in np.unique(steps):
        part = x[steps == step]
        between += len(part) * (part.mean(axis=0) - grand) ** 2
    return between / np.where(total == 0, 1.0, total)


@dataclass
class FeatureProcessor:
    mean_: np.ndarray
    std_: np.ndarray
    constant_columns_: np.ndarray
    clip_value: float
    n_fit_rows: int

    @classmethod
    def fit(cls, x: np.ndarray, clip_value: float) -> "FeatureProcessor":
        if not np.isfinite(x).all():
            raise ValueError("non-finite values in training features")
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        constant = std == 0
        std = np.where(constant, 1.0, std)
        return cls(mean, std, constant, clip_value, len(x))

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean_) / self.std_
        z[:, self.constant_columns_] = 0.0
        return np.clip(z, -self.clip_value, self.clip_value).astype(np.float32)
