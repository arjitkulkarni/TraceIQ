"""Chronological split assignment. Ranges must be ordered, contiguous and non-overlapping."""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def validate_split(split_cfg: dict, min_step: int = 1, max_step: int = 49) -> None:
    train = split_cfg["train_steps"]
    val = split_cfg["val_steps"]
    test = split_cfg["test_steps"]
    if train[0] != min_step or test[1] != max_step:
        raise ValueError("split must cover every time step")
    if not (train[0] <= train[1] < val[0] <= val[1] < test[0] <= test[1]):
        raise ValueError("split ranges must be ascending and non-overlapping")
    if val[0] != train[1] + 1 or test[0] != val[1] + 1:
        raise ValueError("split ranges must be contiguous")


def assign_split(time_steps: pd.Series | np.ndarray, split_cfg: dict) -> np.ndarray:
    validate_split(split_cfg)
    steps = np.asarray(time_steps)
    out = np.full(steps.shape, "", dtype=object)
    for name, key in zip(SPLITS, ("train_steps", "val_steps", "test_steps")):
        lo, hi = split_cfg[key]
        out[(steps >= lo) & (steps <= hi)] = name
    if (out == "").any():
        raise ValueError("found a time step outside the declared split")
    return out
