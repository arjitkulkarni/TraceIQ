"""Tabular baselines. They see only the features of the transaction being scored."""
from __future__ import annotations

from itertools import product

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def param_grid(name: str, cfg: dict) -> list[dict]:
    """Expand list-valued settings into a grid; scalar settings are fixed."""
    spec = cfg["baselines"][name]
    keys = [k for k, v in spec.items() if isinstance(v, list)]
    fixed = {k: v for k, v in spec.items() if not isinstance(v, list)}
    return [{**fixed, **dict(zip(keys, values))} for values in product(*(spec[k] for k in keys))]


def make_model(name: str, params: dict, seed: int, pos_weight: float, device: str):
    if name == "logistic":
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed, **params)
    if name == "xgboost":
        return XGBClassifier(
            tree_method="hist", device=device, scale_pos_weight=pos_weight, eval_metric="aucpr",
            random_state=seed, n_jobs=-1, verbosity=0, **params,
        )
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=seed, **params)
    raise ValueError(f"unknown baseline: {name}")
