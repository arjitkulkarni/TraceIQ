"""Save one run: scores for every node, plus validation metrics and metadata.

Test labels are deliberately not read here. Test metrics exist only in src.eval.compare,
after every selection decision has been frozen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import PREDICTIONS, REPORTS, environment_info, load_config, read_json, write_json
from src.eval.metrics import binary_metrics, pr_auc, select_threshold

RUNS_DIR = REPORTS / "runs"


def run_name(model: str, feature_set: str, edge_mode: str, seed: int) -> str:
    return f"{model}__{feature_set}__{edge_mode}__seed{seed}"


def run_exists(name: str) -> bool:
    return (RUNS_DIR / f"{name}.json").exists() and (PREDICTIONS / f"{name}.parquet").exists()


def run_is_current(name: str, epochs: int) -> bool:
    """A saved neural run counts only if it was trained for this many epochs."""
    return run_exists(name) and read_json(RUNS_DIR / f"{name}.json").get("epochs") == epochs


def validation_ap(nodes: pd.DataFrame, score: np.ndarray) -> float:
    mask = (nodes["split"].to_numpy() == "val") & (nodes["y"].to_numpy() != -1)
    return pr_auc(nodes["y"].to_numpy()[mask], score[mask])


def save_run(
    nodes: pd.DataFrame,
    score: np.ndarray,
    model: str,
    feature_set: str,
    edge_mode: str,
    seed: int,
    params: dict,
    extras: dict | None = None,
) -> dict:
    if len(score) != len(nodes) or not np.isfinite(score).all():
        raise ValueError("expected one finite score per node")
    name = run_name(model, feature_set, edge_mode, seed)
    y = nodes["y"].to_numpy()
    val = (nodes["split"].to_numpy() == "val") & (y != -1)
    threshold = select_threshold(y[val], score[val])
    budgets = tuple(load_config()["threshold"]["review_budgets"])
    record = {
        "name": name,
        "model": model,
        "feature_set": feature_set,
        "edge_mode": edge_mode,
        "seed": seed,
        "params": params,
        "threshold": threshold,
        "val_metrics": binary_metrics(y[val], score[val], threshold, budgets),
        "environment": environment_info(),
        **(extras or {}),
    }
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_index": nodes["node_index"].to_numpy(), "score": score.astype(np.float32)}).to_parquet(
        PREDICTIONS / f"{name}.parquet", index=False
    )
    write_json(RUNS_DIR / f"{name}.json", record)
    return record


def load_runs() -> list[dict]:
    return [read_json(p) for p in sorted(RUNS_DIR.glob("*.json"))]
