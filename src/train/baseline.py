"""Train tabular baselines. Grid search on validation PR-AUC, then repeat over seeds."""
from __future__ import annotations

import time

import joblib
import numpy as np
import torch

from src.common import MODELS, REPORTS, load_config, read_json, write_json
from src.data.snapshots import feature_columns, load_processed
from src.models.baseline import make_model, param_grid
from src.train.record import run_exists, run_name, save_run, validation_ap

MODEL_NAMES = ("logistic", "xgboost", "random_forest")
SEEDLESS = {"logistic"}  # deterministic solver: repeated seeds would give identical runs


def main() -> None:
    cfg = load_config()
    data = load_processed()
    nodes = data.nodes
    y = nodes["y"].to_numpy()
    train = (nodes["split"].to_numpy() == "train") & (y != -1)
    pos_weight = float((y[train] == 0).sum() / (y[train] == 1).sum())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_path = REPORTS / "baseline_selection.json"
    selected = read_json(selected_path) if selected_path.exists() else {}

    for name in MODEL_NAMES:
        for feature_set in cfg["features"]["sets"]:
            cols = feature_columns(feature_set, cfg["features"]["local_count"], data.x.shape[1])
            x = data.x[:, cols]
            key = f"{name}__{feature_set}"
            if key not in selected:
                scored = []
                for params in param_grid(name, cfg):
                    model = make_model(name, params, 0, pos_weight, device).fit(x[train], y[train])
                    ap = validation_ap(nodes, model.predict_proba(x)[:, 1])
                    scored.append({"params": params, "val_pr_auc": ap})
                    print(key, params, round(ap, 4), flush=True)
                selected[key] = max(scored, key=lambda r: r["val_pr_auc"])
                write_json(selected_path, selected)
            best = selected[key]["params"]
            seeds = cfg["seeds"][:1] if name in SEEDLESS else cfg["seeds"]
            for seed in seeds:
                if run_exists(run_name(name, feature_set, "none", seed)):
                    continue
                started = time.perf_counter()
                model = make_model(name, best, seed, pos_weight, device).fit(x[train], y[train])
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                score = model.predict_proba(x)[:, 1]
                infer_seconds = time.perf_counter() - started
                record = save_run(
                    nodes, score, name, feature_set, "none", seed, best,
                    {"fit_seconds": fit_seconds, "inference_seconds_all_nodes": infer_seconds},
                )
                joblib.dump(model, MODELS / f"{record['name']}.joblib")
                print(record["name"], "val", round(record["val_metrics"]["pr_auc"], 4), flush=True)

    # Ablation: the same XGBoost settings, with the time-proxy columns left in.
    best = selected["xgboost__all"]["params"]
    x = data.x
    for seed in cfg["seeds"]:
        if run_exists(run_name("xgboost", "all_with_time_proxies", "none", seed)):
            continue
        model = make_model("xgboost", best, seed, pos_weight, device).fit(x[train], y[train])
        save_run(nodes, model.predict_proba(x)[:, 1], "xgboost", "all_with_time_proxies", "none", seed, best)
    print("baselines done", flush=True)


if __name__ == "__main__":
    main()
