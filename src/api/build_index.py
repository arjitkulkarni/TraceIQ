"""Build the SQLite index the API serves. Every value comes from a saved artifact.

Run after src.eval.compare and src.explain.run.
"""
from __future__ import annotations

import csv
import json
import sqlite3

import numpy as np
import pandas as pd

from src.common import ARTIFACTS, PREDICTIONS, PROCESSED_DIR, REPORTS, load_config, read_json

DB_PATH = ARTIFACTS / "traceiq.db"


def risk_tier(score: np.ndarray, threshold: float, medium_fraction: float) -> np.ndarray:
    return np.where(score >= threshold, "high", np.where(score >= threshold * medium_fraction, "medium", "low"))


def read_csv(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    cfg = load_config()
    selection = read_json(REPORTS / "frozen_selection.json")
    serving = selection["serving_run"]
    family = selection["serving_model_family"]
    run = read_json(REPORTS / "runs" / f"{serving}.json")
    threshold = run["threshold"]
    medium_fraction = cfg["risk_tiers"]["medium_fraction_of_threshold"]

    nodes = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    edges = np.load(PROCESSED_DIR / "edges.npy")
    score = pd.read_parquet(PREDICTIONS / f"{serving}.parquet").sort_values("node_index")["score"].to_numpy()
    tabular_run = selection["families"]["xgboost"]["median_seed_run"]
    tabular = pd.read_parquet(PREDICTIONS / f"{tabular_run}.parquet").sort_values("node_index")["score"].to_numpy()
    anomaly = pd.read_parquet(PREDICTIONS / "isolation_forest.parquet").sort_values("node_index")
    in_degree = np.bincount(edges[:, 1], minlength=len(nodes))
    out_degree = np.bincount(edges[:, 0], minlength=len(nodes))

    explanations_path = REPORTS / "explanations.json"
    explanations = read_json(explanations_path) if explanations_path.exists() else {}

    table = pd.DataFrame(
        {
            "node_index": nodes["node_index"].astype(int),
            "tx_id": nodes["tx_id"].astype(str),
            "time_step": nodes["time_step"].astype(int),
            "split": nodes["split"],
            "dataset_label": nodes["label_state"],
            "model_score": score.astype(float),
            "risk_tier": risk_tier(score, threshold, medium_fraction),
            "tabular_score": tabular.astype(float),
            "anomaly_percentile": anomaly["anomaly_percentile"].to_numpy().astype(float),
            "in_degree": in_degree.astype(int),
            "out_degree": out_degree.astype(int),
            "explanation_available": nodes["tx_id"].astype(str).isin(explanations).astype(int),
        }
    )

    meta = {
        "serving": {
            "model_name": family,
            "run": serving,
            "feature_set": run["feature_set"],
            "edge_mode": run["edge_mode"],
            "seed": run["seed"],
            "threshold": threshold,
            "medium_cutoff": threshold * medium_fraction,
            "score_type": "uncalibrated_sigmoid",
            "tabular_comparison_run": tabular_run,
        },
        "selection": selection,
        "split": read_json(REPORTS / "split_summary.json"),
        "comparison": read_csv(REPORTS / "final_comparison.csv"),
        "ablations": read_csv(REPORTS / "ablations.csv"),
        "training": {"epochs": cfg["gnn"]["epochs"]},
        "epoch_curves": read_csv(REPORTS / "epoch_curves.csv"),
        "per_step": read_csv(REPORTS / "per_step_test.csv"),
        "periods": read_csv(REPORTS / "test_periods.csv"),
        "significance": read_json(REPORTS / "significance.json"),
        "confusion": read_json(REPORTS / "confusion_selected.json"),
        "leakage_checks": read_json(REPORTS / "leakage_checks.json"),
        "isolation_forest": read_json(REPORTS / "isolation_forest.json"),
        "time_proxies": read_json(PROCESSED_DIR / "time_proxy_columns.json"),
        "eda": read_json(REPORTS / "eda.json"),
        "explanation_summary": read_json(REPORTS / "explanation_summary.json")
        if (REPORTS / "explanation_summary.json").exists() else None,
    }

    DB_PATH.unlink(missing_ok=True)
    connection = sqlite3.connect(DB_PATH)
    table.to_sql("transactions", connection, index=False)
    pd.DataFrame(edges, columns=["src", "dst"]).to_sql("edges", connection, index=False)
    for statement in (
        "CREATE UNIQUE INDEX idx_tx ON transactions(tx_id)",
        "CREATE UNIQUE INDEX idx_node ON transactions(node_index)",
        "CREATE INDEX idx_rank ON transactions(model_score DESC)",
        "CREATE INDEX idx_filter ON transactions(split, risk_tier, dataset_label, time_step)",
        "CREATE INDEX idx_src ON edges(src)",
        "CREATE INDEX idx_dst ON edges(dst)",
        "CREATE TABLE explanations (tx_id TEXT PRIMARY KEY, payload TEXT NOT NULL)",
        "CREATE TABLE meta (key TEXT PRIMARY KEY, payload TEXT NOT NULL)",
    ):
        connection.execute(statement)
    connection.executemany(
        "INSERT INTO explanations VALUES (?, ?)", [(k, json.dumps(v)) for k, v in explanations.items()]
    )
    connection.executemany("INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
    connection.commit()
    connection.close()
    print(f"wrote {DB_PATH} with {len(table)} transactions and {len(explanations)} explanations")


if __name__ == "__main__":
    main()
