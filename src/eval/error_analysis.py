"""Error analysis on the frozen serving model, in the held-out period.

Describes graph statistics only. Anonymized data cannot reveal the motive behind a transfer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import PREDICTIONS, PROCESSED_DIR, REPORTS, load_config, read_json, write_json

CASES_PER_TYPE = 10


def neighbor_counts(edges: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Known illicit and known licit direct neighbors, in either direction."""
    illicit = np.zeros(n, dtype=np.int64)
    licit = np.zeros(n, dtype=np.int64)
    for a, b in ((0, 1), (1, 0)):
        np.add.at(illicit, edges[:, a], (y[edges[:, b]] == 1).astype(int))
        np.add.at(licit, edges[:, a], (y[edges[:, b]] == 0).astype(int))
    return illicit, licit


def main() -> None:
    selection = read_json(REPORTS / "frozen_selection.json")
    run_name = selection["serving_run"]
    run = read_json(REPORTS / "runs" / f"{run_name}.json")
    threshold = run["threshold"]
    nodes = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    edges = np.load(PROCESSED_DIR / "edges.npy")
    score = pd.read_parquet(PREDICTIONS / f"{run_name}.parquet").sort_values("node_index")["score"].to_numpy()

    y = nodes["y"].to_numpy()
    n = len(nodes)
    known_illicit, known_licit = neighbor_counts(edges, y, n)
    frame = nodes[["node_index", "tx_id", "time_step", "label_state", "y", "split"]].copy()
    frame["score"] = score
    frame["in_degree"] = np.bincount(edges[:, 1], minlength=n)
    frame["out_degree"] = np.bincount(edges[:, 0], minlength=n)
    frame["illicit_neighbors"] = known_illicit
    frame["licit_neighbors"] = known_licit

    test = frame[(frame["split"] == "test") & (frame["y"] != -1)].copy()
    flagged = test["score"] >= threshold
    test["outcome"] = np.select(
        [flagged & (test["y"] == 1), flagged & (test["y"] == 0), ~flagged & (test["y"] == 1)],
        ["TP", "FP", "FN"], default="TN",
    )

    summary = (
        test.groupby("outcome")
        .agg(count=("score", "size"), mean_score=("score", "mean"), median_time_step=("time_step", "median"),
             mean_in_degree=("in_degree", "mean"), mean_out_degree=("out_degree", "mean"),
             share_with_illicit_neighbor=("illicit_neighbors", lambda s: float((s > 0).mean())),
             share_after_step_42=("time_step", lambda s: float((s >= 43).mean())))
        .reset_index()
    )
    orderings = {"TP": False, "FP": False, "FN": False}
    cases = []
    for outcome, ascending in orderings.items():
        part = test[test["outcome"] == outcome].sort_values("score", ascending=ascending).head(CASES_PER_TYPE)
        for row in part.to_dict("records"):
            cases.append({**row, "tx_id": str(row["tx_id"])})

    write_json(REPORTS / "error_analysis.json", {
        "run": run_name, "threshold": threshold,
        "summary": summary.to_dict("records"), "cases": cases,
    })
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    load_config()
    main()
