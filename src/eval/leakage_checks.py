"""Sanity checks for label leakage. Validation data only.

1. Shuffled labels: a model trained on permuted labels must fall to the base rate.
2. Single-column screen: no one column should predict the validation label almost perfectly.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.common import REPORTS, load_config, write_json
from src.data.snapshots import feature_columns, load_processed
from src.eval.metrics import pr_auc


def main() -> None:
    cfg = load_config()
    data = load_processed()
    nodes = data.nodes
    y = nodes["y"].to_numpy()
    split = nodes["split"].to_numpy()
    train = (split == "train") & (y != -1)
    val = (split == "val") & (y != -1)
    cols = feature_columns("all", cfg["features"]["local_count"], data.x.shape[1])
    x = data.x[:, cols]

    rng = np.random.default_rng(0)
    shuffled_results = []
    for _ in range(3):
        shuffled = rng.permutation(y[train])
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, max_features=0.3)
        model.fit(x[train], shuffled)
        shuffled_results.append(pr_auc(y[val], model.predict_proba(x[val])[:, 1]))

    single = []
    for position, column in enumerate(cols):
        values = x[val, position]
        best = max(pr_auc(y[val], values), pr_auc(y[val], -values))
        single.append({"feature": f"f{column:03d}", "val_pr_auc": best})
    single.sort(key=lambda row: row["val_pr_auc"], reverse=True)

    report = {
        "validation_base_rate": float(y[val].mean()),
        "shuffled_label_val_pr_auc": shuffled_results,
        "best_single_columns": single[:5],
    }
    write_json(REPORTS / "leakage_checks.json", report)
    print(report)


if __name__ == "__main__":
    main()
