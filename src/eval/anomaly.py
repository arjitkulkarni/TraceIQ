"""Optional Isolation Forest overlay (guide 15.4).

Anomaly means unusual under this model, not illicit. It is fitted on training-period rows
without labels, reported separately, and never merged into the risk score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.common import PREDICTIONS, REPORTS, load_config, write_json
from src.data.snapshots import feature_columns, load_processed
from src.eval.metrics import pr_auc


def main() -> None:
    cfg = load_config()
    data = load_processed()
    nodes = data.nodes
    cols = feature_columns("all", cfg["features"]["local_count"], data.x.shape[1])
    x = data.x[:, cols]
    split = nodes["split"].to_numpy()
    y = nodes["y"].to_numpy()

    model = IsolationForest(n_estimators=300, random_state=0, n_jobs=-1)
    model.fit(x[split == "train"])
    anomaly = -model.score_samples(x)
    reference = np.sort(anomaly[split == "train"])
    percentile = np.searchsorted(reference, anomaly) / len(reference)

    pd.DataFrame({"node_index": nodes["node_index"].to_numpy(), "anomaly_percentile": percentile}).to_parquet(
        PREDICTIONS / "isolation_forest.parquet", index=False
    )
    report = {"note": "Unsupervised overlay. Not part of any risk score."}
    for name in ("val", "test"):
        mask = (split == name) & (y != -1)
        report[f"{name}_pr_auc"] = pr_auc(y[mask], anomaly[mask])
        report[f"{name}_base_rate"] = float(y[mask].mean())
    write_json(REPORTS / "isolation_forest.json", report)
    print(report)


if __name__ == "__main__":
    main()
