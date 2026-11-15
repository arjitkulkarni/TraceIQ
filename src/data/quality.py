"""Dataset quality report (guide 5.5 and 8.6). Reports problems, never repairs them silently."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import REPORTS, write_json
from src.data.load import FEATURE_COLUMNS, read_classes, read_edges, read_features


def build_report(features: pd.DataFrame, classes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    x = features[FEATURE_COLUMNS].to_numpy()
    ids = set(features["tx_id"])
    step_of = dict(zip(features["tx_id"], features["time_step"]))

    src_known = edges["src_tx_id"].isin(ids)
    dst_known = edges["dst_tx_id"].isin(ids)
    src_step = edges["src_tx_id"].map(step_of)
    dst_step = edges["dst_tx_id"].map(step_of)

    merged = features[["tx_id", "time_step"]].merge(classes, on="tx_id", how="outer", indicator=True)
    per_step = (
        merged.groupby(["time_step", "label_state"]).size().unstack(fill_value=0)
        .reindex(columns=["illicit", "licit", "unknown"], fill_value=0)
    )
    reciprocal = edges.merge(
        edges, left_on=["src_tx_id", "dst_tx_id"], right_on=["dst_tx_id", "src_tx_id"]
    )
    std = x.std(axis=0)
    return {
        "features": {
            "rows": len(features),
            "columns": features.shape[1],
            "duplicate_tx_ids": int(features["tx_id"].duplicated().sum()),
            "missing_values": int(np.isnan(x).sum()),
            "nonfinite_values": int((~np.isfinite(x)).sum()),
            "time_step_min": int(features["time_step"].min()),
            "time_step_max": int(features["time_step"].max()),
            "time_steps": int(features["time_step"].nunique()),
            "zero_variance_columns": [c for c, s in zip(FEATURE_COLUMNS, std) if s == 0],
            "global_mean_abs_max": float(np.abs(x.mean(axis=0)).max()),
            "global_std_min": float(std.min()),
            "global_std_max": float(std.max()),
            "abs_value_max": float(np.abs(x).max()),
        },
        "classes": {
            "rows": len(classes),
            "duplicate_tx_ids": int(classes["tx_id"].duplicated().sum()),
            "counts": {str(k): int(v) for k, v in classes["label_state"].value_counts().items()},
            "nodes_without_class": int((merged["_merge"] == "left_only").sum()),
            "classes_without_node": int((merged["_merge"] == "right_only").sum()),
        },
        "edges": {
            "rows": len(edges),
            "unknown_source_endpoints": int((~src_known).sum()),
            "unknown_destination_endpoints": int((~dst_known).sum()),
            "self_loops": int((edges["src_tx_id"] == edges["dst_tx_id"]).sum()),
            "duplicate_edges": int(edges.duplicated().sum()),
            "reciprocal_edge_rows": len(reciprocal),
            "cross_time_step_edges": int((src_step != dst_step).sum()),
        },
        "labels_per_step": {int(k): v for k, v in per_step.to_dict("index").items()},
    }


def main() -> None:
    report = build_report(read_features(), read_classes(), read_edges())
    write_json(REPORTS / "data_quality.json", report)
    for section in ("features", "classes", "edges"):
        print(section, report[section])
    print("step illicit licit unknown")
    for step, row in report["labels_per_step"].items():
        print(step, row["illicit"], row["licit"], row["unknown"])


if __name__ == "__main__":
    main()
