"""Stage C and D: canonical tables, ID map, chronological split and train-fitted processor.

Outputs in data/processed/:
  nodes.parquet          node_index, tx_id, time_step, label_state, y, split
  features_raw.npy       [N, 165] float64 source features (time_step removed)
  features_scaled.npy    [N, 165] float32 train-fitted, clipped features
  edges.npy              [E, 2] directed node indices (src, dst)
  feature_processor.joblib
"""
from __future__ import annotations

import hashlib

import joblib
import numpy as np
import pandas as pd

from src.common import PROCESSED_DIR, REPORTS, ensure_dirs, load_config, write_json
from src.data.load import FEATURE_COLUMNS, read_classes, read_edges, read_features
from src.data.split import SPLITS, assign_split
from src.features.preprocess import FeatureProcessor, between_step_variance_share

LABEL_TO_Y = {"licit": 0, "illicit": 1, "unknown": -1}


def build_nodes(features: pd.DataFrame, classes: pd.DataFrame, split_cfg: dict) -> pd.DataFrame:
    nodes = features[["tx_id", "time_step"]].copy()
    nodes.insert(0, "node_index", np.arange(len(nodes), dtype=np.int64))
    nodes = nodes.merge(classes[["tx_id", "label_state"]], on="tx_id", how="left", validate="one_to_one")
    if nodes["label_state"].isna().any():
        raise ValueError("every node must have exactly one label state")
    nodes["label_state"] = nodes["label_state"].astype(str)
    nodes["y"] = nodes["label_state"].map(LABEL_TO_Y).astype(np.int8)
    nodes["split"] = assign_split(nodes["time_step"], split_cfg)
    return nodes


def map_edges(edges: pd.DataFrame, nodes: pd.DataFrame) -> np.ndarray:
    """Map transaction IDs to tensor indices. Refuses to drop an edge silently."""
    lookup = pd.Series(nodes["node_index"].to_numpy(), index=nodes["tx_id"].to_numpy())
    src = edges["src_tx_id"].map(lookup)
    dst = edges["dst_tx_id"].map(lookup)
    if src.isna().any() or dst.isna().any():
        raise ValueError("edge endpoint missing from node table")
    return np.stack([src.to_numpy(np.int64), dst.to_numpy(np.int64)], axis=1)


def split_summary(nodes: pd.DataFrame) -> dict:
    out = {}
    for name in SPLITS:
        part = nodes[nodes["split"] == name]
        illicit = int((part["y"] == 1).sum())
        licit = int((part["y"] == 0).sum())
        out[name] = {
            "steps": [int(part["time_step"].min()), int(part["time_step"].max())],
            "nodes": len(part),
            "illicit": illicit,
            "licit": licit,
            "unknown": int((part["y"] == -1).sum()),
            "illicit_share_of_known": illicit / (illicit + licit),
        }
    return out


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    features, classes, edges = read_features(), read_classes(), read_edges()

    nodes = build_nodes(features, classes, cfg["split"])
    edge_array = map_edges(edges, nodes)
    x_raw = features[FEATURE_COLUMNS].to_numpy(np.float64)

    # Fit on labeled training rows only. Time partitions are saved before any learned step.
    fit_mask = (nodes["split"].to_numpy() == "train") & (nodes["y"].to_numpy() != -1)
    processor = FeatureProcessor.fit(x_raw[fit_mask], cfg["features"]["clip_value"])
    x_scaled = processor.transform(x_raw)

    # Time-proxy detection uses training-period rows only and needs no labels.
    train_rows = nodes["split"].to_numpy() == "train"
    eta2 = between_step_variance_share(x_raw[train_rows], nodes["time_step"].to_numpy()[train_rows])
    limit = cfg["features"]["time_proxy_eta2_threshold"]
    proxies = np.flatnonzero(eta2 > limit)
    write_json(
        PROCESSED_DIR / "time_proxy_columns.json",
        {
            "threshold": limit,
            "columns": proxies.tolist(),
            "names": [FEATURE_COLUMNS[i] for i in proxies],
            "eta2": {FEATURE_COLUMNS[i]: float(eta2[i]) for i in proxies},
            "highest_retained_eta2": float(np.delete(eta2, proxies).max()),
        },
    )

    nodes.to_parquet(PROCESSED_DIR / "nodes.parquet", index=False)
    np.save(PROCESSED_DIR / "features_raw.npy", x_raw)
    np.save(PROCESSED_DIR / "features_scaled.npy", x_scaled)
    np.save(PROCESSED_DIR / "edges.npy", edge_array)
    joblib.dump(processor, PROCESSED_DIR / "feature_processor.joblib")

    summary = {
        "split": cfg["split"],
        "partitions": split_summary(nodes),
        "processor_fit_rows": int(fit_mask.sum()),
        "constant_columns_in_fit_rows": [
            c for c, flag in zip(FEATURE_COLUMNS, processor.constant_columns_) if flag
        ],
        "clipped_value_share": float((np.abs(x_scaled) >= cfg["features"]["clip_value"]).mean()),
        "split_hash": hashlib.sha256(nodes["split"].to_numpy().astype("U5").tobytes()).hexdigest(),
    }
    write_json(REPORTS / "split_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
