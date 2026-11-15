"""Load processed arrays and build one PyG Data object per time step."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.common import PROCESSED_DIR, load_config, read_json


@dataclass
class Processed:
    nodes: pd.DataFrame
    x: np.ndarray
    edges: np.ndarray


def load_processed() -> Processed:
    return Processed(
        nodes=pd.read_parquet(PROCESSED_DIR / "nodes.parquet"),
        x=np.load(PROCESSED_DIR / "features_scaled.npy"),
        edges=np.load(PROCESSED_DIR / "edges.npy"),
    )


def feature_columns(feature_set: str, local_count: int, total: int) -> np.ndarray:
    """Column indices for a feature set. Time-proxy columns are excluded unless asked for."""
    proxies = read_json(PROCESSED_DIR / "time_proxy_columns.json")["columns"]
    if feature_set == "all_with_time_proxies":
        return np.arange(total)
    if feature_set == "all":
        return np.setdiff1d(np.arange(total), proxies)
    if feature_set == "local":
        return np.setdiff1d(np.arange(local_count), proxies)
    raise ValueError(f"unknown feature set: {feature_set}")


def build_snapshots(
    data: Processed, feature_set: str = "all", edge_mode: str = "directed"
) -> list[Data]:
    """One Data per step, in ascending order. No edge ever crosses a step.

    edge_index follows fund flow, so directed message passing moves information from a
    source transaction to the transaction that received its funds. "symmetrized" adds reverse
    edges for learning only and is never a claim that money moved both ways.
    """
    if edge_mode not in ("directed", "symmetrized"):
        raise ValueError(f"unknown edge mode: {edge_mode}")
    cfg = load_config()
    cols = feature_columns(feature_set, cfg["features"]["local_count"], data.x.shape[1])
    nodes = data.nodes
    steps = nodes["time_step"].to_numpy()
    y_all = nodes["y"].to_numpy()
    split = nodes["split"].to_numpy()
    edge_step = steps[data.edges[:, 0]]

    local_index = np.empty(len(nodes), dtype=np.int64)
    snapshots: list[Data] = []
    for step in np.unique(steps):
        node_ids = np.flatnonzero(steps == step)
        local_index[node_ids] = np.arange(len(node_ids))
        step_edges = data.edges[edge_step == step]
        if (steps[step_edges] != step).any():
            raise ValueError("edge crosses a time step")
        edge_index = torch.as_tensor(local_index[step_edges].T.copy(), dtype=torch.long)
        if edge_mode == "symmetrized":
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        y = y_all[node_ids]
        known = y != -1
        partition = split[node_ids][0]
        snapshots.append(
            Data(
                x=torch.as_tensor(data.x[node_ids][:, cols]),
                edge_index=edge_index,
                y=torch.as_tensor(y.astype(np.int64)),
                known_mask=torch.as_tensor(known),
                node_ids=torch.as_tensor(node_ids),
                time_step=int(step),
                split=str(partition),
            )
        )
    return snapshots
