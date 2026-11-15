"""Read the three Elliptic source files. No cleaning and no silent row drops."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import RAW_DIR

FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"

N_RAW_FEATURES = 165  # columns after txId and time_step
FEATURE_COLUMNS = [f"f{i:03d}" for i in range(N_RAW_FEATURES)]
LABEL_MAP = {"1": "illicit", "2": "licit", "unknown": "unknown"}


def read_features(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """The features file has no header: txId, time_step, then 165 anonymous columns."""
    columns = ["tx_id", "time_step", *FEATURE_COLUMNS]
    dtype = {"tx_id": "int64", "time_step": "int16", **{c: "float64" for c in FEATURE_COLUMNS}}
    return pd.read_csv(raw_dir / FEATURES_FILE, header=None, names=columns, dtype=dtype)


def read_classes(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / CLASSES_FILE, dtype={"txId": "int64", "class": "string"})
    df = df.rename(columns={"txId": "tx_id", "class": "raw_class"})
    unexpected = set(df["raw_class"].unique()) - set(LABEL_MAP)
    if unexpected:
        raise ValueError(f"unexpected class values: {unexpected}")
    df["label_state"] = df["raw_class"].map(LABEL_MAP).astype("string")
    return df


def read_edges(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Directed fund flow: source txId1 -> destination txId2."""
    df = pd.read_csv(raw_dir / EDGES_FILE, dtype={"txId1": "int64", "txId2": "int64"})
    return df.rename(columns={"txId1": "src_tx_id", "txId2": "dst_tx_id"})
