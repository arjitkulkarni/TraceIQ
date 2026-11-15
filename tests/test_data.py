"""Data, graph and split tests. Unit tests use tiny tables; integrity tests use processed data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common import PROCESSED_DIR, load_config, read_json
from src.data.build_graph import build_nodes, map_edges
from src.data.load import LABEL_MAP
from src.data.snapshots import build_snapshots, feature_columns, load_processed
from src.data.split import assign_split, validate_split
from src.features.preprocess import FeatureProcessor, between_step_variance_share

SPLIT = {"train_steps": [1, 30], "val_steps": [31, 39], "test_steps": [40, 49]}
needs_processed = pytest.mark.skipif(
    not (PROCESSED_DIR / "nodes.parquet").exists(), reason="run python -m src.data.build_graph first"
)


def tiny_tables():
    features = pd.DataFrame({"tx_id": [900, 7, 55, 12], "time_step": [1, 1, 31, 40]})
    classes = pd.DataFrame(
        {"tx_id": [7, 900, 12, 55], "label_state": ["illicit", "licit", "unknown", "licit"]}
    )
    return features, classes


def test_label_map_matches_dataset_documentation():
    assert LABEL_MAP == {"1": "illicit", "2": "licit", "unknown": "unknown"}


def test_labels_align_with_ids_when_files_have_different_order():
    nodes = build_nodes(*tiny_tables(), SPLIT)
    assert nodes.set_index("tx_id")["y"].to_dict() == {900: 0, 7: 1, 55: 0, 12: -1}
    assert nodes["node_index"].tolist() == [0, 1, 2, 3]
    assert nodes["split"].tolist() == ["train", "train", "val", "test"]


def test_edges_are_remapped_and_keep_direction():
    nodes = build_nodes(*tiny_tables(), SPLIT)
    edges = pd.DataFrame({"src_tx_id": [7], "dst_tx_id": [900]})
    assert map_edges(edges, nodes).tolist() == [[1, 0]]


def test_unknown_edge_endpoint_is_an_error_not_a_silent_drop():
    nodes = build_nodes(*tiny_tables(), SPLIT)
    with pytest.raises(ValueError):
        map_edges(pd.DataFrame({"src_tx_id": [7], "dst_tx_id": [123456]}), nodes)


def test_node_without_label_state_is_an_error():
    features, classes = tiny_tables()
    with pytest.raises(ValueError):
        build_nodes(features, classes.iloc[:3], SPLIT)


@pytest.mark.parametrize(
    "bad",
    [
        {"train_steps": [1, 30], "val_steps": [30, 39], "test_steps": [40, 49]},  # overlap
        {"train_steps": [1, 30], "val_steps": [32, 39], "test_steps": [40, 49]},  # gap
        {"train_steps": [40, 49], "val_steps": [31, 39], "test_steps": [1, 30]},  # reversed
        {"train_steps": [1, 30], "val_steps": [31, 39], "test_steps": [40, 48]},  # incomplete
    ],
)
def test_invalid_splits_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_split(bad)


def test_split_boundaries():
    assert assign_split(np.array([1, 30, 31, 39, 40, 49]), SPLIT).tolist() == [
        "train", "train", "val", "val", "test", "test",
    ]


def test_processor_uses_fit_rows_only():
    train = np.array([[0.0], [2.0]])
    processor = FeatureProcessor.fit(train, clip_value=100.0)
    assert processor.mean_[0] == 1.0
    assert processor.n_fit_rows == 2
    # A later, very different row must not change the learned parameters.
    assert processor.transform(np.array([[1001.0]]))[0, 0] == pytest.approx(100.0)
    assert processor.mean_[0] == 1.0


def test_processor_handles_constant_column_and_rejects_nan():
    processor = FeatureProcessor.fit(np.array([[1.0, 0.0], [1.0, 2.0]]), clip_value=10.0)
    assert processor.transform(np.array([[5.0, 1.0]]))[0, 0] == 0.0
    with pytest.raises(ValueError):
        FeatureProcessor.fit(np.array([[np.nan]]), clip_value=10.0)


def test_eta_squared_flags_a_time_encoding_column():
    steps = np.repeat([1, 2, 3, 4], 50)
    rng = np.random.default_rng(0)
    x = np.stack([steps + rng.normal(0, 0.01, 200), rng.normal(0, 1, 200)], axis=1)
    eta2 = between_step_variance_share(x, steps)
    assert eta2[0] > 0.99
    assert eta2[1] < 0.1


@pytest.fixture(scope="module")
def data():
    return load_processed()


@pytest.fixture(scope="module")
def snapshots(data):
    return build_snapshots(data, "all", "directed")


@needs_processed
class TestProcessedData:
    def test_published_counts(self, data):
        counts = data.nodes["label_state"].value_counts().to_dict()
        assert len(data.nodes) == 203769 and len(data.edges) == 234355
        assert counts == {"unknown": 157205, "licit": 42019, "illicit": 4545}

    def test_unique_ids_and_contiguous_indices(self, data):
        assert data.nodes["tx_id"].is_unique
        assert data.nodes["node_index"].tolist() == list(range(len(data.nodes)))

    def test_label_encoding_is_consistent(self, data):
        pairs = data.nodes.groupby("label_state")["y"].unique().to_dict()
        assert {k: v.tolist() for k, v in pairs.items()} == {"illicit": [1], "licit": [0], "unknown": [-1]}

    def test_time_steps_and_split_are_ordered(self, data):
        assert data.nodes["time_step"].between(1, 49).all()
        ranges = data.nodes.groupby("split")["time_step"].agg(["min", "max"])
        assert ranges.loc["train", "max"] < ranges.loc["val", "min"]
        assert ranges.loc["val", "max"] < ranges.loc["test", "min"]

    def test_edge_indices_valid_and_within_a_step(self, data):
        assert data.edges.min() >= 0 and data.edges.max() < len(data.nodes)
        steps = data.nodes["time_step"].to_numpy()
        assert (steps[data.edges[:, 0]] == steps[data.edges[:, 1]]).all()

    def test_snapshot_count_order_and_totals(self, snapshots):
        assert [s.time_step for s in snapshots] == list(range(1, 50))
        assert sum(s.num_nodes for s in snapshots) == 203769
        assert sum(s.num_edges for s in snapshots) == 234355

    def test_snapshot_shapes_and_alignment(self, data, snapshots):
        for snapshot in snapshots[::7]:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.max() < snapshot.num_nodes
            assert snapshot.x.shape[0] == snapshot.y.shape[0] == snapshot.num_nodes
            ids = snapshot.node_ids.numpy()
            assert (data.nodes["y"].to_numpy()[ids] == snapshot.y.numpy()).all()
            cols = feature_columns("all", 93, data.x.shape[1])
            assert np.array_equal(snapshot.x.numpy(), data.x[ids][:, cols])

    def test_snapshot_edges_match_source_edges_exactly(self, data, snapshots):
        snapshot = snapshots[10]
        ids = snapshot.node_ids.numpy()
        rebuilt = set(map(tuple, ids[snapshot.edge_index.numpy().T].tolist()))
        step = snapshot.time_step
        source = data.edges[data.nodes["time_step"].to_numpy()[data.edges[:, 0]] == step]
        assert rebuilt == set(map(tuple, source.tolist()))

    def test_symmetrized_only_adds_reversed_edges(self, data, snapshots):
        symmetrized = build_snapshots(data, "all", "symmetrized")[10]
        original = snapshots[10].edge_index
        assert symmetrized.num_edges == 2 * original.size(1)
        assert (symmetrized.edge_index[:, original.size(1):] == original.flip(0)).all()

    def test_known_mask_never_contains_unknown(self, snapshots):
        for snapshot in snapshots:
            assert (snapshot.y[snapshot.known_mask] != -1).all()
            assert (snapshot.y[~snapshot.known_mask] == -1).all()

    def test_processor_fitted_only_on_labeled_training_rows(self, data):
        import joblib

        processor = joblib.load(PROCESSED_DIR / "feature_processor.joblib")
        rows = (data.nodes["split"] == "train") & (data.nodes["y"] != -1)
        assert processor.n_fit_rows == int(rows.sum())
        raw = np.load(PROCESSED_DIR / "features_raw.npy")
        assert np.allclose(processor.mean_, raw[rows.to_numpy()].mean(axis=0))

    def test_time_is_not_a_model_input(self, data):
        proxies = read_json(PROCESSED_DIR / "time_proxy_columns.json")["columns"]
        cfg = load_config()
        all_cols = feature_columns("all", cfg["features"]["local_count"], data.x.shape[1])
        local_cols = feature_columns("local", cfg["features"]["local_count"], data.x.shape[1])
        assert set(proxies).isdisjoint(all_cols)
        assert len(all_cols) == 165 - len(proxies)
        assert local_cols.max() < cfg["features"]["local_count"]
