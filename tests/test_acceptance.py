"""Prototype acceptance test (guide 17.5, 22.2), against the real index and source artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.build_index import DB_PATH
from src.api.main import app
from src.common import PREDICTIONS, PROCESSED_DIR, REPORTS, load_config, read_json

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="run python -m src.api.build_index first")


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def source():
    selection = read_json(REPORTS / "frozen_selection.json")
    run = selection["serving_run"]
    nodes = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    scores = pd.read_parquet(PREDICTIONS / f"{run}.parquet").sort_values("node_index")["score"].to_numpy()
    edges = np.load(PROCESSED_DIR / "edges.npy")
    return {
        "selection": selection,
        "run": read_json(REPORTS / "runs" / f"{run}.json"),
        "nodes": nodes.set_index(nodes["tx_id"].astype(str)),
        "scores": scores,
        "edges": set(map(tuple, edges.tolist())),
    }


def sample_ids(client, n=25):
    return [i["transaction_id"] for i in client.get(f"/transactions?split=test&page_size={n}").json()["items"]]


def test_transaction_values_match_source_artifacts(client, source):
    for tx in sample_ids(client):
        body = client.get(f"/transactions/{tx}").json()
        node = source["nodes"].loc[tx]
        assert body["time_step"] == node["time_step"]
        assert body["dataset_label"] == node["label_state"]
        assert body["split"] == node["split"]
        assert body["model_score"] == pytest.approx(float(source["scores"][node["node_index"]]))
        assert body["run"] == source["selection"]["serving_run"]


def test_graph_edges_are_real_directed_edges_in_one_step(client, source):
    for tx in sample_ids(client, 10):
        for hops in (1, 2):
            graph = client.get(f"/transactions/{tx}/graph?hops={hops}").json()
            step = source["nodes"].loc[tx, "time_step"]
            for node in graph["nodes"]:
                assert node["time_step"] == step
            for edge in graph["edges"]:
                pair = (int(source["nodes"].loc[edge["source"], "node_index"]),
                        int(source["nodes"].loc[edge["target"], "node_index"]))
                assert pair in source["edges"]


def test_explanation_edges_are_real_and_name_the_deployed_model(client, source):
    explained = client.get("/transactions?explained=true&page_size=10").json()["items"]
    assert explained
    for item in explained:
        body = client.get(f"/transactions/{item['transaction_id']}/explanation").json()
        assert body["available"] is True
        assert body["explained_model"] == "deployed"
        assert body["run"] == source["selection"]["serving_run"]
        assert body["prediction_score"] == pytest.approx(item["model_score"])
        for edge in body["edges"]:
            pair = (int(source["nodes"].loc[edge["source"], "node_index"]),
                    int(source["nodes"].loc[edge["target"], "node_index"]))
            assert pair in source["edges"]


def test_tier_follows_the_frozen_threshold(client, source):
    threshold = source["run"]["threshold"]
    for item in client.get("/transactions?split=test&page_size=200").json()["items"]:
        assert (item["risk_tier"] == "high") == (item["model_score"] >= threshold)


def test_unknown_labels_are_shown_as_unknown(client):
    body = client.get("/transactions?label=unknown&split=test&page_size=5").json()
    assert body["total"] == 35463
    assert all(i["dataset_label"] == "unknown" for i in body["items"])


def test_test_metrics_exclude_unknown_labels(client):
    confusion = client.get("/metrics").json()["confusion"][0]
    assert confusion["known_nodes"] == 11184
    assert confusion["tp"] + confusion["fn"] == 636
    assert confusion["tp"] + confusion["fp"] + confusion["fn"] + confusion["tn"] == 11184


def test_temporal_runs_are_stateful_and_comparable(source):
    selection = source["selection"]["families"]
    assert selection["evolvegcn_o"]["runs"] and selection["evolvegcn_h"]["runs"]
    runs = [read_json(REPORTS / "runs" / f"{name}.json") for family in selection.values() for name in family["runs"]]
    assert {r["val_metrics"]["known_nodes"] for r in runs} == {8475}


def test_neural_models_trained_for_the_configured_epochs(client):
    body = client.get("/metrics").json()
    epochs = body["training"]["epochs"]
    assert epochs == load_config()["gnn"]["epochs"]
    for row in body["comparison"]:
        if row["family"] in ("mlp", "graphsage", "static_gcn", "evolvegcn_o", "evolvegcn_h"):
            assert float(row["epochs"]) == epochs
    curves = body["epoch_curves"]
    assert {int(row["epoch"]) for row in curves} == set(range(1, epochs + 1))
    assert {row["model"] for row in curves} == {"mlp", "graphsage", "static_gcn", "evolvegcn_o", "evolvegcn_h"}
