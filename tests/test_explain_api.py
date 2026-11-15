"""Explanation wrapper and API tests. API tests run against a small purpose-built database."""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from torch_geometric.nn import MessagePassing

from src.api import main as api
from src.api.build_index import risk_tier
from src.explain.frozen import FrozenGCN, explainable_model, temporal_weights
from src.models.gnn import build_model
from tests.test_models_eval import graph


@pytest.mark.parametrize("name", ["evolvegcn_o", "evolvegcn_h"])
def test_frozen_temporal_model_equals_deployed_model(name):
    snapshots = [graph(), graph()]
    for index, snapshot in enumerate(snapshots):
        snapshot.time_step = index + 1
    model = build_model(name, 8, 16, 0.5).eval()
    state = model.initial_state()
    outputs = []
    with torch.no_grad():
        for snapshot in snapshots:
            h, state = model.step(snapshot.x, snapshot.edge_index, state)
            outputs.append(model.head(h))
        frames = explainable_model(model, snapshots)
        for snapshot, expected in zip(snapshots, outputs):
            assert torch.allclose(frames(snapshot.time_step)(snapshot.x, snapshot.edge_index), expected, atol=1e-5)


def test_frozen_static_gcn_equals_deployed_model():
    data = graph()
    model = build_model("static_gcn", 8, 16, 0.5).eval()
    with torch.no_grad():
        frozen = explainable_model(model, [data])(1)
        assert torch.allclose(frozen(data.x, data.edge_index), model(data.x, data.edge_index), atol=1e-5)


def test_frozen_model_exposes_message_passing_for_edge_masks():
    data = graph()
    data.time_step = 1
    model = build_model("evolvegcn_o", 8, 16, 0.0)
    frozen = FrozenGCN(model.input, temporal_weights(model, [data])[1], model.head)
    assert any(isinstance(module, MessagePassing) for module in frozen.modules())


def test_risk_tiers():
    assert risk_tier(pd.Series([0.9, 0.3, 0.1]).to_numpy(), 0.8, 0.5).tolist() == ["high", "low", "low"]
    assert risk_tier(pd.Series([0.45]).to_numpy(), 0.8, 0.5).tolist() == ["medium"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Chain 1 -> 2 -> 3 -> 4 plus a hub 5 sending to many nodes, all in one step."""
    path = tmp_path_factory.mktemp("db") / "test.db"
    connection = sqlite3.connect(path)
    n = 100
    rows = pd.DataFrame({
        "node_index": range(n),
        "tx_id": [str(1000 + i) for i in range(n)],
        "time_step": [41] * n,
        "split": ["test"] * n,
        "dataset_label": (["illicit", "licit", "unknown", "unknown"] * n)[:n],
        "model_score": [1 - i / n for i in range(n)],
        "risk_tier": ["high"] * 10 + ["low"] * (n - 10),
        "tabular_score": [0.5] * n,
        "anomaly_percentile": [0.5] * n,
        "in_degree": [1] * n,
        "out_degree": [1] * n,
        "explanation_available": [1] + [0] * (n - 1),
    })
    rows.to_sql("transactions", connection, index=False)
    edges = [(0, 1), (1, 2), (2, 3)] + [(4, i) for i in range(5, n)]
    pd.DataFrame(edges, columns=["src", "dst"]).to_sql("edges", connection, index=False)
    serving = {"model_name": "graphsage", "run": "unit", "threshold": 0.9, "score_type": "uncalibrated_sigmoid"}
    explanation = {"edges": [{"source": "1000", "target": "1001", "importance": 0.9}], "features": [],
                   "model_name": "graphsage", "explained_model": "deployed"}
    connection.execute("CREATE TABLE explanations (tx_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    connection.execute("INSERT INTO explanations VALUES (?, ?)", ("1000", json.dumps(explanation)))
    connection.execute("INSERT INTO meta VALUES (?, ?)", ("serving", json.dumps(serving)))
    connection.commit()
    connection.close()
    original = api.DB_PATH
    api.DB_PATH = path
    yield TestClient(api.app)
    api.DB_PATH = original


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_lookup_returns_stored_values(client):
    body = client.get("/transactions/1000").json()
    assert body["model_score"] == 1.0
    assert body["dataset_label"] == "illicit"
    assert body["time_step"] == 41
    assert body["score_type"] == "uncalibrated_sigmoid"


def test_missing_transaction_is_404_everywhere(client):
    for suffix in ("", "/graph", "/explanation"):
        assert client.get(f"/transactions/999999{suffix}").status_code == 404


def test_unknown_is_never_reported_as_safe(client):
    labels = {item["dataset_label"] for item in client.get("/transactions?page_size=10").json()["items"]}
    assert "unknown" in labels


def test_list_is_sorted_and_paginated(client):
    first = client.get("/transactions?sort=risk&page=1&page_size=5").json()
    second = client.get("/transactions?sort=risk&page=2&page_size=5").json()
    scores = [i["model_score"] for i in first["items"] + second["items"]]
    assert scores == sorted(scores, reverse=True)
    assert first["total"] == 100 and len(first["items"]) == 5


def test_filters(client):
    body = client.get("/transactions?tier=high&label=illicit").json()
    assert body["total"] > 0
    assert all(i["risk_tier"] == "high" and i["dataset_label"] == "illicit" for i in body["items"])
    assert client.get("/transactions?tier=nonsense").status_code == 422


def test_page_size_is_capped(client):
    assert client.get("/transactions?page_size=100000").json()["page_size"] == api.cfg["max_page_size"]


def test_graph_keeps_direction_and_only_real_edges(client):
    body = client.get("/transactions/1001/graph?hops=1").json()
    assert {(e["source"], e["target"]) for e in body["edges"]} == {("1000", "1001"), ("1001", "1002")}
    assert next(n for n in body["nodes"] if n["is_focal"])["transaction_id"] == "1001"


def test_graph_is_bounded_on_a_hub(client):
    body = client.get("/transactions/1004/graph?hops=2&max_nodes=100000").json()
    assert len(body["nodes"]) <= api.cfg["graph_max_nodes"]
    assert body["truncated"] is True
    ids = {n["transaction_id"] for n in body["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in body["edges"])


def test_hops_are_capped(client):
    assert client.get("/transactions/1000/graph?hops=50").json()["hops"] == api.cfg["graph_max_hops"]


def test_explanation_states_model_and_absence(client):
    present = client.get("/transactions/1000/explanation").json()
    assert present["available"] is True and present["explained_model"] == "deployed"
    absent = client.get("/transactions/1050/explanation").json()
    assert absent["available"] is False
    assert "does not imply" in present["disclaimer"]


def test_edge_importance_attaches_to_the_right_edge(client):
    edges = client.get("/transactions/1000/graph").json()["edges"]
    assert next(e for e in edges if e["target"] == "1001")["importance"] == 0.9


def test_api_is_read_only(client):
    assert client.post("/transactions").status_code == 405
