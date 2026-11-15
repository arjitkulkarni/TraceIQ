"""Evaluator tests against hand-computed values, and model smoke tests on a small graph."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from src.eval.metrics import binary_metrics, confusion, pr_auc, recall_at_k, select_threshold
from src.models.gnn import ALL_MODELS, TEMPORAL_MODELS, build_model, normalized_adjacency
from src.eval.compare import epoch_curves, summarize, training_summary
from src.train import record
from src.train.gnn import fit, masked_loss, score_all, validation_row


def guide_worked_example():
    """Guide 2.5: 100 labeled, 10 illicit, 12 flagged, 8 correct."""
    y = np.array([1] * 10 + [0] * 90)
    score = np.zeros(100)
    score[:8] = 0.9    # eight true positives
    score[10:14] = 0.8  # four false positives
    return y, score


def test_metrics_match_guide_worked_example():
    y, score = guide_worked_example()
    m = binary_metrics(y, score, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (8, 4, 2, 86)
    assert m["precision"] == pytest.approx(8 / 12)
    assert m["recall"] == pytest.approx(0.8)
    assert m["f1"] == pytest.approx(0.727, abs=0.001)
    assert m["accuracy"] == pytest.approx(0.94)
    assert m["alerts"] == 12


def test_metrics_refuse_unknown_labels():
    with pytest.raises(ValueError):
        binary_metrics(np.array([1, 0, -1]), np.array([0.1, 0.2, 0.3]), 0.5)


def test_confusion_and_empty_alerts_do_not_divide_by_zero():
    assert confusion([1, 0], [False, False]) == {"tp": 0, "fp": 0, "fn": 1, "tn": 1}
    m = binary_metrics(np.array([1, 0]), np.array([0.1, 0.2]), 0.9)
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_pr_auc_hand_values():
    assert pr_auc(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(1.0)
    # Ranks: positive, negative, positive -> AP = (1/1 + 2/3) / 2
    assert pr_auc(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7])) == pytest.approx(5 / 6)


def test_recall_at_k():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.1, 0.9, 0.8, 0.2])
    assert recall_at_k(y, score, 2) == pytest.approx(0.5)


def test_select_threshold_separates_a_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    threshold = select_threshold(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < threshold <= 0.8


def graph(n=12):
    torch.manual_seed(0)
    src = torch.arange(n - 1)
    return Data(
        x=torch.randn(n, 8),
        edge_index=torch.stack([src, src + 1]),
        y=torch.tensor([1, 0, -1, -1] * (n // 4)),
        known_mask=torch.tensor([True, True, False, False] * (n // 4)),
        node_ids=torch.arange(n),
    )


@pytest.mark.parametrize("name", ALL_MODELS)
def test_forward_gives_one_finite_score_per_node(name):
    data = graph()
    model = build_model(name, 8, 16, 0.0).eval()
    with torch.no_grad():
        if name in TEMPORAL_MODELS:
            h, _ = model.step(data.x, data.edge_index, model.initial_state())
            out = model.head(h)
        else:
            out = model(data.x, data.edge_index)
    assert out.shape == (12,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["evolvegcn_o", "evolvegcn_h"])
def test_temporal_state_changes_and_stays_detachable(name):
    data = graph()
    model = build_model(name, 8, 16, 0.0)
    state = model.initial_state()
    _, next_state = model.step(data.x, data.edge_index, state)
    assert not torch.equal(state[0][0], next_state[0][0])
    assert next_state[0][0].requires_grad
    assert not model.detach_state(next_state)[0][0].requires_grad


@pytest.mark.parametrize("name", ["evolvegcn_o", "evolvegcn_h"])
def test_temporal_output_depends_on_earlier_snapshots(name):
    first, other = graph(), graph()
    other.x = other.x + 1.0
    target = graph()
    model = build_model(name, 8, 16, 0.0).eval()
    with torch.no_grad():
        _, state_a = model.step(first.x, first.edge_index, model.initial_state())
        _, state_b = model.step(other.x, other.edge_index, model.initial_state())
        out_a, _ = model.step(target.x, target.edge_index, state_a)
        out_b, _ = model.step(target.x, target.edge_index, state_b)
    if name == "evolvegcn_h":
        assert not torch.allclose(out_a, out_b)
    else:
        # EvolveGCN-O state ignores node features by design, so identical shape means identical state.
        assert torch.allclose(out_a, out_b)


def test_static_output_ignores_snapshot_history():
    data = graph()
    model = build_model("graphsage", 8, 16, 0.0).eval()
    with torch.no_grad():
        assert torch.equal(model(data.x, data.edge_index), model(data.x, data.edge_index))


def test_directed_message_passing_flows_source_to_destination():
    data = graph(4)
    model = build_model("static_gcn", 8, 16, 0.0).eval()
    changed = data.x.clone()
    changed[3] += 5.0  # node 3 is the last destination: it sends to nobody
    with torch.no_grad():
        a = model(data.x, data.edge_index)
        b = model(changed, data.edge_index)
    assert torch.allclose(a[:3], b[:3])
    assert not torch.allclose(a[3], b[3])


def test_adjacency_entry_points_from_source_to_destination():
    edge_index = torch.tensor([[0], [1]])
    dense = normalized_adjacency(edge_index, 2).to_dense()
    assert dense[1, 0] != 0   # row 1 receives from column 0
    assert dense[0, 1] == 0


def test_loss_ignores_unknown_nodes():
    data = graph()
    logits = torch.zeros(12, requires_grad=True)
    masked_loss(logits, data, torch.tensor(1.0)).backward()
    assert (logits.grad[~data.known_mask] == 0).all()
    assert (logits.grad[data.known_mask] != 0).all()


def test_changing_an_unknown_label_value_does_not_change_loss():
    data = graph()
    logits = torch.ones(12)
    before = masked_loss(logits, data, torch.tensor(1.0))
    data.y = data.y.clone()
    data.y[~data.known_mask] = 1
    assert masked_loss(logits, data, torch.tensor(1.0)) == before


@pytest.mark.parametrize("name", ["graphsage", "evolvegcn_h"])
def test_score_all_fills_every_node_in_range(name):
    data = graph()
    model = build_model(name, 8, 16, 0.0)
    scores = score_all(model, [data], 12, name in TEMPORAL_MODELS)
    assert scores.shape == (12,)
    assert ((scores >= 0) & (scores <= 1)).all()


def two_period_graph():
    """Four separable transactions per snapshot: first snapshot trains, second validates."""
    x = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
    edges = torch.tensor([[0, 2], [1, 3]])
    y = torch.tensor([1, 1, 0, 0])
    known = torch.ones(4, dtype=torch.bool)
    train = Data(x=x, edge_index=edges, y=y, known_mask=known, node_ids=torch.arange(4))
    val = Data(x=x, edge_index=edges, y=y, known_mask=known, node_ids=torch.arange(4, 8))
    train.split, val.split = "train", "val"
    nodes = pd.DataFrame({"y": [1, 1, 0, 0] * 2, "split": ["train"] * 4 + ["val"] * 4})
    return [train, val], nodes


@pytest.mark.parametrize("name", ["mlp", "graphsage", "evolvegcn_o"])
def test_fit_trains_exactly_the_configured_epochs_without_stopping_early(name):
    snapshots, nodes = two_period_graph()
    cfg = {"gnn": {"epochs": 7, "weight_decay": 0.0, "bptt_window": 5}}
    params = {"hidden": 8, "lr": 0.01, "dropout": 0.0}
    _, extras = fit(name, snapshots, nodes, params, 0, cfg, torch.device("cpu"))
    assert extras["epochs"] == 7
    assert [row["epoch"] for row in extras["history"]] == list(range(1, 8))
    assert 1 <= extras["best_epoch"] <= 7
    for row in extras["history"]:
        assert 0 <= row["val_accuracy"] <= 1 and np.isfinite(row["train_loss"])


def test_validation_row_reports_accuracy_at_the_max_f1_threshold():
    row = validation_row(3, 0.5, np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1]))
    assert row == {"epoch": 3, "train_loss": 0.5, "val_accuracy": 1.0, "val_f1": 1.0, "val_pr_auc": 1.0}


def test_run_from_a_different_epoch_count_is_not_current(tmp_path, monkeypatch):
    monkeypatch.setattr(record, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(record, "PREDICTIONS", tmp_path)
    (tmp_path / "r.parquet").write_bytes(b"")
    (tmp_path / "r.json").write_text('{"epochs": 150}', encoding="utf-8")
    assert record.run_is_current("r", 150)
    assert not record.run_is_current("r", 20)
    (tmp_path / "old.parquet").write_bytes(b"")
    (tmp_path / "old.json").write_text("{}", encoding="utf-8")
    assert not record.run_is_current("old", 20)


def test_summarize_reports_accuracy():
    m = binary_metrics(np.array([1, 0, 0, 0]), np.array([0.9, 0.1, 0.1, 0.8]), 0.5)
    assert summarize([m])["accuracy_mean"] == pytest.approx(0.75)


def test_epoch_curves_average_seeds_and_skip_tabular_runs():
    def point(epoch, accuracy):
        return {"epoch": epoch, "train_loss": 1.0, "val_accuracy": accuracy, "val_f1": 0.5, "val_pr_auc": 0.5}

    base = {"model": "mlp", "feature_set": "local", "edge_mode": "none"}
    runs = [{**base, "history": [point(1, 0.8)]}, {**base, "history": [point(1, 0.6)]},
            {"model": "xgboost", "feature_set": "all", "edge_mode": "none"}]
    curves = epoch_curves(runs)
    assert len(curves) == 1
    assert curves.iloc[0]["val_accuracy"] == pytest.approx(0.7)


def test_training_summary_leaves_tabular_models_without_epochs():
    records = {"a": {"epochs": 20, "best_epoch": 10}, "b": {"epochs": 20, "best_epoch": 20}, "c": {}}
    assert training_summary(records, ["a", "b"]) == {"epochs": 20, "best_epoch_mean": 15.0}
    assert training_summary(records, ["c"]) == {"epochs": None, "best_epoch_mean": None}
