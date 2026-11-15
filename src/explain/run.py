"""Stage J: precompute GNNExplainer evidence for selected held-out transactions.

Every explanation targets the deployed serving model. Reported with each one:
  stability  How much the top edges and features agree across repeated explainer runs.
  fidelity   How the model's own logit changes when the chosen edges are removed or kept alone.
  occlusion  The real logit change from deleting one edge, and its rank agreement with GNNExplainer.

Logits are used because high scores saturate at 1.0, where a probability cannot show any change.
For the same reason GNNExplainer has little gradient on saturated predictions, so occlusion is the
ground truth that its importances are checked against.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from src.common import MODELS, PREDICTIONS, REPORTS, load_config, read_json, set_seed, write_json
from src.data.snapshots import build_snapshots, feature_columns, load_processed
from src.explain.frozen import explainable_model
from src.models.gnn import build_model

LIMITATIONS = [
    "GNNExplainer shows what this model used, not why a transfer happened.",
    "Importance is a local approximation and varies between runs, so the spread is shown.",
    "On saturated scores GNNExplainer has little gradient. Trust the measured logit drop over importance.",
    "Feature columns are anonymized. Their names carry no business meaning.",
]


def jaccard(sets: list[set]) -> float:
    pairs = list(combinations(sets, 2))
    if not pairs:
        return 1.0
    return float(np.mean([len(a & b) / max(len(a | b), 1) for a, b in pairs]))


def load_serving(selection: dict, data, device):
    run = selection["serving_run"]
    checkpoint = torch.load(MODELS / f"{run}.pt", map_location=device, weights_only=False)
    params = checkpoint["params"]
    model = build_model(checkpoint["model"], checkpoint["in_dim"], params["hidden"], params["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mode = "directed" if checkpoint["edge_mode"] == "none" else checkpoint["edge_mode"]
    snapshots = [s.to(device) for s in build_snapshots(data, checkpoint["feature_set"], mode)]
    return model, snapshots, checkpoint


def choose_transactions(nodes: pd.DataFrame, score: np.ndarray, top_k: int) -> list[int]:
    """Highest-scoring test transactions, plus every error-analysis case."""
    test = np.flatnonzero(nodes["split"].to_numpy() == "test")
    ranked = test[np.argsort(-score[test], kind="stable")[:top_k]]
    cases = read_json(REPORTS / "error_analysis.json")["cases"]
    chosen = list(dict.fromkeys([*ranked.tolist(), *[int(c["node_index"]) for c in cases]]))
    return chosen


@torch.no_grad()
def logit_with_edges(module, x, edge_index, keep: torch.Tensor, local: int) -> float:
    return float(module(x, edge_index[:, keep])[local].item())


def explain_one(module, snapshot, local: int, tx_of, columns, cfg, symmetrized: bool) -> dict | None:
    edge_index = snapshot.edge_index
    x = snapshot.x
    _, _, _, hop_edges = k_hop_subgraph(local, cfg["max_hops"], edge_index, num_nodes=x.size(0), flow="source_to_target")
    candidates = hop_edges.nonzero().view(-1)
    if len(candidates) == 0:
        return None

    edge_runs, feature_runs = [], []
    for repeat in range(cfg["repeats"]):
        set_seed(repeat)
        explainer = Explainer(
            model=module,
            algorithm=GNNExplainer(epochs=cfg["explainer_epochs"]).to(x.device),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config={"mode": "binary_classification", "task_level": "node", "return_type": "raw"},
        )
        result = explainer(x, edge_index, index=local)
        edge_runs.append(result.edge_mask[candidates].detach().cpu().numpy())
        feature_runs.append(result.node_mask[local].detach().cpu().numpy())
    edges = np.stack(edge_runs)
    features = np.stack(feature_runs)

    # Fund flow is reported in its true direction. Under symmetrized training a reverse copy
    # of an edge shares its importance with the real edge and is never shown as money flow.
    local_ids = snapshot.node_ids.cpu().numpy()
    original_count = edge_index.size(1) // 2 if symmetrized else edge_index.size(1)
    records: dict[tuple[str, str], dict] = {}
    for position, edge_position in enumerate(candidates.tolist()):
        real = edge_position if edge_position < original_count else edge_position - original_count
        source, target = edge_index[:, real].tolist()
        key = (tx_of[local_ids[source]], tx_of[local_ids[target]])
        values = edges[:, position]
        if key not in records or values.mean() > records[key]["importance"]:
            records[key] = {"source": key[0], "target": key[1], "importance": float(values.mean()),
                            "importance_std": float(values.std()), "position": real}
    mean_feature = features.mean(axis=0)
    std_feature = features.std(axis=0)
    order = np.argsort(-mean_feature)[: cfg["feature_importance_top"]]
    feature_list = [
        {"feature": f"f{columns[i]:03d}", "group": "local" if columns[i] < 93 else "aggregated",
         "importance": float(mean_feature[i]), "importance_std": float(std_feature[i])}
        for i in order
    ]

    def mask_removing(positions: torch.Tensor) -> torch.Tensor:
        keep = torch.ones(edge_index.size(1), dtype=torch.bool, device=x.device)
        keep[positions] = False
        if symmetrized:
            keep[positions + original_count] = False
        return keep

    everything = torch.ones(edge_index.size(1), dtype=torch.bool, device=x.device)
    logit_original = logit_with_edges(module, x, edge_index, everything, local)

    # Occlusion: delete each real edge in turn and measure the true effect on the logit.
    ranked = sorted(records.values(), key=lambda e: e["importance"], reverse=True)[: cfg["occlusion_max_edges"]]
    for edge in ranked:
        keep = mask_removing(torch.tensor([edge["position"]], device=x.device))
        edge["occlusion_logit_drop"] = logit_original - logit_with_edges(module, x, edge_index, keep, local)
    explainer_top = ranked[:5]
    edge_list = sorted(ranked, key=lambda e: abs(e["occlusion_logit_drop"]), reverse=True)[
        : cfg["edge_importance_top"]
    ]
    agreement = None
    if len(ranked) >= 3:
        explained = [e["importance"] for e in ranked]
        measured = [abs(e["occlusion_logit_drop"]) for e in ranked]
        if len(set(explained)) > 1 and len(set(measured)) > 1:
            agreement = float(spearmanr(explained, measured).statistic)

    top_positions = torch.tensor([e["position"] for e in explainer_top], device=x.device)
    remove = mask_removing(top_positions)
    keep_only = torch.ones(edge_index.size(1), dtype=torch.bool, device=x.device)
    keep_only[candidates] = False
    keep_only[top_positions] = True
    if symmetrized:
        keep_only[top_positions + original_count] = True
    logit_without = logit_with_edges(module, x, edge_index, remove, local)
    logit_only = logit_with_edges(module, x, edge_index, keep_only, local)
    no_edges = torch.zeros(edge_index.size(1), dtype=torch.bool, device=x.device)
    logit_no_edges = logit_with_edges(module, x, edge_index, no_edges, local)

    def top_edge_set(run: np.ndarray) -> set:
        return set(np.argsort(-run)[:5].tolist())

    def top_feature_set(run: np.ndarray) -> set:
        return set(np.argsort(-run)[:5].tolist())

    for edge in edge_list:
        edge.pop("position")
    return {
        "edges": edge_list,
        "features": feature_list,
        "stability": {
            "repeats": cfg["repeats"],
            "edge_top5_jaccard": jaccard([top_edge_set(run) for run in edges]),
            "feature_top5_jaccard": jaccard([top_feature_set(run) for run in features]),
        },
        "fidelity": {
            "logit_original": logit_original,
            "logit_without_top_edges": logit_without,
            "logit_only_top_edges": logit_only,
            "logit_without_any_edges": logit_no_edges,
        },
        "occlusion_spearman": agreement,
    }


def main() -> None:
    cfg = load_config()
    selection = read_json(REPORTS / "frozen_selection.json")
    data = load_processed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, snapshots, checkpoint = load_serving(selection, data, device)
    frames = explainable_model(model, snapshots)
    by_step = {s.time_step: s for s in snapshots}
    columns = feature_columns(checkpoint["feature_set"], cfg["features"]["local_count"], data.x.shape[1])

    nodes = data.nodes
    score = pd.read_parquet(PREDICTIONS / f"{selection['serving_run']}.parquet").sort_values("node_index")["score"].to_numpy()
    tx_of = nodes["tx_id"].astype(str).to_numpy()
    steps = nodes["time_step"].to_numpy()
    chosen = choose_transactions(nodes, score, cfg["explain"]["top_k_transactions"])

    explanations, skipped = {}, []
    for count, node in enumerate(chosen, start=1):
        step = int(steps[node])
        snapshot = by_step[step]
        local = int(torch.nonzero(snapshot.node_ids == node)[0, 0])
        evidence = explain_one(frames(step), snapshot, local, tx_of, columns, cfg["explain"],
                               checkpoint["edge_mode"] == "symmetrized")
        if evidence is None:
            skipped.append(tx_of[node])
            continue
        explanations[tx_of[node]] = {
            "tx_id": tx_of[node],
            "time_step": step,
            "model_name": selection["serving_model_family"],
            "run": selection["serving_run"],
            "explained_model": "deployed",
            "method": "GNNExplainer",
            "prediction_score": float(score[node]),
            **evidence,
            "limitations": LIMITATIONS,
        }
        print(count, len(chosen), tx_of[node], flush=True)
    write_json(REPORTS / "explanations.json", explanations)

    values = list(explanations.values())
    agreements = [v["occlusion_spearman"] for v in values if v["occlusion_spearman"] is not None]
    summary = {
        "explained": len(values),
        "skipped_no_incoming_context": len(skipped),
        "explained_model": "deployed",
        "mean_edge_top5_jaccard": float(np.mean([v["stability"]["edge_top5_jaccard"] for v in values])),
        "mean_feature_top5_jaccard": float(np.mean([v["stability"]["feature_top5_jaccard"] for v in values])),
        "mean_logit_drop_without_top_edges": float(np.mean(
            [v["fidelity"]["logit_original"] - v["fidelity"]["logit_without_top_edges"] for v in values])),
        "mean_logit_change_only_top_edges": float(np.mean(
            [v["fidelity"]["logit_only_top_edges"] - v["fidelity"]["logit_original"] for v in values])),
        "mean_logit_drop_without_any_edges": float(np.mean(
            [v["fidelity"]["logit_original"] - v["fidelity"]["logit_without_any_edges"] for v in values])),
        "occlusion_spearman_mean": float(np.mean(agreements)) if agreements else None,
        "occlusion_spearman_median": float(np.median(agreements)) if agreements else None,
        "occlusion_spearman_cases": len(agreements),
    }
    write_json(REPORTS / "explanation_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
