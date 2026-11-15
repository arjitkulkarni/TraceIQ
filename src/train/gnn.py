"""Train neural models.

Protocol:
  1. Grid search per model on the directed "all" variant, seed 0, by validation PR-AUC.
  2. Repeat the chosen settings over every seed, feature set and edge mode.

Every run trains for exactly gnn.epochs epochs and records its validation accuracy, F1 and
PR-AUC per epoch. Runs and grid choices made with a different epoch count are redone.

Rollout: temporal models process steps 1..49 in order. Training updates use steps 1..30 and
state is detached every bptt_window snapshots. Later steps only advance state from features
and edges, which are available at prediction time. Validation and test labels never reach the
model. Test scores are saved but test metrics are not computed here.

Every epoch is printed as it finishes, for the grid search and the final runs, and the same
lines go to artifacts/reports/train_gnn.log so another terminal can follow along.

Usage: python -m src.train.gnn [--models graphsage evolvegcn_o] [--seeds 0 1] [--force]
  --force  retrain the chosen models even if saved runs and grid choices are current
"""
from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

from src.common import MODELS, REPORTS, load_config, read_json, set_seed, write_json
from src.data.snapshots import Processed, build_snapshots, load_processed
from src.eval.metrics import binary_metrics, select_threshold
from src.models.gnn import ALL_MODELS, TEMPORAL_MODELS, build_model
from src.train.record import run_is_current, run_name, save_run


LOG_PATH = REPORTS / "train_gnn.log"
_log_file = None


def log(message: str = "") -> None:
    """Print a line now, and copy it to the training log when one is open."""
    print(message, flush=True)
    if _log_file is not None:
        _log_file.write(message + "\n")
        _log_file.flush()


def epoch_line(epoch: int, epochs: int, row: dict, improved: bool, seconds: float) -> str:
    return (f"  epoch {epoch:>3}/{epochs}  loss {row['train_loss']:.4f}  val acc {row['val_accuracy']:.4f}  "
            f"f1 {row['val_f1']:.4f}  pr-auc {row['val_pr_auc']:.4f}  {seconds:5.1f}s{'  * best' if improved else ''}")


def compute_pos_weight(snapshots) -> float:
    positive = sum(int((s.y == 1).sum()) for s in snapshots if s.split == "train")
    negative = sum(int((s.y == 0).sum()) for s in snapshots if s.split == "train")
    return negative / positive


def masked_loss(logits, snapshot, pos_weight: torch.Tensor) -> torch.Tensor:
    """Summed loss over known labels. Unknown nodes pass messages but carry no loss."""
    mask = snapshot.known_mask
    return F.binary_cross_entropy_with_logits(
        logits[mask], snapshot.y[mask].float(), pos_weight=pos_weight, reduction="sum"
    )


def train_epoch_static(model, train_snaps, optimizer, pos_weight, total_known, rng) -> float:
    model.train()
    running = 0.0
    for i in rng.permutation(len(train_snaps)):
        snapshot = train_snaps[i]
        optimizer.zero_grad()
        loss = masked_loss(model(snapshot.x, snapshot.edge_index), snapshot, pos_weight)
        (loss * len(train_snaps) / total_known).backward()
        optimizer.step()
        running += loss.item()
    return running / total_known


def train_epoch_temporal(model, train_snaps, optimizer, pos_weight, total_known, window: int) -> float:
    """Ordered snapshots with truncated backpropagation through time."""
    model.train()
    state = model.initial_state()
    running = 0.0
    optimizer.zero_grad()
    accumulated = torch.zeros((), device=train_snaps[0].x.device)
    for t, snapshot in enumerate(train_snaps, start=1):
        h, state = model.step(snapshot.x, snapshot.edge_index, state)
        accumulated = accumulated + masked_loss(model.head(h), snapshot, pos_weight)
        if t % window == 0 or t == len(train_snaps):
            (accumulated * len(train_snaps) / window / total_known).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()
            running += accumulated.item()
            accumulated = torch.zeros((), device=snapshot.x.device)
            state = model.detach_state(state)
    return running / total_known


@torch.no_grad()
def score_all(model, snapshots, n_nodes: int, temporal: bool) -> np.ndarray:
    model.eval()
    scores = np.zeros(n_nodes, dtype=np.float64)
    state = model.initial_state() if temporal else None
    for snapshot in snapshots:
        if temporal:
            h, state = model.step(snapshot.x, snapshot.edge_index, state)
            logits = model.head(h)
        else:
            logits = model(snapshot.x, snapshot.edge_index)
        scores[snapshot.node_ids.cpu().numpy()] = torch.sigmoid(logits).double().cpu().numpy()
    return scores


def validation_row(epoch: int, loss: float, y: np.ndarray, score: np.ndarray) -> dict:
    """One epoch of the training curve. Validation labels only, thresholded by the same max-F1 rule."""
    metrics = binary_metrics(y, score, select_threshold(y, score), budgets=())
    return {"epoch": epoch, "train_loss": loss, "val_accuracy": metrics["accuracy"],
            "val_f1": metrics["f1"], "val_pr_auc": metrics["pr_auc"]}


def fit(
    name: str, snapshots, nodes, params: dict, seed: int, cfg: dict, device, label: str | None = None
) -> tuple[dict, dict]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    temporal = name in TEMPORAL_MODELS
    gnn_cfg = cfg["gnn"]
    model = build_model(name, snapshots[0].x.size(1), params["hidden"], params["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=gnn_cfg["weight_decay"])
    train_snaps = [s for s in snapshots if s.split == "train"]
    total_known = sum(int(s.known_mask.sum()) for s in train_snaps)
    pos_weight = torch.tensor(compute_pos_weight(snapshots), device=device)

    y = nodes["y"].to_numpy()
    val_mask = (nodes["split"].to_numpy() == "val") & (y != -1)
    best = {"ap": -1.0, "epoch": 0, "state": None}
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(1, gnn_cfg["epochs"] + 1):
        if temporal:
            loss = train_epoch_temporal(model, train_snaps, optimizer, pos_weight, total_known, gnn_cfg["bptt_window"])
        else:
            loss = train_epoch_static(model, train_snaps, optimizer, pos_weight, total_known, rng)
        scores = score_all(model, snapshots, len(nodes), temporal)
        row = validation_row(epoch, loss, y[val_mask], scores[val_mask])
        history.append(row)
        improved = row["val_pr_auc"] > best["ap"]
        if improved:
            best = {"ap": row["val_pr_auc"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if label:
            log(epoch_line(epoch, gnn_cfg["epochs"], row, improved, time.perf_counter() - started))
    model.load_state_dict(best["state"])
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    final = score_all(model, snapshots, len(nodes), temporal)
    extras = {
        "epochs": gnn_cfg["epochs"],
        "best_epoch": best["epoch"],
        "history": history,
        "fit_seconds": fit_seconds,
        "inference_seconds_all_nodes": time.perf_counter() - started,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
    }
    return {"scores": final, "state": best["state"], "val_ap": best["ap"]}, extras


def to_device(snapshots, device):
    return [s.to(device) for s in snapshots]


def grid_search(name: str, data: Processed, cfg: dict, device, force: bool = False) -> dict:
    path = REPORTS / "gnn_selection.json"
    selected = read_json(path) if path.exists() else {}
    if not force and selected.get(name, {}).get("epochs") == cfg["gnn"]["epochs"]:
        log(f"{name}: reusing grid choice {selected[name]['params']}")
        return selected[name]["params"]
    snapshots = to_device(build_snapshots(data, "all", "directed"), device)
    grid = cfg["gnn"]["grid"]
    combos = list(product(grid["hidden"], grid["lr"], grid["dropout"]))
    results = []
    for index, (hidden, lr, dropout) in enumerate(combos, start=1):
        params = {"hidden": hidden, "lr": lr, "dropout": dropout}
        log(f"\n[grid {name} {index}/{len(combos)}] hidden {hidden}, lr {lr}, dropout {dropout}")
        fitted, extras = fit(name, snapshots, data.nodes, params, 0, cfg, device, label="grid")
        results.append({"params": params, "val_pr_auc": fitted["val_ap"], "best_epoch": extras["best_epoch"]})
        log(f"  -> best val pr-auc {fitted['val_ap']:.4f} at epoch {extras['best_epoch']}")
    selected = read_json(path) if path.exists() else {}
    selected[name] = {**max(results, key=lambda r: r["val_pr_auc"]), "epochs": cfg["gnn"]["epochs"], "grid": results}
    write_json(path, selected)
    log(f"{name}: chose {selected[name]['params']}")
    return selected[name]["params"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(ALL_MODELS))
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--force", action="store_true", help="retrain even if saved runs are current")
    args = parser.parse_args()

    cfg = load_config()
    data = load_processed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    epochs = cfg["gnn"]["epochs"]

    def variants(name):
        edge_modes = ["none"] if name == "mlp" else cfg["gnn"]["edge_modes"]
        return list(product(cfg["features"]["sets"], edge_modes))

    planned = [run_name(name, f, e, seed) for name in args.models for f, e in variants(name) for seed in seeds]
    todo = [run for run in planned if args.force or not run_is_current(run, epochs)]
    if not todo:
        log(f"All {len(planned)} runs are already trained for {epochs} epochs. Use --force to retrain.")
        return

    global _log_file
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(LOG_PATH, "w", encoding="utf-8")
    log(f"training started {datetime.now():%Y-%m-%d %H:%M:%S} on {device}, {epochs} epochs per run")
    log("columns: training loss, then validation accuracy, F1 and PR-AUC; '* best' marks the kept weights")
    log(f"{len(todo)} of {len(planned)} runs to train; the rest are already trained for {epochs} epochs")
    done = 0

    for name in args.models:
        if not any(run.startswith(f"{name}__") for run in todo):
            continue
        params = grid_search(name, data, cfg, device, force=args.force)
        for feature_set, edge_mode in variants(name):
            snapshots = to_device(
                build_snapshots(data, feature_set, "directed" if edge_mode == "none" else edge_mode), device
            )
            for seed in seeds:
                run = run_name(name, feature_set, edge_mode, seed)
                if run not in todo:
                    continue
                done += 1
                log(f"\n[run {done}/{len(todo)}] {run}  (hidden {params['hidden']}, lr {params['lr']}, "
                    f"dropout {params['dropout']})")
                fitted, extras = fit(name, snapshots, data.nodes, params, seed, cfg, device, label=run)
                record = save_run(data.nodes, fitted["scores"], name, feature_set, edge_mode, seed, params, extras)
                torch.save(
                    {"state_dict": fitted["state"], "params": params, "in_dim": snapshots[0].x.size(1),
                     "feature_set": feature_set, "edge_mode": edge_mode, "model": name},
                    MODELS / f"{run}.pt",
                )
                memory = f", peak GPU {extras['peak_gpu_mb']:.0f} MB" if extras["peak_gpu_mb"] else ""
                log(f"  -> kept epoch {extras['best_epoch']}: val accuracy {record['val_metrics']['accuracy']:.4f}, "
                    f"pr-auc {record['val_metrics']['pr_auc']:.4f}, {extras['fit_seconds']:.0f}s{memory}")
    log(f"\ngnn done {datetime.now():%Y-%m-%d %H:%M:%S}")
    if todo:
        log("Refresh the comparison and dashboard with: python -m src.pipeline --from compare")
    _log_file.close()
    _log_file = None


if __name__ == "__main__":
    main()
