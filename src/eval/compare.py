"""Stage I: final evaluation.

Phase 1 uses validation only. It picks the best variant of each model family and writes
artifacts/reports/frozen_selection.json.
Phase 2 is the only place in the project that computes test metrics.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import precision_recall_curve

from src.common import FIGURES, PREDICTIONS, PROCESSED_DIR, REPORTS, load_config, read_json, write_json
from src.eval.metrics import binary_metrics, pr_auc
from src.eval.plot_style import INK_2, MODEL_COLORS, apply_style, save, shade_splits
from src.train.record import load_runs

EPOCH_METRICS = ("train_loss", "val_accuracy", "val_f1", "val_pr_auc")
FAMILIES = ("logistic", "random_forest", "xgboost", "mlp", "graphsage", "static_gcn", "evolvegcn_o", "evolvegcn_h")
DISPLAY = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "mlp": "MLP (no edges)",
    "graphsage": "GraphSAGE",
    "static_gcn": "Static GCN",
    "evolvegcn_o": "EvolveGCN-O",
    "evolvegcn_h": "EvolveGCN-H",
}
GRAPH_FAMILIES = ("graphsage", "static_gcn", "evolvegcn_o", "evolvegcn_h")
BOOTSTRAP_ROUNDS = 1000


def scores_for(run: str) -> np.ndarray:
    return pd.read_parquet(PREDICTIONS / f"{run}.parquet").sort_values("node_index")["score"].to_numpy()


def variants_table(runs: list[dict]) -> pd.DataFrame:
    rows = [
        {"model": r["model"], "feature_set": r["feature_set"], "edge_mode": r["edge_mode"], "seed": r["seed"],
         "name": r["name"], "threshold": r["threshold"], "val_pr_auc": r["val_metrics"]["pr_auc"],
         "val_f1": r["val_metrics"]["f1"]}
        for r in runs
    ]
    return pd.DataFrame(rows)


def freeze_selection(table: pd.DataFrame) -> dict:
    """Validation only. The time-proxy ablation variant can never be selected."""
    eligible = table[table["feature_set"] != "all_with_time_proxies"]
    means = (
        eligible.groupby(["model", "feature_set", "edge_mode"])["val_pr_auc"].mean().reset_index()
        .sort_values("val_pr_auc", ascending=False)
    )
    selection: dict = {"families": {}}
    for family in FAMILIES:
        best = means[means["model"] == family].iloc[0]
        members = eligible[
            (eligible["model"] == family)
            & (eligible["feature_set"] == best["feature_set"])
            & (eligible["edge_mode"] == best["edge_mode"])
        ].sort_values("val_pr_auc")
        selection["families"][family] = {
            "feature_set": best["feature_set"],
            "edge_mode": best["edge_mode"],
            "val_pr_auc_mean": float(best["val_pr_auc"]),
            "median_seed_run": members.iloc[len(members) // 2]["name"],
            "runs": members["name"].tolist(),
        }
    graph = {k: selection["families"][k] for k in GRAPH_FAMILIES}
    serving = max(graph, key=lambda k: graph[k]["val_pr_auc_mean"])
    selection["serving_model_family"] = serving
    selection["serving_run"] = graph[serving]["median_seed_run"]
    selection["best_overall_by_validation"] = max(
        selection["families"], key=lambda k: selection["families"][k]["val_pr_auc_mean"]
    )
    selection["rule"] = (
        "Per family: variant with the highest mean validation PR-AUC. Serving model: best graph family, "
        "median-validation seed. Threshold: per run, maximum validation F1."
    )
    return selection


def summarize(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    out: dict = {"runs": len(frame)}
    for column in ("accuracy", "precision", "recall", "f1", "pr_auc", "recall_at_100", "recall_at_500", "alerts"):
        out[f"{column}_mean"] = float(frame[column].mean())
        out[f"{column}_std"] = float(frame[column].std(ddof=1)) if len(frame) > 1 else 0.0
    return out


def epoch_curves(runs: list[dict]) -> pd.DataFrame:
    """Mean validation curve per model variant, across seeds. Tabular models have no epochs."""
    rows = [
        {"model": r["model"], "feature_set": r["feature_set"], "edge_mode": r["edge_mode"], **point}
        for r in runs
        for point in r.get("history", [])
    ]
    columns = ["model", "feature_set", "edge_mode", "epoch"]
    if not rows:
        return pd.DataFrame(columns=[*columns, *EPOCH_METRICS])
    return pd.DataFrame(rows).groupby(columns, as_index=False)[list(EPOCH_METRICS)].mean()


def training_summary(records: dict, names: list[str]) -> dict:
    """Epochs and best epoch for a neural family. Tabular models have neither."""
    trained = [records[n] for n in names if "epochs" in records[n]]
    if not trained:
        return {"epochs": None, "best_epoch_mean": None}
    return {"epochs": trained[0]["epochs"], "best_epoch_mean": float(np.mean([r["best_epoch"] for r in trained]))}


def test_rows(records: dict, names: list[str], y: np.ndarray, mask: np.ndarray, budgets) -> list[dict]:
    return [
        binary_metrics(y[mask], scores_for(n)[mask], records[n]["threshold"], budgets) for n in names
    ]


def bootstrap_difference(y, a_scores, b_scores) -> dict:
    """Paired bootstrap over test nodes of mean-seed PR-AUC(a) minus PR-AUC(b)."""
    rng = np.random.default_rng(0)
    n = len(y)
    observed = np.mean([pr_auc(y, s) for s in a_scores]) - np.mean([pr_auc(y, s) for s in b_scores])
    differences = []
    for _ in range(BOOTSTRAP_ROUNDS):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        differences.append(
            np.mean([pr_auc(y[idx], s[idx]) for s in a_scores])
            - np.mean([pr_auc(y[idx], s[idx]) for s in b_scores])
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": float(observed), "ci95_low": float(low), "ci95_high": float(high),
            "ci_excludes_zero": bool(low > 0 or high < 0)}


def per_step(selection, records, nodes, families) -> pd.DataFrame:
    y = nodes["y"].to_numpy()
    steps = nodes["time_step"].to_numpy()
    rows = []
    for family in families:
        for run in selection["families"][family]["runs"]:
            score = scores_for(run)
            threshold = records[run]["threshold"]
            for step in range(40, 50):
                mask = (steps == step) & (y != -1)
                m = binary_metrics(y[mask], score[mask], threshold, budgets=())
                rows.append({"family": family, "run": run, "time_step": step, "recall": m["recall"],
                             "precision": m["precision"], "f1": m["f1"], "illicit": m["illicit_nodes"]})
    return pd.DataFrame(rows)


def fig_comparison(table: pd.DataFrame) -> None:
    ordered = table.sort_values("pr_auc_mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(ordered))
    ax.errorbar(ordered["pr_auc_mean"], positions, xerr=ordered["pr_auc_std"], fmt="none",
                ecolor=INK_2, elinewidth=1, capsize=3, zorder=2)
    ax.scatter(ordered["pr_auc_mean"], positions, s=52, zorder=3, edgecolors="white",
               color=[MODEL_COLORS[m] for m in ordered["family"]])
    for pos, (_, row) in zip(positions, ordered.iterrows()):
        ax.annotate(f"{row['pr_auc_mean']:.3f}", (row["pr_auc_mean"], pos), xytext=(8, 5),
                    textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_yticks(positions, [DISPLAY[m] for m in ordered["family"]])
    ax.tick_params(axis="y", labelcolor="#0b0b0b")
    ax.set_xlabel("Test PR-AUC (mean and standard deviation across seeds)")
    ax.set_title("Test PR-AUC by model")
    ax.grid(axis="y", visible=False)
    save(fig, FIGURES / "model_comparison_pr_auc.png")


def fig_accuracy(table: pd.DataFrame, majority: float) -> None:
    ordered = table.sort_values("accuracy_mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(ordered))
    ax.errorbar(ordered["accuracy_mean"], positions, xerr=ordered["accuracy_std"], fmt="none",
                ecolor=INK_2, elinewidth=1, capsize=3, zorder=2)
    ax.scatter(ordered["accuracy_mean"], positions, s=52, zorder=3, edgecolors="white",
               color=[MODEL_COLORS[m] for m in ordered["family"]])
    for pos, (_, row) in zip(positions, ordered.iterrows()):
        ax.annotate(f"{row['accuracy_mean']:.3f}", (row["accuracy_mean"], pos), xytext=(8, 5),
                    textcoords="offset points", color=INK_2, fontsize=9)
    ax.axvline(majority, color=INK_2, ls="--", lw=1)
    ax.annotate(f"always licit {majority:.3f}", (majority, 0), xytext=(-4, 4), textcoords="offset points",
                ha="right", color=INK_2, fontsize=9)
    ax.set_yticks(positions, [DISPLAY[m] for m in ordered["family"]])
    ax.tick_params(axis="y", labelcolor="#0b0b0b")
    ax.set_xlabel("Test accuracy at the frozen threshold (mean and standard deviation across seeds)")
    ax.set_title("Test accuracy by model")
    ax.grid(axis="y", visible=False)
    save(fig, FIGURES / "model_comparison_accuracy.png")


def fig_epoch_curves(curves: pd.DataFrame, selection: dict) -> None:
    panels = [("val_accuracy", "Validation accuracy"), ("val_pr_auc", "Validation PR-AUC")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (metric, title) in zip(axes, panels):
        for family in ("mlp", *GRAPH_FAMILIES):
            info = selection["families"][family]
            part = curves[(curves["model"] == family) & (curves["feature_set"] == info["feature_set"])
                          & (curves["edge_mode"] == info["edge_mode"])]
            ax.plot(part["epoch"], part[metric], color=MODEL_COLORS[family], lw=1.8, label=DISPLAY[family])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Epoch")
        ax.set_title(title)
    axes[1].legend(loc="lower right")
    save(fig, FIGURES / "training_curves_val.png")


def fig_pr_curves(selection, y, mask, families) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for family in families:
        run = selection["families"][family]["median_seed_run"]
        precision, recall, _ = precision_recall_curve(y[mask], scores_for(run)[mask])
        ax.plot(recall, precision, color=MODEL_COLORS[family], lw=1.8, label=DISPLAY[family])
    ax.axhline(y[mask].mean(), color=INK_2, ls="--", lw=1)
    ax.annotate(f"base rate {y[mask].mean():.3f}", (0.01, y[mask].mean()), xytext=(0, 4),
                textcoords="offset points", ha="left", color=INK_2, fontsize=9)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall on the test period")
    ax.legend(loc="upper right")
    save(fig, FIGURES / "pr_curves_test.png")


def fig_per_step(table: pd.DataFrame, families, split_cfg, nodes) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1])
    grouped = table.groupby(["family", "time_step"])["recall"].mean().reset_index()
    for family in families:
        part = grouped[grouped["family"] == family]
        axes[0].plot(part["time_step"], part["recall"], color=MODEL_COLORS[family], marker="o", ms=4,
                     label=DISPLAY[family])
    axes[0].axvline(42.5, color=INK_2, ls="--", lw=1)
    axes[0].annotate("step 43", (42.5, 0.95), xytext=(4, 0), textcoords="offset points", color=INK_2, fontsize=9)
    axes[0].set_ylabel("Recall at frozen threshold")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Recall collapses from step 43 in every model")
    axes[0].legend(loc="upper right")
    illicit = table.drop_duplicates("time_step").sort_values("time_step")
    axes[1].bar(illicit["time_step"], illicit["illicit"], color="#eb6834", width=0.7)
    axes[1].set_ylabel("Known illicit")
    axes[1].set_xlabel("Time step")
    axes[1].grid(axis="x", visible=False)
    save(fig, FIGURES / "per_step_recall_test.png")


def fig_confusions(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(rows), figsize=(3.4 * len(rows), 3.2))
    for ax, row in zip(np.atleast_1d(axes), rows):
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        ax.imshow(np.log1p(matrix), cmap="Blues", vmin=0, vmax=np.log1p(matrix.max()) * 1.4)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}", ha="center", va="center", color="#0b0b0b", fontsize=11)
        ax.set_xticks([0, 1], ["predicted licit", "flagged illicit"])
        ax.set_yticks([0, 1], ["actual licit", "actual illicit"])
        ax.set_title(row["display"], fontsize=10)
        ax.grid(False)
    save(fig, FIGURES / "confusion_matrices_test.png")


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns) + "\n"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in frame.to_numpy())
    return header + body + "\n"


def main() -> None:
    apply_style()
    cfg = load_config()
    budgets = tuple(cfg["threshold"]["review_budgets"])
    runs = load_runs()
    records = {r["name"]: r for r in runs}
    table = variants_table(runs)

    # Phase 1: validation only. Written to disk before any test label is read.
    selection = freeze_selection(table)
    write_json(REPORTS / "frozen_selection.json", selection)

    # Phase 2: test metrics.
    nodes = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    y = nodes["y"].to_numpy()
    test_mask = (nodes["split"].to_numpy() == "test") & (y != -1)

    main_rows = []
    for family in FAMILIES:
        info = selection["families"][family]
        summary = summarize(test_rows(records, info["runs"], y, test_mask, budgets))
        main_rows.append({"family": family, "model": DISPLAY[family], "feature_set": info["feature_set"],
                          "edge_mode": info["edge_mode"], "val_pr_auc": info["val_pr_auc_mean"],
                          **training_summary(records, info["runs"]), **summary})
    main_table = pd.DataFrame(main_rows).sort_values("pr_auc_mean", ascending=False)

    ablation_rows = []
    for (model, feature_set, edge_mode), part in table.groupby(["model", "feature_set", "edge_mode"]):
        summary = summarize(test_rows(records, part["name"].tolist(), y, test_mask, budgets))
        ablation_rows.append({"model": model, "feature_set": feature_set, "edge_mode": edge_mode,
                              "val_pr_auc": float(part["val_pr_auc"].mean()), **summary})
    ablations = pd.DataFrame(ablation_rows).sort_values(["model", "feature_set", "edge_mode"])

    curves = epoch_curves(runs)

    step_table = per_step(selection, records, nodes, FAMILIES)
    early = (nodes["time_step"].to_numpy() <= 42) & test_mask
    late = (nodes["time_step"].to_numpy() >= 43) & test_mask
    period_rows = []
    for family in FAMILIES:
        names = selection["families"][family]["runs"]
        for label, mask in (("steps_40_42", early), ("steps_43_49", late)):
            summary = summarize(test_rows(records, names, y, mask, budgets))
            period_rows.append({"model": DISPLAY[family], "period": label, "precision": summary["precision_mean"],
                                "recall": summary["recall_mean"], "f1": summary["f1_mean"],
                                "pr_auc": summary["pr_auc_mean"]})
    periods = pd.DataFrame(period_rows)

    def seed_scores(family: str) -> list[np.ndarray]:
        return [scores_for(n)[test_mask] for n in selection["families"][family]["runs"]]

    pairs = [("graphsage", "xgboost"), ("graphsage", "random_forest"), ("graphsage", "mlp"),
             ("evolvegcn_o", "static_gcn"), ("evolvegcn_h", "static_gcn"),
             ("evolvegcn_o", "graphsage"), ("evolvegcn_h", "graphsage")]
    significance = {f"{a}_minus_{b}": bootstrap_difference(y[test_mask], seed_scores(a), seed_scores(b))
                    for a, b in pairs}

    main_table.to_csv(REPORTS / "final_comparison.csv", index=False)
    ablations.to_csv(REPORTS / "ablations.csv", index=False)
    curves.to_csv(REPORTS / "epoch_curves.csv", index=False)
    step_table.to_csv(REPORTS / "per_step_test.csv", index=False)
    periods.to_csv(REPORTS / "test_periods.csv", index=False)
    write_json(REPORTS / "significance.json", significance)

    display_families = ["xgboost", "mlp", "graphsage", "evolvegcn_o"]
    fig_comparison(main_table)
    fig_accuracy(main_table, 1 - y[test_mask].mean())
    fig_epoch_curves(curves, selection)
    fig_pr_curves(selection, y, test_mask, display_families)
    fig_per_step(step_table, ["xgboost", "graphsage", "evolvegcn_o"], cfg["split"], nodes)
    confusion_rows = []
    for family in ("xgboost", "graphsage", "evolvegcn_o"):
        run = selection["families"][family]["median_seed_run"]
        row = binary_metrics(y[test_mask], scores_for(run)[test_mask], records[run]["threshold"], budgets)
        confusion_rows.append({**row, "display": DISPLAY[family]})
    fig_confusions(confusion_rows)
    write_json(REPORTS / "confusion_selected.json", confusion_rows)

    pretty = main_table[["model", "feature_set", "edge_mode", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean",
                         "pr_auc_mean", "pr_auc_std", "recall_at_100_mean", "alerts_mean"]].round(3)
    print(pretty.to_string(index=False))
    print(periods.pivot(index="model", columns="period", values="pr_auc").round(3).to_string())
    print(json.dumps(significance, indent=1))
    (REPORTS / "final_comparison.md").write_text(markdown_table(pretty), encoding="utf-8")


if __name__ == "__main__":
    main()
