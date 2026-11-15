"""Exploratory analysis (guide section 09). Writes figures and artifacts/reports/eda.json."""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import ks_2samp

from src.common import FIGURES, PROCESSED_DIR, REPORTS, load_config, write_json
from src.eval.plot_style import INK_2, LABEL_COLORS, apply_style, save, shade_splits

DARK_MARKET_STEP = 43


def per_step_table(nodes: pd.DataFrame) -> pd.DataFrame:
    table = nodes.groupby(["time_step", "label_state"]).size().unstack(fill_value=0)
    return table.reindex(columns=["illicit", "licit", "unknown"], fill_value=0)


def fig_label_counts(nodes: pd.DataFrame) -> None:
    counts = nodes["label_state"].value_counts().reindex(["unknown", "licit", "illicit"])
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(counts.index, counts.values, color=[LABEL_COLORS[k] for k in counts.index], height=0.6)
    for bar, value in zip(bars, counts.values):
        ax.text(value + 1500, bar.get_y() + bar.get_height() / 2,
                f"{value:,} ({value / len(nodes):.1%})", va="center", color=INK_2, fontsize=9)
    ax.set_title("Most transactions have no label")
    ax.set_xlabel("Transactions")
    ax.set_xlim(0, counts.max() * 1.25)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    save(fig, FIGURES / "eda_label_counts.png")


def fig_steps(table: pd.DataFrame, split_cfg: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    total = table.sum(axis=1)
    axes[0].bar(total.index, total.values, color=LABEL_COLORS["unknown"], width=0.75)
    axes[0].set_ylabel("All transactions")
    axes[0].set_title("Transactions per time step", pad=18)
    axes[1].bar(table.index, table["illicit"], color=LABEL_COLORS["illicit"], width=0.75)
    axes[1].set_ylabel("Known illicit")
    axes[1].set_xlabel("Time step")
    axes[1].set_title("Known illicit transactions per time step", pad=18)
    for ax in axes:
        shade_splits(ax, split_cfg)
        ax.grid(axis="x", visible=False)
    axes[1].axvline(DARK_MARKET_STEP - 0.5, color=INK_2, ls="--", lw=1)
    axes[1].annotate("step 43", (DARK_MARKET_STEP - 0.5, table["illicit"].max() * 0.85),
                     xytext=(4, 0), textcoords="offset points", color=INK_2, fontsize=9)
    save(fig, FIGURES / "eda_transactions_per_step.png")


def fig_split_prevalence(nodes: pd.DataFrame) -> dict:
    known = nodes[nodes["y"] != -1]
    prevalence = known.groupby("split")["y"].mean().reindex(["train", "val", "test"])
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(["train", "validation", "test"], prevalence.values, color=LABEL_COLORS["illicit"], width=0.5)
    for bar, value in zip(bars, prevalence.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.1%}",
                ha="center", color=INK_2, fontsize=9)
    ax.set_title("Illicit share of known labels drops in the test period")
    ax.set_ylabel("Illicit / known")
    ax.grid(axis="x", visible=False)
    save(fig, FIGURES / "eda_prevalence_by_split.png")
    return {k: float(v) for k, v in prevalence.items()}


def graph_stats(nodes: pd.DataFrame, edges: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    n = len(nodes)
    in_degree = np.bincount(edges[:, 1], minlength=n)
    out_degree = np.bincount(edges[:, 0], minlength=n)
    adjacency = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    n_components, component = connected_components(adjacency, directed=True, connection="weak")
    sizes = np.bincount(component)
    per_step_components = (
        pd.DataFrame({"step": nodes["time_step"], "component": component})
        .groupby("step")["component"].nunique()
    )
    stats = {
        "density_edges_per_node": len(edges) / n,
        "isolated_nodes": int(((in_degree + out_degree) == 0).sum()),
        "weak_components": int(n_components),
        "weak_components_per_step_min": int(per_step_components.min()),
        "weak_components_per_step_max": int(per_step_components.max()),
        "largest_component_share_max": float(sizes.max() / n),
        "in_degree_median": float(np.median(in_degree)),
        "in_degree_mean": float(in_degree.mean()),
        "in_degree_max": int(in_degree.max()),
        "out_degree_median": float(np.median(out_degree)),
        "out_degree_mean": float(out_degree.mean()),
        "out_degree_max": int(out_degree.max()),
        "max_nodes_in_a_step": int(nodes["time_step"].value_counts().max()),
    }
    return stats, in_degree, out_degree


def fig_degrees(in_degree: np.ndarray, out_degree: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, values, name in ((axes[0], in_degree, "In-degree"), (axes[1], out_degree, "Out-degree")):
        counts = np.bincount(values)
        degree = np.flatnonzero(counts)
        degree = degree[degree > 0]
        ax.scatter(degree, counts[degree], s=14, color=LABEL_COLORS["licit"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{name} (log)")
        ax.set_title(f"{name}: {(values == 0).mean():.1%} of nodes are zero")
    axes[0].set_ylabel("Transactions (log)")
    save(fig, FIGURES / "eda_degree_distributions.png")


def drift(x_raw: np.ndarray, nodes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Unlabeled feature drift between train and test periods, on all rows of each period."""
    split = nodes["split"].to_numpy()
    rng = np.random.default_rng(0)
    train = rng.choice(np.flatnonzero(split == "train"), 20000, replace=False)
    test = rng.choice(np.flatnonzero(split == "test"), 20000, replace=False)
    rows = []
    for col in range(x_raw.shape[1]):
        result = ks_2samp(x_raw[train, col], x_raw[test, col])
        rows.append({"feature": f"f{col:03d}", "ks": float(result.statistic)})
    table = pd.DataFrame(rows).sort_values("ks", ascending=False)
    summary = {
        "ks_median": float(table["ks"].median()),
        "ks_max": float(table["ks"].max()),
        "features_with_ks_above_0_1": int((table["ks"] > 0.1).sum()),
        "top_features": table.head(5).to_dict("records"),
    }
    return table, summary


def fig_drift(x_raw: np.ndarray, nodes: pd.DataFrame, table: pd.DataFrame, split_cfg: dict) -> None:
    top = table.head(3)["feature"].tolist()
    steps = nodes["time_step"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, top)):
        values = x_raw[:, int(name[1:])]
        frame = pd.DataFrame({"step": steps, "v": values}).groupby("step")["v"]
        ax.plot(frame.median().index, frame.median().values, color=LABEL_COLORS["licit"], label="median")
        ax.fill_between(frame.median().index, frame.quantile(0.25).values, frame.quantile(0.75).values,
                        color=LABEL_COLORS["licit"], alpha=0.2, lw=0, label="interquartile range")
        ax.set_ylabel(name)
        shade_splits(ax, split_cfg)
        ax.grid(axis="x", visible=False)
        if i == 0:
            ax.legend(loc="upper left", ncol=2)
    axes[0].set_title("Features that drift most between train and test (anonymous columns)", pad=18)
    axes[-1].set_xlabel("Time step")
    save(fig, FIGURES / "eda_feature_drift.png")


def fig_neighborhoods(nodes: pd.DataFrame, edges: np.ndarray) -> None:
    """Two real neighborhoods from the training period, with direction and known labels only."""
    labels = nodes["label_state"].to_numpy()
    split = nodes["split"].to_numpy()
    graph = nx.DiGraph()
    graph.add_edges_from(edges[split[edges[:, 0]] == "train"].tolist())
    undirected = graph.to_undirected(as_view=True)
    rng = np.random.default_rng(0)

    def pick(state: str) -> int:
        candidates = [int(n) for n in rng.permutation(np.flatnonzero((labels == state) & (split == "train")))
                      if n in graph and 8 <= len(nx.ego_graph(undirected, int(n), radius=2)) <= 25]
        return candidates[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, state in zip(axes, ("illicit", "licit")):
        focal = pick(state)
        nbrs = list(nx.ego_graph(undirected, focal, radius=2).nodes)
        sub = graph.subgraph(nbrs)
        pos = nx.spring_layout(sub.to_undirected(), seed=3)
        nx.draw_networkx_edges(sub, pos, ax=ax, edge_color=INK_2, arrows=True, arrowsize=11,
                               width=0.9, node_size=170)
        for state_name, marker in (("unknown", "o"), ("licit", "s"), ("illicit", "^")):
            subset = [n for n in sub.nodes if labels[n] == state_name]
            nx.draw_networkx_nodes(sub, pos, nodelist=subset, ax=ax, node_shape=marker, node_size=170,
                                   node_color=LABEL_COLORS[state_name], edgecolors="white", label=state_name)
        nx.draw_networkx_nodes(sub, pos, nodelist=[focal], ax=ax, node_shape="o", node_size=420,
                               node_color="none", edgecolors="#0b0b0b", linewidths=1.5)
        step = int(nodes["time_step"].iloc[focal])
        ax.set_title(f"Around a known {state} transaction (step {step})")
        ax.legend(loc="lower right", scatterpoints=1)
        ax.axis("off")
    save(fig, FIGURES / "eda_sample_neighborhoods.png")


def main() -> None:
    apply_style()
    cfg = load_config()
    nodes = pd.read_parquet(PROCESSED_DIR / "nodes.parquet")
    edges = np.load(PROCESSED_DIR / "edges.npy")
    x_raw = np.load(PROCESSED_DIR / "features_raw.npy")

    table = per_step_table(nodes)
    fig_label_counts(nodes)
    fig_steps(table, cfg["split"])
    prevalence = fig_split_prevalence(nodes)
    stats, in_degree, out_degree = graph_stats(nodes, edges)
    fig_degrees(in_degree, out_degree)
    drift_table, drift_summary = drift(x_raw, nodes)
    fig_drift(x_raw, nodes, drift_table, cfg["split"])
    fig_neighborhoods(nodes, edges)

    late = table.loc[DARK_MARKET_STEP:]
    report = {
        "labeled_share": float((nodes["y"] != -1).mean()),
        "illicit_share_of_known_by_split": prevalence,
        "illicit_share_of_known_steps_43_49": float(late["illicit"].sum() / (late["illicit"] + late["licit"]).sum()),
        "illicit_share_of_known_steps_40_42": float(
            table.loc[40:42, "illicit"].sum() / (table.loc[40:42, "illicit"] + table.loc[40:42, "licit"]).sum()
        ),
        "graph": stats,
        "feature_drift_ks_train_vs_test": drift_summary,
    }
    write_json(REPORTS / "eda.json", report)
    print(report)


if __name__ == "__main__":
    main()
