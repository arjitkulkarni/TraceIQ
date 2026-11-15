"""Shared matplotlib style: one palette, recessive axes, fixed identity colors."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed identity colors: the same entity gets the same color in every figure.
LABEL_COLORS = {"licit": "#2a78d6", "illicit": "#eb6834", "unknown": "#898781"}
MODEL_COLORS = {
    "logistic": "#2a78d6",
    "xgboost": "#eb6834",
    "random_forest": "#1baf7a",
    "mlp": "#eda100",
    "graphsage": "#e87ba4",
    "static_gcn": "#008300",
    "evolvegcn_o": "#6250d6",
    "evolvegcn_h": "#e34948",
}
SPLIT_COLORS = {"train": "#f0efec", "val": "#e1e0d9", "test": "#c3c2b7"}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK,
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "figure.dpi": 150,
        }
    )


def shade_splits(ax, split_cfg: dict) -> None:
    """Grey bands mark train / validation / test on a time-step axis."""
    for name, key in (("train", "train_steps"), ("val", "val_steps"), ("test", "test_steps")):
        lo, hi = split_cfg[key]
        ax.axvspan(lo - 0.5, hi + 0.5, color=SPLIT_COLORS[name], alpha=0.35, lw=0, zorder=0)
        ax.text(
            (lo + hi) / 2, 1.0, {"val": "validation"}.get(name, name),
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=INK_2, fontsize=9,
        )


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
