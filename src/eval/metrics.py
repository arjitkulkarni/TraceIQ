"""Metric definitions shared by every model, so all comparisons use identical code."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def confusion(y: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    return {
        "tp": int((y & flagged).sum()),
        "fp": int((~y & flagged).sum()),
        "fn": int((y & ~flagged).sum()),
        "tn": int((~y & ~flagged).sum()),
    }


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def recall_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    """Share of illicit nodes found among the K highest scores. Ties break by position."""
    order = np.argsort(-score, kind="stable")[:k]
    return _ratio(np.asarray(y)[order].sum(), np.asarray(y).sum())


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    order = np.argsort(-score, kind="stable")[:k]
    return _ratio(np.asarray(y)[order].sum(), len(order))


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC reported as sklearn average precision (step-wise, no interpolation)."""
    if np.asarray(y).sum() == 0:
        return float("nan")
    return float(average_precision_score(y, score))


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float, budgets=(100, 500)) -> dict:
    """y must contain known labels only: 1 illicit, 0 licit."""
    y = np.asarray(y)
    score = np.asarray(score)
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("metrics accept known labels only")
    cm = confusion(y, score >= threshold)
    precision = _ratio(cm["tp"], cm["tp"] + cm["fp"])
    recall = _ratio(cm["tp"], cm["tp"] + cm["fn"])
    out = {
        **cm,
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "accuracy": _ratio(cm["tp"] + cm["tn"], len(y)),
        "pr_auc": pr_auc(y, score),
        "alerts": cm["tp"] + cm["fp"],
        "known_nodes": len(y),
        "illicit_nodes": int(y.sum()),
    }
    for k in budgets:
        out[f"recall_at_{k}"] = recall_at_k(y, score, k)
        out[f"precision_at_{k}"] = precision_at_k(y, score, k)
    return out


def select_threshold(y_val: np.ndarray, score_val: np.ndarray) -> float:
    """Validation-only rule: the score cutoff that maximizes F1 on validation labels."""
    precision, recall, thresholds = precision_recall_curve(y_val, score_val)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(thresholds[int(np.argmax(f1[:-1]))])
