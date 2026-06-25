from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


CRITERIA = ("c1", "c2", "c3")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision is undefined if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC is undefined if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy is undefined if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute CVS multi-label classification metrics.

    Parameters
    ----------
    y_true:
        Array of shape [N, 3] containing binary ground-truth labels.
    y_prob:
        Array of shape [N, 3] containing predicted probabilities.
    threshold:
        Probability threshold for converting probabilities into binary predictions.

    Returns
    -------
    dict
        Per-criterion AP, ROC-AUC, balanced accuracy, and mean metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_prob {y_prob.shape}")

    if y_true.ndim != 2 or y_true.shape[1] != 3:
        raise ValueError(f"Expected shape [N, 3], got {y_true.shape}")

    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, float] = {}

    ap_values = []
    auc_values = []
    bacc_values = []

    for i, name in enumerate(CRITERIA):
        ap = _safe_average_precision(y_true[:, i], y_prob[:, i])
        auc = _safe_roc_auc(y_true[:, i], y_prob[:, i])
        bacc = _safe_balanced_accuracy(y_true[:, i], y_pred[:, i])

        metrics[f"{name}_ap"] = ap
        metrics[f"{name}_auc"] = auc
        metrics[f"{name}_bacc"] = bacc

        ap_values.append(ap)
        auc_values.append(auc)
        bacc_values.append(bacc)

    metrics["mAP"] = float(np.nanmean(ap_values))
    metrics["mean_auc"] = float(np.nanmean(auc_values))
    metrics["mean_bacc"] = float(np.nanmean(bacc_values))

    return metrics


def compute_multilabel_metrics_from_logits(
    y_true: np.ndarray,
    logits: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute CVS metrics from raw model logits.
    """
    y_prob = sigmoid(logits)
    return compute_multilabel_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)