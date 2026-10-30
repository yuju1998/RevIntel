"""Shared evaluation for every lead conversion candidate (and the naive baseline)."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

THRESHOLD_RULE = "maximize F1 on the validation split; ties -> higher threshold"


@dataclass(frozen=True)
class Threshold:
    value: float
    validation_f1: float


def select_threshold(y_validation: np.ndarray, p_validation: np.ndarray) -> Threshold:
    """Pick the decision threshold from validation data only (see THRESHOLD_RULE)."""
    precision, recall, thresholds = precision_recall_curve(y_validation, p_validation)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    best = np.flatnonzero(f1 == f1.max())[-1]  # thresholds ascend: last = highest
    return Threshold(value=float(thresholds[best]), validation_f1=round(float(f1[best]), 6))


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[False, True]).ravel()
    single_class = len(np.unique(y)) < 2
    return {
        "rows": len(y),
        "positive_rate": round(float(np.mean(y)), 6),
        "roc_auc": None if single_class else round(float(roc_auc_score(y, p)), 6),
        "average_precision": None
        if single_class
        else round(float(average_precision_score(y, p)), 6),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "threshold": threshold,
        "precision": round(float(precision_score(y, predicted, zero_division=0)), 6),
        "recall": round(float(recall_score(y, predicted, zero_division=0)), 6),
        "f1": round(float(f1_score(y, predicted, zero_division=0)), 6),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def constant_scores(rate: float, rows: int) -> np.ndarray:
    """Naive baseline: every snapshot scored with the training positive rate."""
    return np.full(rows, rate, dtype="float64")
