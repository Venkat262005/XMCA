"""Threshold optimization strictly on validation splits using Youden's J or F1."""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve


def tune_decision_threshold(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    metric: str = "youden_j",
    num_steps: int = 101
) -> Tuple[float, Dict[str, float]]:
    """
    Search for best classification threshold strictly on validation set.
    Supports:
    - 'youden_j': Maximizes Youden's J statistic (J = Sensitivity + Specificity - 1 = TPR - FPR).
    - 'f1': Maximizes harmonic mean of precision and recall.
    Never evaluates on held-out test set.
    """
    y_true_val = np.asarray(y_true_val).astype(int)
    y_prob_val = np.asarray(y_prob_val).astype(float)

    if metric == "youden_j":
        # Analytical optimal threshold via ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_val, y_prob_val)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_threshold = float(np.clip(thresholds[best_idx], 0.01, 0.99))

        preds = (y_prob_val >= best_threshold).astype(int)
        f1 = float(f1_score(y_true_val, preds, zero_division=0))
        prec = float(precision_score(y_true_val, preds, zero_division=0))
        rec = float(recall_score(y_true_val, preds, zero_division=0))
        youden_j = float(j_scores[best_idx])

        return best_threshold, {
            "threshold": best_threshold,
            "youden_j": youden_j,
            "f1": f1,
            "precision": prec,
            "recall": rec
        }

    # Grid search for F1
    best_threshold = 0.5
    best_metric_value = -1.0
    best_metrics = {}

    thresholds = np.linspace(0.01, 0.99, num_steps)
    for thresh in thresholds:
        preds = (y_prob_val >= thresh).astype(int)
        f1 = float(f1_score(y_true_val, preds, zero_division=0))
        prec = float(precision_score(y_true_val, preds, zero_division=0))
        rec = float(recall_score(y_true_val, preds, zero_division=0))

        if f1 > best_metric_value:
            best_metric_value = f1
            best_threshold = float(thresh)
            best_metrics = {
                "threshold": best_threshold,
                "f1": f1,
                "precision": prec,
                "recall": rec
            }

    return best_threshold, best_metrics
