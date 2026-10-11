"""Threshold optimization strictly on validation splits."""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def tune_decision_threshold(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    metric: str = "f1",
    num_steps: int = 101
) -> Tuple[float, Dict[str, float]]:
    """
    Search for best classification threshold on validation set.
    Never evaluates on test set.
    """
    best_threshold = 0.5
    best_metric_value = -1.0
    best_metrics = {}

    thresholds = np.linspace(0.01, 0.99, num_steps)
    for thresh in thresholds:
        preds = (y_prob_val >= thresh).astype(int)
        f1 = float(f1_score(y_true_val, preds, zero_division=0))
        prec = float(precision_score(y_true_val, preds, zero_division=0))
        rec = float(recall_score(y_true_val, preds, zero_division=0))

        score = f1 if metric == "f1" else (2 * prec * rec / (prec + rec + 1e-9))
        if score > best_metric_value:
            best_metric_value = score
            best_threshold = float(thresh)
            best_metrics = {"f1": f1, "precision": prec, "recall": rec, "threshold": best_threshold}

    return best_threshold, best_metrics
