"""Core diagnostic evaluation metrics: AUROC, AUPRC, F1, Precision, Recall, ECE."""

from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Partition predictions into equally spaced bins and weight |acc(b) - conf(b)| by bin size.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 10
) -> Dict[str, float]:
    """Calculate AUROC, AUPRC, F1, Precision, Recall, and ECE."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # AUROC
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5

    # AUPRC
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = float(np.mean(y_true))

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "ece": ece,
        "threshold": float(threshold)
    }
