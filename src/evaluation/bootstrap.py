"""Bootstrap confidence intervals (95%) for evaluation metrics."""

from typing import Dict, List, Tuple
import numpy as np
from src.evaluation.metrics import compute_all_metrics


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute non-parametric bootstrap confidence intervals for all metrics.
    Returns dict for each metric: {mean, lower, upper, std}.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    rng = np.random.RandomState(seed)

    alpha = (1.0 - ci_level) / 2.0
    lower_pct = alpha * 100
    upper_pct = (1.0 - alpha) * 100

    metric_samples: Dict[str, List[float]] = {
        "auroc": [],
        "auprc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "ece": []
    }

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n, size=n)
        b_true = y_true[boot_idx]
        b_prob = y_prob[boot_idx]

        # Ensure both classes present in bootstrap sample for ROC/PRC
        if len(np.unique(b_true)) < 2:
            continue

        b_metrics = compute_all_metrics(b_true, b_prob, threshold=threshold)
        for k in metric_samples:
            metric_samples[k].append(b_metrics[k])

    results: Dict[str, Dict[str, float]] = {}
    for k, values in metric_samples.items():
        if values:
            arr = np.array(values)
            results[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "lower_ci": float(np.percentile(arr, lower_pct)),
                "upper_ci": float(np.percentile(arr, upper_pct))
            }
        else:
            results[k] = {"mean": 0.0, "std": 0.0, "lower_ci": 0.0, "upper_ci": 0.0}

    return results
