"""Diagnostic plotting: Reliability diagrams, ROC curves, PR curves, and proxy attribution."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path,
    n_bins: int = 10,
    title: str = "Reliability Diagram (Calibration Curve)"
) -> None:
    """Plot reliability diagram comparing confidence to empirical accuracy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(y_true[mask]))
            bin_confidences.append(np.mean(y_prob[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(bin_confidences, bin_accuracies, "s-", color="#1f77b4", label="XMCA-Net-Lite")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Observed Hallucination Fraction")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path,
    title: str = "ROC and Precision-Recall Curves"
) -> None:
    """Generate side-by-side ROC and PR curves."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve
    except ImportError:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(fpr, tpr, color="#2ca02c", lw=2, label="ROC")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(rec, prec, color="#d62728", lw=2, label="Precision-Recall")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
