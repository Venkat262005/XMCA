"""DeLong test for comparing areas under correlated ROC curves (AUROC).

Reference:
DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
Comparing the areas under two or more correlated receiver operating characteristic curves:
a nonparametric approach. Biometrics, 44(3), 837-845.
"""

from typing import Dict, Tuple
import numpy as np
from scipy import stats


def compute_delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray
) -> Dict[str, float]:
    """
    Computes DeLong's test statistic and p-value for the difference in AUROC
    between model A (e.g. XMCA-Net) and model B (baseline).

    Returns:
        dict with:
            - 'auroc_a': AUROC of model A
            - 'auroc_b': AUROC of model B
            - 'auroc_diff': auroc_a - auroc_b
            - 'z_score': DeLong z-statistic
            - 'p_value': two-tailed p-value
    """
    y_true = np.asarray(y_true).astype(int)
    y_score_a = np.asarray(y_score_a).astype(float)
    y_score_b = np.asarray(y_score_b).astype(float)

    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)

    m = np.sum(pos_mask)  # number of positives
    n = np.sum(neg_mask)  # number of negatives

    if m == 0 or n == 0:
        return {"auroc_a": 0.5, "auroc_b": 0.5, "auroc_diff": 0.0, "z_score": 0.0, "p_value": 1.0}

    # Structural components for model A
    xa = y_score_a[pos_mask]
    ya = y_score_a[neg_mask]

    # Structural components for model B
    xb = y_score_b[pos_mask]
    yb = y_score_b[neg_mask]

    # Kernel function: psi(x, y) = 1 if x > y, 0.5 if x == y, 0 if x < y
    # V10: (m x 1) component
    v10_a = np.mean(xa[:, None] > ya[None, :] + 0.5 * (xa[:, None] == ya[None, :]), axis=1)
    v01_a = np.mean(xa[:, None] < ya[None, :] + 0.5 * (xa[:, None] == ya[None, :]), axis=0)

    v10_b = np.mean(xb[:, None] > yb[None, :] + 0.5 * (xb[:, None] == yb[None, :]), axis=1)
    v01_b = np.mean(xb[:, None] < yb[None, :] + 0.5 * (xb[:, None] == yb[None, :]), axis=0)

    theta_a = np.mean(v10_a)
    theta_b = np.mean(v10_b)

    # Covariance matrices
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))

    s = s10 / m + s01 / n

    diff = theta_a - theta_b
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]

    if var_diff <= 0:
        z = 0.0
        p = 1.0
    else:
        z = diff / np.sqrt(var_diff)
        p = 2 * (1.0 - stats.norm.cdf(abs(z)))

    return {
        "auroc_a": float(theta_a),
        "auroc_b": float(theta_b),
        "auroc_diff": float(diff),
        "z_score": float(z),
        "p_value": float(p)
    }
