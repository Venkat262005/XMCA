"""Retrieval score fusion: Learned isotonic regression vs fixed weighted fusion."""

from typing import List
import numpy as np
from sklearn.isotonic import IsotonicRegression


class RetrievalScoreFusion:
    """
    Computes aggregate Retrieval Consistency Score (S_R).
    Supports learned isotonic calibration and fixed-weight ablation.
    """

    def __init__(self, mode: str = "isotonic", fixed_weight_dense: float = 0.5):
        self.mode = mode  # 'isotonic' or 'fixed_weighted'
        self.fixed_weight_dense = fixed_weight_dense
        self.isotonic_regressor = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit(self, raw_scores: np.ndarray, relevance_labels: np.ndarray) -> None:
        """
        Fit isotonic regression on validation raw scores.
        raw_scores: shape (N,) combining dense and rerank scores (e.g. mean).
        relevance_labels: shape (N,) in {0, 1}.
        """
        if self.mode == "isotonic":
            self.isotonic_regressor.fit(raw_scores, relevance_labels)
            self.is_fitted = True

    def fuse(self, dense_scores: List[float], rerank_scores: List[float]) -> float:
        """
        Combine top passage dense and rerank scores into a unified S_R in [0, 1].
        """
        if not dense_scores or not rerank_scores:
            return 0.0

        # Mean score across retrieved top passages
        d_val = float(np.mean(dense_scores))
        r_val = float(np.mean(rerank_scores))

        # Clamp to [0, 1]
        d_val = max(0.0, min(1.0, (d_val + 1.0) / 2.0 if d_val < 0 else d_val))
        r_val = max(0.0, min(1.0, r_val))

        if self.mode == "fixed_weighted" or not self.is_fitted:
            raw = self.fixed_weight_dense * d_val + (1.0 - self.fixed_weight_dense) * r_val
            return float(np.clip(raw, 0.0, 1.0))

        # Learned isotonic calibration
        raw = 0.5 * d_val + 0.5 * r_val
        calibrated = float(self.isotonic_regressor.predict([raw])[0])
        return float(np.clip(calibrated, 0.0, 1.0))
