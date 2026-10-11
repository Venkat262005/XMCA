"""Calibrated XMCA Combiner using LightGBM / XGBoost and Isotonic Regression."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from src.combiner.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class XMCACombiner:
    """
    Predicts calibrated probability of hallucination by combining
    Perception (S_P), Retrieval (S_R), and Utilization (S_U) consistency signals.
    """

    def __init__(
        self,
        model_type: str = "lightgbm",
        calibration_method: str = "isotonic",
        random_state: int = 42
    ):
        self.model_type = model_type
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.feature_columns = FEATURE_COLUMNS
        self.classifier = None
        self.calibrator = None
        self.is_fitted = False

    def _init_base_classifier(self):
        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    num_leaves=15,
                    random_state=self.random_state,
                    verbose=-1
                )
            except ImportError:
                logger.info("LightGBM not installed, trying XGBoost...")
        if self.model_type == "xgboost" or self.classifier is None:
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=self.random_state,
                    eval_metric="logloss"
                )
            except ImportError:
                logger.info("XGBoost not installed, using calibrated LogisticRegression fallback.")

        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=self.random_state, max_iter=200)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str = "target"
    ) -> None:
        """
        Train classifier on training split and calibrate probabilities
        using isotonic regression on validation split.
        Strict rule: Never use test split for calibration!
        """
        X_train = train_df[self.feature_columns].values
        y_train = train_df[target_col].values

        X_val = val_df[self.feature_columns].values
        y_val = val_df[target_col].values

        self.classifier = self._init_base_classifier()
        self.classifier.fit(X_train, y_train)

        # Obtain raw validation probabilities
        val_raw_probs = self.classifier.predict_proba(X_val)[:, 1]

        # Fit isotonic calibrator on validation probabilities
        if self.calibration_method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(val_raw_probs, y_val)
        else:
            self.calibrator = None

        self.is_fitted = True

    def predict_proba(self, df: pd.DataFrame, baseline_min_rule: bool = False) -> np.ndarray:
        """
        Predict calibrated hallucination probabilities in [0, 1].
        If baseline_min_rule is True: Heuristic baseline 1 - min(S_P, S_R, S_U).
        """
        if baseline_min_rule:
            min_scores = df["min_score"].values
            return np.clip(1.0 - min_scores, 0.0, 1.0)

        assert self.is_fitted, "Model must be fitted before prediction!"
        X = df[self.feature_columns].values
        raw_probs = self.classifier.predict_proba(X)[:, 1]

        if self.calibrator is not None:
            calibrated = self.calibrator.predict(raw_probs)
            return np.clip(calibrated, 0.0, 1.0)

        return raw_probs

    def save(self, output_dir: str | Path) -> None:
        """Serialize model, calibrator, and feature definitions."""
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "classifier": self.classifier,
            "calibrator": self.calibrator,
            "feature_columns": self.feature_columns,
            "model_type": self.model_type,
            "calibration_method": self.calibration_method,
            "random_state": self.random_state
        }, p / "xmca_combiner_model.joblib")

    @classmethod
    def load(cls, model_path: str | Path) -> "XMCACombiner":
        """Load trained combiner model."""
        target_file = Path(model_path)
        if target_file.is_dir():
            target_file = target_file / "xmca_combiner_model.joblib"
        data = joblib.load(target_file)
        instance = cls(
            model_type=data["model_type"],
            calibration_method=data["calibration_method"],
            random_state=data["random_state"]
        )
        instance.classifier = data["classifier"]
        instance.calibrator = data["calibrator"]
        instance.feature_columns = data["feature_columns"]
        instance.is_fitted = True
        return instance
