"""Multi-Head Calibrated XMCA Combiner: Hallucination Probability & Cause Attribution."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from src.combiner.features import (
    FEATURE_COLUMNS,
    compute_cause_attribution,
    determine_proxy_attribution,
    extract_combiner_features
)

logger = logging.getLogger(__name__)


class XMCACombiner:
    """
    Multi-Head Diagnostic Combiner:
    - Head 1: Predicts calibrated P(hallucination | X) in [0, 1].
    - Head 2: Decomposes and predicts specific failure cause (Perception, Retrieval, Utilization, Compound, or Faithful).
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
                    n_estimators=150,
                    learning_rate=0.03,
                    max_depth=5,
                    num_leaves=31,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=self.random_state,
                    verbose=-1
                )
            except (ImportError, OSError, Exception) as e:
                logger.info(f"LightGBM unavailable ({e}), trying XGBoost...")
        if self.model_type == "xgboost" or self.classifier is None:
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=150,
                    learning_rate=0.03,
                    max_depth=5,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=self.random_state,
                    eval_metric="logloss"
                )
            except (ImportError, OSError, Exception) as e:
                logger.info(f"XGBoost unavailable ({e}), using calibrated LogisticRegression fallback.")

        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=self.random_state, max_iter=300, C=1.0)

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
        X_train = train_df[self.feature_columns]
        y_train = train_df[target_col].values

        X_val = val_df[self.feature_columns]
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
        min_scores = df["min_score"].values
        deficit_probs = np.clip(1.0 - min_scores, 0.0, 1.0)

        if baseline_min_rule or not self.is_fitted:
            return deficit_probs

        X = df[self.feature_columns]
        raw_probs = self.classifier.predict_proba(X)[:, 1]

        if self.calibrator is not None:
            calibrated = self.calibrator.predict(raw_probs)
            model_probs = np.clip(calibrated, 0.0, 1.0)
        else:
            model_probs = raw_probs

        # Blend learned calibration with domain consistency constraint
        # (Guarantees that severe consistency collapse min(S) < 0.25 flags hallucination)
        final_probs = np.maximum(model_probs, np.where(min_scores < 0.30, deficit_probs, model_probs))
        return np.clip(final_probs, 0.0, 1.0)

    def diagnose_dataframe(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Run complete multi-head diagnosis on a batch dataframe:
        Returns:
            list of dicts with calibrated prob, binary verdict, primary cause, and soft cause distribution.
        """
        probs = self.predict_proba(df)
        diagnostics = []

        for idx, row in df.iterrows():
            prob = float(probs[idx])
            sp = float(row.get("s_p", 0.5))
            sr = float(row.get("s_r", 0.5))
            su = float(row.get("s_u", 0.5))

            primary_cause, cause_probs = compute_cause_attribution(
                s_p=sp,
                s_r=sr,
                s_u=su,
                hallucination_prob=prob,
                threshold=threshold
            )

            diagnostics.append({
                "id": row.get("id", f"sample_{idx}"),
                "hallucination_probability": prob,
                "is_hallucination": bool(prob >= threshold),
                "primary_cause": primary_cause,
                "cause_probabilities": cause_probs,
                "proxy_attribution": determine_proxy_attribution(sp, sr, su),
                "consistency_scores": {"s_p": sp, "s_r": sr, "s_u": su}
            })

        return diagnostics

    def diagnose_single(
        self,
        s_p: float,
        s_r: float,
        s_u: float,
        evidence_count: int = 1,
        claim_count: int = 1,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Convenience method to diagnose a single instance from raw scores."""
        feat_dict = extract_combiner_features(s_p, s_r, s_u, evidence_count, claim_count)
        df = pd.DataFrame([feat_dict])
        return self.diagnose_dataframe(df, threshold=threshold)[0]

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
