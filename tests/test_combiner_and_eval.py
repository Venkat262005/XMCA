"""Unit tests for feature engineering, calibration, thresholding, and ablations."""

import numpy as np
from src.combiner.features import build_feature_dataframe, determine_proxy_attribution, extract_combiner_features
from src.combiner.model import XMCACombiner
from src.combiner.thresholds import tune_decision_threshold
from src.evaluation.ablation_suite import run_ablation_experiment
from src.evaluation.bootstrap import compute_bootstrap_confidence_intervals
from src.evaluation.metrics import compute_all_metrics, compute_expected_calibration_error


def test_proxy_attribution():
    attr = determine_proxy_attribution(s_p=0.2, s_r=0.8, s_u=0.9)
    assert attr == "perception_proxy_failure"

    attr = determine_proxy_attribution(s_p=0.9, s_r=0.1, s_u=0.9)
    assert attr == "retrieval_proxy_failure"

    attr = determine_proxy_attribution(s_p=0.9, s_r=0.8, s_u=0.15)
    assert attr == "utilization_proxy_failure"


def test_metrics_and_calibration():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.85, 0.15, 0.75, 0.05, 0.95, 0.3])

    metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["auroc"] > 0.9
    assert metrics["f1"] == 1.0
    assert metrics["ece"] < 0.2

    # Test bootstrap
    ci = compute_bootstrap_confidence_intervals(y_true, y_prob, threshold=0.5, n_bootstraps=50)
    assert "auroc" in ci
    assert ci["auroc"]["lower_ci"] <= ci["auroc"]["upper_ci"]


def test_combiner_fit_and_ablation():
    # Synthetic samples
    np.random.seed(42)
    records = []
    for i in range(100):
        # Correlate lower scores with hallucination
        is_h = 1 if np.random.rand() < 0.4 else 0
        sp = np.random.uniform(0.1, 0.5) if is_h else np.random.uniform(0.6, 1.0)
        sr = np.random.uniform(0.1, 0.6) if is_h else np.random.uniform(0.5, 1.0)
        su = np.random.uniform(0.1, 0.5) if is_h else np.random.uniform(0.6, 1.0)
        records.append({
            "id": f"s_{i}",
            "s_p": sp,
            "s_r": sr,
            "s_u": su,
            "evidence_count": 3,
            "claim_count": 2,
            "is_hallucination_gt": is_h
        })

    train_rec = records[:70]
    val_rec = records[70:85]
    test_rec = records[85:]

    res = run_ablation_experiment(
        train_records=train_rec,
        val_records=val_rec,
        test_records=test_rec,
        ablation_name="full_model",
        seeds=[42, 101]
    )

    assert "auroc_mean" in res
    assert 0.0 <= res["auroc_mean"] <= 1.0
    assert "ece_mean" in res
