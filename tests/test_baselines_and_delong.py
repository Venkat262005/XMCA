"""Unit tests for DeLong AUROC significance test and 7 baseline models."""

import numpy as np
from src.baselines.suite import BaselineEvaluationSuite
from src.combiner.thresholds import tune_decision_threshold
from src.evaluation.delong import compute_delong_test


def test_delong_test_computation():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # Model A: Strong detector
    score_a = np.array([0.9, 0.85, 0.8, 0.75, 0.2, 0.15, 0.1, 0.05])
    # Model B: Weak / random detector
    score_b = np.array([0.5, 0.4, 0.6, 0.5, 0.5, 0.6, 0.4, 0.5])

    res = compute_delong_test(y_true, score_a, score_b)

    assert "auroc_a" in res
    assert "auroc_b" in res
    assert res["auroc_a"] > res["auroc_b"]
    assert res["p_value"] <= 0.05  # Statistically significant difference
    assert isinstance(res["z_score"], float)


def test_seven_baselines_suite():
    suite = BaselineEvaluationSuite()

    records = [
        {"s_p": 0.9, "s_r": 0.85, "s_u": 0.95, "generated_answer": "Valid factual statement.", "token_probs": [0.95, 0.90]},
        {"s_p": 0.1, "s_r": 0.85, "s_u": 0.80, "generated_answer": "Perception hallucination.", "token_probs": [0.30, 0.40]},
        {"s_p": 0.9, "s_r": 0.10, "s_u": 0.80, "generated_answer": "Retrieval hallucination.", "token_probs": [0.50, 0.60]},
        {"s_p": 0.9, "s_r": 0.90, "s_u": 0.10, "generated_answer": "Utilization hallucination.", "token_probs": [0.45, 0.55]}
    ]

    preds = suite.evaluate_all(records)

    expected_baselines = [
        "vanilla_rag", "pope_probe", "faithscore", "opera",
        "refind", "rora_vlm", "realign"
    ]

    for b in expected_baselines:
        assert b in preds
        assert len(preds[b]) == 4
        assert np.all(preds[b] >= 0.0)
        assert np.all(preds[b] <= 1.0)


def test_youden_j_threshold_optimization():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.85, 0.75, 0.65, 0.35, 0.25, 0.15, 0.05])

    best_thresh, metrics = tune_decision_threshold(y_true, y_prob, metric="youden_j")

    assert "youden_j" in metrics
    assert metrics["youden_j"] == 1.0  # Perfect separation
    assert 0.35 <= best_thresh <= 0.65
