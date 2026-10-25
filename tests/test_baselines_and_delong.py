"""Unit tests for DeLong AUROC significance test and 7 literature-faithful baseline models."""

import numpy as np
from src.baselines.suite import (
    BaselineEvaluationSuite,
    FaithScoreBaseline,
    OPERABaseline,
    POPEProbeBaseline,
    REFINDAdaptationBaseline,
    ReAlignBaseline,
    RoRAVLMBaseline,
    VanillaRAGBaseline
)
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


def test_pope_probe_trainable():
    probe = POPEProbeBaseline()
    # Synthetic perception features: high s_p -> faithful (0), low s_p -> hallucination (1)
    sp_train = np.array([0.95, 0.90, 0.85, 0.80, 0.15, 0.10, 0.05, 0.08])
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    probe.fit(sp_train, y_train)
    assert probe.is_fitted is True

    # Test prediction on new samples
    p_faithful = probe.predict_hallucination_prob(0.92)
    p_hallucinated = probe.predict_hallucination_prob(0.12)
    assert p_faithful < 0.40
    assert p_hallucinated > 0.60
    assert p_faithful < p_hallucinated


def test_faithscore_atomic_facts_pipeline():
    faith = FaithScoreBaseline(verification_threshold=0.5)

    # 1. Fact extraction
    facts = faith.extract_atomic_facts("A white cat is sleeping on a blue chair.")
    assert len(facts) >= 1

    # 2. Fully grounded atomic facts -> low hallucination
    prob_faithful = faith.predict_hallucination_prob(claim_scores=[0.9, 0.85, 0.92])
    assert prob_faithful < 0.25

    # 3. Ungrounded atomic facts -> high hallucination
    prob_unfaithful = faith.predict_hallucination_prob(claim_scores=[0.1, 0.2, 0.05])
    assert prob_unfaithful > 0.75


def test_opera_overtrust_penalty():
    opera = OPERABaseline()

    # Case A: High visual grounding, uniform tokens -> low penalty
    p_faithful = opera.predict_hallucination_prob(s_p=0.90, token_probs=[0.85, 0.88, 0.87])
    assert p_faithful < 0.25

    # Case B: Low visual grounding, token confidence spike (over-trust) -> high penalty
    p_overtrust = opera.predict_hallucination_prob(s_p=0.15, token_probs=[0.99, 0.20, 0.15])
    assert p_overtrust > 0.65


def test_refind_csr_and_contradiction():
    refind = REFINDAdaptationBaseline()

    # High entailment, 0 contradiction -> low hallucination
    nli_faithful = {"entailment": 0.92, "neutral": 0.06, "contradiction": 0.02}
    p_faithful = refind.predict_hallucination_prob("A fact", s_u=0.90, nli_probs=nli_faithful)
    assert p_faithful < 0.20

    # High contradiction -> high hallucination
    nli_contradict = {"entailment": 0.05, "neutral": 0.10, "contradiction": 0.85}
    p_contra = refind.predict_hallucination_prob("A false claim", s_u=0.10, nli_probs=nli_contradict)
    assert p_contra > 0.75


def test_rora_retrieval_margin():
    rora = RoRAVLMBaseline()

    # High margin (top1 is 0.95, distractors 0.1) -> high confidence, low hallucination
    p_robust = rora.predict_hallucination_prob(s_r=0.90, s_u=0.85, passage_scores=[0.95, 0.15, 0.10])
    # Low margin / noise distraction (top1 is 0.40, distractors 0.38)
    p_distracted = rora.predict_hallucination_prob(s_r=0.35, s_u=0.30, passage_scores=[0.40, 0.39, 0.38])

    assert p_robust < p_distracted


def test_realign_dual_preference():
    realign = ReAlignBaseline()

    # Both visual and knowledge aligned
    p_aligned = realign.predict_hallucination_prob(s_p=0.90, s_u=0.92)
    # Cross-modal contradiction (high visual, low utilization or vice versa)
    p_conflict = realign.predict_hallucination_prob(s_p=0.90, s_u=0.10)

    assert p_aligned < 0.20
    assert p_conflict > 0.50


def test_youden_j_threshold_optimization():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.85, 0.75, 0.65, 0.35, 0.25, 0.15, 0.05])

    best_thresh, metrics = tune_decision_threshold(y_true, y_prob, metric="youden_j")

    assert "youden_j" in metrics
    assert metrics["youden_j"] == 1.0  # Perfect separation
    assert 0.35 <= best_thresh <= 0.65
