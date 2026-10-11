"""Unit tests for XMCADiagnosticEngine and root cause attribution."""

from src.combiner.features import compute_cause_attribution
from src.combiner.model import XMCACombiner
from src.diagnostic_engine import XMCADiagnosticEngine


def test_cause_attribution_perception():
    # S_P is low (0.15), while S_R (0.85) and S_U (0.80) are high
    cause, probs = compute_cause_attribution(s_p=0.15, s_r=0.85, s_u=0.80, hallucination_prob=0.82)
    assert cause == "perception_failure"
    assert probs["perception_failure"] > probs["retrieval_failure"]
    assert probs["perception_failure"] > probs["utilization_failure"]


def test_cause_attribution_retrieval():
    # S_R is low (0.10), while S_P (0.90) and S_U (0.85) are high
    cause, probs = compute_cause_attribution(s_p=0.90, s_r=0.10, s_u=0.85, hallucination_prob=0.79)
    assert cause == "retrieval_failure"
    assert probs["retrieval_failure"] > probs["perception_failure"]
    assert probs["retrieval_failure"] > probs["utilization_failure"]


def test_cause_attribution_utilization():
    # S_U is low (0.12), while S_P (0.85) and S_R (0.88) are high
    cause, probs = compute_cause_attribution(s_p=0.85, s_r=0.88, s_u=0.12, hallucination_prob=0.85)
    assert cause == "utilization_failure"
    assert probs["utilization_failure"] > probs["perception_failure"]
    assert probs["utilization_failure"] > probs["retrieval_failure"]


def test_cause_attribution_faithful():
    # High scores across all three consistency signals
    cause, probs = compute_cause_attribution(s_p=0.95, s_r=0.92, s_u=0.90, hallucination_prob=0.08)
    assert cause == "faithful"
    assert probs["faithful"] > 0.85


def test_diagnostic_engine_end_to_end():
    engine = XMCADiagnosticEngine(threshold=0.5)

    # Test diagnosis call
    report = engine.diagnose(
        image="dummy.jpg",
        question="Is there a golden retriever sitting on the couch?",
        answer="A golden retriever is resting on the green couch in the living room.",
        retrieved_passages=["Domestic pets often rest on furniture in living rooms."]
    )

    assert isinstance(report.hallucination_probability, float)
    assert 0.0 <= report.hallucination_probability <= 1.0
    assert report.primary_cause in ["perception_failure", "retrieval_failure", "utilization_failure", "compound_failure", "faithful"]
    assert len(report.claims_audited) >= 1
    assert "remediation" in report.actionable_recommendation.lower() or "no" in report.actionable_recommendation.lower()
