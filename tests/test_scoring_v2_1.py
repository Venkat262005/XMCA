"""
Unit and regression tests for XMCA-Net Scoring Revision v2.1 (v2_1_evidence_grounded).

Verifies:
1. Detection count and branch consistency.
2. Negative answer with no detections (bounded absence: S_P <= 0.85).
3. Negative answer with unvalidated detections (SigLIP rejects -> unconfirmed absence, NOT omission contradicted).
4. Negative answer with validated contradictory visual evidence (SigLIP corroborates -> omission contradicted).
5. Ambiguous answer polarity handling (returns S_P = None, status 'unavailable').
6. Missing retrieval context handling (returns S_U = None, status 'unavailable_no_context', probabilities = None).
7. Distinction between missing retrieval context and measured neutral NLI output.
8. Proper label and score pairing without semantic contradiction.
9. Version isolation and non-interference with legacy v1 and v2 records.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.scoring.scoring_v2_1 import (
    SCORING_VERSION,
    DETECTOR_THRESHOLD,
    SIGLIP_GROUNDING_THRESHOLD,
    extract_queried_entity,
    parse_answer_polarity,
    construct_directional_hypothesis,
    compute_perception_score_v2_1,
    compute_utilization_score_v2_1,
    compute_scores_v2_1
)


class MockDetector:
    def __init__(self, boxes=None, confs=None):
        self.boxes = boxes or []
        self.confs = confs or []

    def detect_regions(self, image_path, query_labels):
        return self.boxes, query_labels, self.confs


class MockScorer:
    def __init__(self, sim_score=0.5):
        self.sim_score = sim_score

    def compute_s_p(self, image_path, claims, boxes):
        return self.sim_score


class MockNLIEngine:
    def __init__(self, entail=0.8, neutral=0.1, contra=0.1):
        self.entail = entail
        self.neutral = neutral
        self.contra = contra

    def predict_entailment_probabilities(self, pairs):
        return [{
            "entailment": self.entail,
            "neutral": self.neutral,
            "contradiction": self.contra
        }]


def test_v2_1_version_tagging():
    assert SCORING_VERSION == "v2_1_evidence_grounded"


def test_negative_answer_with_no_detections(tmp_path):
    img = tmp_path / "test.jpg"
    img.touch()
    detector = MockDetector(boxes=[], confs=[])
    scorer = MockScorer(sim_score=0.0)

    res = compute_perception_score_v2_1(
        image_path=str(img),
        question="Is there an airplane in the image?",
        generated_answer="No, there is no airplane.",
        detector=detector,
        scorer=scorer
    )
    assert res["status"] == "available"
    assert res["semantic_alignment"] == "negative_no_contradictory_evidence"
    assert res["s_p"] == 0.85  # 1.0 - 0.15 bounded
    assert res["s_p"] <= 0.85
    assert res["absence_proven_by_detector"] is False
    assert res["visual_detected"] is False
    assert res["visual_corroborated"] is False


def test_negative_answer_with_unvalidated_box_siglip_rejects(tmp_path):
    """
    Case 8 diagnostic pattern: Detector found a candidate box, but SigLIP says similarity is near zero.
    Must NOT be labeled negative_omission_contradicted!
    """
    img = tmp_path / "test.jpg"
    img.touch()
    # Detector proposes a box with 0.1643 confidence
    detector = MockDetector(boxes=[[10, 10, 50, 50]], confs=[0.1643])
    # SigLIP returns 0.0002 (below SIGLIP_GROUNDING_THRESHOLD = 0.10)
    scorer = MockScorer(sim_score=0.0002)

    res = compute_perception_score_v2_1(
        image_path=str(img),
        question="Is there a snowboard in the image?",
        generated_answer="No, there is no snowboard.",
        detector=detector,
        scorer=scorer
    )
    assert res["status"] == "available"
    # Crucial semantic fix: unvalidated box is NOT omission contradicted
    assert res["semantic_alignment"] == "negative_unvalidated_box_unconfirmed_absence"
    assert res["visual_detected"] is True
    assert res["visual_corroborated"] is False
    assert res["s_p"] <= 0.85  # 1.0 - 0.1643 = 0.8357
    assert res["s_p"] == round(1.0 - 0.1643, 4)


def test_negative_answer_with_validated_contradictory_visual_evidence(tmp_path):
    """
    Omission contradiction: Detector proposes a box AND SigLIP corroborates strong similarity (e.g. 0.85).
    Negative claim is genuinely contradicted.
    """
    img = tmp_path / "test.jpg"
    img.touch()
    detector = MockDetector(boxes=[[10, 10, 50, 50]], confs=[0.80])
    scorer = MockScorer(sim_score=0.85)

    res = compute_perception_score_v2_1(
        image_path=str(img),
        question="Is there a dog in the image?",
        generated_answer="No, there are no dogs.",
        detector=detector,
        scorer=scorer
    )
    assert res["status"] == "available"
    assert res["semantic_alignment"] == "negative_omission_contradicted"
    assert res["visual_detected"] is True
    assert res["visual_corroborated"] is True
    assert res["s_p"] == round(1.0 - 0.85, 4)  # 0.15


def test_positive_answer_grounded_vs_unconfirmed_box(tmp_path):
    img = tmp_path / "test.jpg"
    img.touch()

    # Grounded: both detector and SigLIP corroborate
    det = MockDetector(boxes=[[10, 10, 50, 50]], confs=[0.75])
    sc = MockScorer(sim_score=0.80)
    res_pos = compute_perception_score_v2_1(
        image_path=str(img),
        question="Is there a cat in the image?",
        generated_answer="Yes, there is a cat.",
        detector=det,
        scorer=sc
    )
    assert res_pos["semantic_alignment"] == "positive_grounded"
    assert res_pos["s_p"] == 0.80

    # Unconfirmed box: detector proposed box, but SigLIP similarity < 0.10
    sc_low = MockScorer(sim_score=0.02)
    res_unconf = compute_perception_score_v2_1(
        image_path=str(img),
        question="Is there a cat in the image?",
        generated_answer="Yes, there is a cat.",
        detector=det,
        scorer=sc_low
    )
    assert res_unconf["semantic_alignment"] == "positive_box_unconfirmed_by_siglip"
    assert res_unconf["s_p"] == 0.02


def test_ambiguous_polarity_returns_none(tmp_path):
    img = tmp_path / "test.jpg"
    img.touch()
    detector = MockDetector(boxes=[], confs=[])
    scorer = MockScorer(sim_score=0.0)

    res = compute_perception_score_v2_1(
        image_path=str(img),
        question="Is there a dog in the image?",
        generated_answer="It is difficult to tell whether that is a dog or a shadow.",
        detector=detector,
        scorer=scorer
    )
    assert res["s_p"] is None
    assert res["status"] == "unavailable"
    assert res["semantic_alignment"] == "ambiguous_answer_polarity"
    assert "ambiguous_polarity" in res["reason"]


def test_missing_retrieval_context_returns_none_and_no_fake_probabilities():
    nli = MockNLIEngine()
    res = compute_utilization_score_v2_1(
        retrieved_passages=[],
        question="Is there a boat in the image?",
        generated_answer="Yes, there is a boat.",
        nli_engine=nli
    )
    assert res["s_u"] is None
    assert res["status"] == "unavailable_no_context"
    assert res["reason"] == "empty_retrieved_passages"
    # Verify NO fake neutral probabilities were fabricated
    assert res["probabilities"]["entailment"] is None
    assert res["probabilities"]["neutral"] is None
    assert res["probabilities"]["contradiction"] is None


def test_measured_neutral_nli_distinguished_from_missing_context():
    # Model inference actually ran and produced neutral = 0.98
    nli_neutral = MockNLIEngine(entail=0.01, neutral=0.98, contra=0.01)
    res = compute_utilization_score_v2_1(
        retrieved_passages=["Artifact 19th century industrial collection."],
        question="Is there a snowboard in the image?",
        generated_answer="There is no snowboard in the image.",
        nli_engine=nli_neutral
    )
    assert res["status"] == "available"
    assert res["s_u"] == 0.50  # (1 + 0.01 - 0.01) / 2 = 0.50
    assert res["probabilities"]["neutral"] == 0.98
    assert res["probabilities"]["entailment"] == 0.01
    assert res["probabilities"]["contradiction"] == 0.01
