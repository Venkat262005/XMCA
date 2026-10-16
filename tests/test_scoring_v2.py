"""
Comprehensive Unit Tests for Version 2 Grounded Proposition Scoring Engine.

Validates:
1. Positive and negative POPE answers and directional hypothesis construction.
2. Missing detections (faithful negative assertion vs perceptual hallucination).
3. Contradictory, neutral, and entailing evidence in NLI utilization.
4. Malformed questions and failed proposition extraction behavior.
5. Explicit unavailability statuses and reasons.
6. Comparability and version separation (zero silent mixing with v1).
"""

import pytest
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock

from src.scoring.scoring_v2 import (
    SCORING_VERSION,
    extract_queried_entity,
    parse_answer_polarity,
    construct_directional_hypothesis,
    compute_perception_score_v2,
    compute_utilization_score_v2,
    compute_scores_v2,
)


def test_entity_extraction_pope_variations():
    # Standard POPE questions
    e1, r1 = extract_queried_entity("Is there a snowboard in the image?")
    assert e1 == "snowboard"
    assert "success" in r1

    e2, r2 = extract_queried_entity("Is there an elephant in the image?")
    assert e2 == "elephant"
    assert "success" in r2

    e3, r3 = extract_queried_entity("Are there any cars in this image?")
    assert e3 == "cars"
    assert "success" in r3

    e4, r4 = extract_queried_entity("Do you see a dog in the image?")
    assert e4 == "dog"
    assert "success" in r4


def test_entity_extraction_malformed_and_edge_cases():
    # Empty string
    e1, r1 = extract_queried_entity("")
    assert e1 is None
    assert r1 == "empty_or_invalid_question"

    # None
    e2, r2 = extract_queried_entity(None)
    assert e2 is None
    assert r2 == "empty_or_invalid_question"

    # Too short
    e3, r3 = extract_queried_entity("Why?")
    assert e3 is None
    assert r3 == "question_too_short"

    # Malformed unstructured text
    e4, r4 = extract_queried_entity("Random unparseable sentence 123456789")
    assert e4 is None
    assert "failed" in r4


def test_answer_polarity_parsing():
    assert parse_answer_polarity("Yes")[0] == 1
    assert parse_answer_polarity("Yes, there is a person in the image.")[0] == 1
    assert parse_answer_polarity("Yeah")[0] == 1

    assert parse_answer_polarity("No")[0] == -1
    assert parse_answer_polarity("No, there is no backpack in the image.")[0] == -1
    assert parse_answer_polarity("False")[0] == -1

    assert parse_answer_polarity("The image shows a sunny sky.")[0] == 0
    assert parse_answer_polarity("")[0] == 0


def test_directional_hypothesis_construction():
    # Positive POPE
    h1, m1 = construct_directional_hypothesis(
        question="Is there a snowboard in the image?",
        generated_answer="Yes",
        extracted_entity="snowboard"
    )
    assert h1 == "There is a snowboard in the image."
    assert "affirmative" in m1

    # Negative POPE
    h2, m2 = construct_directional_hypothesis(
        question="Is there a backpack in the image?",
        generated_answer="No",
        extracted_entity="backpack"
    )
    assert h2 == "There is no backpack in the image."
    assert "negative" in m2

    # Fallback when entity is missing
    h3, m3 = construct_directional_hypothesis(
        question="Is there a car?",
        generated_answer="Yes",
        extracted_entity=None
    )
    assert "true" in h3.lower()


def test_perception_score_positive_and_negative_cases(tmp_path):
    dummy_img = tmp_path / "test.jpg"
    dummy_img.write_bytes(b"fake_image_bytes")

    mock_detector = MagicMock()
    mock_scorer = MagicMock()

    # Case 1: Positive assertion + object detected -> High S_P
    mock_detector.detect_regions.return_value = (
        [[10, 10, 50, 50]], ["snowboard"], [0.85]
    )
    mock_scorer.compute_s_p.return_value = 0.82
    res_pos = compute_perception_score_v2(
        image_path=str(dummy_img),
        question="Is there a snowboard in the image?",
        generated_answer="Yes, there is a snowboard.",
        detector=mock_detector,
        scorer=mock_scorer
    )
    assert res_pos["status"] == "available"
    assert res_pos["s_p"] == 0.82
    assert res_pos["visual_detected"] is True
    assert res_pos["semantic_alignment"] == "positive_grounded"
    assert res_pos["detections_count"] == 1
    assert res_pos["max_detector_confidence"] == 0.85
    assert res_pos["metric_type"] == "heuristic_consistency_proxy"

    # Case 2: Positive assertion + NO detection (Unsupported by detector) -> S_P = 0.0
    mock_detector.detect_regions.return_value = ([], [], [])
    res_halluc = compute_perception_score_v2(
        image_path=str(dummy_img),
        question="Is there a car in the image?",
        generated_answer="Yes, there is a car.",
        detector=mock_detector,
        scorer=mock_scorer
    )
    assert res_halluc["status"] == "available"
    assert res_halluc["s_p"] == 0.0
    assert res_halluc["visual_detected"] is False
    assert res_halluc["semantic_alignment"] == "positive_unsupported_by_detector"
    assert res_halluc["detections_count"] == 0

    # Case 3: Negative assertion + NO detection (No contradictory visual evidence) -> S_P = 1.0
    mock_detector.detect_regions.return_value = ([], [], [])
    res_neg = compute_perception_score_v2(
        image_path=str(dummy_img),
        question="Is there a backpack in the image?",
        generated_answer="No, there is no backpack.",
        detector=mock_detector,
        scorer=mock_scorer
    )
    assert res_neg["status"] == "available"
    assert res_neg["s_p"] == 0.85
    assert res_neg["visual_detected"] is False
    assert res_neg["semantic_alignment"] == "negative_no_contradictory_evidence"
    assert res_neg["absence_proven_by_detector"] is False
    assert res_neg["max_detector_confidence"] is None
    assert res_neg["detections_count"] == 0

    # Case 4: Negative assertion + clear detection (Omission / contradiction) -> Low S_P
    mock_detector.detect_regions.return_value = (
        [[10, 10, 50, 50]], ["person"], [0.90]
    )
    mock_scorer.compute_s_p.return_value = 0.88
    res_omission = compute_perception_score_v2(
        image_path=str(dummy_img),
        question="Is there a person in the image?",
        generated_answer="No",
        detector=mock_detector,
        scorer=mock_scorer
    )
    assert res_omission["status"] == "available"
    assert res_omission["s_p"] == round(1.0 - 0.88, 4)
    assert res_omission["semantic_alignment"] == "negative_omission_contradicted"
    assert res_omission["detections_count"] == 1


def test_perception_unavailable_on_missing_image_or_malformed_question():
    mock_detector = MagicMock()
    mock_scorer = MagicMock()

    # Missing image
    res_no_img = compute_perception_score_v2(
        image_path="/non/existent/path.jpg",
        question="Is there a car in the image?",
        generated_answer="Yes",
        detector=mock_detector,
        scorer=mock_scorer
    )
    assert res_no_img["s_p"] is None
    assert res_no_img["status"] == "unavailable"
    assert "image_not_found" in res_no_img["reason"]

    # Malformed question
    dummy_img = Path("data/smoke_test_images/COCO_val2014_000000310196.jpg")
    res_bad_q = compute_perception_score_v2(
        image_path=str(dummy_img),
        question="gibberish non question",
        generated_answer="Yes",
        detector=mock_detector,
        scorer=mock_scorer
    )
    assert res_bad_q["s_p"] is None
    assert res_bad_q["status"] == "unavailable"
    assert "entity_extraction_failed" in res_bad_q["reason"]


def test_utilization_score_entailment_contradiction_neutral():
    mock_nli = MagicMock()

    # Case 1: Strongly entailing context
    mock_nli.predict_entailment_probabilities.return_value = [
        {"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01}
    ]
    res_ent = compute_utilization_score_v2(
        retrieved_passages=["The museum collection features an authentic 19th-century snowboard."],
        question="Is there a snowboard in the image?",
        generated_answer="Yes",
        nli_engine=mock_nli,
        extracted_entity="snowboard"
    )
    assert res_ent["status"] == "available"
    # S_U = (1.0 + 0.95 - 0.01) / 2 = 0.97
    assert res_ent["s_u"] == 0.97
    assert res_ent["probabilities"]["entailment"] == 0.95
    assert res_ent["hypothesis"] == "There is a snowboard in the image."

    # Case 2: Strongly contradictory context
    mock_nli.predict_entailment_probabilities.return_value = [
        {"entailment": 0.02, "neutral": 0.03, "contradiction": 0.95}
    ]
    res_contra = compute_utilization_score_v2(
        retrieved_passages=["Snowboards are completely prohibited in this historical archive."],
        question="Is there a snowboard in the image?",
        generated_answer="Yes",
        nli_engine=mock_nli,
        extracted_entity="snowboard"
    )
    # S_U = (1.0 + 0.02 - 0.95) / 2 = 0.035
    assert res_contra["s_u"] == 0.035
    assert res_contra["probabilities"]["contradiction"] == 0.95

    # Case 3: Neutral / uninformative context
    mock_nli.predict_entailment_probabilities.return_value = [
        {"entailment": 0.01, "neutral": 0.98, "contradiction": 0.01}
    ]
    res_neu = compute_utilization_score_v2(
        retrieved_passages=["Artifact_office_11 originated in 19th-century industrial collection."],
        question="Is there a snowboard in the image?",
        generated_answer="No",
        nli_engine=mock_nli,
        extracted_entity="snowboard"
    )
    # S_U = (1.0 + 0.01 - 0.01) / 2 = 0.50
    assert res_neu["s_u"] == 0.50
    assert res_neu["probabilities"]["neutral"] == 0.98
    assert res_neu["hypothesis"] == "There is no snowboard in the image."


def test_scoring_version_tagging_and_integrity():
    mock_detector = MagicMock()
    mock_scorer = MagicMock()
    mock_nli = MagicMock()

    mock_detector.detect_regions.return_value = ([], [], [])
    mock_nli.predict_entailment_probabilities.return_value = [
        {"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1}
    ]

    dummy_img = Path("data/smoke_test_images/COCO_val2014_000000310196.jpg")
    unified = compute_scores_v2(
        image_path=str(dummy_img),
        question="Is there a car in the image?",
        generated_answer="No",
        retrieved_passages=["General industrial context."],
        s_r=0.045,
        detector=mock_detector,
        scorer=mock_scorer,
        nli_engine=mock_nli
    )

    assert unified["scoring_version"] == "v2_proposition_grounded"
    assert unified["s_p"] is not None
    assert unified["s_r"] == 0.045
    assert unified["s_u"] is not None
    assert "perception_details" in unified
    assert "utilization_details" in unified


def test_nli_label_mapping_validation_pass_and_fail():
    from src.utilization.nli import validate_and_resolve_nli_mapping

    # 1. Valid reported MoritzLaurer mapping
    cfg1 = MagicMock(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
    m1 = validate_and_resolve_nli_mapping(cfg1, 3)
    assert m1 == {0: "entailment", 1: "neutral", 2: "contradiction"}

    # 2. Alternative valid RoBERTa mapping
    cfg2 = MagicMock(id2label={0: "contradiction", 1: "neutral", 2: "entailment"})
    m2 = validate_and_resolve_nli_mapping(cfg2, 3)
    assert m2 == {0: "contradiction", 1: "neutral", 2: "entailment"}

    # 3. Missing id2label
    cfg_none = MagicMock(id2label=None)
    with pytest.raises(ValueError, match="missing valid 'id2label'"):
        validate_and_resolve_nli_mapping(cfg_none, 3)

    # 4. Unknown label
    cfg_unknown = MagicMock(id2label={0: "positive", 1: "neutral", 2: "contradiction"})
    with pytest.raises(ValueError, match="Unrecognized NLI class label"):
        validate_and_resolve_nli_mapping(cfg_unknown, 3)

    # 5. Duplicate label
    cfg_dup = MagicMock(id2label={0: "entailment", 1: "entailment", 2: "neutral"})
    with pytest.raises(ValueError, match="Duplicate class mapping"):
        validate_and_resolve_nli_mapping(cfg_dup, 3)

    # 6. Dimension mismatch
    with pytest.raises(ValueError, match="Logits dimension .* does not match"):
        validate_and_resolve_nli_mapping(cfg1, 2)


def test_combiner_features_rejects_none():
    from src.combiner.features import extract_combiner_features

    # Non-null values work cleanly
    feats = extract_combiner_features(s_p=0.5, s_r=0.5, s_u=0.5)
    assert len(feats) == 25

    # None s_p is strictly rejected with ValueError
    with pytest.raises(ValueError, match="s_p is None"):
        extract_combiner_features(s_p=None, s_r=0.5, s_u=0.5)

    # None s_r is strictly rejected
    with pytest.raises(ValueError, match="s_r is None"):
        extract_combiner_features(s_p=0.5, s_r=None, s_u=0.5)

    # None s_u is strictly rejected
    with pytest.raises(ValueError, match="s_u is None"):
        extract_combiner_features(s_p=0.5, s_r=0.5, s_u=None)
