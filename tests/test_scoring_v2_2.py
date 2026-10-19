"""Unit tests for Version 2.2 Generalized Multimodal Grounded Scoring Engine."""

import pytest
from unittest.mock import MagicMock
from src.scoring.scoring_v2_2 import (
    SCORING_VERSION,
    extract_queried_entity_v2_2,
    parse_answer_polarity_v2_2,
    construct_directional_hypothesis_v2_2,
    compute_perception_score_v2_2,
    compute_utilization_score_v2_2,
)


def test_entity_extraction_pope_and_open_domain():
    # POPE existence
    e1, r1 = extract_queried_entity_v2_2("Is there a snowboard in the image?")
    assert e1 == "snowboard"
    assert "success_pope" in r1

    # A-OKVQA demonstrative subject (sample 01200)
    e2, r2 = extract_queried_entity_v2_2("This airline company is headquartered in which country?")
    assert e2 == "airline company"
    assert "success_demonstrative" in r2

    # A-OKVQA open WH subject (sample 01201)
    e3, r3 = extract_queried_entity_v2_2("What is the cat on the toilet lid staring at?")
    assert e3 == "cat"
    assert "success_wh" in r3

    # A-OKVQA preposition noun (sample 01202)
    e4, r4 = extract_queried_entity_v2_2("What is not unique about this animals?")
    assert e4 == "animals"
    assert "success_preposition_noun" in r4

    # A-OKVQA WH preposition subject
    e5, r5 = extract_queried_entity_v2_2("What is in the motorcyclist's mouth?")
    assert "motorcyclist" in e5
    assert "success_wh" in r5


def test_parse_answer_polarity_factual_assertion():
    # Yes / No
    assert parse_answer_polarity_v2_2("Yes, there is.")[0] == 1
    assert parse_answer_polarity_v2_2("No, there isn't.")[0] == -1

    # Open-ended factual assertions are positive assertions of the described facts
    pol, reason = parse_answer_polarity_v2_2("The airline company in the image is headquartered in the United States.")
    assert pol == 1
    assert reason == "factual_assertion"

    pol2, reason2 = parse_answer_polarity_v2_2("The cat on the toilet lid is staring at the snowboarder.")
    assert pol2 == 1
    assert reason2 == "factual_assertion"


def test_utilization_unavailable_on_empty_premise():
    mock_nli = MagicMock()
    res = compute_utilization_score_v2_2(
        premise="",
        question="What is the cat staring at?",
        generated_answer="The cat is staring at the wall.",
        nli_evaluator=mock_nli
    )
    assert res["s_u"] is None
    assert res["status"] == "unavailable_no_context"
    mock_nli.predict_nli_probabilities.assert_not_called()
