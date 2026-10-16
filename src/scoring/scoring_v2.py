"""
XMCA-Net — Version 2 Grounded Proposition Scoring Engine.

Addresses the semantic limitations of v1 token-level scoring:
1. Perception (S_P v2):
   - Derives physical queried entity/concept from the question rather than passing raw 'Yes'/'No' particles to OWL-ViT.
   - Evaluates grounded visual evidence (detector confidence + SigLIP similarity).
   - Compares grounded visual evidence directionally with generated answer assertion.
   - Records explicit 'unavailable' status and reason if entity extraction or image loading fails.

2. Utilization (S_U v2):
   - Constructs a directionally complete proposition hypothesis rather than evaluating isolated particles.
   - Evaluates premise-hypothesis pair using DeBERTa-v3 cross-encoder NLI.
   - Retains entailment, neutral, and contradiction probabilities separately.
   - Computes balanced directional support formula:
       S_U = (1.0 + p_entailment - p_contradiction) / 2.0  in [0, 1]

3. Versioning & Comparability:
   - Tags all outputs with scoring_version='v2_proposition_grounded'.
   - Prevents silent mixing with legacy 'v1_raw_token' scores.
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCORING_VERSION = "v2_proposition_grounded"


def extract_queried_entity(question: str) -> Tuple[Optional[str], str]:
    """
    Extract the queried target entity from visual questions (e.g. POPE binary queries).
    
    Returns:
        (entity_str, status_reason): Extracted noun phrase or None with reason.
    """
    if not question or not isinstance(question, str):
        return None, "empty_or_invalid_question"

    cleaned = question.strip()
    if len(cleaned) < 5:
        return None, "question_too_short"

    # Pattern 1: POPE standard "Is there a/an/any <entity> in/on/at the image?"
    pattern_pope = re.compile(
        r"^(?:is|are)\s+there\s+(?:an\b|a\b|any\b|the\b)?\s*([a-zA-Z0-9_\-\s]+?)\s+(?:in|on|at|around)\s+(?:the|this)?\s*image\??$",
        re.IGNORECASE
    )
    m = pattern_pope.match(cleaned)
    if m:
        entity = m.group(1).strip()
        if entity:
            return entity.lower(), "success_pope_pattern"

    # Pattern 2: "Do you see a/an <entity> in the image?"
    pattern_see = re.compile(
        r"^(?:do\s+you\s+see|can\s+you\s+see)\s+(?:an\b|a\b|any\b|the\b)?\s*([a-zA-Z0-9_\-\s]+?)\s+(?:in|on|at)?\s*(?:the|this)?\s*image\??$",
        re.IGNORECASE
    )
    m2 = pattern_see.match(cleaned)
    if m2:
        entity = m2.group(1).strip()
        if entity:
            return entity.lower(), "success_see_pattern"

    # Pattern 3: General "What is in the <noun>?" or "What <noun> is ...?"
    pattern_what = re.compile(r"^what\s+(?:is|are|type\s+of)?\s*([a-zA-Z0-9_\-\s]+?)\b", re.IGNORECASE)
    m3 = pattern_what.match(cleaned)
    if m3:
        entity = m3.group(1).strip()
        if len(entity) > 2 and entity.lower() not in {"is", "the", "this", "that"}:
            return entity.lower(), "success_wh_pattern"

    return None, "failed_unrecognized_question_syntax"


def parse_answer_polarity(generated_answer: str) -> Tuple[int, str]:
    """
    Parse generated answer polarity:
    +1: Affirmative (entity claimed present/true)
    -1: Negative (entity claimed absent/false)
     0: Ambiguous or unparseable
    """
    if not generated_answer or not isinstance(generated_answer, str):
        return 0, "empty_answer"

    text = generated_answer.strip().lower()
    # Normalize leading tokens
    first_word = re.sub(r"[^\w]", "", text.split()[0]) if text.split() else ""

    if first_word in {"yes", "yeah", "yep", "true", "correct"}:
        return 1, "affirmative"
    if first_word in {"no", "nope", "false", "incorrect"}:
        return -1, "negative"

    if re.search(r"\bthere\s+is\s+(?:a|an)\b", text) and not re.search(r"\bno\b|\bnot\b", text):
        return 1, "affirmative_phrase"
    if re.search(r"\bthere\s+(?:is|are)\s+no\b|\bnot\s+present\b|\bnot\s+in\b", text):
        return -1, "negative_phrase"

    return 0, "ambiguous_polarity"


def construct_directional_hypothesis(
    question: str,
    generated_answer: str,
    extracted_entity: Optional[str]
) -> Tuple[str, str]:
    """
    Construct a complete, directionally sound proposition hypothesis for NLI.
    
    Returns:
        (hypothesis_text, construction_method)
    """
    polarity, pol_reason = parse_answer_polarity(generated_answer)

    if extracted_entity:
        if polarity == 1:
            return f"There is a {extracted_entity} in the image.", f"entity_affirmative_{pol_reason}"
        elif polarity == -1:
            return f"There is no {extracted_entity} in the image.", f"entity_negative_{pol_reason}"

    # Fallback when entity not explicitly extracted or answer is complex
    clean_ans = generated_answer.strip()
    if clean_ans.lower() in {"yes", "no"}:
        # If bare yes/no and entity extraction failed
        clean_q = question.strip().rstrip("?")
        if polarity == 1:
            return f"It is true that {clean_q}.", "bare_yes_fallback"
        elif polarity == -1:
            return f"It is false that {clean_q}.", "bare_no_fallback"

    # Default to direct answer sentence
    return clean_ans, "raw_sentence_hypothesis"


def compute_perception_score_v2(
    image_path: str,
    question: str,
    generated_answer: str,
    detector: Any,
    scorer: Any
) -> Dict[str, Any]:
    """
    Version 2 Perception Score (S_P v2):
    1. Extracts queried entity from question.
    2. Runs detector and visual alignment on entity.
    3. Evaluates directional consistency between visual evidence and model assertion.
    """
    if not Path(image_path).exists():
        return {
            "s_p": None,
            "status": "unavailable",
            "reason": f"image_not_found: {image_path}",
            "entity": None,
            "visual_evidence_score": None,
            "visual_detected": False,
            "polarity": 0
        }

    entity, ext_status = extract_queried_entity(question)
    polarity, pol_status = parse_answer_polarity(generated_answer)

    if not entity:
        return {
            "s_p": None,
            "status": "unavailable",
            "reason": f"entity_extraction_failed: {ext_status}",
            "entity": None,
            "visual_evidence_score": None,
            "visual_detected": False,
            "polarity": polarity
        }

    try:
        # Detect relevant visual evidence for the extracted target entity
        boxes, labels, confs = detector.detect_regions(image_path, [entity])
        has_detection = len(boxes) > 0 and any(c >= 0.15 for c in confs)
        max_conf = max(confs) if confs else 0.0

        if has_detection:
            # Score visual grounding with SigLIP
            siglip_sim = float(scorer.compute_s_p(image_path, [entity], boxes))
            vis_score = max(0.0, min(1.0, siglip_sim))
        else:
            vis_score = 0.0

        # Heuristic directional consistency alignment:
        # NOTE: Non-detection by an open-vocabulary detector is NOT ground-truth proof
        # of physical absence; it indicates absence of corroborating or contradicting visual evidence.
        if polarity == 1:
            # Model asserts entity is PRESENT
            # High score if visually detected; zero score if absent above threshold
            s_p = vis_score if has_detection else 0.0
            alignment_type = "positive_grounded" if has_detection else "positive_unsupported_by_detector"
        elif polarity == -1:
            # Model asserts entity is ABSENT
            # NOTE: Non-detection by an open-vocabulary detector is NOT proof of physical absence.
            # When no detections are found, we bound S_P by the detector's operational threshold (0.15)
            # so unconfirmed absence never achieves a perfect 1.0 certainty score.
            if not has_detection:
                eff_conf = max_conf if max_conf > 0.0 else 0.15
                s_p = float(1.0 - eff_conf)
                alignment_type = "negative_no_contradictory_evidence"
            else:
                # Confirmed visual detection contradicts the negative claim (omission error)
                s_p = float(max(0.0, 1.0 - vis_score))
                alignment_type = "negative_omission_contradicted"
        else:
            # Ambiguous answer polarity: score directly by detected visual presence
            s_p = vis_score
            alignment_type = "ambiguous_polarity_grounding"

        return {
            "s_p": round(float(s_p), 4),
            "status": "available",
            "semantic_alignment": alignment_type,
            "entity": entity,
            "visual_evidence_score": round(vis_score, 4),
            "visual_detected": has_detection,
            "detections_count": len(boxes),
            "raw_confidences": [round(float(c), 4) for c in confs],
            "max_detector_confidence": round(float(max_conf), 4) if confs else None,
            "detector_threshold": 0.15,
            "absence_proven_by_detector": False,
            "bounding_boxes": boxes,
            "polarity": polarity,
            "metric_type": "heuristic_consistency_proxy",
            "scoring_version": SCORING_VERSION
        }

    except Exception as e:
        logger.error(f"Perception v2 computation error on {image_path}: {e}")
        return {
            "s_p": None,
            "status": "unavailable",
            "semantic_alignment": "detector_execution_error",
            "reason": str(e),
            "entity": entity,
            "visual_evidence_score": None,
            "visual_detected": False,
            "detections_count": 0,
            "raw_confidences": [],
            "max_detector_confidence": 0.0,
            "bounding_boxes": [],
            "polarity": polarity,
            "metric_type": "heuristic_consistency_proxy",
            "scoring_version": SCORING_VERSION
        }


def compute_utilization_score_v2(
    retrieved_passages: List[str],
    question: str,
    generated_answer: str,
    nli_engine: Any,
    extracted_entity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Version 2 Utilization Score (S_U v2):
    1. Builds complete directional hypothesis.
    2. Evaluates DeBERTa-v3 NLI on (premise, hypothesis).
    3. Preserves entailment, neutral, and contradiction probabilities.
    4. Computes normalized directional support formula:
         S_U = (1.0 + p_entailment - p_contradiction) / 2.0 in [0, 1]
    """
    if not retrieved_passages:
        return {
            "s_u": 0.5,
            "status": "neutral_no_context",
            "reason": "empty_retrieved_passages",
            "probabilities": {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0},
            "hypothesis": generated_answer,
            "scoring_version": SCORING_VERSION
        }

    if extracted_entity is None:
        extracted_entity, _ = extract_queried_entity(question)

    hypothesis, constr_method = construct_directional_hypothesis(
        question=question,
        generated_answer=generated_answer,
        extracted_entity=extracted_entity
    )
    premise = retrieved_passages[0].strip()

    try:
        nli_res = nli_engine.predict_entailment_probabilities([(premise, hypothesis)])
        probs = nli_res[0] if nli_res else {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

        p_ent = float(probs.get("entailment", 0.0))
        p_neu = float(probs.get("neutral", 0.0))
        p_con = float(probs.get("contradiction", 0.0))

        # Normalized directional consistency:
        # Range is strictly [0.0, 1.0]
        # p_ent=1.0, p_con=0.0 -> S_U = 1.0 (strongly supported)
        # p_ent=0.0, p_con=1.0 -> S_U = 0.0 (strongly contradicted)
        # p_neu=1.0, p_ent=0.0, p_con=0.0 -> S_U = 0.5 (neutral/unsupported evidence)
        s_u = float((1.0 + p_ent - p_con) / 2.0)
        s_u_bounded = max(0.0, min(1.0, s_u))

        return {
            "s_u": round(s_u_bounded, 4),
            "status": "available",
            "semantic_support": f"nli_evaluated_{constr_method}",
            "probabilities": {
                "entailment": round(p_ent, 4),
                "neutral": round(p_neu, 4),
                "contradiction": round(p_con, 4)
            },
            "hypothesis": hypothesis,
            "premise_snippet": premise[:80] + ("..." if len(premise) > 80 else ""),
            "metric_type": "textual_semantic_entailment_proxy",
            "causal_attribution_claimed": False,
            "scoring_version": SCORING_VERSION
        }

    except Exception as e:
        logger.error(f"Utilization v2 computation error: {e}")
        return {
            "s_u": None,
            "status": "unavailable",
            "reason": f"nli_execution_error: {str(e)}",
            "probabilities": {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0},
            "hypothesis": hypothesis,
            "scoring_version": SCORING_VERSION
        }


def compute_scores_v2(
    image_path: str,
    question: str,
    generated_answer: str,
    retrieved_passages: List[str],
    s_r: float,
    detector: Any,
    scorer: Any,
    nli_engine: Any
) -> Dict[str, Any]:
    """
    Compute unified Version 2 scores with component details and explicit metadata.
    """
    perc_res = compute_perception_score_v2(
        image_path=image_path,
        question=question,
        generated_answer=generated_answer,
        detector=detector,
        scorer=scorer
    )
    util_res = compute_utilization_score_v2(
        retrieved_passages=retrieved_passages,
        question=question,
        generated_answer=generated_answer,
        nli_engine=nli_engine,
        extracted_entity=perc_res.get("entity")
    )

    return {
        "scoring_version": SCORING_VERSION,
        "s_p": perc_res.get("s_p"),
        "s_r": round(float(s_r), 4),
        "s_u": util_res.get("s_u"),
        "perception_details": perc_res,
        "utilization_details": util_res
    }
