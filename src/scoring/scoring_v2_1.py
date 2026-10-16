"""
XMCA-Net — Version 2.1 Evidence-Grounded Proposition Scoring Engine.

Implements rigorous semantic distinctions for multimodal consistency proxies:
1. Perception (S_P v2.1):
   - Derives physical queried entity/concept from the question syntax.
   - Distinguishes:
       a. Detector proposed a region (OWL-ViT bounding boxes).
       b. Detector confidence exceeded threshold (tau_det = 0.15).
       c. SigLIP visual grounding score indicates meaningful alignment (tau_vis = 0.10).
       d. Answer assertion polarity (+1, -1, 0).
   - Unvalidated boxes (box detected but SigLIP similarity < tau_vis) do NOT trigger
     'negative_omission_contradicted' for negative claims.
   - Non-detection is bounded by detector threshold: S_P <= 0.85, never 1.0.
   - Ambiguous polarity (0) returns S_P = None with status 'unavailable' (does not penalize hedging as 0.0).
   - Missing images or entity extraction failures return S_P = None with explicit reasons.

2. Utilization (S_U v2.1):
   - Constructs directionally complete proposition hypothesis.
   - Evaluates (premise, hypothesis) via DeBERTa-v3 cross-encoder.
   - Dynamically validates NLI label mapping and output dimensions.
   - Empty retrieval context explicitly returns S_U = None with status 'unavailable_no_context'
     and probabilities = None (no fake 1.0 neutral probability fabrication).
   - S_U formula: (1.0 + p_entailment - p_contradiction) / 2.0 in [0, 1].

3. Provenance & Versioning:
   - Tagged as SCORING_VERSION = 'v2_1_evidence_grounded'.
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCORING_VERSION = "v2_1_evidence_grounded"
DETECTOR_THRESHOLD = 0.15
SIGLIP_GROUNDING_THRESHOLD = 0.10


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
    """
    polarity, pol_reason = parse_answer_polarity(generated_answer)

    if extracted_entity:
        if polarity == 1:
            return f"There is a {extracted_entity} in the image.", f"entity_affirmative_{pol_reason}"
        elif polarity == -1:
            return f"There is no {extracted_entity} in the image.", f"entity_negative_{pol_reason}"

    clean_ans = generated_answer.strip()
    if clean_ans.lower() in {"yes", "no"}:
        clean_q = question.strip().rstrip("?")
        if polarity == 1:
            return f"It is true that {clean_q}.", "bare_yes_fallback"
        elif polarity == -1:
            return f"It is false that {clean_q}.", "bare_no_fallback"

    return clean_ans, "raw_sentence_hypothesis"


def compute_perception_score_v2_1(
    image_path: str,
    question: str,
    generated_answer: str,
    detector: Any,
    scorer: Any
) -> Dict[str, Any]:
    """
    Version 2.1 Perception Score (S_P v2.1):
    Rigorously decouples:
    1. Detector presence (boxes found)
    2. Operational threshold attainment (max_conf >= 0.15)
    3. SigLIP visual grounding validation (vis_score >= 0.10)
    4. Directional consistency alignment
    """
    if not Path(image_path).exists():
        return {
            "s_p": None,
            "status": "unavailable",
            "semantic_alignment": "image_missing",
            "reason": f"image_not_found: {image_path}",
            "entity": None,
            "visual_evidence_score": None,
            "visual_detected": False,
            "visual_corroborated": False,
            "detections_count": 0,
            "raw_confidences": [],
            "max_detector_confidence": None,
            "bounding_boxes": [],
            "polarity": 0,
            "metric_type": "heuristic_consistency_proxy",
            "scoring_version": SCORING_VERSION
        }

    entity, ext_status = extract_queried_entity(question)
    polarity, pol_status = parse_answer_polarity(generated_answer)

    if not entity:
        return {
            "s_p": None,
            "status": "unavailable",
            "semantic_alignment": "entity_extraction_failed",
            "reason": f"entity_extraction_failed: {ext_status}",
            "entity": None,
            "visual_evidence_score": None,
            "visual_detected": False,
            "visual_corroborated": False,
            "detections_count": 0,
            "raw_confidences": [],
            "max_detector_confidence": None,
            "bounding_boxes": [],
            "polarity": polarity,
            "metric_type": "heuristic_consistency_proxy",
            "scoring_version": SCORING_VERSION
        }

    try:
        boxes, labels, confs = detector.detect_regions(image_path, [entity])
        has_detection = len(boxes) > 0 and any(c >= DETECTOR_THRESHOLD for c in confs)
        max_conf = max(confs) if confs else 0.0

        if has_detection:
            siglip_sim = float(scorer.compute_s_p(image_path, [entity], boxes))
            vis_score = max(0.0, min(1.0, siglip_sim))
        else:
            vis_score = 0.0

        # Check whether SigLIP corroborates that the detected region depicts the entity
        is_corroborated = has_detection and (vis_score >= SIGLIP_GROUNDING_THRESHOLD)

        # Directional consistency proxy logic
        if polarity == 1:
            # Model asserts entity is PRESENT
            if is_corroborated:
                s_p = vis_score
                alignment_type = "positive_grounded"
            elif has_detection:
                # Detector proposed a box, but SigLIP similarity is below grounding threshold
                s_p = vis_score
                alignment_type = "positive_box_unconfirmed_by_siglip"
            else:
                s_p = 0.0
                alignment_type = "positive_unsupported_by_detector"

        elif polarity == -1:
            # Model asserts entity is ABSENT
            if is_corroborated:
                # Both detector AND SigLIP confirm presence -> contradicts negative claim (omission error)
                s_p = float(max(0.0, 1.0 - vis_score))
                alignment_type = "negative_omission_contradicted"
            elif has_detection:
                # Detector proposed a box, but SigLIP rejects entity alignment.
                # Crucial fix: Do NOT call it omission contradicted when visual evidence is rejected by SigLIP!
                eff_conf = max_conf if max_conf > 0.0 else DETECTOR_THRESHOLD
                s_p = float(1.0 - eff_conf)
                alignment_type = "negative_unvalidated_box_unconfirmed_absence"
            else:
                # Zero boxes found -> non-detection does NOT prove physical absence
                s_p = float(1.0 - DETECTOR_THRESHOLD)  # 0.85
                alignment_type = "negative_no_contradictory_evidence"

        else:
            # Ambiguous answer polarity (e.g., model hedges "It is difficult to tell...")
            # Forcing 0.0 or 1.0 penalizes epistemic hedging. Mark unavailable for polarity ambiguity.
            return {
                "s_p": None,
                "status": "unavailable",
                "semantic_alignment": "ambiguous_answer_polarity",
                "reason": "cannot_evaluate_directional_consistency_for_ambiguous_polarity",
                "entity": entity,
                "visual_evidence_score": round(vis_score, 4),
                "visual_detected": has_detection,
                "visual_corroborated": is_corroborated,
                "detections_count": len(boxes),
                "raw_confidences": [round(float(c), 4) for c in confs],
                "max_detector_confidence": round(float(max_conf), 4) if confs else None,
                "detector_threshold": DETECTOR_THRESHOLD,
                "siglip_grounding_threshold": SIGLIP_GROUNDING_THRESHOLD,
                "absence_proven_by_detector": False,
                "bounding_boxes": boxes,
                "polarity": polarity,
                "metric_type": "heuristic_consistency_proxy",
                "scoring_version": SCORING_VERSION
            }

        return {
            "s_p": round(float(s_p), 4),
            "status": "available",
            "semantic_alignment": alignment_type,
            "entity": entity,
            "visual_evidence_score": round(vis_score, 4),
            "visual_detected": has_detection,
            "visual_corroborated": is_corroborated,
            "detections_count": len(boxes),
            "raw_confidences": [round(float(c), 4) for c in confs],
            "max_detector_confidence": round(float(max_conf), 4) if confs else None,
            "detector_threshold": DETECTOR_THRESHOLD,
            "siglip_grounding_threshold": SIGLIP_GROUNDING_THRESHOLD,
            "absence_proven_by_detector": False,
            "bounding_boxes": boxes,
            "polarity": polarity,
            "metric_type": "heuristic_consistency_proxy",
            "scoring_version": SCORING_VERSION
        }

    except Exception as e:
        logger.error(f"Perception v2.1 computation error on {image_path}: {e}")
        return {
            "s_p": None,
            "status": "unavailable",
            "semantic_alignment": "detector_execution_error",
            "reason": str(e),
            "entity": entity,
            "visual_evidence_score": None,
            "visual_detected": False,
            "visual_corroborated": False,
            "detections_count": 0,
            "raw_confidences": [],
            "max_detector_confidence": None,
            "bounding_boxes": [],
            "polarity": polarity,
            "metric_type": "heuristic_consistency_proxy",
            "scoring_version": SCORING_VERSION
        }


def compute_utilization_score_v2_1(
    retrieved_passages: List[str],
    question: str,
    generated_answer: str,
    nli_engine: Any,
    extracted_entity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Version 2.1 Utilization Score (S_U v2.1):
    Evaluates semantic directional consistency.
    If retrieved_passages is empty, returns S_U = None with status 'unavailable_no_context'
    and probabilities = None. No fake neutral probabilities are synthesized.
    """
    # Strict missing context handling:
    valid_passages = [p.strip() for p in (retrieved_passages or []) if p and isinstance(p, str) and p.strip()]
    if not valid_passages:
        return {
            "s_u": None,
            "status": "unavailable_no_context",
            "semantic_support": "no_retrieval_context",
            "reason": "empty_retrieved_passages",
            "probabilities": {
                "entailment": None,
                "neutral": None,
                "contradiction": None
            },
            "hypothesis": construct_directional_hypothesis(question, generated_answer, extracted_entity)[0],
            "premise_snippet": None,
            "metric_type": "textual_semantic_entailment_proxy",
            "causal_attribution_claimed": False,
            "scoring_version": SCORING_VERSION
        }

    if extracted_entity is None:
        extracted_entity, _ = extract_queried_entity(question)

    hypothesis, constr_method = construct_directional_hypothesis(
        question=question,
        generated_answer=generated_answer,
        extracted_entity=extracted_entity
    )
    premise = valid_passages[0]

    try:
        nli_res = nli_engine.predict_entailment_probabilities([(premise, hypothesis)])
        if not nli_res:
            raise RuntimeError("NLI engine returned empty probability list.")
        probs = nli_res[0]

        p_ent = float(probs["entailment"])
        p_neu = float(probs["neutral"])
        p_con = float(probs["contradiction"])

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
        logger.error(f"Utilization v2.1 computation error: {e}")
        return {
            "s_u": None,
            "status": "unavailable",
            "semantic_support": "nli_error",
            "reason": f"nli_execution_error: {str(e)}",
            "probabilities": {
                "entailment": None,
                "neutral": None,
                "contradiction": None
            },
            "hypothesis": hypothesis,
            "premise_snippet": premise[:80] + ("..." if len(premise) > 80 else ""),
            "metric_type": "textual_semantic_entailment_proxy",
            "causal_attribution_claimed": False,
            "scoring_version": SCORING_VERSION
        }


def compute_scores_v2_1(
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
    Compute unified Version 2.1 scores with decoupled grounding and context availability.
    """
    perc_res = compute_perception_score_v2_1(
        image_path=image_path,
        question=question,
        generated_answer=generated_answer,
        detector=detector,
        scorer=scorer
    )
    util_res = compute_utilization_score_v2_1(
        retrieved_passages=retrieved_passages,
        question=question,
        generated_answer=generated_answer,
        nli_engine=nli_engine,
        extracted_entity=perc_res.get("entity")
    )

    return {
        "scoring_version": SCORING_VERSION,
        "s_p": perc_res.get("s_p"),
        "s_r": round(float(s_r), 4) if s_r is not None else None,
        "s_u": util_res.get("s_u"),
        "perception_details": perc_res,
        "utilization_details": util_res
    }
