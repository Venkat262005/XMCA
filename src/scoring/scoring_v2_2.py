"""
XMCA-Net — Version 2.2 Generalized Multimodal Grounded Scoring Engine.
Extends v2.1 to handle open-domain questions while strictly preserving unavailable states.

Maintains strict compliance with research boundaries:
1. Preserves explicit 'unavailable' / None status when images, entities, or premises are missing.
2. Bounded negative absence: S_P <= 0.85 (non-detection does NOT prove physical absence).
3. SigLIP visual grounding threshold tau_vis = 0.10.
4. Separate scoring_version tag 'v2_2_multimodal_grounded'.
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCORING_VERSION = "v2_2_multimodal_grounded"
DETECTOR_THRESHOLD = 0.15
SIGLIP_GROUNDING_THRESHOLD = 0.10


def extract_queried_entity_v2_2(question: str, generated_answer: str = "") -> Tuple[Optional[str], str]:
    """
    Extracts the physical entity to ground from question syntax or answer claims.
    Returns (entity, reason). Does NOT invent entities.
    """
    if not question or not isinstance(question, str):
        return None, "empty_or_invalid_question"

    cleaned = question.strip().rstrip("?")
    if len(cleaned) < 5:
        return None, "question_too_short"

    # Pattern 1: Standard POPE existence queries
    m_pope = re.search(
        r"^(?:is|are)\s+there\s+(?:an\b|a\b|any\b|the\b)?\s*([a-zA-Z0-9_\-\s]+?)\s+(?:in|on|at|around)\s+(?:the|this)?\s*(?:image|picture|photo|scene)\??$",
        cleaned,
        re.IGNORECASE
    )
    if m_pope:
        entity = m_pope.group(1).strip()
        if entity:
            return entity.lower(), "success_pope_syntax"

    # Pattern 2: Perception queries ("Do you see a...")
    m_see = re.search(
        r"^(?:do\s+you\s+see|can\s+you\s+see)\s+(?:an\b|a\b|any\b|the\b)?\s*([a-zA-Z0-9_\-\s]+?)\s+(?:in|on|at)?\s*(?:the|this)?\s*(?:image|picture|photo|scene)\??$",
        cleaned,
        re.IGNORECASE
    )
    if m_see:
        entity = m_see.group(1).strip()
        if entity:
            return entity.lower(), "success_see_syntax"

    # Pattern 3: Loose POPE variations (typos or non-standard prepositions)
    m_loose = re.search(
        r"^(?:is|are)\s+there\s+(?:an\b|a\b|any\b|the\b)?\s*([a-zA-Z0-9_\-\s]+?)\s+(?:in|on|at|around)\b",
        cleaned,
        re.IGNORECASE
    )
    if m_loose:
        entity = m_loose.group(1).strip()
        if entity:
            return entity.lower(), "success_pope_loose_syntax"

    # Pattern 4: Demonstrative subject queries ("This <noun> is..." e.g. "This airline company is headquartered...")
    m_dem = re.search(
        r"\b(?:this|that|the)\s+([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+)?)\s+(?:is|are|was|were|has|have)\b",
        cleaned,
        re.IGNORECASE
    )
    if m_dem and m_dem.group(1).lower() not in {"image", "picture", "photo", "scene"}:
        entity = m_dem.group(1).strip().lower()
        # Strip trailing preposition if any
        entity = re.sub(r"\s+(?:in|on|at|by|with|for|from|to|about|of)$", "", entity).strip()
        if len(entity) > 2:
            return entity, "success_demonstrative_syntax"

    # Pattern 5: Open-ended WH questions with specific visual subject ("What is the <noun> staring at/doing?")
    m_wh = re.search(
        r"\bwhat\s+(?:is|are)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+)?)\b",
        cleaned,
        re.IGNORECASE
    )
    if m_wh:
        cand = m_wh.group(1).strip().lower()
        # Strip trailing preposition if any
        cand = re.sub(r"\s+(?:in|on|at|by|with|for|from|to|about|of)$", "", cand).strip()
        if cand not in {"color", "name", "kind", "type", "difference", "reason", "purpose", "not", "it", "this", "that"} and not cand.startswith("not"):
            if len(cand) > 2:
                return cand, "success_wh_subject_syntax"

    # Pattern 6: Visual noun extraction from question phrase ("...about this animals?" -> "animals")
    m_noun = re.search(
        r"\b(?:about|of|on)\s+(?:this|the|these|those)\s+([a-zA-Z0-9_\-]+)\b",
        cleaned,
        re.IGNORECASE
    )
    if m_noun:
        cand = m_noun.group(1).strip().lower()
        if cand not in {"image", "photo", "picture", "scene"} and len(cand) > 2:
            return cand, "success_preposition_noun_syntax"

    return None, "failed_unrecognized_question_syntax"


def parse_answer_polarity_v2_2(generated_answer: str) -> Tuple[int, str]:
    """
    Parse generated answer polarity:
    +1: Affirmative / factual positive assertion
    -1: Negative (entity claimed absent / false)
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

    # For affirmative declarative statements (e.g. "The cat is on the toilet lid"):
    if not re.search(r"\b(?:not|no|neither|none|cannot|never)\b", text):
        return 1, "factual_assertion"

    return 0, "ambiguous_polarity"


def construct_directional_hypothesis_v2_2(
    question: str,
    generated_answer: str,
    extracted_entity: Optional[str]
) -> Tuple[str, str]:
    """
    Constructs a complete directional hypothesis for NLI evaluation.
    """
    polarity, pol_reason = parse_answer_polarity_v2_2(generated_answer)
    clean_ans = generated_answer.strip()

    if clean_ans.lower() in {"yes", "no"}:
        clean_q = question.strip().rstrip("?")
        if polarity == 1:
            return f"It is true that {clean_q}.", "bare_yes_fallback"
        elif polarity == -1:
            return f"It is false that {clean_q}.", "bare_no_fallback"

    if extracted_entity and polarity == -1 and re.match(r"^no\b", clean_ans, re.IGNORECASE):
        return f"There is no {extracted_entity} in the image.", f"entity_negative_{pol_reason}"

    return clean_ans, "raw_sentence_hypothesis"


def compute_perception_score_v2_2(
    image_path: str,
    question: str,
    generated_answer: str,
    detector: Any,
    scorer: Any
) -> Dict[str, Any]:
    """
    Version 2.2 Generalized Perception Score (S_P v2.2):
    Rigorously evaluates visual grounding while preserving explicit unavailable states.
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

    entity, ext_status = extract_queried_entity_v2_2(question, generated_answer)
    polarity, pol_status = parse_answer_polarity_v2_2(generated_answer)

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

        is_corroborated = has_detection and (vis_score >= SIGLIP_GROUNDING_THRESHOLD)

        if polarity == 1:
            if is_corroborated:
                s_p = vis_score
                alignment_type = "positive_grounded"
            elif has_detection:
                s_p = vis_score
                alignment_type = "positive_box_unconfirmed_by_siglip"
            else:
                s_p = 0.0
                alignment_type = "positive_unsupported_by_detector"
        elif polarity == -1:
            if is_corroborated:
                s_p = float(max(0.0, 1.0 - vis_score))
                alignment_type = "negative_omission_contradicted"
            elif has_detection:
                eff_conf = max_conf if max_conf > 0.0 else DETECTOR_THRESHOLD
                s_p = float(1.0 - eff_conf)
                alignment_type = "negative_unvalidated_box_unconfirmed_absence"
            else:
                eff_conf = max_conf if max_conf > 0.0 else DETECTOR_THRESHOLD
                s_p = float(1.0 - eff_conf)
                alignment_type = "negative_no_detection_bounded_absence"
        else:
            return {
                "s_p": None,
                "status": "unavailable",
                "semantic_alignment": "ambiguous_answer_polarity",
                "reason": "cannot_evaluate_directional_consistency_for_ambiguous_polarity",
                "entity": entity,
                "visual_evidence_score": vis_score,
                "visual_detected": has_detection,
                "visual_corroborated": is_corroborated,
                "detections_count": len(boxes),
                "raw_confidences": [round(float(c), 4) for c in confs],
                "max_detector_confidence": round(float(max_conf), 4) if confs else None,
                "bounding_boxes": boxes,
                "polarity": polarity,
                "metric_type": "heuristic_consistency_proxy",
                "scoring_version": SCORING_VERSION
            }

        return {
            "s_p": round(float(np.clip(s_p, 0.0, 1.0)), 4),
            "status": "available",
            "semantic_alignment": alignment_type,
            "reason": None,
            "entity": entity,
            "visual_evidence_score": round(vis_score, 4),
            "visual_detected": has_detection,
            "visual_corroborated": is_corroborated,
            "detections_count": len(boxes),
            "raw_confidences": [round(float(c), 4) for c in confs],
            "max_detector_confidence": round(float(max_conf), 4) if confs else None,
            "bounding_boxes": boxes,
            "polarity": polarity,
            "metric_type": "heuristic_consistency_proxy",
            "scoring_version": SCORING_VERSION
        }
    except Exception as e:
        logger.error(f"Perception scoring exception: {e}")
        return {
            "s_p": None,
            "status": "unavailable",
            "semantic_alignment": "detector_runtime_error",
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


def compute_utilization_score_v2_2(
    premise: str,
    question: str,
    generated_answer: str,
    nli_evaluator: Any,
    extracted_entity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Version 2.2 Utilization Score (S_U v2.2):
    Evaluates NLI entailment between retrieved premise and proposition hypothesis.
    """
    if not premise or not premise.strip():
        return {
            "s_u": None,
            "status": "unavailable_no_context",
            "semantic_support": "empty_premise_context",
            "probabilities": None,
            "hypothesis": None,
            "premise_snippet": None,
            "metric_type": "textual_semantic_entailment_proxy",
            "causal_attribution_claimed": False,
            "scoring_version": SCORING_VERSION
        }

    hypothesis, method = construct_directional_hypothesis_v2_2(
        question=question,
        generated_answer=generated_answer,
        extracted_entity=extracted_entity
    )

    probs = nli_evaluator.predict_nli_probabilities(premise, hypothesis)
    p_ent = probs.get("entailment", 0.0)
    p_neu = probs.get("neutral", 0.0)
    p_con = probs.get("contradiction", 0.0)

    s_u = (1.0 + p_ent - p_con) / 2.0

    return {
        "s_u": round(float(np.clip(s_u, 0.0, 1.0)), 4),
        "status": "available",
        "semantic_support": f"nli_evaluated_{method}",
        "probabilities": {
            "entailment": round(float(p_ent), 4),
            "neutral": round(float(p_neu), 4),
            "contradiction": round(float(p_con), 4)
        },
        "hypothesis": hypothesis,
        "premise_snippet": premise[:120].strip(),
        "metric_type": "textual_semantic_entailment_proxy",
        "causal_attribution_claimed": False,
        "scoring_version": SCORING_VERSION
    }
