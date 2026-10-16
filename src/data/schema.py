"""Canonical data schemas for XMCA-Net-Lite."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class XMCADataPoint:
    """Standardized multimodal QA record across POPE, FaithScore, A-OKVQA, InfoSeek."""
    id: str
    dataset: str
    image_id: str
    image_path: str  # Local path or URI
    question: str
    ground_truth_answers: List[str]
    task_type: str  # binary_vqa, free_form, knowledge_vqa, entity_qa
    is_hallucination_gt: Optional[int] = None  # 1 for hallucinated, 0 for faithful/factual, None if unknown
    image_sha256: Optional[str] = None
    split: Optional[str] = None  # train, val, test
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationOutput:
    """Model generation output shard record."""
    id: str
    dataset: str
    image_id: str
    vlm_name: str
    question: str
    retrieved_passages: List[str]
    generated_answer: str
    is_hallucination_gt: Optional[int] = None
    hallucination_fp: Optional[int] = None  # Specific for POPE: 1 if GT=='no' and generated=='yes'
    is_correct: Optional[int] = None  # Answer correctness vs GT
    omission_fn: Optional[int] = None  # Specific for POPE: 1 if GT=='yes' and generated=='no'
    parse_status: str = "success"  # "success", "unparseable", "failed"
    status: str = "completed"  # "completed", "failed"
    error_msg: Optional[str] = None
    execution_time_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_pope_answer(ground_truth: str, generated_answer: str) -> Dict[str, Any]:
    """
    Standard POPE answer parser adhering strictly to official protocol:
    - Verifies ground_truth is strictly 'yes' or 'no'. If invalid, returns parse_status='invalid_gt' and None targets.
    - hallucination_fp = 1 if and only if GT is 'no' and generated answer is 'yes'.
    - hallucination_fp = 0 for other successfully parsed cases.
    - is_correct = 1 if predicted answer matches ground truth, 0 otherwise.
    - omission_fn = 1 if GT is 'yes' and generated answer is 'no'.
    - If empty, ambiguous (both yes and no present), negated ("not yes", "not no"), or unparseable,
      returns parse_status='unparseable' and None targets (no fabricated labels).
    """
    import re
    if not isinstance(ground_truth, str):
        clean_gt = ""
    else:
        clean_gt = ground_truth.strip().lower()

    if clean_gt not in ("yes", "no"):
        return {
            "parsed_answer": "invalid_gt",
            "parse_status": "invalid_gt",
            "is_correct": None,
            "hallucination_fp": None,
            "omission_fn": None,
            "is_hallucination_gt": None
        }

    if not isinstance(generated_answer, str):
        text = ""
    else:
        text = generated_answer.strip().lower()

    if not text:
        return {
            "parsed_answer": "empty",
            "parse_status": "unparseable",
            "is_correct": None,
            "hallucination_fp": None,
            "omission_fn": None,
            "is_hallucination_gt": None
        }

    # Check for explicit negation / ambiguity
    if re.search(r"\bnot\s+(yes|no)\b", text) or "neither yes nor no" in text or "not sure" in text:
        return {
            "parsed_answer": "ambiguous",
            "parse_status": "unparseable",
            "is_correct": None,
            "hallucination_fp": None,
            "omission_fn": None,
            "is_hallucination_gt": None
        }

    has_yes = bool(re.search(r"\byes\b", text))
    has_no = bool(re.search(r"\bno\b", text))

    # If both yes and no appear in the text, it is ambiguous unless there's an unambiguous starting token
    if has_yes and has_no:
        # Check if one is an immediate prefix followed by punctuation (e.g. "No, there is..." vs "Yes, but no...")
        tokens = re.findall(r"\b(yes|no)\b", text)
        if len(set(tokens)) > 1:
            # Conflicting yes and no tokens in response
            return {
                "parsed_answer": "ambiguous",
                "parse_status": "unparseable",
                "is_correct": None,
                "hallucination_fp": None,
                "omission_fn": None,
                "is_hallucination_gt": None
            }
        predicted = tokens[0]
    elif has_yes:
        predicted = "yes"
    elif has_no:
        predicted = "no"
    elif text.startswith("yes"):
        predicted = "yes"
    elif text.startswith("no"):
        predicted = "no"
    else:
        return {
            "parsed_answer": "unparseable",
            "parse_status": "unparseable",
            "is_correct": None,
            "hallucination_fp": None,
            "omission_fn": None,
            "is_hallucination_gt": None
        }

    is_corr = 1 if predicted == clean_gt else 0
    halluc_fp = 1 if (clean_gt == "no" and predicted == "yes") else 0
    omiss_fn = 1 if (clean_gt == "yes" and predicted == "no") else 0

    return {
        "parsed_answer": predicted,
        "parse_status": "success",
        "is_correct": is_corr,
        "hallucination_fp": halluc_fp,
        "omission_fn": omiss_fn,
        "is_hallucination_gt": halluc_fp
    }


@dataclass
class PerceptionScoreRecord:
    """Perception evaluation results for an answer."""
    id: str
    extracted_claims: List[str]
    detected_boxes: List[List[float]]  # [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax]
    detected_labels: List[str]
    claim_visual_similarities: List[float]
    s_p: float  # Aggregated Perception Consistency Score in [0, 1]


@dataclass
class RetrievalScoreRecord:
    """Retrieval evaluation results for question-evidence pair."""
    id: str
    retrieved_passages: List[str]
    retriever_similarities: List[float]
    reranker_scores: List[float]
    s_r: float  # Aggregated Retrieval Consistency Score in [0, 1]


@dataclass
class UtilizationScoreRecord:
    """Utilization evaluation results for claims against evidence."""
    id: str
    extracted_claims: List[str]
    nli_entailment_probs: List[float]
    nli_contradiction_probs: List[float]
    s_u: float  # Aggregated Utilization Consistency Score in [0, 1]
    is_text_only: bool = False
