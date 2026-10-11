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
    is_hallucination_gt: int  # 1 for hallucinated, 0 for faithful/factual
    task_type: str  # binary_vqa, free_form, knowledge_vqa, entity_qa
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
    is_hallucination_gt: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
