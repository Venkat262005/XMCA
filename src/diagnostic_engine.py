"""XMCA-Net-Lite Diagnostic Engine: Unified inference, claim audits, and cause explanations."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.combiner.model import XMCACombiner
from src.config import load_config
from src.perception.claims import extract_claims_from_answer
from src.perception.detector import OwlViTDetector
from src.perception.scorer import PerceptionScorer
from src.retrieval.fusion import RetrievalScoreFusion
from src.retrieval.indexer import DenseRetriever
from src.retrieval.reranker import MiniLMReranker
from src.utilization.nli import DeBERTaNLIEngine
from src.utilization.scorer import UtilizationScorer


@dataclass
class ClaimDiagnosticAudit:
    """Diagnostic audit for an individual atomic claim in the VLM answer."""
    claim_index: int
    claim_text: str
    perception_support: float
    retrieval_support: float
    utilization_support: float
    verdict: str  # "grounded", "unsupported_perception", "unsupported_retrieval", "contradiction"


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic evaluation report produced by XMCA-Net-Lite."""
    is_hallucination: bool
    hallucination_probability: float
    primary_cause: str  # "perception_failure", "retrieval_failure", "utilization_failure", "compound_failure", "faithful"
    cause_probabilities: Dict[str, float]
    consistency_scores: Dict[str, float]
    threshold_used: float
    claims_audited: List[ClaimDiagnosticAudit]
    natural_language_explanation: str
    actionable_recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["claims_audited"] = [asdict(c) for c in self.claims_audited]
        return d


class XMCADiagnosticEngine:
    """
    State-of-the-art diagnostic engine for Vision-Language Models.
    Combines Perception, Retrieval, and Utilization consistency signals
    to determine hallucination probability and pinpoint root cause failures.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        combiner_model: Optional[XMCACombiner] = None,
        threshold: float = 0.5
    ):
        self.config = config or load_config()
        self.threshold = threshold

        # Initialize sub-modules (with lazy loading / fallbacks)
        self.detector = OwlViTDetector(
            model_id=self.config["models"]["perception"]["detector_model_id"]
        )
        self.p_scorer = PerceptionScorer(
            model_id=self.config["models"]["perception"]["similarity_model_id"]
        )
        self.retriever = DenseRetriever(
            model_id=self.config["models"]["retriever"]["model_id"]
        )
        self.reranker = MiniLMReranker(
            model_id=self.config["models"]["reranker"]["model_id"]
        )
        self.retrieval_fusion = RetrievalScoreFusion(mode="isotonic")
        self.nli_engine = DeBERTaNLIEngine(
            model_id=self.config["models"]["utilization"]["model_id"],
            use_lora=False
        )
        self.u_scorer = UtilizationScorer(self.nli_engine)

        # Load or initialize combiner
        if combiner_model is not None:
            self.combiner = combiner_model
        else:
            combiner_path = Path(self.config["paths"]["combiner_dir"])
            if (combiner_path / "xmca_combiner_model.joblib").exists():
                self.combiner = XMCACombiner.load(combiner_path)
            else:
                self.combiner = XMCACombiner(random_state=42)

    def diagnose(
        self,
        image: str | Path,
        question: str,
        answer: str,
        retrieved_passages: Optional[List[str]] = None,
        visual_context: Optional[str] = None,
        override_scores: Optional[Dict[str, float]] = None
    ) -> DiagnosticReport:
        """
        Diagnose a multimodal QA generation instance.

        Args:
            image: Path to image file or dummy placeholder URI.
            question: Multimodal question query.
            answer: Generated VLM answer.
            retrieved_passages: Optional list of retrieved knowledge passages. If None, retrieves from indexed corpus.
            visual_context: Optional text summary of image context.
            override_scores: Optional dictionary of precomputed/preset consistency scores (s_p, s_r, s_u).

        Returns:
            DiagnosticReport containing calibrated hallucination probability,
            primary failure cause, soft cause distribution, and claim audits.
        """
        image_str = str(image)

        # 1. Atomic Claim Extraction
        claims = extract_claims_from_answer(answer, question=question)

        # 2. Perception Consistency (S_P)
        boxes, labels, confidences = self.detector.detect_regions(image_str, claims)
        s_p = self.p_scorer.compute_s_p(image_str, claims, boxes, box_scores=confidences)

        # 3. Retrieval Consistency (S_R)
        if retrieved_passages is None or len(retrieved_passages) == 0:
            search_results = self.retriever.search(question, top_k=self.config["models"]["retriever"]["top_k"])
            passages = [doc["text"] for doc, _ in search_results] if search_results else []
            dense_scores = [score for _, score in search_results] if search_results else []
        else:
            passages = list(retrieved_passages)
            # Evaluate genuine semantic similarity between answer and passages using retriever
            dense_scores = [self.retriever.compute_similarity(answer, p) if hasattr(self.retriever, "compute_similarity") else 0.5 for p in passages]

        if passages and len(passages) > 0:
            rerank_scores = self.reranker.score_pairs(answer, passages)
            s_r = self.retrieval_fusion.fuse(dense_scores, rerank_scores)
        else:
            s_r = None

        # 4. Utilization Consistency (S_U)
        s_u = self.u_scorer.compute_s_u(claims, passages, visual_context=visual_context, is_text_only=False)

        # Apply override scores if explicitly provided (e.g. from preset simulation)
        if override_scores:
            s_p = float(override_scores.get("s_p", s_p))
            s_r = float(override_scores.get("s_r", s_r))
            s_u = float(override_scores.get("s_u", s_u))

        # 5. Multi-Head Combiner Diagnosis
        diag_dict = self.combiner.diagnose_single(
            s_p=s_p,
            s_r=s_r,
            s_u=s_u,
            evidence_count=len(passages),
            claim_count=len(claims),
            threshold=self.threshold
        )

        h_prob = float(diag_dict["hallucination_probability"])
        is_hallucinated = bool(diag_dict["is_hallucination"])
        primary_cause = diag_dict["primary_cause"]
        cause_probs = diag_dict["cause_probabilities"]

        # 6. Granular Claim-Level Audit
        claim_audits: List[ClaimDiagnosticAudit] = []
        for i, c in enumerate(claims):
            c_sp = self.p_scorer.compute_s_p(image_str, [c], boxes, box_scores=confidences)
            c_sr = s_r  # Retrieval context is at query level
            c_su = self.u_scorer.compute_s_u([c], passages, visual_context=visual_context)

            if c_sp < 0.4:
                verdict = "unsupported_perception"
            elif c_su < 0.35:
                verdict = "contradiction_or_unsupported_utilization"
            elif c_sr < 0.35:
                verdict = "unsupported_retrieval"
            else:
                verdict = "grounded"

            claim_audits.append(ClaimDiagnosticAudit(
                claim_index=i + 1,
                claim_text=c,
                perception_support=float(np.round(c_sp, 4)),
                retrieval_support=float(np.round(c_sr, 4)),
                utilization_support=float(np.round(c_su, 4)),
                verdict=verdict
            ))

        # 7. Generate Natural Language Explanation & Actionable Recommendation
        explanation, recommendation = self._synthesize_explanation_and_recommendation(
            is_hallucinated=is_hallucinated,
            prob=h_prob,
            primary_cause=primary_cause,
            s_p=s_p,
            s_r=s_r,
            s_u=s_u,
            claims=claims,
            claim_audits=claim_audits
        )

        return DiagnosticReport(
            is_hallucination=is_hallucinated,
            hallucination_probability=float(np.round(h_prob, 4)),
            primary_cause=primary_cause,
            cause_probabilities=cause_probs,
            consistency_scores={
                "s_p": float(np.round(s_p, 4)),
                "s_r": float(np.round(s_r, 4)),
                "s_u": float(np.round(s_u, 4))
            },
            threshold_used=self.threshold,
            claims_audited=claim_audits,
            natural_language_explanation=explanation,
            actionable_recommendation=recommendation
        )

    def _synthesize_explanation_and_recommendation(
        self,
        is_hallucinated: bool,
        prob: float,
        primary_cause: str,
        s_p: float,
        s_r: float,
        s_u: float,
        claims: List[str],
        claim_audits: List[ClaimDiagnosticAudit]
    ) -> tuple[str, str]:
        """Generate human-readable diagnostic analysis and engineering remediation."""
        if not is_hallucinated:
            exp = (
                f"The generated answer appears FAITHFUL (hallucination probability: {prob:.1%}). "
                f"All three consistency signals indicate strong grounding: "
                f"Perception S_P={s_p:.2f}, Retrieval S_R={s_r:.2f}, Utilization S_U={s_u:.2f}."
            )
            rec = "No pipeline remediation needed. Answer is well-grounded in both visual and retrieved evidence."
            return exp, rec

        # Hallucination cases
        failing_claims = [c.claim_text for c in claim_audits if c.verdict != "grounded"]
        claim_highlight = f" Claims with deficits: {failing_claims}" if failing_claims else ""

        if primary_cause == "perception_failure":
            exp = (
                f"High hallucination probability ({prob:.1%}) attributed to PERCEPTION FAILURE. "
                f"The VLM made visual assertions unsupported by detected image regions "
                f"(S_P={s_p:.2f} is significantly below retrieval S_R={s_r:.2f} and utilization S_U={s_u:.2f})."
                f"{claim_highlight}"
            )
            rec = (
                "Remediation: Enhance visual resolution, apply object grounding verification (e.g. OWL-ViT filters), "
                "or adjust system prompts to instruct the VLM to decline answering if visual entities cannot be resolved."
            )
        elif primary_cause == "retrieval_failure":
            exp = (
                f"High hallucination probability ({prob:.1%}) attributed to RETRIEVAL FAILURE. "
                f"The retrieved knowledge context lacked sufficient relevance to answer the multimodal question "
                f"(S_R={s_r:.2f} vs S_P={s_p:.2f}, S_U={s_u:.2f})."
                f"{claim_highlight}"
            )
            rec = (
                "Remediation: Improve dense retrieval index (e.g. fine-tune BGE/E5 embeddings on domain corpus), "
                "increase top-k passage retrieval, or apply cross-encoder query-passage reranking."
            )
        elif primary_cause == "utilization_failure":
            exp = (
                f"High hallucination probability ({prob:.1%}) attributed to UTILIZATION FAILURE. "
                f"Although knowledge was retrieved (S_R={s_r:.2f}) and image features were visible (S_P={s_p:.2f}), "
                f"the VLM failed to faithfully synthesize the evidence (NLI entailment S_U={s_u:.2f})."
                f"{claim_highlight}"
            )
            rec = (
                "Remediation: Apply constrained decoding, enforce chain-of-thought grounding over the retrieved passages, "
                "or fine-tune the VLM generator with an NLI faithfulness loss penalty."
            )
        else:  # compound failure
            exp = (
                f"High hallucination probability ({prob:.1%}) attributed to COMPOUND MULTI-MODAL FAILURE. "
                f"Multiple consistency signals collapsed simultaneously: "
                f"S_P={s_p:.2f}, S_R={s_r:.2f}, S_U={s_u:.2f}."
                f"{claim_highlight}"
            )
            rec = (
                "Remediation: Both retrieval recall and visual verification failed. Check whether the question is out-of-distribution "
                "and whether the knowledge corpus contains relevant passages."
            )

        return exp, rec
