"""Utilization consistency scorer (S_U) comparing answer claims against evidence."""

from typing import List, Optional
import numpy as np
from src.utilization.nli import DeBERTaNLIEngine


class UtilizationScorer:
    """Computes Utilization Consistency Score (S_U) in [0, 1]."""

    def __init__(self, nli_engine: Optional[DeBERTaNLIEngine] = None, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.nli_engine = nli_engine or DeBERTaNLIEngine(strict_mode=strict_mode)

    def compute_s_u(
        self,
        claims: List[str],
        retrieved_passages: List[str],
        visual_context: Optional[str] = None,
        is_text_only: bool = False
    ) -> float:
        """
        Compute S_U.
        If is_text_only is True (Ablation #5), evidence premise is strictly retrieved passages.
        Otherwise, combines retrieved passages with visual context (cross-modal utilization).
        """
        if not claims:
            return None

        if not retrieved_passages and not visual_context:
            return None  # Evidence premise unavailable; cannot evaluate utilization

        # Build composite premise
        premise_parts = list(retrieved_passages)
        if visual_context and not is_text_only:
            premise_parts.append(str(visual_context))

        premise = " ".join(premise_parts).strip()
        if not premise:
            return None

        pairs = [(premise, claim) for claim in claims]
        nli_outputs = self.nli_engine.predict_entailment_probabilities(pairs)

        entailment_scores = [out["entailment"] for out in nli_outputs]
        contradiction_scores = [out["contradiction"] for out in nli_outputs]

        # Standardized linear margin formula mapping [-1, 1] into [0, 1]
        # Neutral evidence (entail=0, contra=0) maps to 0.5000.
        mean_entail = float(np.mean(entailment_scores))
        mean_contra = float(np.mean(contradiction_scores))

        s_u = (1.0 + mean_entail - mean_contra) / 2.0
        return float(np.clip(s_u, 0.0, 1.0))
