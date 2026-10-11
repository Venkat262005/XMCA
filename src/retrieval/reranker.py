"""Cross-encoder reranker using MiniLM for passage re-scoring."""

import logging
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class MiniLMReranker:
    """MiniLM Cross-Encoder reranker."""

    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None

    def _init_model(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()
        except Exception as e:
            logger.info(f"Transformers cross-encoder unavailable ({e}). Using semantic overlap scoring.")
            self._model = None

    def score_pairs(self, query: str, passages: List[str]) -> List[float]:
        """Compute cross-encoder relevance scores for query and passages in [0, 1]."""
        if not passages:
            return []
        self._init_model()

        if self._model is not None and self._tokenizer is not None:
            import torch
            pairs = [[query, p] for p in passages]
            with torch.no_grad():
                inputs = self._tokenizer(pairs, padding=True, truncation=True, max_length=256, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                logits = self._model(**inputs).logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                return [float(p) for p in (probs if isinstance(probs, list) else [probs])]

        # Fallback lexical/jaccard overlap
        scores = []
        q_words = set(query.lower().split())
        for p in passages:
            p_words = set(p.lower().split())
            if not p_words or not q_words:
                scores.append(0.1)
                continue
            inter = len(q_words.intersection(p_words))
            union = len(q_words.union(p_words))
            scores.append(float(inter / union if union > 0 else 0.0))
        return scores
