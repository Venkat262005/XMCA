"""DeBERTa-v3-base MNLI model with optional LoRA adapter."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class DeBERTaNLIEngine:
    """Natural Language Inference engine with optional PEFT LoRA training/inference."""

    def __init__(
        self,
        model_id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        self.model_id = model_id
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_seq_length = max_seq_length
        self._tokenizer = None
        self._model = None

    def _init_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)

            if self.use_lora:
                try:
                    from peft import LoraConfig, get_peft_model, TaskType
                    peft_config = LoraConfig(
                        task_type=TaskType.SEQ_CLS,
                        inference_mode=True,
                        r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                        target_modules=["query_proj", "value_proj"]
                    )
                    self._model = get_peft_model(self._model, peft_config)
                    logger.info("Attached LoRA adapter to DeBERTa-v3.")
                except ImportError:
                    logger.info("peft library not installed, using base DeBERTa.")

            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()
            logger.info(f"Loaded {self.model_id} for NLI.")
        except Exception as e:
            logger.info(f"Transformers NLI unavailable ({e}). Using semantic overlap fallback.")
            self._model = None

    def predict_entailment_probabilities(
        self,
        premise_hypothesis_pairs: List[tuple[str, str]]
    ) -> List[Dict[str, float]]:
        """
        Predict [entailment, neutral, contradiction] probabilities for (premise, hypothesis) pairs.
        Returns list of dicts with 'entailment', 'neutral', 'contradiction'.
        """
        if not premise_hypothesis_pairs:
            return []
        self._init_model()

        if self._model is not None and self._tokenizer is not None:
            import torch
            try:
                premises = [p for p, _ in premise_hypothesis_pairs]
                hypotheses = [h for _, h in premise_hypothesis_pairs]
                inputs = self._tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                with torch.no_grad():
                    logits = self._model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                results = []
                # Typical DeBERTa MNLI mapping: 0 -> contradiction, 1 -> neutral, 2 -> entailment
                for p in probs:
                    entail = float(p[-1]) if len(p) >= 3 else float(p[0])
                    contra = float(p[0]) if len(p) >= 3 else float(1.0 - entail)
                    results.append({
                        "entailment": entail,
                        "neutral": float(p[1]) if len(p) >= 3 else 0.0,
                        "contradiction": contra
                    })
                return results
            except Exception as e:
                logger.warning(f"DeBERTa prediction error ({e}). Returning fallback probabilities.")

        # Fallback keyword overlap
        results = []
        for premise, hyp in premise_hypothesis_pairs:
            p_words = set(premise.lower().split())
            h_words = set(hyp.lower().split())
            overlap = len(p_words.intersection(h_words)) / max(len(h_words), 1)
            results.append({
                "entailment": float(np.clip(overlap * 0.9, 0.0, 1.0)),
                "neutral": float(0.1),
                "contradiction": float(np.clip(1.0 - overlap, 0.0, 1.0))
            })
        return results
