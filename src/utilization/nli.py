"""DeBERTa-v3-base MNLI model with optional LoRA adapter."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


def validate_and_resolve_nli_mapping(config: Any, num_logits: int) -> Dict[int, str]:
    """
    Safely validate and resolve id2label mapping for 3-class NLI.
    Verifies that:
    1. config.id2label exists and is a valid dictionary.
    2. Exactly one recognized label exists for 'entailment', 'neutral', and 'contradiction'.
    3. All logits indices 0..num_logits-1 are covered.
    4. No duplicate, missing, or unknown classes exist.
    """
    raw_id2label = getattr(config, "id2label", None)
    if not raw_id2label or not isinstance(raw_id2label, dict):
        raise ValueError("Model configuration missing valid 'id2label' dictionary.")

    resolved: Dict[int, str] = {}
    label_to_index: Dict[str, int] = {}
    STANDARD_LABELS = {"entailment", "neutral", "contradiction"}

    for raw_k, raw_v in raw_id2label.items():
        try:
            idx = int(raw_k)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid non-integer label index in id2label: {raw_k}")

        lbl_str = str(raw_v).strip().lower()
        if "entail" in lbl_str or lbl_str == "label_0":
            norm_lbl = "entailment"
        elif "contra" in lbl_str or lbl_str == "label_2":
            norm_lbl = "contradiction"
        elif "neut" in lbl_str or lbl_str == "label_1":
            norm_lbl = "neutral"
        else:
            raise ValueError(f"Unrecognized NLI class label '{raw_v}' at index {idx}.")

        if norm_lbl in label_to_index:
            raise ValueError(
                f"Duplicate class mapping detected: '{norm_lbl}' mapped to both index "
                f"{label_to_index[norm_lbl]} and index {idx}."
            )
        label_to_index[norm_lbl] = idx
        resolved[idx] = norm_lbl

    missing = STANDARD_LABELS - set(label_to_index.keys())
    if missing:
        raise ValueError(f"Incomplete NLI label mapping: missing required classes {missing}.")

    if num_logits != len(resolved):
        raise ValueError(f"Logits dimension ({num_logits}) does not match id2label mapping size ({len(resolved)}).")

    for i in range(num_logits):
        if i not in resolved:
            raise ValueError(f"Output logit index {i} not covered in resolved id2label mapping.")

    return resolved


class DeBERTaNLIEngine:
    """Natural Language Inference engine with optional PEFT LoRA training/inference."""

    def __init__(
        self,
        model_id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        max_seq_length: int = 512,
        strict_mode: bool = False
    ):
        self.model_id = model_id
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_seq_length = max_seq_length
        self.strict_mode = strict_mode
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
            if self.strict_mode:
                raise RuntimeError(f"FATAL in strict_mode: Failed to load {self.model_id}: {e}") from e
            logger.info(f"Transformers NLI unavailable ({e}). Using semantic overlap fallback.")
            self._model = None

    def predict_nli_probabilities(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Predict [entailment, neutral, contradiction] for a single (premise, hypothesis) pair."""
        res = self.predict_entailment_probabilities([(premise, hypothesis)])
        return res[0] if res else {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

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
                    num_classes = logits.shape[-1]
                    id2label = validate_and_resolve_nli_mapping(self._model.config, num_classes)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                results = []
                for p_vec in probs:
                    # Validate probability vector properties
                    if not all(np.isfinite(p_vec)):
                        raise ValueError(f"Non-finite probabilities detected in NLI output: {p_vec}")
                    if any(v < -1e-5 or v > 1.0 + 1e-5 for v in p_vec):
                        raise ValueError(f"Probabilities out of valid [0, 1] range: {p_vec}")
                    if abs(float(np.sum(p_vec)) - 1.0) > 1e-3:
                        raise ValueError(f"Probabilities do not sum to 1.0 (sum={np.sum(p_vec):.6f})")

                    label_probs = {id2label[i]: float(np.clip(p_vec[i], 0.0, 1.0)) for i in range(len(p_vec))}
                    results.append({
                        "entailment": label_probs["entailment"],
                        "neutral": label_probs["neutral"],
                        "contradiction": label_probs["contradiction"]
                    })
                return results
            except Exception as e:
                if self.strict_mode:
                    raise RuntimeError(f"FATAL in strict_mode: DeBERTa prediction error: {e}") from e
                logger.warning(f"DeBERTa prediction error ({e}). Returning fallback probabilities.")

        if self.strict_mode:
            raise RuntimeError("FATAL in strict_mode: Cannot compute NLI entailment without genuine neural model.")

        # Fallback directional keyword coverage without stopwords
        STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "by", "for", "with",
            "about", "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "to", "from", "up", "down", "out", "off", "over", "under",
            "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should",
            "now", "of", "and", "this", "that", "these", "those", "displays", "shows"
        }
        results = []
        for premise, hyp in premise_hypothesis_pairs:
            p_words = {w.strip(".,;:!?\"'()").lower() for w in premise.split()} - STOPWORDS
            h_words = {w.strip(".,;:!?\"'()").lower() for w in hyp.split()} - STOPWORDS
            if not h_words:
                coverage = 0.85
            else:
                # Substring / partial match fallback
                matched = 0
                for hw in h_words:
                    if any(hw in pw or pw in hw for pw in p_words):
                        matched += 1
                coverage = matched / len(h_words)

            entail = float(np.clip(coverage, 0.05, 1.0))
            contra = float(np.clip(1.0 - coverage, 0.0, 1.0)) if coverage < 0.25 else 0.05
            results.append({
                "entailment": entail,
                "neutral": 0.1,
                "contradiction": contra
            })
        return results
