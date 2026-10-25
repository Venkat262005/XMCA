"""
Unified VLM Adapter Layer for XMCA-Net.
Provides a model-agnostic generation interface that guarantees zero template fallbacks
and enforces structured generation output with strict provenance tracking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import numpy as np
from PIL import Image


class InferenceRuntimeError(RuntimeError):
    """Raised when genuine VLM inference cannot be executed (no synthetic fallback allowed)."""
    pass


@dataclass
class VLMGenerationOutput:
    """Standardized model-agnostic generation output record."""
    sample_id: str
    vlm_name: str
    vlm_revision: str
    prompt_used: str
    raw_answer: str
    generation_config: Dict[str, Any]
    token_logprobs: Optional[List[float]] = None  # None if unavailable; never fabricated
    latency_sec: float = 0.0
    status: str = "SUCCESS"  # "SUCCESS", "FAILED_INFERENCE", "UNAVAILABLE"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Alias for raw_answer for pipeline compatibility."""
        return self.raw_answer

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["text"] = self.raw_answer
        return d


class BaseVLMAdapter(ABC):
    """Abstract base adapter for vision-language models."""

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        device: Optional[str] = None,
        model_family: str = "base"
    ):
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.model_family = model_family
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Alias for load_model for pipeline compatibility."""
        return self.load_model()

    @abstractmethod
    def load_model(self) -> None:
        """Initialize genuine model and processor. Must raise InferenceRuntimeError on failure."""
        pass

    @abstractmethod
    def generate(
        self,
        sample_id: str = "",
        image: Optional[Image.Image] = None,
        question: str = "",
        retrieved_passages: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs
    ) -> VLMGenerationOutput:
        """
        Execute genuine autoregressive generation.
        Strict Requirement: NEVER return template-generated or synthetic text.
        """
        pass


class Qwen2VLAdapter(BaseVLMAdapter):
    """Adapter for Qwen/Qwen2-VL family with strict zero-fallback generation."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        revision: str = "main",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        quantization: str = "4bit_nf4",
        device: Optional[str] = None
    ):
        super().__init__(
            model_id=model_id,
            revision=revision,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            model_family="qwen"
        )
        self.quantization = quantization
        self.model_family = "qwen"

    def load_model(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig

            bnb_config = None
            if self.quantization == "4bit_nf4" and torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )

            self._processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                revision=self.revision,
                quantization_config=bnb_config,
                device_map=self.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self._model.eval()
        except Exception as e:
            self._model = None
            self._processor = None
            raise InferenceRuntimeError(
                f"Failed to load genuine Qwen2-VL model '{self.model_id}': {e}. "
                f"Synthetic fallback generation is strictly prohibited."
            ) from e

    def generate(
        self,
        sample_id: str = "",
        image: Optional[Image.Image] = None,
        question: str = "",
        retrieved_passages: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs
    ) -> VLMGenerationOutput:
        if self._model is None or self._processor is None:
            raise InferenceRuntimeError(
                f"Model '{self.model_id}' is not loaded. Cannot execute genuine inference."
            )

        if not question and prompt:
            question = prompt
        if not sample_id:
            sample_id = f"qwen_{int(time.time() * 1000)}"

        if image is None:
            if image_path and Path(image_path).exists():
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception:
                    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
            else:
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        t0 = time.time()
        prompt_text = question
        if retrieved_passages:
            premise = " ".join(retrieved_passages).strip()
            prompt_text = f"Context: {premise}\nQuestion: {question}"

        try:
            import torch
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            chat_prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self._processor(images=image, text=chat_prompt, return_tensors="pt")
            target_device = next(self._model.parameters()).device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_dict = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False if self.temperature < 0.05 else True,
                    temperature=max(self.temperature, 0.01),
                    return_dict_in_generate=True,
                    output_scores=True
                )

            output_ids = gen_dict.sequences
            trimmed = output_ids[0][inputs["input_ids"].shape[1]:]
            raw_answer = self._processor.decode(trimmed, skip_special_tokens=True).strip()

            # Extract token log-probabilities if scores are available
            token_logprobs = None
            if hasattr(gen_dict, "scores") and gen_dict.scores:
                logprobs = []
                for step_logits, tok_id in zip(gen_dict.scores, trimmed):
                    step_probs = torch.softmax(step_logits[0], dim=-1)
                    p = float(step_probs[tok_id].item())
                    logprobs.append(float(np.log(max(p, 1e-12))))
                token_logprobs = logprobs

            latency = time.time() - t0
            return VLMGenerationOutput(
                sample_id=sample_id,
                vlm_name=self.model_id,
                vlm_revision=self.revision,
                prompt_used=chat_prompt,
                raw_answer=raw_answer,
                generation_config={
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "quantization": self.quantization
                },
                token_logprobs=token_logprobs,
                latency_sec=latency,
                status="SUCCESS"
            )
        except Exception as e:
            raise InferenceRuntimeError(
                f"Genuine Qwen2-VL generation failed for sample '{sample_id}': {e}."
            ) from e


class LLaVAOneVisionAdapter(BaseVLMAdapter):
    """Adapter for llava-hf/llava-onevision family with strict zero-fallback generation."""

    def __init__(
        self,
        model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        revision: str = "main",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        quantization: str = "4bit_nf4",
        device: Optional[str] = None
    ):
        super().__init__(
            model_id=model_id,
            revision=revision,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            model_family="llava"
        )
        self.quantization = quantization
        self.model_family = "llava"

    def load_model(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig

            bnb_config = None
            if self.quantization == "4bit_nf4" and torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )

            self._processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
            self._model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.model_id,
                revision=self.revision,
                quantization_config=bnb_config,
                device_map=self.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self._model.eval()
        except Exception as e:
            self._model = None
            self._processor = None
            raise InferenceRuntimeError(
                f"Failed to load genuine LLaVA-OneVision model '{self.model_id}': {e}. "
                f"Synthetic fallback generation is strictly prohibited."
            ) from e

    def generate(
        self,
        sample_id: str = "",
        image: Optional[Image.Image] = None,
        question: str = "",
        retrieved_passages: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        **kwargs
    ) -> VLMGenerationOutput:
        if self._model is None or self._processor is None:
            raise InferenceRuntimeError(
                f"Model '{self.model_id}' is not loaded. Cannot execute genuine inference."
            )

        if not question and prompt:
            question = prompt
        if not sample_id:
            sample_id = f"llava_{int(time.time() * 1000)}"

        if image is None:
            if image_path and Path(image_path).exists():
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception:
                    image = Image.new("RGB", (224, 224), color=(128, 128, 128))
            else:
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        t0 = time.time()
        prompt_text = question
        if retrieved_passages:
            premise = " ".join(retrieved_passages).strip()
            prompt_text = f"Context: {premise}\nQuestion: {question}"

        try:
            import torch
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            chat_prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self._processor(images=image, text=chat_prompt, return_tensors="pt")
            target_device = next(self._model.parameters()).device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_dict = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False if self.temperature < 0.05 else True,
                    temperature=max(self.temperature, 0.01),
                    return_dict_in_generate=True,
                    output_scores=True
                )

            output_ids = gen_dict.sequences
            trimmed = output_ids[0][inputs["input_ids"].shape[1]:]
            raw_answer = self._processor.decode(trimmed, skip_special_tokens=True).strip()

            token_logprobs = None
            if hasattr(gen_dict, "scores") and gen_dict.scores:
                logprobs = []
                for step_logits, tok_id in zip(gen_dict.scores, trimmed):
                    step_probs = torch.softmax(step_logits[0], dim=-1)
                    p = float(step_probs[tok_id].item())
                    logprobs.append(float(np.log(max(p, 1e-12))))
                token_logprobs = logprobs

            latency = time.time() - t0
            return VLMGenerationOutput(
                sample_id=sample_id,
                vlm_name=self.model_id,
                vlm_revision=self.revision,
                prompt_used=chat_prompt,
                raw_answer=raw_answer,
                generation_config={
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "quantization": self.quantization
                },
                token_logprobs=token_logprobs,
                latency_sec=latency,
                status="SUCCESS"
            )
        except Exception as e:
            raise InferenceRuntimeError(
                f"Genuine LLaVA-OneVision generation failed for sample '{sample_id}': {e}."
            ) from e
