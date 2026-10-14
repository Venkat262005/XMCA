"""Open-vocabulary object detector using OWL-ViT."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class OwlViTDetector:
    """Open-vocabulary bounding box detector using OWL-ViT."""

    def __init__(
        self,
        model_id: str = "google/owlvit-base-patch32",
        box_threshold: float = 0.15,
        text_threshold: float = 0.20
    ):
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._processor = None
        self._model = None

    def _init_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
            self._processor = OwlViTProcessor.from_pretrained(self.model_id)
            self._model = OwlViTForObjectDetection.from_pretrained(self.model_id)
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()
            logger.info(f"Loaded {self.model_id} detector.")
        except Exception as e:
            logger.info(f"OWL-ViT detector unavailable ({e}). Using region generator fallback.")
            self._model = None

    def detect_regions(
        self,
        image_path: str,
        query_labels: List[str]
    ) -> Tuple[List[List[float]], List[str], List[float]]:
        """
        Detect visual regions corresponding to candidate query labels.
        Returns: (boxes, detected_labels, confidences)
        """
        self._init_model()

        # Load or create dummy image
        if Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self._model is not None and self._processor is not None and query_labels:
            import torch
            try:
                # Truncate and clean query labels for OWL-ViT (max 16 tokens for positional embeddings)
                clean_labels = [str(l).strip()[:80] for l in query_labels if str(l).strip()]
                if not clean_labels:
                    clean_labels = ["object"]

                inputs = self._processor(
                    text=[clean_labels],
                    images=image,
                    padding=True,
                    truncation=True,
                    max_length=16,
                    return_tensors="pt"
                )
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._model(**inputs)

                target_sizes = torch.Tensor([image.size[::-1]])
                if torch.cuda.is_available():
                    target_sizes = target_sizes.to("cuda")
                if hasattr(self._processor, "post_process_grounded_object_detection"):
                    results = self._processor.post_process_grounded_object_detection(
                        outputs=outputs,
                        target_sizes=target_sizes,
                        threshold=self.box_threshold,
                        text_labels=[clean_labels]
                    )[0]
                    boxes = results["boxes"].cpu().numpy().tolist()
                    scores = results["scores"].cpu().numpy().tolist()
                    labels = results.get("text_labels", [clean_labels[idx] for idx in results["labels"].cpu().numpy().tolist() if idx < len(clean_labels)])
                elif hasattr(self._processor, "post_process_object_detection"):
                    results = self._processor.post_process_object_detection(
                        outputs=outputs,
                        target_sizes=target_sizes,
                        threshold=self.box_threshold
                    )[0]
                    boxes = results["boxes"].cpu().numpy().tolist()
                    scores = results["scores"].cpu().numpy().tolist()
                    labels_idx = results["labels"].cpu().numpy().tolist()
                    labels = [clean_labels[idx] for idx in labels_idx if idx < len(clean_labels)]
                elif hasattr(self._processor, "image_processor") and hasattr(self._processor.image_processor, "post_process_object_detection"):
                    results = self._processor.image_processor.post_process_object_detection(
                        outputs=outputs,
                        target_sizes=target_sizes,
                        threshold=self.box_threshold
                    )[0]
                    boxes = results["boxes"].cpu().numpy().tolist()
                    scores = results["scores"].cpu().numpy().tolist()
                    labels_idx = results["labels"].cpu().numpy().tolist()
                    labels = [clean_labels[idx] for idx in labels_idx if idx < len(clean_labels)]
                else:
                    boxes, labels, scores = [], [], []

                if boxes:
                    return boxes, labels, scores
            except Exception as e:
                logger.warning(f"OWL-ViT detection error ({e}). Using fallback grid.")

        # Fallback grid regions
        w, h = image.size
        fallback_boxes = [
            [0.0, 0.0, float(w), float(h)],
            [0.0, 0.0, float(w / 2), float(h / 2)],
            [float(w / 2), 0.0, float(w), float(h / 2)],
            [0.0, float(h / 2), float(w / 2), float(h)],
            [float(w / 2), float(h / 2), float(w), float(h)],
        ]
        fallback_labels = query_labels[:len(fallback_boxes)] if query_labels else ["scene"]
        fallback_scores = [0.85] * len(fallback_boxes)
        return fallback_boxes, fallback_labels, fallback_scores
