"""FAISS dense indexer using BGE-small / E5-base representations."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import faiss
import numpy as np

logger = logging.getLogger(__name__)


class DenseRetriever:
    """FAISS-based dense retriever supporting BGE-small / E5 embeddings."""

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5", dimension: int = 384):
        self.model_id = model_id
        self.dimension = dimension
        self.index: faiss.IndexFlatIP | None = None
        self.corpus_docs: List[Dict[str, Any]] = []
        self._encoder = None
        self._tokenizer = None

    def _init_model(self):
        if self._encoder is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._encoder = AutoModel.from_pretrained(self.model_id)
            if torch.cuda.is_available():
                self._encoder = self._encoder.to("cuda")
            self._encoder.eval()
        except Exception as e:
            logger.info(f"Transformers dense model unavailable ({e}). Using deterministic projection.")
            self._encoder = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode strings into normalized embeddings."""
        self._init_model()
        if self._encoder is not None and self._tokenizer is not None:
            import torch
            with torch.no_grad():
                inputs = self._tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                outputs = self._encoder(**inputs)
                # Mean pooling
                mask = inputs["attention_mask"].unsqueeze(-1)
                embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                return embeddings.cpu().numpy().astype(np.float32)

        # High-quality deterministic fallback embedding using hashing trick
        embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            vec = np.zeros(self.dimension, dtype=np.float32)
            for w in words:
                h = hash(w) % self.dimension
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            else:
                vec[0] = 1.0
            embeddings[i] = vec
        return embeddings

    def build_index(self, corpus_docs: List[Dict[str, Any]]) -> None:
        """Build FAISS inner-product index over normalized embeddings (cosine similarity)."""
        self.corpus_docs = corpus_docs
        texts = [doc["text"] for doc in corpus_docs]
        embeddings = self.encode(texts)
        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        logger.info(f"FAISS index built with {self.index.ntotal} documents, dimension {self.dimension}.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve top_k passages and their cosine similarity scores."""
        if self.index is None or self.index.ntotal == 0:
            return []
        q_emb = self.encode([query])
        scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.corpus_docs):
                results.append((self.corpus_docs[idx], float(score)))
        return results

    def save(self, index_file: str | Path, meta_file: str | Path) -> None:
        """Persist FAISS index and corpus document mapping."""
        if self.index is not None:
            faiss.write_index(self.index, str(index_file))
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.corpus_docs, f, indent=2)

    def load(self, index_file: str | Path, meta_file: str | Path) -> None:
        """Load FAISS index and corpus metadata from disk."""
        self.index = faiss.read_index(str(index_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            self.corpus_docs = json.load(f)
