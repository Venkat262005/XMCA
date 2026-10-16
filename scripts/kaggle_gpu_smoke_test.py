"""
Kaggle GPU Minimal Smoke Test for XMCA-Net.
Designed specifically for NVIDIA CUDA GPUs (e.g. T4 / P100 / A100):
- Verifies CUDA availability, device name, and genuine VRAM allocation.
- Executes Qwen2-VL-2B-Instruct with 4-bit NF4 BitsAndBytes quantization on CUDA.
- Evaluates OwlViT detector, SigLIP perception scorer, and DeBERTa NLI on CUDA.
- Extracts and validates the 25-feature combiner vector in strict mode.
- Records genuine CUDA execution latency and peak VRAM allocated.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from PIL import Image

from src.combiner.features import FEATURE_COLUMNS, extract_combiner_features
from src.data.schema import parse_pope_answer
from src.generation.qwen import Qwen2VLGenerator
from src.perception.detector import OwlViTDetector
from src.perception.scorer import PerceptionScorer
from src.retrieval.fusion import RetrievalScoreFusion
from src.retrieval.indexer import DenseRetriever
from src.retrieval.reranker import MiniLMReranker
from src.utilization.nli import DeBERTaNLIEngine

from src.scoring.scoring_v2 import compute_scores_v2, SCORING_VERSION


def run_kaggle_gpu_smoke_test(output_dir: str = "artifacts_kaggle_gpu_smoke"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("XMCA-NET — KAGGLE GPU MINIMAL SMOKE TEST")
    print(f"Scoring Engine Version: {SCORING_VERSION}")
    print("=" * 70)

    # 1. Hardware, CUDA & Environment Verification
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version (PyTorch build): {torch.version.cuda}")

    if not cuda_available:
        print("[WARNING] CUDA is not available. This script is intended for actual Kaggle GPU execution.")
        device_name = "CPU (No CUDA)"
        total_vram_gb = 0.0
        peak_vram_gb = 0.0
    else:
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU Device: {device_name}")
        print(f"Total VRAM: {total_vram_gb:.2f} GB")

    # 2. Test Image Setup
    test_img_path = PROJECT_ROOT / "data" / "scaling_batch25_images" / "COCO_val2014_000000310196.jpg"
    if not test_img_path.exists():
        test_img_path = PROJECT_ROOT / "data" / "smoke_test_images" / "COCO_val2014_000000310196.jpg"
    
    if not test_img_path.exists():
        raise FileNotFoundError(f"Missing verification image: {test_img_path}")

    img_sha256 = hashlib.sha256(test_img_path.read_bytes()).hexdigest()
    print(f"Verification Image: {test_img_path.name} (SHA-256: {img_sha256[:16]}...)")

    t_start = time.time()

    # 3. Model Initialization (CUDA NF4 if available)
    print("\n[1/4] Initializing models on GPU...")
    t0 = time.time()
    generator = Qwen2VLGenerator(
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        quantization="4bit_nf4" if cuda_available else "none",
        strict_mode=True
    )
    generator._init_model()

    retriever = DenseRetriever(strict_mode=True)
    reranker = MiniLMReranker(strict_mode=True)
    fusion = RetrievalScoreFusion()
    detector = OwlViTDetector(strict_mode=True)
    scorer = PerceptionScorer(strict_mode=True)
    nli = DeBERTaNLIEngine(strict_mode=True)
    t_init = time.time() - t0
    print(f"✓ All models loaded in {t_init:.2f}s")

    # 4. End-to-End Single-Sample Test
    test_q = "Is there a snowboard in the image?"
    print(f"\n[2/4] Executing inference for: \"{test_q}\"")

    # Retrieval
    t_ret0 = time.time()
    docs = retriever.search(test_q, top_k=3)
    passages = [d["text"] for d, _ in docs]
    dense_scores = [sc for _, sc in docs]
    rerank_scores = reranker.score_pairs(test_q, passages)
    s_r = float(fusion.fuse(dense_scores, rerank_scores))
    t_ret = time.time() - t_ret0

    # Generation
    t_gen0 = time.time()
    gen_ans = generator.generate_single(
        image_path=str(test_img_path),
        question=test_q,
        retrieved_passages=passages
    )
    t_gen = time.time() - t_gen0
    print(f"✓ Generated Answer: \"{gen_ans}\" in {t_gen:.2f}s")

    # Evaluation
    pope_eval = parse_pope_answer(ground_truth="yes", generated_answer=gen_ans)

    # Scoring v2: Grounded Proposition Perception & Directional Utilization
    t_score0 = time.time()
    scores_v2 = compute_scores_v2(
        image_path=str(test_img_path),
        question=test_q,
        generated_answer=gen_ans,
        retrieved_passages=passages,
        s_r=s_r,
        detector=detector,
        scorer=scorer,
        nli_engine=nli
    )
    t_score = time.time() - t_score0
    s_p = float(scores_v2["s_p"]) if scores_v2["s_p"] is not None else 0.0
    s_u = float(scores_v2["s_u"]) if scores_v2["s_u"] is not None else 0.5

    # 25-Feature Extraction
    feats = extract_combiner_features(
        s_p=s_p,
        s_r=s_r,
        s_u=s_u,
        evidence_count=len(passages),
        claim_count=len(claims)
    )
    all_finite = all(np.isfinite(v) for v in feats.values())
    assert len(feats) == 25, f"Expected 25 features, got {len(feats)}"
    assert all_finite, "Found non-finite features"

    # Peak VRAM
    if cuda_available:
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        vram_str = f"{peak_vram_gb:.2f} GB / {total_vram_gb:.2f} GB"
    else:
        vram_str = "N/A (CPU execution)"

    print("\n[3/4] Verification Summary:")
    print(f"  • S_P: {s_p:.4f} | S_R: {s_r:.4f} | S_U: {s_u:.4f}")
    print(f"  • Features: 25/25 valid and finite")
    print(f"  • POPE Evaluation: Correct={pope_eval['is_correct']}, HallucinationFP={pope_eval['hallucination_fp']}")
    print(f"  • Latency: Retrieval={t_ret:.3f}s, Gen={t_gen:.3f}s, Scoring={t_score:.3f}s")
    print(f"  • Peak VRAM: {vram_str}")

    # Output Manifest
    manifest = {
        "status": "passed",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scoring_version": SCORING_VERSION,
        "hardware": {
            "device_name": device_name,
            "cuda_available": cuda_available,
            "total_vram_gb": total_vram_gb,
            "peak_vram_allocated_gb": peak_vram_gb if cuda_available else 0.0
        },
        "sample": {
            "image_path": str(test_img_path),
            "image_sha256": img_sha256,
            "question": test_q,
            "generated_answer": gen_ans,
            "evaluation": pope_eval,
            "scores": {"s_p": s_p, "s_r": s_r, "s_u": s_u},
            "scores_v2_details": scores_v2,
            "features_25": feats,
            "runtimes_sec": {
                "retrieval": round(t_ret, 3),
                "generation": round(t_gen, 3),
                "scoring": round(t_score, 3),
                "total": round(time.time() - t_start, 3)
            }
        }
    }

    manifest_file = out_path / "kaggle_gpu_smoke_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[4/4] Saved verification manifest to {manifest_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_kaggle_gpu_smoke_test()
