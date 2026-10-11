"""Command-line runner for XMCA-Net-Lite pipeline and diagnostics."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
from src.combiner.features import build_feature_dataframe
from src.combiner.model import XMCACombiner
from src.combiner.thresholds import tune_decision_threshold
from src.config import load_config, set_seed
from src.data.loader import load_dataset_records
from src.data.splitter import save_and_manifest_splits, split_by_image_id
from src.evaluation.ablation_suite import run_ablation_experiment
from src.evaluation.bootstrap import compute_bootstrap_confidence_intervals
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.plots import plot_reliability_diagram, plot_roc_pr_curves
from src.generation.llava import LLaVAOneVisionGenerator
from src.generation.qwen import Qwen2VLGenerator
from src.perception.claims import extract_claims_from_answer
from src.perception.detector import OwlViTDetector
from src.perception.scorer import PerceptionScorer
from src.retrieval.corpus import build_evidence_corpus
from src.retrieval.fusion import RetrievalScoreFusion
from src.retrieval.indexer import DenseRetriever
from src.retrieval.reranker import MiniLMReranker
from src.utilization.nli import DeBERTaNLIEngine
from src.utilization.scorer import UtilizationScorer
from src.utils.checkpoints import ModelCheckpointTracker, ShardManager
from src.utils.logging_metadata import save_run_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("XMCA-Net-Lite")


def run_data_preparation(config: dict, seed: int):
    logger.info("==> Step 00: Environment & Data Preparation")
    splits_dir = Path(config["paths"]["splits_dir"])
    dataset_configs = config["datasets"]["sources"]

    manifests = {}
    for ds_key, ds_meta in dataset_configs.items():
        max_s = min(ds_meta["max_samples"], config["datasets"]["strict_max_per_dataset"])
        records = load_dataset_records(ds_key, max_samples=max_s, seed=seed)
        train, val, test = split_by_image_id(
            records=records,
            train_ratio=config["datasets"]["split_ratios"]["train"],
            val_ratio=config["datasets"]["split_ratios"]["val"],
            test_ratio=config["datasets"]["split_ratios"]["test"],
            seed=seed,
            max_total_samples=max_s
        )
        manifest = save_and_manifest_splits(ds_key, train, val, test, splits_dir)
        manifests[ds_key] = manifest
        logger.info(f"Dataset {ds_key}: train={len(train)}, val={len(val)}, test={len(test)}")

    save_run_metadata(splits_dir, config, seed, "00_environment_data", {"manifests": manifests})
    logger.info("Data preparation completed successfully.")


def run_prepare_corpora(config: dict, seed: int):
    logger.info("==> Step 01: Build Evidence Corpus & FAISS Index")
    splits_dir = Path(config["paths"]["splits_dir"])
    corpora_dir = Path(config["paths"]["corpora_dir"])

    all_records = []
    for p in splits_dir.glob("*/*train.json"):
        with open(p, "r", encoding="utf-8") as f:
            all_records.extend(json.load(f))

    corpus_file = corpora_dir / "evidence_corpus.jsonl"
    corpus_entries = build_evidence_corpus(all_records, corpus_file)

    retriever = DenseRetriever(model_id=config["models"]["retriever"]["model_id"])
    retriever.build_index(corpus_entries)
    index_file = corpora_dir / "faiss_index.bin"
    meta_file = corpora_dir / "corpus_metadata.json"
    retriever.save(index_file, meta_file)

    save_run_metadata(corpora_dir, config, seed, "01_prepare_corpora", {"corpus_size": len(corpus_entries)})
    logger.info(f"Corpora preparation completed. Total entries: {len(corpus_entries)}")


def main():
    parser = argparse.ArgumentParser(description="XMCA-Net-Lite Pipeline Runner")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--step", type=str, default="data", choices=["data", "corpora", "all"], help="Pipeline step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    if args.step in ["data", "all"]:
        run_data_preparation(config, args.seed)
    if args.step in ["corpora", "all"]:
        run_prepare_corpora(config, args.seed)


if __name__ == "__main__":
    main()
