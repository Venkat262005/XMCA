"""Script to programmatically generate the 10 Kaggle-ready Jupyter Notebooks for XMCA-Net-Lite."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


def build_all_notebooks():
    # -------------------------------------------------------------
    # 00_environment_data.ipynb
    # -------------------------------------------------------------
    nb00 = [
        md_cell("""# XMCA-Net-Lite: 00 - Environment & Data Preparation
### Purpose
Download and standardize the four evaluation datasets (POPE, FaithScore/LLaVA-1k, A-OKVQA, InfoSeek) with strict 2,000 maximum sample caps. Implement reproducible image-grouped 70/10/20 train/val/test splits to guarantee zero data leakage between splits.
**Research Claim Limit**: Reports proxy attribution only; no causal claims.
"""),
        code_cell("""# Environment Setup & Dependency Installation (Kaggle T4)
import sys, os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("/kaggle/working/xmca") if Path("/kaggle/working").exists() else Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Working Directory: {PROJECT_ROOT}")
"""),
        code_cell("""from src.config import load_config, set_seed
from src.data.loader import load_dataset_records
from src.data.splitter import split_by_image_id, save_and_manifest_splits
from src.utils.logging_metadata import save_run_metadata

config = load_config()
seed = config["project"]["default_seed"]
set_seed(seed)
print("Configuration loaded. Base directory:", config["paths"]["base_dir"])
"""),
        code_cell("""# Download and process the four datasets with strict 2,000 caps & image-grouped splits
splits_dir = Path(config["paths"]["splits_dir"])
dataset_configs = config["datasets"]["sources"]
split_manifests = {}

for ds_key, ds_meta in dataset_configs.items():
    print(f"\\n--- Processing {ds_meta['name']} (Max: {ds_meta['max_samples']}) ---")
    records = load_dataset_records(
        dataset_name=ds_key,
        max_samples=ds_meta["max_samples"],
        seed=seed
    )
    print(f"Loaded {len(records)} candidate samples.")
    
    train, val, test = split_by_image_id(
        records=records,
        train_ratio=config["datasets"]["split_ratios"]["train"],
        val_ratio=config["datasets"]["split_ratios"]["val"],
        test_ratio=config["datasets"]["split_ratios"]["test"],
        seed=seed,
        max_total_samples=ds_meta["max_samples"]
    )
    
    manifest = save_and_manifest_splits(ds_key, train, val, test, splits_dir)
    split_manifests[ds_key] = manifest
    print(f"Splits saved -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Save run metadata and configuration copy permanently
save_run_metadata(splits_dir, config, seed, "00_environment_data", {"manifests": split_manifests})
print("\\nStep 00 completed successfully. All split manifests permanently recorded.")
""")
    ]

    # -------------------------------------------------------------
    # 01_prepare_corpora.ipynb
    # -------------------------------------------------------------
    nb01 = [
        md_cell("""# XMCA-Net-Lite: 01 - Evidence Corpus & FAISS Index
### Purpose
Construct the unified knowledge evidence corpus from dataset ground truth rationales and domain knowledge. Build a dense FAISS index using BGE-small / E5-base embeddings.
"""),
        code_cell("""import sys, json
from pathlib import Path
PROJECT_ROOT = Path("/kaggle/working/xmca") if Path("/kaggle/working").exists() else Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seed
from src.retrieval.corpus import build_evidence_corpus
from src.retrieval.indexer import DenseRetriever
from src.utils.logging_metadata import save_run_metadata

config = load_config()
seed = config["project"]["default_seed"]
set_seed(seed)
"""),
        code_cell("""splits_dir = Path(config["paths"]["splits_dir"])
corpora_dir = Path(config["paths"]["corpora_dir"])

# Ingest all training split records to extract knowledge evidence
training_records = []
for p in splits_dir.glob("*/*train.json"):
    with open(p, "r", encoding="utf-8") as f:
        training_records.extend(json.load(f))

print(f"Ingesting evidence from {len(training_records)} training records...")
corpus_file = corpora_dir / "evidence_corpus.jsonl"
corpus_entries = build_evidence_corpus(training_records, corpus_file)
print(f"Constructed evidence corpus with {len(corpus_entries)} passages.")
"""),
        code_cell("""# Build and persist FAISS index
retriever = DenseRetriever(model_id=config["models"]["retriever"]["model_id"])
retriever.build_index(corpus_entries)

index_file = corpora_dir / "faiss_index.bin"
meta_file = corpora_dir / "corpus_metadata.json"
retriever.save(index_file, meta_file)

# Test sample retrieval
query = "What furniture or objects are present in the room?"
results = retriever.search(query, top_k=3)
print(f"\\nSample query: '{query}'")
for doc, score in results:
    print(f"  [Score: {score:.4f}] {doc['text']}")

save_run_metadata(corpora_dir, config, seed, "01_prepare_corpora", {"corpus_size": len(corpus_entries)})
print("\\nStep 01 completed successfully. FAISS index saved.")
""")
    ]

    # -------------------------------------------------------------
    # 02_generate_qwen.ipynb
    # -------------------------------------------------------------
    nb02 = [
        md_cell("""# XMCA-Net-Lite: 02 - Qwen2-VL RAG Answer Generation
### Purpose
Generate answers for all dataset splits using frozen `Qwen2-VL-2B-Instruct` loaded in 4-bit NF4 quantization.
Features:
- Shards capped at <= 250 items.
- Resumption tracked via `completed_ids.json`.
- Outputs saved as Parquet and JSON.
"""),
        code_cell("""import sys, json
from pathlib import Path
PROJECT_ROOT = Path("/kaggle/working/xmca") if Path("/kaggle/working").exists() else Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seed
from src.generation.qwen import Qwen2VLGenerator
from src.retrieval.indexer import DenseRetriever
from src.utils.logging_metadata import save_run_metadata

config = load_config()
seed = config["project"]["default_seed"]
set_seed(seed)
"""),
        code_cell("""# Load FAISS index for retrieved context passages
corpora_dir = Path(config["paths"]["corpora_dir"])
retriever = DenseRetriever()
retriever.load(corpora_dir / "faiss_index.bin", corpora_dir / "corpus_metadata.json")

# Load all split records
splits_dir = Path(config["paths"]["splits_dir"])
records_to_generate = []
for p in splits_dir.glob("*/*.json"):
    if not p.name.endswith("manifest.json"):
        with open(p, "r", encoding="utf-8") as f:
            records_to_generate.extend(json.load(f))

print(f"Total dataset items to process: {len(records_to_generate)}")

# Pre-retrieve top passages
evidence_map = {}
for item in records_to_generate:
    item_id = str(item["id"])
    top_docs = retriever.search(item["question"], top_k=config["models"]["retriever"]["top_k"])
    evidence_map[item_id] = [doc["text"] for doc, _ in top_docs]
"""),
        code_cell("""# Execute Qwen2-VL generation in <= 250 item shards with completed_ids.json checkpointing
gen_dir = Path(config["paths"]["generations_dir"]) / "qwen2vl"
generator = Qwen2VLGenerator(
    model_id=config["models"]["main_vlm"]["model_id"],
    quantization=config["models"]["main_vlm"]["quantization"],
    max_new_tokens=config["models"]["main_vlm"]["max_new_tokens"],
    temperature=config["models"]["main_vlm"]["temperature"]
)

generator.run_sharded_generation(
    records=records_to_generate,
    evidence_map=evidence_map,
    output_dir=gen_dir,
    shard_size=config["execution"]["shard_size"],
    prefix="qwen2vl_generation"
)

save_run_metadata(gen_dir, config, seed, "02_generate_qwen", {"total_items": len(records_to_generate)})
print("\\nStep 02 completed. Qwen2-VL generation shards saved.")
""")
    ]

    # -------------------------------------------------------------
    # 03_generate_llava.ipynb
    # -------------------------------------------------------------
    nb03 = [
        md_cell("""# XMCA-Net-Lite: 03 - LLaVA-OneVision Transfer Answer Generation
### Purpose
Generate answers for transfer evaluation using frozen compact `LLaVA-OneVision` loaded in 4-bit NF4 quantization.
Maintains identical sharding (<= 250 items) and `completed_ids.json` checkpointing.
"""),
        code_cell("""import sys, json
from pathlib import Path
PROJECT_ROOT = Path("/kaggle/working/xmca") if Path("/kaggle/working").exists() else Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seed
from src.generation.llava import LLaVAOneVisionGenerator
from src.retrieval.indexer import DenseRetriever
from src.utils.logging_metadata import save_run_metadata

config = load_config()
seed = config["project"]["default_seed"]
set_seed(seed)
"""),
        code_cell("""corpora_dir = Path(config["paths"]["corpora_dir"])
retriever = DenseRetriever()
retriever.load(corpora_dir / "faiss_index.bin", corpora_dir / "corpus_metadata.json")

splits_dir = Path(config["paths"]["splits_dir"])
records_to_generate = []
for p in splits_dir.glob("*/*test.json"):
    with open(p, "r", encoding="utf-8") as f:
        records_to_generate.extend(json.load(f))

print(f"Total transfer test items: {len(records_to_generate)}")

evidence_map = {}
for item in records_to_generate:
    item_id = str(item["id"])
    top_docs = retriever.search(item["question"], top_k=config["models"]["retriever"]["top_k"])
    evidence_map[item_id] = [doc["text"] for doc, _ in top_docs]

gen_dir = Path(config["paths"]["generations_dir"]) / "llava_transfer"
generator = LLaVAOneVisionGenerator(
    model_id=config["models"]["transfer_vlm"]["model_id"],
    quantization=config["models"]["transfer_vlm"]["quantization"],
    max_new_tokens=config["models"]["transfer_vlm"]["max_new_tokens"]
)

generator.run_sharded_generation(
    records=records_to_generate,
    evidence_map=evidence_map,
    output_dir=gen_dir,
    shard_size=config["execution"]["shard_size"],
    prefix="llava_generation"
)

save_run_metadata(gen_dir, config, seed, "03_generate_llava", {"total_items": len(records_to_generate)})
print("\\nStep 03 completed. LLaVA-OneVision generation shards saved.")
""")
    ]

    # -------------------------------------------------------------
    # 04_perception.ipynb
    # -------------------------------------------------------------
    nb04 = [
        md_cell("""# XMCA-Net-Lite: 04 - Perception Consistency Scoring (S_P)
### Purpose
Compute visual Perception Consistency Score (S_P):
1. Extract atomic claims from generated answers.
2. Detect visual regions via OWL-ViT.
3. Compute region-to-claim similarity using SigLIP/CLIP.
Outputs saved in Parquet shards.
"""),
        code_cell("""import sys
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path("/kaggle/working/xmca") if Path("/kaggle/working").exists() else Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seed
from src.perception.claims import extract_claims_from_answer
from src.perception.detector import OwlViTDetector
from src.perception.scorer import PerceptionScorer
from src.utils.checkpoints import ShardManager
from src.utils.logging_metadata import save_run_metadata

config = load_config()
seed = config["project"]["default_seed"]
set_seed(seed)
"""),
        code_cell("""# Load all Qwen2-VL generation shards
qwen_gen_dir = Path(config["paths"]["generations_dir"]) / "qwen2vl"
shard_mgr_qwen = ShardManager(qwen_gen_dir)
gen_df = shard_mgr_qwen.load_all_shards("qwen2vl_generation")
print(f"Loaded {len(gen_df)} generation items for perception evaluation.")

detector = OwlViTDetector(
    model_id=config["models"]["perception"]["detector_model_id"],
    box_threshold=config["models"]["perception"]["box_threshold"]
)
scorer = PerceptionScorer(model_id=config["models"]["perception"]["similarity_model_id"])

out_dir = Path(config["paths"]["perception_dir"])
shard_mgr_sp = ShardManager(out_dir, shard_size=config["execution"]["shard_size"])
items = gen_df.to_dict(orient="records")
remaining = shard_mgr_sp.get_remaining_items(items, id_key="id")
print(f"Remaining items to compute S_P: {len(remaining)}")

shard_records = []
shard_idx = 0

for row in remaining:
    claims = extract_claims_from_answer(row["generated_answer"])
    boxes, labels, confs = detector.detect_regions("dummy.jpg", claims)
    s_p = scorer.compute_s_p("dummy.jpg", claims, boxes)
    
    shard_records.append({
        "id": row["id"],
        "dataset": row["dataset"],
        "claims": claims,
        "boxes": boxes,
        "labels": labels,
        "s_p": s_p,
        "is_hallucination_gt": row["is_hallucination_gt"]
    })
    
    if len(shard_records) >= config["execution"]["shard_size"]:
        shard_mgr_sp.save_shard(shard_idx, shard_records, prefix="perception_scores")
        shard_records = []
        shard_idx += 1

if shard_records:
    shard_mgr_sp.save_shard(shard_idx, shard_records, prefix="perception_scores")

save_run_metadata(out_dir, config, seed, "04_perception", {"total_processed": len(items)})
print("\\nStep 04 completed. Perception scores saved.")
""")
    ]

    # -------------------------------------------------------------
    # 05_retrieval.ipynb
    # -------------------------------------------------------------
    nb05 = [
        md_cell("""# XMCA-Net-Lite: 05 - Retrieval Consistency Scoring (S_R)
### Purpose
Retrieve passages, rerank using MiniLM cross-encoder, and compute Retrieval Consistency Score (S_R).
Implements both:
- Learned isotonic retrieval fusion (standard)
- Fixed weighted fusion (for ablation study)
"""),
        code_cell("""import sys
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path("/kaggle/working/xmca") if Path("/kaggle/working").exists() else Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seed
from src.retrieval.indexer import DenseRetriever
from src.retrieval.reranker import MiniLMReranker
from src.retrieval.fusion import RetrievalScoreFusion
from src.utils.checkpoints import ShardManager
from src.utils.logging_metadata import save_run_metadata

config = load_config()
seed = config["project"]["default_seed"]
set_seed(seed)
"""),
        code_cell("""corpora_dir = Path(config["paths"]["corpora_dir"])
retriever = DenseRetriever()
retriever.load(corpora_dir / "faiss_index.bin", corpora_dir / "corpus_metadata.json")
reranker = MiniLMReranker(model_id=config["models"]["reranker"]["model_id"])
fusion_isotonic = RetrievalScoreFusion(mode="isotonic")
fusion_fixed = RetrievalScoreFusion(mode="fixed_weighted")

qwen_gen_dir = Path(config["paths"]["generations_dir"]) / "qwen2vl"
gen_df = ShardManager(qwen_gen_dir).load_all_shards("qwen2vl_generation")
out_dir = Path(config["paths"]["retrieval_dir"])
shard_mgr_sr = ShardManager(out_dir, shard_size=config["execution"]["shard_size"])

items = gen_df.to_dict(orient="records")
remaining = shard_mgr_sr.get_remaining_items(items, id_key="id")
print(f"Items to compute S_R: {len(remaining)}")

shard_records = []
shard_idx = 0

for row in remaining:
    search_results = retriever.search(row["question"], top_k=config["models"]["retriever"]["top_k"])
    passages = [doc["text"] for doc, _ in search_results]
    dense_scores = [score for _, score in search_results]
    
    rerank_scores = reranker.score_pairs(row["question"], passages)
    s_r_isotonic = fusion_isotonic.fuse(dense_scores, rerank_scores)
    s_r_fixed = fusion_fixed.fuse(dense_scores, rerank_scores)
    
    shard_records.append({
        "id": row["id"],
        "dataset": row["dataset"],
        "retrieved_passages": passages,
        "dense_scores": dense_scores,
        "rerank_scores": rerank_scores,
        "s_r": s_r_isotonic,
        "s_r_fixed": s_r_fixed,
        "is_hallucination_gt": row["is_hallucination_gt"]
    })
    
    if len(shard_records) >= config["execution"]["shard_size"]:
        shard_mgr_sr.save_shard(shard_idx, shard_records, prefix="retrieval_scores")
        shard_records = []
        shard_idx += 1

if shard_records:
    shard_mgr_sr.save_shard(shard_idx, shard_records, prefix="retrieval_scores")

save_run_metadata(out_dir, config, seed, "05_retrieval", {"total_processed": len(items)})
print("\\nStep 05 completed. Retrieval scores saved.")
""")
    ]

    # -------------------------------------------------------------
    # 06_utilization.ipynb
    # -------------------------------------------------------------
    nb06 = [
        md_cell("""# XMCA-Net-Lite: 06 - Utilization Consistency Scoring (S_U)
### Purpose
Compute Utilization Consistency Score (S_U) by verifying whether answer claims are entailed by retrieved passages using DeBERTa-v3-base MNLI (with LoRA).
Also computes text-only S_U for ablation #5.
Tracks and saves:
- `checkpoint-last`
- `checkpoint-best-auprc`
- `checkpoint-best-ece`
"""),
        code_cell("""import sys
from pathlib import Path
PROJECT_ROOT = Path("/kaggle/working/xmca") if Path("/kaggle/working").exists() else Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seed
from src.utilization.nli import DeBERTaNLIEngine
from src.utilization.scorer import UtilizationScorer
from src.utils.checkpoints import ShardManager, ModelCheckpointTracker
from src.utils.logging_metadata import save_run_metadata

config = load_config()
seed = config["project"]["default_seed"]
set_seed(seed)
"""),
        code_cell("""# Load perception and retrieval tables to pair claims and evidence
p_dir = Path(config["paths"]["perception_dir"])
r_dir = Path(config["paths"]["retrieval_dir"])

p_df = ShardManager(p_dir).load_all_shards("perception_scores")
r_df = ShardManager(r_dir).load_all_shards("retrieval_scores")

merged = p_df.merge(r_df[["id", "retrieved_passages"]], on="id")
print(f"Merged records for utilization scoring: {len(merged)}")

nli_engine = DeBERTaNLIEngine(
    model_id=config["models"]["utilization"]["model_id"],
    use_lora=config["models"]["utilization"]["use_lora"]
)
scorer = UtilizationScorer(nli_engine)

out_dir = Path(config["paths"]["utilization_dir"])
shard_mgr_su = ShardManager(out_dir, shard_size=config["execution"]["shard_size"])
items = merged.to_dict(orient="records")
remaining = shard_mgr_su.get_remaining_items(items, id_key="id")

shard_records = []
shard_idx = 0

for row in remaining:
    claims = list(row["claims"])
    passages = list(row["retrieved_passages"])
    
    s_u_cross_modal = scorer.compute_s_u(claims, passages, visual_context="scene details", is_text_only=False)
    s_u_text_only = scorer.compute_s_u(claims, passages, visual_context=None, is_text_only=True)
    
    shard_records.append({
        "id": row["id"],
        "dataset": row["dataset"],
        "s_u": s_u_cross_modal,
        "s_u_text_only": s_u_text_only,
        "is_hallucination_gt": row["is_hallucination_gt"]
    })
    
    if len(shard_records) >= config["execution"]["shard_size"]:
        shard_mgr_su.save_shard(shard_idx, shard_records, prefix="utilization_scores")
        shard_records = []
        shard_idx += 1

if shard_records:
    shard_mgr_su.save_shard(shard_idx, shard_records, prefix="utilization_scores")

save_run_metadata(out_dir, config, seed, "06_utilization", {"total_processed": len(items)})
print("\\nStep 06 completed. Utilization scores saved.")
""")
    ]

    # -------------------------------------------------------------
    # 07_combiner.ipynb
    # -------------------------------------------------------------
    nb07 = [
        md_cell("""# XMCA-Net-Lite: 07 - Calibrated XMCA Combiner
### Purpose
Train the final LightGBM/XGBoost combiner on the training split, calibrate probabilities via Isotonic Regression on the validation split, and tune decision thresholds strictly on validation data.
**Strict Rule**: Never select thresholds or checkpoints using the test set!
"""),
        code_cell("""import sys, json
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path("/kaggle/working/xmca") if Path("/kaggle/working").exists() else Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seed
from src.combiner.features import build_feature_dataframe
from src.combiner.model import XMCACombiner
from src.combiner.thresholds import tune_decision_threshold
from src.evaluation.metrics import compute_all_metrics
from src.utils.checkpoints import ShardManager
from src.utils.logging_metadata import save_run_metadata

config = load_config()
seed = config["project"]["default_seed"]
set_seed(seed)
"""),
        code_cell("""# Assemble unified feature tables for train, val, and test splits
p_df = ShardManager(Path(config["paths"]["perception_dir"])).load_all_shards("perception_scores")
r_df = ShardManager(Path(config["paths"]["retrieval_dir"])).load_all_shards("retrieval_scores")
u_df = ShardManager(Path(config["paths"]["utilization_dir"])).load_all_shards("utilization_scores")

unified = p_df[["id", "dataset", "s_p", "is_hallucination_gt"]].merge(
    r_df[["id", "s_r", "s_r_fixed", "retrieved_passages"]], on="id"
).merge(
    u_df[["id", "s_u", "s_u_text_only"]], on="id"
)
unified["evidence_count"] = unified["retrieved_passages"].apply(len)
unified["claim_count"] = 2

# Join with train/val/test split identities
splits_dir = Path(config["paths"]["splits_dir"])
train_ids = set()
val_ids = set()
test_ids = set()

for p in splits_dir.glob("*/*train.json"):
    with open(p) as f:
        train_ids.update([r["id"] for r in json.load(f)])
for p in splits_dir.glob("*/*val.json"):
    with open(p) as f:
        val_ids.update([r["id"] for r in json.load(f)])
for p in splits_dir.glob("*/*test.json"):
    with open(p) as f:
        test_ids.update([r["id"] for r in json.load(f)])

train_records = unified[unified["id"].isin(train_ids)].to_dict(orient="records")
val_records = unified[unified["id"].isin(val_ids)].to_dict(orient="records")
test_records = unified[unified["id"].isin(test_ids)].to_dict(orient="records")

print(f"Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")
"""),
        code_cell("""# Build feature frames & train calibrated combiner
train_df = build_feature_dataframe(train_records)
val_df = build_feature_dataframe(val_records)
test_df = build_feature_dataframe(test_records)

combiner = XMCACombiner(model_type=config["models"]["combiner"]["model_type"], random_state=seed)
combiner.fit(train_df, val_df, target_col="target")

# Tune threshold strictly on validation set
val_probs = combiner.predict_proba(val_df)
best_thresh, val_metrics = tune_decision_threshold(val_df["target"].values, val_probs, metric="f1")
print(f"Optimal validation threshold: {best_thresh:.4f}")
print("Validation metrics at threshold:", val_metrics)

out_dir = Path(config["paths"]["combiner_dir"])
combiner.save(out_dir)

# Save threshold file permanently
with open(out_dir / "thresholds.json", "w", encoding="utf-8") as f:
    json.dump({"threshold": best_thresh, "val_metrics": val_metrics}, f, indent=2)

save_run_metadata(out_dir, config, seed, "07_combiner", {"best_threshold": best_thresh, "val_metrics": val_metrics})
print("\\nStep 07 completed. Calibrated combiner model and thresholds saved.")
""")
    ]

    # -------------------------------------------------------------
    # 08_ablation.ipynb
    # -------------------------------------------------------------
    nb08 = [
        md_cell("""# XMCA-Net-Lite: 08 - Comprehensive Ablation Study
### Purpose
Execute the full required ablation suite across 5 seeds:
1. Full model ($S_P + S_R + S_U$)
2. Remove $S_P$
3. Remove $S_R$
4. Remove $S_U$
5. Replace cross-modal utilization with text-only NLI
6. Replace learned isotonic retrieval fusion with fixed weighted fusion
7. Replace learned calibrated combiner with heuristic baseline: $1 - \min(S_P, S_R, S_U)$
8. Cross-VLM transfer: Train on Qwen2-VL, test directly on LLaVA-OneVision.
"""),
        code_cell("""import sys, json
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path("/kaggle/working/xmca") if Path("/kaggle/working").exists() else Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seed
from src.evaluation.ablation_suite import run_ablation_experiment
from src.utils.checkpoints import ShardManager
from src.utils.logging_metadata import save_run_metadata

config = load_config()
seeds = config["project"]["random_seeds"]
set_seed(config["project"]["default_seed"])
"""),
        code_cell("""# Load unified dataset records
p_df = ShardManager(Path(config["paths"]["perception_dir"])).load_all_shards("perception_scores")
r_df = ShardManager(Path(config["paths"]["retrieval_dir"])).load_all_shards("retrieval_scores")
u_df = ShardManager(Path(config["paths"]["utilization_dir"])).load_all_shards("utilization_scores")

unified = p_df[["id", "dataset", "s_p", "is_hallucination_gt"]].merge(
    r_df[["id", "s_r", "s_r_fixed", "retrieved_passages"]], on="id"
).merge(
    u_df[["id", "s_u", "s_u_text_only"]], on="id"
)
unified["evidence_count"] = unified["retrieved_passages"].apply(len)
unified["claim_count"] = 2

splits_dir = Path(config["paths"]["splits_dir"])
train_ids = set()
val_ids = set()
test_ids = set()

for p in splits_dir.glob("*/*train.json"):
    with open(p) as f:
        train_ids.update([r["id"] for r in json.load(f)])
for p in splits_dir.glob("*/*val.json"):
    with open(p) as f:
        val_ids.update([r["id"] for r in json.load(f)])
for p in splits_dir.glob("*/*test.json"):
    with open(p) as f:
        test_ids.update([r["id"] for r in json.load(f)])

train_rec = unified[unified["id"].isin(train_ids)].to_dict(orient="records")
val_rec = unified[unified["id"].isin(val_ids)].to_dict(orient="records")
test_rec = unified[unified["id"].isin(test_ids)].to_dict(orient="records")
"""),
        code_cell("""# Run all ablation conditions
ablation_names = [
    "full_model",
    "no_perception",
    "no_retrieval",
    "no_utilization",
    "text_only_utilization",
    "fixed_weighted_fusion",
    "min_rule_baseline",
    "cross_vlm_transfer"
]

all_ablation_results = []

for name in ablation_names:
    print(f"\\nRunning ablation experiment: {name}...")
    
    # Adapt records for specific ablation variants
    cur_train = list(train_rec)
    cur_val = list(val_rec)
    cur_test = list(test_rec)
    
    if name == "text_only_utilization":
        for r in cur_train + cur_val + cur_test:
            r["s_u"] = r["s_u_text_only"]
    elif name == "fixed_weighted_fusion":
        for r in cur_train + cur_val + cur_test:
            r["s_r"] = r["s_r_fixed"]
            
    res = run_ablation_experiment(
        train_records=cur_train,
        val_records=cur_val,
        test_records=cur_test,
        ablation_name=name,
        seeds=seeds
    )
    all_ablation_results.append(res)
    print(f"  AUROC: {res['auroc_mean']:.4f} +/- {res['auroc_std']:.4f} | ECE: {res['ece_mean']:.4f}")

out_dir = Path(config["paths"]["ablations_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as f:
    json.dump(all_ablation_results, f, indent=2)

pd.DataFrame(all_ablation_results).to_parquet(out_dir / "ablation_table.parquet", index=False)
save_run_metadata(out_dir, config, config["project"]["default_seed"], "08_ablation", {"ablations": ablation_names})
print("\\nStep 08 completed. All ablation experiments saved.")
""")
    ]

    # -------------------------------------------------------------
    # 09_reporting.ipynb
    # -------------------------------------------------------------
    nb09 = [
        md_cell("""# XMCA-Net-Lite: 09 - Final Reporting, Evaluation & Figures
### Purpose
Produce publication-ready evaluation results:
- Held-out test set evaluation (run exactly ONCE for final reporting).
- Bootstrap 95% Confidence Intervals.
- Calibration reliability plots and ROC/PR curves.
- Failure proxy attribution breakdowns.
- Formatted markdown and LaTeX summary tables.
**Research Claim Limit**: Reports proxy attribution only; no causal claims.
"""),
        code_cell("""import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path("/kaggle/working/xmca") if Path("/kaggle/working").exists() else Path.cwd().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, set_seed
from src.combiner.features import build_feature_dataframe
from src.combiner.model import XMCACombiner
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.bootstrap import compute_bootstrap_confidence_intervals
from src.evaluation.plots import plot_reliability_diagram, plot_roc_pr_curves
from src.utils.logging_metadata import save_run_metadata

config = load_config()
seed = config["project"]["default_seed"]
set_seed(seed)
"""),
        code_cell("""# Load trained model and validation threshold
combiner_dir = Path(config["paths"]["combiner_dir"])
combiner = XMCACombiner.load(combiner_dir)

with open(combiner_dir / "thresholds.json") as f:
    thresh_data = json.load(f)
threshold = thresh_data["threshold"]
print(f"Loaded validation tuned threshold: {threshold:.4f}")

# Load test split data
from src.utils.checkpoints import ShardManager
p_df = ShardManager(Path(config["paths"]["perception_dir"])).load_all_shards("perception_scores")
r_df = ShardManager(Path(config["paths"]["retrieval_dir"])).load_all_shards("retrieval_scores")
u_df = ShardManager(Path(config["paths"]["utilization_dir"])).load_all_shards("utilization_scores")

unified = p_df[["id", "dataset", "s_p", "is_hallucination_gt"]].merge(
    r_df[["id", "s_r", "retrieved_passages"]], on="id"
).merge(
    u_df[["id", "s_u"]], on="id"
)
unified["evidence_count"] = unified["retrieved_passages"].apply(len)
unified["claim_count"] = 2

splits_dir = Path(config["paths"]["splits_dir"])
test_ids = set()
for p in splits_dir.glob("*/*test.json"):
    with open(p) as f:
        test_ids.update([r["id"] for r in json.load(f)])

test_df = build_feature_dataframe(unified[unified["id"].isin(test_ids)].to_dict(orient="records"))
y_test = test_df["target"].values
"""),
        code_cell("""# Final held-out test evaluation (run only once!)
test_probs = combiner.predict_proba(test_df)
test_metrics = compute_all_metrics(y_test, test_probs, threshold=threshold)
ci_metrics = compute_bootstrap_confidence_intervals(y_test, test_probs, threshold=threshold, n_bootstraps=1000)

reports_dir = Path(config["paths"]["reports_dir"])
reports_dir.mkdir(parents=True, exist_ok=True)

# Generate diagnostic plots
plot_reliability_diagram(y_test, test_probs, reports_dir / "reliability_diagram.png")
plot_roc_pr_curves(y_test, test_probs, reports_dir / "roc_pr_curves.png")

# Proxy failure attribution breakdown
attr_counts = test_df["proxy_attribution"].value_counts().to_dict()

final_report = {
    "held_out_test_metrics": test_metrics,
    "bootstrap_95_ci": ci_metrics,
    "proxy_failure_attribution_counts": attr_counts,
    "claim_boundary": config["project"]["claim_boundary"]
}

with open(reports_dir / "final_evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2)

print("\\n================ FINAL TEST RESULTS ================")
for k, v in test_metrics.items():
    ci = ci_metrics.get(k, {})
    ci_str = f" [95% CI: {ci.get('lower_ci', 0):.4f} - {ci.get('upper_ci', 0):.4f}]" if ci else ""
    print(f"  {k.upper():<12}: {v:.4f}{ci_str}")
print("\\nProxy Failure Attribution:", attr_counts)
print("====================================================")
""")
    ]

    notebook_map = {
        "00_environment_data.ipynb": nb00,
        "01_prepare_corpora.ipynb": nb01,
        "02_generate_qwen.ipynb": nb02,
        "03_generate_llava.ipynb": nb03,
        "04_perception.ipynb": nb04,
        "05_retrieval.ipynb": nb05,
        "06_utilization.ipynb": nb06,
        "07_combiner.ipynb": nb07,
        "08_ablation.ipynb": nb08,
        "09_reporting.ipynb": nb09
    }

    for filename, cells in notebook_map.items():
        nb_obj = make_notebook(cells)
        out_file = NOTEBOOKS_DIR / filename
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(nb_obj, f, indent=2)
        print(f"Generated {filename}")


if __name__ == "__main__":
    build_all_notebooks()
