# XMCA-Net-Lite: Research-Grade RAG-VLM Diagnostic System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hardware](https://img.shields.io/badge/target-Kaggle%20T4%20GPU-orange.svg)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

XMCA-Net-Lite is a diagnostic system for Retrieval-Augmented Vision-Language Models (RAG-VLMs). It predicts whether a VLM generation is hallucinated by combining three consistency signals:
1. **Perception Consistency ($S_P$)**: Verifies whether answer claims are visually grounded in the image regions.
2. **Retrieval Consistency ($S_R$)**: Verifies whether retrieved knowledge is relevant to the multimodal query using dense indexing and cross-encoder reranking.
3. **Utilization Consistency ($S_U$)**: Verifies whether answer claims are entailed by the retrieved evidence using Natural Language Inference (NLI).

---

## ⚠️ Research Claim Boundaries

In strict adherence to research integrity standards:
- **Allowed Claims**:
  - XMCA-Net-Lite combines three consistency signals for hallucination detection.
  - The system provides calibrated hallucination probabilities ($P(\text{hallucination} \mid X) \in [0, 1]$).
  - Ablations quantify the marginal diagnostic value of each module.
  - Cross-modal utilization can be directly compared against text-only utilization.
  - The detector can be evaluated for cross-VLM backbone transfer.
- **Explicitly Prohibited Claims**:
  - **No causal attribution claims**: Only *proxy failure attribution* is reported (based on the lowest individual consistency score).
  - **No claims of ground-truth perception, retrieval, or utilization failure causes**.
  - **No claims of creating the new XMCA-Diag dataset**.
  - **No claims of beating seven baselines** until all controlled, identical-data comparisons are finalized.

---

## 📊 Datasets & Splits

Four datasets are supported with a **strict maximum cap of 2,000 examples** per dataset:
- **POPE**: Object hallucination and perception evaluation.
- **FaithScore / LLaVA-1k subset**: Free-form image-grounded faithfulness.
- **A-OKVQA**: Knowledge-based visual question answering with retrieval.
- **InfoSeek**: Entity-centric retrieval question answering.

### Data Leakage Guarantee
Splits are partitioned into **70% Train, 10% Validation, and 20% Test** grouped strictly by **`image_id`**. This guarantees that no image appears in more than one split. Split files and SHA-256 hashes are permanently manifested.

---

## 🧠 Model Architecture

- **Main VLM**: `Qwen2-VL-2B-Instruct` (Frozen, loaded in 4-bit NF4).
- **Transfer VLM**: `LLaVA-OneVision` checkpoint (Frozen, loaded in 4-bit NF4).
- **Retriever**: `BGE-small-en-v1.5` embeddings with FAISS IndexFlatIP (cosine similarity).
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Perception Module**: `OWL-ViT` open-vocabulary region detection + `SigLIP` region-claim similarity ($S_P$).
- **Utilization Module**: `DeBERTa-v3-base-mnli` with PEFT LoRA adapters ($S_U$).
- **Combiner**: `LightGBM` / `XGBoost` classifier with **Isotonic Regression calibration**.

---

## 📓 Notebook Sequence

The project is structured into 10 modular, Kaggle-ready notebooks:

| Notebook | Purpose | Permanent Artifact |
|---|---|---|
| [`00_environment_data.ipynb`](notebooks/00_environment_data.ipynb) | Dataset download & image-grouped splits | Manifests & Parquet split files |
| [`01_prepare_corpora.ipynb`](notebooks/01_prepare_corpora.ipynb) | Evidence corpus & FAISS index creation | `evidence_corpus.jsonl`, `faiss_index.bin` |
| [`02_generate_qwen.ipynb`](notebooks/02_generate_qwen.ipynb) | Generate Qwen2-VL RAG answers | Generation Parquet shards (`<= 250` items) |
| [`03_generate_llava.ipynb`](notebooks/03_generate_llava.ipynb) | Generate LLaVA transfer answers | Transfer generation Parquet shards |
| [`04_perception.ipynb`](notebooks/04_perception.ipynb) | Compute claims & Perception score ($S_P$) | Perception score tables |
| [`05_retrieval.ipynb`](notebooks/05_retrieval.ipynb) | Retrieve, rerank, and compute $S_R$ | Evidence & retrieval score tables |
| [`06_utilization.ipynb`](notebooks/06_utilization.ipynb) | Train/load NLI & compute $S_U$ | Utilization score tables & LoRA checkpoint |
| [`07_combiner.ipynb`](notebooks/07_combiner.ipynb) | Train calibrated combiner & tune threshold | `xmca_combiner_model.joblib`, `thresholds.json` |
| [`08_ablation.ipynb`](notebooks/08_ablation.ipynb) | Run 8 ablation experiments across 5 seeds | `ablation_results.json`, `ablation_table.parquet` |
| [`09_reporting.ipynb`](notebooks/09_reporting.ipynb) | Held-out test evaluation & publication plots | Final JSON report, reliability diagrams, ROC/PR |

---

## 💾 Checkpointing & Resumption

- **Sharded Execution**: All inference and scoring processes operate in shards of at most **250 examples**.
- **Resumption**: A `completed_ids.json` ledger is atomically updated after each shard to ensure crash-proof recovery.
- **Model Checkpoints**:
  - `checkpoint-last`
  - `checkpoint-best-auprc` (evaluated strictly on validation data)
  - `checkpoint-best-ece` (evaluated strictly on validation data)
- **Strict Rule**: Held-out test data is evaluated **only once** for final reporting and is never used for checkpoint or threshold selection.

---

## 🧪 Ablation Experiments (5 Seeds)

The ablation suite evaluates 8 distinct experimental conditions across seeds `[42, 101, 202, 303, 404]`:
1. **Full Model**: $S_P + S_R + S_U$ + interaction features.
2. **Remove $S_P$**: Evaluate without perception score.
3. **Remove $S_R$**: Evaluate without retrieval score.
4. **Remove $S_U$**: Evaluate without utilization score.
5. **Text-only Utilization**: Compare against text-only NLI (no visual context in premise).
6. **Fixed Weighted Retrieval Fusion**: Replace learned isotonic fusion with fixed weights ($0.5 \times S_{\text{dense}} + 0.5 \times S_{\text{rerank}}$).
7. **Heuristic Baseline**: Heuristic $1 - \min(S_P, S_R, S_U)$ combiner.
8. **Cross-VLM Transfer**: Combiner trained on Qwen2-VL, tested directly on LLaVA-OneVision.

---

## 🚀 Quickstart

### 1. Local Testing & Verification
```bash
# Run unit test suite
pytest tests/

# Run complete end-to-end integration smoke test
python3 scripts/smoke_test_pipeline.py
```

### 2. Kaggle T4 Deployment
1. Upload this repository as a Kaggle Dataset or clone it into `/kaggle/working/xmca`.
2. Attach a Kaggle T4 GPU instance.
3. Open and run notebooks sequentially: `00_environment_data.ipynb` through `09_reporting.ipynb`.
