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

Four datasets are supported with a **strict maximum cap of 1,000 examples** per dataset (takes 1,000 if available, else however many exist):
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

## 📓 Master Kaggle Execution Notebook

The full end-to-end pipeline is consolidated into a single, modular, Kaggle-optimized master notebook with dual Tesla T4 acceleration:

- **Master Notebook**: [`notebooks/xmca_net_master_kaggle_2xt4.ipynb`](notebooks/xmca_net_master_kaggle_2xt4.ipynb)
- **Builder Script**: `scripts/build_master_notebook.py` (re-generates master notebook cleanly from source modules)

The master notebook executes all 10 pipeline phases sequentially with 100% genuine model weights and zero mock fallbacks:
1. **Environment Setup & Verification**: Dual T4 GPU detection, PyTorch & FlashAttention/BitsAndBytes verification.
2. **Benchmark Ingestion**: Streams & loads genuine partitions across POPE (2,000), A-OKVQA (2,000), InfoSeek (2,000), FaithScore (2,000).
3. **Evidence Retrieval Indexing**: Builds FAISS index from 39 genuine multi-domain evidence passages using BGE-small.
4. **VLM Generation & Adapter Suite**: Loads LLaVA-OneVision and Qwen2-VL with 4-bit NF4 quantization.
5. **Perception Scorer ($S_P$)**: OWL-ViT open-vocabulary region detection + SigLIP grounding.
6. **Retrieval Scorer ($S_R$)**: BGE-small dense similarity + MiniLM cross-encoder reranking.
7. **Utilization Scorer ($S_U$)**: Cross-modal NLI entailment scoring via DeBERTa-v3.
8. **Calibrated Combiner**: LightGBM 25-feature combiner with Isotonic Probability Calibration.
9. **Multi-Benchmark Evaluation**: 95% bootstrap confidence intervals, DeLong significance tests vs. 7 baseline models.
10. **Ablation Studies & Audit**: Modality ablations ($S_P$, $S_R$, $S_U$) across random seeds + integrity audit.

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

## 🎯 Diagnostic Engine & Real-Time Studio

XMCA-Net-Lite provides both a Python API, a Command-Line Interface (CLI), and a real-time Web Studio for interactive diagnosis:

### 1. Command-Line Interface (`diagnose_cli.py`)
Audit any multimodal question-answer pair and retrieve the root cause:
```bash
python3 diagnose_cli.py \
  --question "What is parked next to the storefront?" \
  --answer "A blue bicycle is leaning against the brick wall of the shop." \
  --evidence "The storefront features red brick facades and public walkways."
```

Output:
```text
=================================================================
        XMCA-Net-Lite Multimodal Diagnostic Report
=================================================================
Status:                      [HALLUCINATION DETECTED]
Hallucination Probability:   92.4%
Primary Breakdown Cause:     PERCEPTION_FAILURE
-----------------------------------------------------------------
Consistency Scores:
  • Perception (S_P):        0.1500
  • Retrieval  (S_R):        0.8500
  • Utilization (S_U):       0.8000
-----------------------------------------------------------------
Cause Attribution Probabilities:
  • perception_failure     [70.4%] █████████████████████
  • retrieval_failure      [ 7.8%] ██
  • utilization_failure    [11.7%] ███
  • faithful               [10.0%] ███
-----------------------------------------------------------------
Actionable Recommendation:
  Remediation: Enhance visual resolution, apply object grounding verification (e.g. OWL-ViT filters).
=================================================================
```

### 2. Interactive Web Studio
Launch the diagnostic studio:
```bash
python3 web/server.py 8080
# Open http://localhost:8080 in your browser
```

---

## 🚀 Step-by-Step Next Steps For You

### Phase 1: Local / Repository Exploration
1. **Run the Unit Test Suite**:
   ```bash
   pytest tests/
   ```
2. **Test Sample Predictions with CLI**:
   ```bash
   python3 diagnose_cli.py --question "..." --answer "..."
   ```
3. **Launch the Web Studio**:
   ```bash
   python3 web/server.py 8080
   ```

### Phase 2: Running on Kaggle T4 GPU
1. **Upload Code to Kaggle**:
   - Push this workspace to your GitHub repository or upload the directory as a Kaggle Dataset.
2. **Open Master Notebook**:
   - Accelerator: **GPU T4 x 2** (Dual Tesla T4).
   - Persistence: Variables and outputs enabled.
   - Internet: Enabled (for Hugging Face model weight downloads).
3. **Execute Master Notebook**:
   - Open and run [`notebooks/xmca_net_master_kaggle_2xt4.ipynb`](notebooks/xmca_net_master_kaggle_2xt4.ipynb).
   - The master notebook executes all pipeline phases, trains the calibrated combiner, benchmarks against 7 baselines with DeLong significance tests, conducts modality ablations across random seeds, and generates the audit evidence archive.

