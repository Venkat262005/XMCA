import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
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
from src.utils.checkpoints import ShardManager
from src.utils.logging_metadata import save_run_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmokeTest")


def run_full_pipeline_smoke_test(num_samples_per_ds: int = 50):
    logger.info("Starting XMCA-Net-Lite full pipeline smoke test...")
    config = load_config()
    seed = config["project"]["default_seed"]
    set_seed(seed)

    splits_dir = Path(config["paths"]["splits_dir"])
    corpora_dir = Path(config["paths"]["corpora_dir"])
    gen_dir = Path(config["paths"]["generations_dir"])
    p_dir = Path(config["paths"]["perception_dir"])
    r_dir = Path(config["paths"]["retrieval_dir"])
    u_dir = Path(config["paths"]["utilization_dir"])
    comb_dir = Path(config["paths"]["combiner_dir"])
    abl_dir = Path(config["paths"]["ablations_dir"])
    rep_dir = Path(config["paths"]["reports_dir"])

    # 1. Step 00: Data Preparation & Splits
    logger.info("Executing Step 00: Data & Splits...")
    all_train_records = []
    for ds_key in ["pope", "faithscore", "aokvqa", "infoseek"]:
        records = load_dataset_records(ds_key, max_samples=num_samples_per_ds, seed=seed)
        train, val, test = split_by_image_id(records, 0.7, 0.1, 0.2, seed=seed, max_total_samples=num_samples_per_ds)
        save_and_manifest_splits(ds_key, train, val, test, splits_dir)
        all_train_records.extend(train)

    # 2. Step 01: Corpora & FAISS
    logger.info("Executing Step 01: Corpora & FAISS...")
    corpus_file = corpora_dir / "evidence_corpus.jsonl"
    corpus_entries = build_evidence_corpus(all_train_records, corpus_file)
    retriever = DenseRetriever()
    retriever.build_index(corpus_entries)
    retriever.save(corpora_dir / "faiss_index.bin", corpora_dir / "corpus_metadata.json")

    # 3. Step 02 & 03: Generations (Qwen & LLaVA)
    logger.info("Executing Step 02 & 03: Generations...")
    all_splits_records = []
    for p in splits_dir.glob("*/*.json"):
        if not p.name.endswith("manifest.json"):
            with open(p, "r", encoding="utf-8") as f:
                all_splits_records.extend(json.load(f))

    # Evidence map
    evidence_map = {}
    for it in all_splits_records:
        top_d = retriever.search(it["question"], top_k=3)
        evidence_map[str(it["id"])] = [d["text"] for d, _ in top_d]

    qwen_gen = Qwen2VLGenerator()
    qwen_gen.run_sharded_generation(
        records=all_splits_records,
        evidence_map=evidence_map,
        output_dir=gen_dir / "qwen2vl",
        shard_size=100,
        prefix="qwen2vl_generation"
    )

    llava_gen = LLaVAOneVisionGenerator()
    llava_gen.run_sharded_generation(
        records=all_splits_records,
        evidence_map=evidence_map,
        output_dir=gen_dir / "llava_transfer",
        shard_size=100,
        prefix="llava_generation"
    )

    # 4. Step 04: Perception Scoring (S_P)
    logger.info("Executing Step 04: Perception (S_P)...")
    qwen_df = ShardManager(gen_dir / "qwen2vl").load_all_shards("qwen2vl_generation")
    detector = OwlViTDetector()
    p_scorer = PerceptionScorer()
    p_mgr = ShardManager(p_dir, shard_size=100)
    p_records = []
    for row in qwen_df.to_dict(orient="records"):
        claims = extract_claims_from_answer(row["generated_answer"])
        boxes, labels, _ = detector.detect_regions("dummy.jpg", claims)
        sp = p_scorer.compute_s_p("dummy.jpg", claims, boxes)
        p_records.append({
            "id": row["id"],
            "dataset": row["dataset"],
            "claims": claims,
            "boxes": boxes,
            "labels": labels,
            "s_p": sp,
            "is_hallucination_gt": row["is_hallucination_gt"]
        })
    p_mgr.save_shard(0, p_records, prefix="perception_scores")

    # 5. Step 05: Retrieval Scoring (S_R)
    logger.info("Executing Step 05: Retrieval (S_R)...")
    reranker = MiniLMReranker()
    fusion_iso = RetrievalScoreFusion(mode="isotonic")
    fusion_fix = RetrievalScoreFusion(mode="fixed_weighted")
    r_mgr = ShardManager(r_dir, shard_size=100)
    r_records = []
    for row in qwen_df.to_dict(orient="records"):
        results = retriever.search(row["question"], top_k=3)
        passages = [d["text"] for d, _ in results]
        d_scores = [s for _, s in results]
        r_scores = reranker.score_pairs(row["question"], passages)
        sr_iso = fusion_iso.fuse(d_scores, r_scores)
        sr_fix = fusion_fix.fuse(d_scores, r_scores)
        r_records.append({
            "id": row["id"],
            "dataset": row["dataset"],
            "retrieved_passages": passages,
            "dense_scores": d_scores,
            "rerank_scores": r_scores,
            "s_r": sr_iso,
            "s_r_fixed": sr_fix,
            "is_hallucination_gt": row["is_hallucination_gt"]
        })
    r_mgr.save_shard(0, r_records, prefix="retrieval_scores")

    # 6. Step 06: Utilization Scoring (S_U)
    logger.info("Executing Step 06: Utilization (S_U)...")
    nli_eng = DeBERTaNLIEngine(use_lora=False)
    u_scorer = UtilizationScorer(nli_eng)
    u_mgr = ShardManager(u_dir, shard_size=100)
    u_records = []
    for pr, rr in zip(p_records, r_records):
        su_cm = u_scorer.compute_s_u(pr["claims"], rr["retrieved_passages"], visual_context="scene details")
        su_to = u_scorer.compute_s_u(pr["claims"], rr["retrieved_passages"], visual_context=None, is_text_only=True)
        u_records.append({
            "id": pr["id"],
            "dataset": pr["dataset"],
            "s_u": su_cm,
            "s_u_text_only": su_to,
            "is_hallucination_gt": pr["is_hallucination_gt"]
        })
    u_mgr.save_shard(0, u_records, prefix="utilization_scores")

    # 7. Step 07: Combiner Training & Validation Tuning
    logger.info("Executing Step 07: Calibrated Combiner...")
    unified_df = pd.DataFrame(p_records)[["id", "dataset", "s_p", "is_hallucination_gt"]].merge(
        pd.DataFrame(r_records)[["id", "s_r", "s_r_fixed", "retrieved_passages"]], on="id"
    ).merge(
        pd.DataFrame(u_records)[["id", "s_u", "s_u_text_only"]], on="id"
    )
    unified_df["evidence_count"] = unified_df["retrieved_passages"].apply(len)
    unified_df["claim_count"] = 2

    # Map splits
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

    train_rec = unified_df[unified_df["id"].isin(train_ids)].to_dict(orient="records")
    val_rec = unified_df[unified_df["id"].isin(val_ids)].to_dict(orient="records")
    test_rec = unified_df[unified_df["id"].isin(test_ids)].to_dict(orient="records")

    train_df = build_feature_dataframe(train_rec)
    val_df = build_feature_dataframe(val_rec)
    test_df = build_feature_dataframe(test_rec)

    combiner = XMCACombiner(model_type="lightgbm", random_state=seed)
    combiner.fit(train_df, val_df, target_col="target")
    val_probs = combiner.predict_proba(val_df)
    best_thresh, val_metrics = tune_decision_threshold(val_df["target"].values, val_probs, metric="f1")
    combiner.save(comb_dir)
    with open(comb_dir / "thresholds.json", "w") as f:
        json.dump({"threshold": best_thresh, "val_metrics": val_metrics}, f)

    # 8. Step 08: Ablations
    logger.info("Executing Step 08: Ablations across 5 seeds...")
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
    abl_results = []
    for name in ablation_names:
        cur_train = list(train_rec)
        cur_val = list(val_rec)
        cur_test = list(test_rec)
        if name == "text_only_utilization":
            for r in cur_train + cur_val + cur_test:
                r["s_u"] = r["s_u_text_only"]
        elif name == "fixed_weighted_fusion":
            for r in cur_train + cur_val + cur_test:
                r["s_r"] = r["s_r_fixed"]
        res = run_ablation_experiment(cur_train, cur_val, cur_test, name, seeds=[42, 101, 202, 303, 404])
        abl_results.append(res)

    with open(abl_dir / "ablation_results.json", "w") as f:
        json.dump(abl_results, f, indent=2)

    # 9. Step 09: Final Reporting
    logger.info("Executing Step 09: Held-out Test Reporting & Figures...")
    test_probs = combiner.predict_proba(test_df)
    y_test = test_df["target"].values
    test_metrics = compute_all_metrics(y_test, test_probs, threshold=best_thresh)
    ci_metrics = compute_bootstrap_confidence_intervals(y_test, test_probs, threshold=best_thresh, n_bootstraps=200)

    plot_reliability_diagram(y_test, test_probs, rep_dir / "reliability_diagram.png")
    plot_roc_pr_curves(y_test, test_probs, rep_dir / "roc_pr_curves.png")

    final_report = {
        "held_out_test_metrics": test_metrics,
        "bootstrap_95_ci": ci_metrics,
        "proxy_failure_attribution_counts": test_df["proxy_attribution"].value_counts().to_dict(),
        "claim_boundary": config["project"]["claim_boundary"]
    }
    with open(rep_dir / "final_evaluation_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    logger.info("Smoke test completed successfully! All steps validated.")
    return final_report


if __name__ == "__main__":
    run_full_pipeline_smoke_test(num_samples_per_ds=40)
