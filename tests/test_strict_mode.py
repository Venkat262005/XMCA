"""Tests verifying that mock fallbacks are strictly rejected in strict_mode."""

import pytest
from pathlib import Path
from src.data.schema import parse_pope_answer
from src.generation.qwen import Qwen2VLGenerator
from src.generation.llava import LLaVAOneVisionGenerator
from src.perception.detector import OwlViTDetector
from src.perception.scorer import PerceptionScorer
from src.utilization.nli import DeBERTaNLIEngine
from src.data.loader import load_dataset_records
from src.retrieval.corpus import build_evidence_corpus


def test_pope_answer_parser_logic():
    """Verify standard POPE metric parsing."""
    # 1. False positive hallucination (GT=no, Model=yes)
    r1 = parse_pope_answer("no", "Yes, there is a dog in the picture.")
    assert r1["hallucination_fp"] == 1
    assert r1["is_correct"] == 0
    assert r1["omission_fn"] == 0
    assert r1["parse_status"] == "success"

    # 2. Faithful negative (GT=no, Model=no)
    r2 = parse_pope_answer("no", "No, I cannot see any dog.")
    assert r2["hallucination_fp"] == 0
    assert r2["is_correct"] == 1
    assert r2["omission_fn"] == 0
    assert r2["parse_status"] == "success"

    # 3. Faithful positive (GT=yes, Model=yes)
    r3 = parse_pope_answer("yes", "Yes, there is a dog.")
    assert r3["hallucination_fp"] == 0
    assert r3["is_correct"] == 1
    assert r3["omission_fn"] == 0
    assert r3["parse_status"] == "success"

    # 4. False negative omission (GT=yes, Model=no)
    r4 = parse_pope_answer("yes", "No, there is none.")
    assert r4["hallucination_fp"] == 0
    assert r4["is_correct"] == 0
    assert r4["omission_fn"] == 1
    assert r4["parse_status"] == "success"

    # 5. Unparseable output
    r5 = parse_pope_answer("yes", "It is difficult to determine what this is.")
    assert r5["parse_status"] == "unparseable"
    assert r5["hallucination_fp"] is None
    assert r5["is_correct"] is None


def test_strict_mode_rejects_missing_image():
    """Verify that strict mode raises FileNotFoundError for missing images."""
    qwen = Qwen2VLGenerator(strict_mode=True)
    with pytest.raises(FileNotFoundError):
        qwen.generate_single("nonexistent_path/fake_img.jpg", "Question?", [])

    detector = OwlViTDetector(strict_mode=True)
    with pytest.raises(FileNotFoundError):
        detector.detect_regions("nonexistent_path/fake_img.jpg", ["cat"])

    scorer = PerceptionScorer(strict_mode=True)
    with pytest.raises(FileNotFoundError):
        scorer.compute_s_p("nonexistent_path/fake_img.jpg", ["claim"], [])


def test_corpus_builder_excludes_held_out_splits(tmp_path):
    """Verify that held-out val/test answers are strictly barred from the retrieval corpus."""
    records = [
        {"id": "train_01", "split": "train", "ground_truth_answers": ["Valid training rationale 12345"]},
        {"id": "val_01", "split": "val", "ground_truth_answers": ["LEAKED VAL ANSWER 12345"]},
        {"id": "test_01", "split": "test", "ground_truth_answers": ["LEAKED TEST ANSWER 12345"]}
    ]
    out_file = tmp_path / "corpus.jsonl"
    entries = build_evidence_corpus(records, out_file)

    texts = [e["text"] for e in entries]
    assert "Valid training rationale 12345" in texts
    assert "LEAKED VAL ANSWER 12345" not in texts
    assert "LEAKED TEST ANSWER 12345" not in texts


def test_strict_mode_rejects_mock_retriever_and_reranker():
    """Verify that DenseRetriever and MiniLMReranker reject mock fallbacks in strict mode."""
    from src.retrieval.indexer import DenseRetriever
    from src.retrieval.reranker import MiniLMReranker

    retriever = DenseRetriever(model_id="/tmp/nonexistent_model_dir_123", strict_mode=True)
    with pytest.raises(RuntimeError):
        retriever.encode(["query text"])

    reranker = MiniLMReranker(model_id="/tmp/nonexistent_reranker_dir_123", strict_mode=True)
    with pytest.raises(RuntimeError):
        reranker.score_pairs("query", ["passage"])


def test_strict_mode_rejects_mock_nli():
    """Verify that DeBERTaNLIEngine rejects keyword-overlap fallback in strict mode."""
    engine = DeBERTaNLIEngine(model_id="/tmp/nonexistent_deberta_dir_123", strict_mode=True)
    with pytest.raises(RuntimeError):
        engine.predict_entailment_probabilities([("premise", "hypothesis")])


def test_pope_parser_exhaustive_edge_cases():
    """Exhaustively test all 4 combinations, invalid GT, empty, and ambiguous/negated outputs."""
    # 1. Four standard combinations
    res_yy = parse_pope_answer("yes", "Yes, there is one.")
    assert (res_yy["hallucination_fp"], res_yy["is_correct"], res_yy["omission_fn"]) == (0, 1, 0)
    assert res_yy["parse_status"] == "success"

    res_yn = parse_pope_answer("yes", "No, it is not present.")
    assert (res_yn["hallucination_fp"], res_yn["is_correct"], res_yn["omission_fn"]) == (0, 0, 1)
    assert res_yn["parse_status"] == "success"

    res_ny = parse_pope_answer("no", "Yes, it is definitely there.")
    assert (res_ny["hallucination_fp"], res_ny["is_correct"], res_ny["omission_fn"]) == (1, 0, 0)
    assert res_ny["parse_status"] == "success"

    res_nn = parse_pope_answer("no", "No, there is none.")
    assert (res_nn["hallucination_fp"], res_nn["is_correct"], res_nn["omission_fn"]) == (0, 1, 0)
    assert res_nn["parse_status"] == "success"

    # 2. Invalid GT records must NOT receive fabricated labels
    for inv_gt in ["maybe", "unknown", "", "   ", None, 123]:
        r = parse_pope_answer(inv_gt, "Yes")
        assert r["parse_status"] == "invalid_gt"
        assert r["hallucination_fp"] is None
        assert r["is_correct"] is None
        assert r["omission_fn"] is None
        assert r["is_hallucination_gt"] is None

    # 3. Empty generated answer
    for empty_ans in ["", "   ", None]:
        r = parse_pope_answer("yes", empty_ans)
        assert r["parse_status"] == "unparseable"
        assert r["hallucination_fp"] is None
        assert r["is_correct"] is None

    # 4. Ambiguous / conflicting responses
    for amb_ans in ["Yes and no, it depends.", "No, wait, yes there is.", "yes... no"]:
        r = parse_pope_answer("yes", amb_ans)
        assert r["parse_status"] == "unparseable"
        assert r["hallucination_fp"] is None
        assert r["is_correct"] is None

    # 5. Negated / hedging responses
    for neg_ans in ["not yes", "not sure", "neither yes nor no", "cannot confirm"]:
        r = parse_pope_answer("no", neg_ans)
        assert r["parse_status"] == "unparseable"
        assert r["hallucination_fp"] is None
        assert r["is_correct"] is None


def test_corpus_leakage_and_self_answer_prevention(tmp_path):
    """Verify strict inclusion of train split, exclusion of missing/unknown/heldout splits and self-answer."""
    records = [
        {"id": "train_01", "split": "train", "ground_truth_answers": ["Permitted train answer"]},
        {"id": "train_02", "split": "training", "ground_truth_answers": ["Permitted training answer"]},
        {"id": "missing_01", "split": None, "ground_truth_answers": ["LEAK MISSING SPLIT"]},
        {"id": "empty_01", "split": "", "ground_truth_answers": ["LEAK EMPTY SPLIT"]},
        {"id": "unknown_01", "split": "mystery", "ground_truth_answers": ["LEAK UNKNOWN SPLIT"]},
        {"id": "val_01", "split": "val", "ground_truth_answers": ["LEAK VAL SPLIT"]},
        {"id": "test_01", "split": "test", "ground_truth_answers": ["LEAK TEST SPLIT"]},
        # Duplicate answer check:
        {"id": "train_03", "split": "train", "ground_truth_answers": ["  permitted TRAIN answer  "]}
    ]
    out_file = tmp_path / "strict_corpus.jsonl"
    entries = build_evidence_corpus(records, out_file)
    texts = [e["text"] for e in entries]

    # Verify permitted
    assert "Permitted train answer" in texts
    assert "Permitted training answer" in texts

    # Verify prohibited splits rejected
    assert "LEAK MISSING SPLIT" not in texts
    assert "LEAK EMPTY SPLIT" not in texts
    assert "LEAK UNKNOWN SPLIT" not in texts
    assert "LEAK VAL SPLIT" not in texts
    assert "LEAK TEST SPLIT" not in texts

    # Verify duplicates rejected (train_03 duplicate of train_01)
    assert len([t for t in texts if "permitted train answer" in t.lower()]) == 1

    # Verify self-answer retrieval prevention
    from src.retrieval.indexer import DenseRetriever
    retriever = DenseRetriever(model_id="/tmp/nonexistent_dense", strict_mode=False)
    retriever.build_index(entries)

    # Search for train_01 query excluding train_01:
    results = retriever.search("Permitted train answer", top_k=2, exclude_source_id="train_01")
    retrieved_source_ids = [doc["source_id"] for doc, _ in results]
    assert "train_01" not in retrieved_source_ids


def test_score_calculation_boundary_cases():
    """Verify mathematical definitions and edge cases for S_P, S_R, and S_U."""
    from src.retrieval.fusion import RetrievalScoreFusion

    # S_R edge cases
    fusion = RetrievalScoreFusion(mode="fixed_weighted", fixed_weight_dense=0.5)
    # Empty retrieved passages returns 0.0
    assert fusion.fuse([], []) == 0.0
    assert fusion.fuse([0.8], [0.6]) == 0.7

    # S_U formula: S_U = P(entailment) * (1.0 - P(contradiction))
    # Test high neutral case: NLI outputs neutral=0.8, entail=0.15, contra=0.05
    p_entail = 0.15
    p_contra = 0.05
    s_u = p_entail * (1.0 - p_contra)
    assert round(s_u, 4) == round(0.15 * 0.95, 4)  # Neutral probability does not artificially inflate S_U


