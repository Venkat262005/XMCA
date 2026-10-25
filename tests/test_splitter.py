"""Unit tests for image-grouped splitting and leakage prevention."""

from pathlib import Path
from src.data.splitter import save_and_manifest_splits, split_by_image_id


def _create_mock_records(num_samples: int):
    # Groups roughly 3 records per image_id to verify image-grouped splitting
    return [
        {
            "id": f"rec_{i}",
            "image_id": f"img_{i // 3}",
            "question": f"Question {i}?",
            "ground_truth_answer": "yes",
            "task_type": "binary_vqa"
        }
        for i in range(num_samples)
    ]


def test_zero_image_leakage():
    records = _create_mock_records(300)
    train, val, test = split_by_image_id(records, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42)

    train_imgs = {r["image_id"] for r in train}
    val_imgs = {r["image_id"] for r in val}
    test_imgs = {r["image_id"] for r in test}

    assert len(train_imgs.intersection(val_imgs)) == 0, "Train and Val share images!"
    assert len(train_imgs.intersection(test_imgs)) == 0, "Train and Test share images!"
    assert len(val_imgs.intersection(test_imgs)) == 0, "Val and Test share images!"
    assert len(train) + len(val) + len(test) == 300, "Sample count mismatch!"


def test_dataset_sample_cap():
    records = _create_mock_records(2500)
    train, val, test = split_by_image_id(records, max_total_samples=2000, seed=42)
    total = len(train) + len(val) + len(test)
    assert total <= 2000, f"Total exceeded 2000 max cap: {total}"


def test_split_manifest_and_hashes(tmp_path: Path):
    records = _create_mock_records(50)
    train, val, test = split_by_image_id(records, seed=42)
    manifest = save_and_manifest_splits("infoseek", train, val, test, tmp_path)

    assert manifest["total_samples"] == 50
    assert "train_parquet_sha256" in manifest["hashes"]
    assert len(manifest["hashes"]["train_parquet_sha256"]) == 64  # SHA-256 length
