"""Image-grouped reproducible splitting logic preventing cross-split leakage."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from src.utils.logging_metadata import compute_file_sha256


def split_by_image_id(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
    max_total_samples: int = 2000
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split records deterministically by image_id into train, val, test.
    Guarantees no image_id appears in more than one split.
    Enforces strict max total samples cap.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Enforce dataset maximum size cap
    if len(records) > max_total_samples:
        rng = np.random.RandomState(seed)
        shuffled_indices = rng.permutation(len(records))[:max_total_samples]
        records = [records[i] for i in shuffled_indices]

    # Group records by image_id
    image_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        img_id = str(r.get("image_id", r.get("id")))
        image_groups[img_id].append(r)

    unique_image_ids = sorted(list(image_groups.keys()))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_image_ids)

    # Compute target sample counts
    total_samples = len(records)
    target_train_count = int(total_samples * train_ratio)
    target_val_count = int(total_samples * val_ratio)

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    train_images = set()
    val_images = set()
    test_images = set()

    for img_id in unique_image_ids:
        group = image_groups[img_id]
        if len(train_records) + len(group) <= target_train_count:
            train_records.extend(group)
            train_images.add(img_id)
        elif len(val_records) + len(group) <= target_val_count:
            val_records.extend(group)
            val_images.add(img_id)
        else:
            test_records.extend(group)
            test_images.add(img_id)

    # Sanity checks for zero data leakage
    assert len(train_images.intersection(val_images)) == 0, "Leakage detected between train and val images!"
    assert len(train_images.intersection(test_images)) == 0, "Leakage detected between train and test images!"
    assert len(val_images.intersection(test_images)) == 0, "Leakage detected between val and test images!"

    return train_records, val_records, test_records


def save_and_manifest_splits(
    dataset_name: str,
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    output_dir: str | Path
) -> Dict[str, Any]:
    """Persist split datasets as parquet/json and generate an immutable manifest."""
    out_dir = Path(output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_data = {"train": train, "val": val, "test": test}
    file_hashes = {}
    sample_counts = {}

    for split_name, split_list in splits_data.items():
        df = pd.DataFrame(split_list)
        parquet_path = out_dir / f"{split_name}.parquet"
        json_path = out_dir / f"{split_name}.json"

        df.to_parquet(parquet_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(split_list, f, indent=2)

        file_hashes[f"{split_name}_parquet_sha256"] = compute_file_sha256(parquet_path)
        file_hashes[f"{split_name}_json_sha256"] = compute_file_sha256(json_path)
        sample_counts[split_name] = len(split_list)

    manifest = {
        "dataset": dataset_name,
        "sample_counts": sample_counts,
        "total_samples": sum(sample_counts.values()),
        "hashes": file_hashes
    }

    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
