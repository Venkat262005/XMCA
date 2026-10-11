"""Unit tests for sharding, completed_ids.json, and model checkpointing."""

import json
from pathlib import Path
from src.utils.checkpoints import ModelCheckpointTracker, ShardManager


def test_shard_manager_and_completed_ids(tmp_path: Path):
    mgr = ShardManager(output_dir=tmp_path, shard_size=5)

    items = [{"id": f"item_{i}", "val": i} for i in range(12)]
    remaining = mgr.get_remaining_items(items)
    assert len(remaining) == 12

    # Save shard 0 (5 items)
    shard_0_items = remaining[:5]
    mgr.save_shard(0, shard_0_items, prefix="test_shard")

    assert mgr.is_completed("item_0")
    assert mgr.is_completed("item_4")
    assert not mgr.is_completed("item_5")

    # Load completed_ids.json
    with open(tmp_path / "completed_ids.json", "r") as f:
        comp = json.load(f)
    assert len(comp) == 5

    # Check resumption
    new_remaining = mgr.get_remaining_items(items)
    assert len(new_remaining) == 7


def test_model_checkpoint_tracker(tmp_path: Path):
    tracker = ModelCheckpointTracker(checkpoints_dir=tmp_path)

    saved_states = []

    def dummy_save(target_dir: Path):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "weights.txt").write_text("model_data")
        saved_states.append(target_dir.name)

    # Epoch 1: AUPRC 0.70, ECE 0.15
    res1 = tracker.update(1, {"auprc": 0.70, "ece": 0.15}, dummy_save)
    assert res1["is_best_auprc"] is True
    assert res1["is_best_ece"] is True
    assert (tmp_path / "checkpoint-last").exists()
    assert (tmp_path / "checkpoint-best-auprc").exists()
    assert (tmp_path / "checkpoint-best-ece").exists()

    # Epoch 2: AUPRC 0.65 (worse), ECE 0.10 (better)
    res2 = tracker.update(2, {"auprc": 0.65, "ece": 0.10}, dummy_save)
    assert res2["is_best_auprc"] is False
    assert res2["is_best_ece"] is True
