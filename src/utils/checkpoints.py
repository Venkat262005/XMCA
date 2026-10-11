"""Shard processing, resumption tracking, and model checkpoint management."""

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set
import pandas as pd


class ShardManager:
    """Manages sharding (<= 250 items/shard) and idempotent resume via completed_ids.json."""

    def __init__(self, output_dir: str | Path, shard_size: int = 250):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = min(shard_size, 250)
        self.completed_ids_file = self.output_dir / "completed_ids.json"
        self._completed_ids: Set[str] = self._load_completed_ids()

    def _load_completed_ids(self) -> Set[str]:
        if self.completed_ids_file.exists():
            try:
                with open(self.completed_ids_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data)
            except Exception:
                return set()
        return set()

    def save_completed_ids(self) -> None:
        """Atomically persist completed IDs."""
        temp_file = self.output_dir / "completed_ids.json.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(sorted(list(self._completed_ids)), f, indent=2)
        temp_file.replace(self.completed_ids_file)

    def is_completed(self, item_id: str) -> bool:
        return item_id in self._completed_ids

    def mark_completed(self, item_ids: Iterable[str]) -> None:
        self._completed_ids.update(item_ids)
        self.save_completed_ids()

    def get_remaining_items(self, items: List[Dict[str, Any]], id_key: str = "id") -> List[Dict[str, Any]]:
        return [it for it in items if str(it.get(id_key)) not in self._completed_ids]

    def save_shard(self, shard_index: int, records: List[Dict[str, Any]], prefix: str = "shard") -> Path:
        """Save shard records as Parquet and complementary JSON."""
        if not records:
            return self.output_dir

        df = pd.DataFrame(records)
        parquet_file = self.output_dir / f"{prefix}_{shard_index:04d}.parquet"
        json_file = self.output_dir / f"{prefix}_{shard_index:04d}.json"

        df.to_parquet(parquet_file, index=False)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        # Update completed IDs
        id_keys = [str(r["id"]) for r in records if "id" in r]
        self.mark_completed(id_keys)

        return parquet_file

    def load_all_shards(self, prefix: str = "shard") -> pd.DataFrame:
        """Load and concatenate all Parquet shards in directory."""
        shard_files = sorted(list(self.output_dir.glob(f"{prefix}_*.parquet")))
        if not shard_files:
            return pd.DataFrame()
        dfs = [pd.read_parquet(f) for f in shard_files]
        combined = pd.concat(dfs, ignore_index=True)
        # Drop duplicates by id if present
        if "id" in combined.columns:
            combined = combined.drop_duplicates(subset=["id"]).reset_index(drop=True)
        return combined


class ModelCheckpointTracker:
    """Tracks training states: checkpoint-last, checkpoint-best-auprc, checkpoint-best-ece."""

    def __init__(self, checkpoints_dir: str | Path):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.best_auprc = -1.0
        self.best_ece = float("inf")
        self.metrics_history: List[Dict[str, Any]] = []

    def update(
        self,
        epoch_or_step: int,
        metrics: Dict[str, float],
        save_fn: Callable[[Path], None]
    ) -> Dict[str, bool]:
        """
        Evaluate validation metrics and save checkpoint states.
        Strict rule: metrics MUST be from validation set, NEVER test set.
        """
        auprc = metrics.get("auprc", 0.0)
        ece = metrics.get("ece", 1.0)

        record = {"epoch_or_step": epoch_or_step, **metrics}
        self.metrics_history.append(record)

        # 1. Always save checkpoint-last
        last_dir = self.checkpoints_dir / "checkpoint-last"
        save_fn(last_dir)

        is_best_auprc = False
        is_best_ece = False

        # 2. Check best AUPRC
        if auprc > self.best_auprc:
            self.best_auprc = auprc
            is_best_auprc = True
            best_auprc_dir = self.checkpoints_dir / "checkpoint-best-auprc"
            save_fn(best_auprc_dir)

        # 3. Check best ECE
        if ece < self.best_ece:
            self.best_ece = ece
            is_best_ece = True
            best_ece_dir = self.checkpoints_dir / "checkpoint-best-ece"
            save_fn(best_ece_dir)

        # Save history log
        with open(self.checkpoints_dir / "eval_history.json", "w", encoding="utf-8") as f:
            json.dump({
                "history": self.metrics_history,
                "best_auprc": self.best_auprc,
                "best_ece": self.best_ece
            }, f, indent=2)

        return {"is_best_auprc": is_best_auprc, "is_best_ece": is_best_ece}
