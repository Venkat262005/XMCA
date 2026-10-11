"""Configuration loader and environment manager for XMCA-Net-Lite."""

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def is_kaggle_environment() -> bool:
    """Detect if running inside Kaggle notebook environment."""
    return os.path.exists("/kaggle/working")


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load and resolve config dictionary with platform-specific paths."""
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve paths based on environment
    if is_kaggle_environment():
        base_working = Path("/kaggle/working/xmca_artifacts")
    else:
        # Local workspace artifacts directory
        workspace_root = Path(__file__).resolve().parent.parent
        base_working = workspace_root / "artifacts"

    paths = config.get("paths", {})
    paths["base_dir"] = str(base_working)
    paths["data_dir"] = str(base_working / "data")
    paths["splits_dir"] = str(base_working / "splits")
    paths["corpora_dir"] = str(base_working / "corpora")
    paths["generations_dir"] = str(base_working / "generations")
    paths["perception_dir"] = str(base_working / "perception")
    paths["retrieval_dir"] = str(base_working / "retrieval")
    paths["utilization_dir"] = str(base_working / "utilization")
    paths["combiner_dir"] = str(base_working / "combiner")
    paths["checkpoints_dir"] = str(base_working / "checkpoints")
    paths["ablations_dir"] = str(base_working / "ablations")
    paths["reports_dir"] = str(base_working / "reports")
    config["paths"] = paths

    # Ensure local working directories exist
    for k, p in paths.items():
        if isinstance(p, str) and not p.startswith("/kaggle"):
            Path(p).mkdir(parents=True, exist_ok=True)
        elif is_kaggle_environment() and isinstance(p, str) and not p.startswith("/kaggle/input"):
            Path(p).mkdir(parents=True, exist_ok=True)

    return config


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
