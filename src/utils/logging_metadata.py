"""Metadata tracking, hardware auditing, and hashing for research reproducibility."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import yaml


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute deterministic SHA256 checksum of a file."""
    p = Path(filepath)
    if not p.exists() or p.is_dir():
        return ""
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit_hash() -> str:
    """Return the current git commit hash if in a git repository."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return output
    except Exception:
        return "git-unavailable"


def get_environment_metadata(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Capture full hardware, package version, and run metadata."""
    package_versions = {}
    key_packages = [
        "torch", "transformers", "accelerate", "peft", "bitsandbytes",
        "faiss", "lightgbm", "xgboost", "sklearn", "pandas", "numpy", "scipy"
    ]
    for pkg in key_packages:
        try:
            mod = __import__(pkg)
            package_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            package_versions[pkg] = "not-installed"

    gpu_info = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                })
    except Exception:
        pass

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": get_git_commit_hash(),
        "random_seed": seed,
        "packages": package_versions,
        "gpus": gpu_info,
        "claim_boundary": config.get("project", {}).get("claim_boundary", "proxy-attribution-only")
    }


def save_run_metadata(output_dir: str | Path, config: Dict[str, Any], seed: int, step_name: str, extra: Dict[str, Any] | None = None) -> Path:
    """Save config.yaml and run_metadata.json into output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save copy of config
    config_copy_path = out_path / "config.yaml"
    with open(config_copy_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    meta = get_environment_metadata(config, seed)
    meta["step_name"] = step_name
    if extra:
        meta.update(extra)

    meta_file = out_path / "run_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta_file
