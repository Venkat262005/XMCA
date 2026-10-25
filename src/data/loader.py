"""Dataset loading and preprocessing for POPE, FaithScore, A-OKVQA, and InfoSeek."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.data.schema import XMCADataPoint

logger = logging.getLogger(__name__)


def generate_benchmark_records(
    dataset_name: str,
    num_samples: int = 1000,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    PROHIBITED: Generating synthetic benchmark records violates research integrity rules.
    """
    raise RuntimeError(
        f"Synthetic record generation for '{dataset_name}' is strictly prohibited. "
        "Only genuine datasets with verified provenance may be loaded."
    )
    import numpy as np
    rng = np.random.RandomState(seed)

    records = []
    # Create realistic image pools to simulate multiple questions per image
    num_unique_images = max(10, num_samples // 3)
    image_ids = [f"img_{dataset_name}_{i:05d}" for i in range(num_unique_images)]

    categories = ["kitchen", "street", "wildlife", "office", "museum", "space"]

    for i in range(num_samples):
        img_id = image_ids[rng.randint(0, num_unique_images)]
        cat = categories[rng.randint(0, len(categories))]

        if dataset_name == "pope":
            obj_name = rng.choice(["chair", "dog", "fridge", "car", "laptop", "bottle"])
            is_present = bool(rng.choice([True, False]))
            q = f"Is there a {obj_name} in the image?"
            gt_ans = ["yes"] if is_present else ["no"]
            is_hallucinated = int(not is_present) if rng.rand() < 0.3 else 0
            task_type = "binary_vqa"
        elif dataset_name == "faithscore":
            q = f"Describe the details observed in this {cat} scene."
            gt_ans = [f"The scene shows a modern {cat} environment with high visual detail."]
            is_hallucinated = int(rng.rand() < 0.35)
            task_type = "free_form"
        elif dataset_name == "aokvqa":
            q = f"What historical or operational purpose does the apparatus in this {cat} setting serve?"
            gt_ans = [f"It serves for regular maintenance and inspection in {cat} tasks."]
            is_hallucinated = int(rng.rand() < 0.25)
            task_type = "knowledge_vqa"
        else:  # infoseek
            entity = f"Artifact_{cat}_{i%100}"
            q = f"What is the identity and provenance of {entity} shown here?"
            gt_ans = [f"{entity} originated in 19th-century industrial collection."]
            is_hallucinated = int(rng.rand() < 0.30)
            task_type = "entity_qa"

        record = XMCADataPoint(
            id=f"{dataset_name}_{i:06d}",
            dataset=dataset_name,
            image_id=img_id,
            image_path=f"data/images/{img_id}.jpg",
            question=q,
            ground_truth_answers=gt_ans,
            is_hallucination_gt=is_hallucinated,
            task_type=task_type,
            metadata={"category": cat, "sample_idx": i}
        ).to_dict()
        records.append(record)

    return records


def load_dataset_records(
    dataset_name: str,
    max_samples: int = 1000,
    data_dir: Optional[str | Path] = None,
    seed: int = 42,
    strict_mode: bool = False,
    image_save_dir: Optional[str | Path] = "data/images"
) -> List[Dict[str, Any]]:
    """
    Load real dataset from disk/HuggingFace, materializing real images to disk with SHA-256 hashes.
    In strict_mode, fails loudly if HuggingFace/disk data is unavailable (no synthetic fallbacks).
    """
    import hashlib
    from PIL import Image

    dataset_key = dataset_name.lower().replace("-", "").replace("_", "")
    records: List[Dict[str, Any]] = []
    img_dir_path = Path(image_save_dir) if image_save_dir else Path("data/images")
    img_dir_path.mkdir(parents=True, exist_ok=True)

    # Check local pre-existing raw files
    if data_dir:
        candidate_file = Path(data_dir) / f"{dataset_name}.json"
        if candidate_file.exists():
            try:
                with open(candidate_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    records = raw_data[:max_samples] if len(raw_data) > max_samples else raw_data
                logger.info(f"Loaded {len(records)} records from {candidate_file}")
            except Exception as e:
                logger.warning(f"Failed to read {candidate_file}: {e}")

    # Check local split files in artifacts/splits/ if empty
    if not records:
        repo_root = Path(__file__).resolve().parent.parent.parent
        search_dirs = [
            Path("artifacts/splits") / dataset_key,
            repo_root / "artifacts/splits" / dataset_key,
            Path("/kaggle/working/xmca/artifacts/splits") / dataset_key,
        ]
        for sdir in search_dirs:
            for sname in ["test.json", "val.json", "train.json"]:
                split_candidate = sdir / sname
                if split_candidate.exists():
                    try:
                        with open(split_candidate, "r", encoding="utf-8") as f:
                            raw_data = json.load(f)
                            records = raw_data[:max_samples] if len(raw_data) > max_samples else raw_data
                        if records:
                            logger.info(f"Loaded {len(records)} records from {split_candidate}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to read {split_candidate}: {e}")
            if records:
                break

    # Attempt HuggingFace datasets if empty
    if not records:
        try:
            from datasets import load_dataset
            if "pope" in dataset_key:
                # POPE dataset on HuggingFace
                ds = load_dataset("lmms-lab/POPE", split="test")
                for idx, row in enumerate(ds):
                    if idx >= max_samples:
                        break
                    clean_meta = {
                        k: v for k, v in row.items()
                        if k != "image" and not hasattr(v, "save") and isinstance(v, (str, int, float, bool, list, dict))
                    }
                    img_id = str(row.get("image_source", f"pope_img_{idx}"))
                    img_file = img_dir_path / f"{img_id}.jpg"

                    # Materialize PIL Image if present
                    raw_img = row.get("image")
                    img_sha256 = None
                    if raw_img is not None and hasattr(raw_img, "save"):
                        if not img_file.exists():
                            raw_img.convert("RGB").save(img_file, format="JPEG", quality=95)
                    if img_file.exists():
                        img_sha256 = hashlib.sha256(img_file.read_bytes()).hexdigest()

                    gt_ans = str(row.get("answer", row.get("ground_truth", "yes"))).strip().lower()
                    records.append(XMCADataPoint(
                        id=f"pope_{idx:05d}",
                        dataset="pope",
                        image_id=img_id,
                        image_path=str(img_file),
                        question=str(row.get("question", "")),
                        ground_truth_answers=[gt_ans],
                        is_hallucination_gt=None,  # Populated post-generation by parse_pope_answer()
                        image_sha256=img_sha256,
                        task_type="binary_vqa",
                        metadata=clean_meta
                    ).to_dict())
                    if (idx + 1) % 200 == 0 or (idx + 1) == max_samples:
                        print(f"[POPE] Loading: {idx + 1}/{max_samples} samples done", flush=True)

            elif "aokvqa" in dataset_key:
                ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
                for idx, row in enumerate(ds):
                    if idx >= max_samples:
                        break
                    clean_meta = {
                        k: v for k, v in row.items()
                        if k != "image" and not hasattr(v, "save") and isinstance(v, (str, int, float, bool, list, dict))
                    }
                    img_id = str(row.get("image_id", f"aokvqa_img_{idx}"))
                    img_file = img_dir_path / f"{img_id}.jpg"

                    raw_img = row.get("image")
                    img_sha256 = None
                    if raw_img is not None and hasattr(raw_img, "save"):
                        if not img_file.exists():
                            raw_img.convert("RGB").save(img_file, format="JPEG", quality=95)
                    if img_file.exists():
                        img_sha256 = hashlib.sha256(img_file.read_bytes()).hexdigest()

                    rationales = [str(r) for r in row.get("rationales", []) if str(r).strip()]
                    direct_answers = [str(a) for a in row.get("direct_answers", []) if str(a).strip()]
                    clean_meta["direct_answers"] = direct_answers
                    clean_meta["choices"] = row.get("choices", [])

                    records.append(XMCADataPoint(
                        id=f"aokvqa_{idx:05d}",
                        dataset="aokvqa",
                        image_id=img_id,
                        image_path=str(img_file),
                        question=str(row.get("question", "")),
                        ground_truth_answers=rationales if rationales else direct_answers,
                        is_hallucination_gt=None,  # Evaluated post-generation
                        image_sha256=img_sha256,
                        task_type="knowledge_vqa",
                        metadata=clean_meta
                    ).to_dict())
                    if (idx + 1) % 200 == 0 or (idx + 1) == max_samples:
                        print(f"[A-OKVQA] Loading: {idx + 1}/{max_samples} samples done", flush=True)
            elif "infoseek" in dataset_key:
                # InfoSeek loader: Requires verified access to open-compass/InfoSeek or official Wikipedia corpus
                try:
                    ds = load_dataset("open-compass/InfoSeek", split="val")
                    for idx, row in enumerate(ds):
                        if idx >= max_samples:
                            break
                        clean_meta = {
                            k: v for k, v in row.items()
                            if k != "image" and not hasattr(v, "save") and isinstance(v, (str, int, float, bool, list, dict))
                        }
                        img_id = str(row.get("image_id", f"infoseek_img_{idx}"))
                        img_file = img_dir_path / f"{img_id}.jpg"
                        raw_img = row.get("image")
                        img_sha256 = None
                        if raw_img is not None and hasattr(raw_img, "save"):
                            if not img_file.exists():
                                raw_img.convert("RGB").save(img_file, format="JPEG", quality=95)
                        if img_file.exists():
                            img_sha256 = hashlib.sha256(img_file.read_bytes()).hexdigest()

                        records.append(XMCADataPoint(
                            id=f"infoseek_{idx:05d}",
                            dataset="infoseek",
                            image_id=img_id,
                            image_path=str(img_file),
                            question=str(row.get("question", "")),
                            ground_truth_answers=[str(a) for a in row.get("answer", [])] if isinstance(row.get("answer"), list) else [str(row.get("answer", ""))],
                            is_hallucination_gt=None,
                            image_sha256=img_sha256,
                            task_type="entity_qa",
                            metadata=clean_meta
                        ).to_dict())
                except Exception as e:
                    raise RuntimeError(
                        f"Genuine InfoSeek dataset ingestion from 'open-compass/InfoSeek' failed: {e}. "
                        "Synthetic fallback generation is strictly prohibited."
                    ) from e

            elif "faithscore" in dataset_key:
                # FaithScore loader: Requires verified access to genuine image-grounded prompts
                try:
                    ds = load_dataset("lmms-lab/LLaVA-Bench", split="test")
                    for idx, row in enumerate(ds):
                        if idx >= max_samples:
                            break
                        clean_meta = {
                            k: v for k, v in row.items()
                            if k != "image" and not hasattr(v, "save") and isinstance(v, (str, int, float, bool, list, dict))
                        }
                        img_id = str(row.get("image_id", f"faithscore_img_{idx}"))
                        img_file = img_dir_path / f"{img_id}.jpg"
                        raw_img = row.get("image")
                        img_sha256 = None
                        if raw_img is not None and hasattr(raw_img, "save"):
                            if not img_file.exists():
                                raw_img.convert("RGB").save(img_file, format="JPEG", quality=95)
                        if img_file.exists():
                            img_sha256 = hashlib.sha256(img_file.read_bytes()).hexdigest()

                        records.append(XMCADataPoint(
                            id=f"faithscore_{idx:05d}",
                            dataset="faithscore",
                            image_id=img_id,
                            image_path=str(img_file),
                            question=str(row.get("question", row.get("prompt", ""))),
                            ground_truth_answers=[str(row.get("ground_truth", row.get("gpt_answer", "")))],
                            is_hallucination_gt=None,
                            image_sha256=img_sha256,
                            task_type="free_form",
                            metadata=clean_meta
                        ).to_dict())
                except Exception as e:
                    raise RuntimeError(
                        f"Genuine FaithScore dataset ingestion failed: {e}. "
                        "Synthetic fallback generation is strictly prohibited."
                    ) from e
            else:
                raise ValueError(f"Unrecognized dataset name: '{dataset_name}'")

        except Exception as e:
            msg = f"Dataset loading error for '{dataset_name}': {e}"
            logger.error(msg)
            raise RuntimeError(f"FATAL: {msg}. Synthetic fallback generation is strictly prohibited.") from e

    # Strict non-fabrication guarantee: never fall back to synthetic data
    if not records:
        raise RuntimeError(
            f"FATAL: Zero genuine records could be loaded for dataset '{dataset_name}'. "
            "Synthetic record generation is strictly prohibited by research integrity rules."
        )

    # Strictly cap at max_samples
    if len(records) > max_samples:
        records = records[:max_samples]

    print(f"[{dataset_name}] Ingested {len(records)} samples (cap: {max_samples}).", flush=True)
    return records
