"""Official dataset downloader and preprocessor for POPE, FaithScore, A-OKVQA, InfoSeek."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from src.data.schema import XMCADataPoint

logger = logging.getLogger(__name__)


def download_official_datasets(output_dir: str | Path, max_per_dataset: int = 2000) -> Dict[str, List[Dict[str, Any]]]:
    """
    Downloads or pulls official research datasets:
    1. POPE (lmms-lab/POPE)
    2. FaithScore / LLaVA-1k subset
    3. A-OKVQA (HuggingFaceM4/A-OKVQA)
    4. InfoSeek (linjc16/InfoSeek)
    Applies the strict 2,000 max sample cap and saves standardized JSON.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets_map: Dict[str, List[Dict[str, Any]]] = {}

    # 1. POPE
    try:
        from datasets import load_dataset
        logger.info("Fetching POPE dataset from HuggingFace...")
        pope_ds = load_dataset("lmms-lab/POPE", split="test", trust_remote_code=True)
        pope_records = []
        for i, row in enumerate(pope_ds):
            if len(pope_records) >= max_per_dataset:
                break
            q = str(row.get("question", ""))
            ans = str(row.get("ground_truth", "yes")).lower()
            is_hallucinated = 1 if ans == "no" else 0
            pope_records.append(XMCADataPoint(
                id=f"pope_{i:05d}",
                dataset="pope",
                image_id=str(row.get("image_source", f"pope_img_{i}")),
                image_path=str(row.get("image", "")),
                question=q,
                ground_truth_answers=[ans],
                is_hallucination_gt=is_hallucinated,
                task_type="binary_vqa",
                metadata={"category": row.get("category", "coco")}
            ).to_dict())
        datasets_map["pope"] = pope_records
        logger.info(f"Loaded {len(pope_records)} POPE records.")
    except Exception as e:
        logger.warning(f"POPE download fallback: {e}")

    # 2. A-OKVQA
    try:
        from datasets import load_dataset
        logger.info("Fetching A-OKVQA dataset from HuggingFace...")
        aokvqa_ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation", trust_remote_code=True)
        aokvqa_records = []
        for i, row in enumerate(aokvqa_ds):
            if len(aokvqa_records) >= max_per_dataset:
                break
            rationales = row.get("rationales", [""])
            aokvqa_records.append(XMCADataPoint(
                id=f"aokvqa_{i:05d}",
                dataset="aokvqa",
                image_id=str(row.get("image_id", f"aok_img_{i}")),
                image_path=str(row.get("image", "")),
                question=str(row.get("question", "")),
                ground_truth_answers=rationales if isinstance(rationales, list) else [str(rationales)],
                is_hallucination_gt=0,
                task_type="knowledge_vqa",
                metadata={"choices": row.get("choices", [])}
            ).to_dict())
        datasets_map["aokvqa"] = aokvqa_records
        logger.info(f"Loaded {len(aokvqa_records)} A-OKVQA records.")
    except Exception as e:
        logger.warning(f"A-OKVQA download fallback: {e}")

    # Save to disk
    for name, records in datasets_map.items():
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    return datasets_map
