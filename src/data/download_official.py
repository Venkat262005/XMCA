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

    # 1. POPE (lmms-lab/POPE)
    try:
        from datasets import load_dataset
        logger.info("Fetching POPE dataset from HuggingFace...")
        pope_ds = load_dataset("lmms-lab/POPE", split="test")
        pope_records = []
        for i, row in enumerate(pope_ds):
            if len(pope_records) >= max_per_dataset:
                break
            q = str(row.get("question", ""))
            ans = str(row.get("answer") or row.get("ground_truth", "yes")).strip().lower()
            is_hallucinated = 1 if ans == "no" else 0
            pope_records.append(XMCADataPoint(
                id=f"pope_{i:05d}",
                dataset="pope",
                image_id=str(row.get("image_source", f"COCO_val2014_{i:012d}")),
                image_path=str(row.get("image_source", "")),
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

    # 2. A-OKVQA (HuggingFaceM4/A-OKVQA)
    try:
        from datasets import load_dataset
        logger.info("Fetching A-OKVQA dataset from HuggingFace...")
        aokvqa_ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
        aokvqa_records = []
        for i, row in enumerate(aokvqa_ds):
            if len(aokvqa_records) >= max_per_dataset:
                break
            choices = row.get("choices", [])
            corr_idx = row.get("correct_choice_idx")
            correct_choice = choices[corr_idx] if (corr_idx is not None and corr_idx < len(choices)) else None
            direct_answers = row.get("direct_answers", [])
            valid_answers = [str(a).strip() for a in direct_answers]
            if correct_choice and correct_choice not in valid_answers:
                valid_answers.append(correct_choice)

            aokvqa_records.append(XMCADataPoint(
                id=f"aokvqa_{i:05d}",
                dataset="aokvqa",
                image_id=str(row.get("image_id", f"aok_img_{i}")),
                image_path=str(row.get("image_id", "")),
                question=str(row.get("question", "")),
                ground_truth_answers=valid_answers,
                is_hallucination_gt=None,
                task_type="knowledge_vqa",
                metadata={
                    "choices": choices,
                    "correct_choice_idx": corr_idx,
                    "correct_choice": correct_choice,
                    "direct_answers": direct_answers
                }
            ).to_dict())
        datasets_map["aokvqa"] = aokvqa_records
        logger.info(f"Loaded {len(aokvqa_records)} A-OKVQA records.")
    except Exception as e:
        logger.warning(f"A-OKVQA download fallback: {e}")

    # 3. InfoSeek (Entity-Centric Knowledge QA from M2KR Parquet)
    try:
        from datasets import load_dataset
        logger.info("Fetching InfoSeek dataset from M2KR benchmark...")
        infoseek_ds = load_dataset(
            "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
            data_files={"test": "Infoseek_data/test-*.parquet"},
            split="test"
        )
        infoseek_records = []
        for i, row in enumerate(infoseek_ds):
            if len(infoseek_records) >= max_per_dataset:
                break
            q = str(row.get("question", ""))
            gold = str(row.get("gold_answer", "")).strip()
            ans_list = [str(a).strip() for a in row.get("answers", []) if str(a).strip()]
            if gold and gold not in ans_list:
                ans_list.insert(0, gold)
            infoseek_records.append(XMCADataPoint(
                id=f"infoseek_{i:05d}",
                dataset="infoseek",
                image_id=str(row.get("img_path", f"infoseek_img_{i}")),
                image_path=str(row.get("img_path", "")),
                question=q,
                ground_truth_answers=ans_list,
                is_hallucination_gt=None,
                task_type="entity_qa",
                metadata={
                    "caption": row.get("img_caption", ""),
                    "question_type": row.get("question_type", "String")
                }
            ).to_dict())
        datasets_map["infoseek"] = infoseek_records
        logger.info(f"Loaded {len(infoseek_records)} InfoSeek records.")
    except Exception as e:
        logger.warning(f"InfoSeek download fallback: {e}")

    # 4. FaithScore / LLaVA-1k (Atomic Faithfulness from M2KR Parquet)
    try:
        from datasets import load_dataset
        logger.info("Fetching FaithScore / LLaVA subset from M2KR benchmark...")
        llava_ds = load_dataset(
            "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
            data_files={"test": "LLaVA_data/test-*.parquet"},
            split="test"
        )
        faith_records = []
        for i, row in enumerate(llava_ds):
            if len(faith_records) >= max_per_dataset:
                break
            q = str(row.get("question", ""))
            pos_contents = row.get("pos_item_contents", [""])
            ans = str(pos_contents[0]) if pos_contents else ""
            faith_records.append(XMCADataPoint(
                id=f"faithscore_{i:05d}",
                dataset="faithscore",
                image_id=str(row.get("image_id", f"faith_img_{i}")),
                image_path=str(row.get("img_path", "")),
                question=q,
                ground_truth_answers=[ans] if ans else [],
                is_hallucination_gt=None,
                task_type="free_form",
                metadata={
                    "instruction": row.get("instruction", ""),
                    "llava_split": row.get("llava_split", "")
                }
            ).to_dict())
        datasets_map["faithscore"] = faith_records
        logger.info(f"Loaded {len(faith_records)} FaithScore records.")
    except Exception as e:
        logger.warning(f"FaithScore download fallback: {e}")

    # Save to disk
    for name, records in datasets_map.items():
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    return datasets_map

