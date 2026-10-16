"""Corpus construction and chunking for knowledge retrieval."""

import json
from pathlib import Path
from typing import Any, Dict, List


def build_evidence_corpus(
    dataset_records: List[Dict[str, Any]],
    output_path: str | Path,
    extra_passages: List[str] | None = None
) -> List[Dict[str, Any]]:
    """
    Construct a standardized evidence corpus JSONL file from dataset annotations
    and external domain passages.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    corpus_entries: List[Dict[str, Any]] = []
    seen_texts = set()

    ALLOWED_SPLITS = {"train", "training"}

    # Ingest ONLY from explicitly permitted training records (reject unknown/missing/held-out splits)
    for r in dataset_records:
        raw_split = r.get("split")
        if not raw_split or str(raw_split).strip().lower() not in ALLOWED_SPLITS:
            # STRICT LEAKAGE PREVENTION: Reject missing, unknown, validation, and test splits
            continue

        ans_list = r.get("ground_truth_answers", [])
        for ans in ans_list:
            clean_ans = ans.strip()
            norm_key = " ".join(clean_ans.lower().split())
            if clean_ans and len(clean_ans) > 5 and norm_key not in seen_texts:
                seen_texts.add(norm_key)
                corpus_entries.append({
                    "corpus_id": f"doc_{len(corpus_entries):06d}",
                    "text": clean_ans,
                    "source_dataset": r.get("dataset", "unknown"),
                    "source_id": str(r.get("id", ""))
                })

    # Add extra external domain passages if provided
    if extra_passages:
        for p in extra_passages:
            clean_p = p.strip()
            if clean_p and clean_p not in seen_texts:
                seen_texts.add(clean_p)
                corpus_entries.append({
                    "corpus_id": f"doc_{len(corpus_entries):06d}",
                    "text": clean_p,
                    "source_dataset": "external_knowledge",
                    "source_id": ""
                })

    # Ensure corpus is non-empty
    if not corpus_entries:
        sample_knowledge = [
            "Industrial artifacts dating from the 19th century represent mechanical advances.",
            "Modern domestic spaces feature standard furniture arrangements including chairs and tables.",
            "Visual question answering evaluates reasoning over paired imagery and linguistic queries."
        ]
        for p in sample_knowledge:
            corpus_entries.append({
                "corpus_id": f"doc_{len(corpus_entries):06d}",
                "text": p,
                "source_dataset": "seed_domain",
                "source_id": ""
            })

    # Write out as JSONL
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in corpus_entries:
            f.write(json.dumps(entry) + "\n")

    return corpus_entries
