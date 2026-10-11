"""Feature extraction and engineering for the XMCA combiner."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "s_p",
    "s_r",
    "s_u",
    "diff_sp_sr",
    "diff_sp_su",
    "diff_sr_su",
    "min_score",
    "max_score",
    "mean_score",
    "evidence_count",
    "claim_count"
]


def extract_combiner_features(
    s_p: float,
    s_r: float,
    s_u: float,
    evidence_count: int = 1,
    claim_count: int = 1,
    ablation: Optional[str] = None
) -> Dict[str, float]:
    """
    Extract feature dictionary for a single sample.
    Supports feature masking for ablation studies:
    - 'no_perception': zero-out / mask s_p
    - 'no_retrieval': zero-out / mask s_r
    - 'no_utilization': zero-out / mask s_u
    """
    feat_sp = 0.0 if ablation == "no_perception" else float(s_p)
    feat_sr = 0.0 if ablation == "no_retrieval" else float(s_r)
    feat_su = 0.0 if ablation == "no_utilization" else float(s_u)

    active_scores = []
    if ablation != "no_perception":
        active_scores.append(feat_sp)
    if ablation != "no_retrieval":
        active_scores.append(feat_sr)
    if ablation != "no_utilization":
        active_scores.append(feat_su)

    if not active_scores:
        active_scores = [0.0]

    min_s = float(np.min(active_scores))
    max_s = float(np.max(active_scores))
    mean_s = float(np.mean(active_scores))

    features = {
        "s_p": feat_sp,
        "s_r": feat_sr,
        "s_u": feat_su,
        "diff_sp_sr": feat_sp - feat_sr,
        "diff_sp_su": feat_sp - feat_su,
        "diff_sr_su": feat_sr - feat_su,
        "min_score": min_s,
        "max_score": max_s,
        "mean_score": mean_s,
        "evidence_count": float(evidence_count),
        "claim_count": float(claim_count)
    }

    return features


def determine_proxy_attribution(s_p: float, s_r: float, s_u: float) -> str:
    """
    Proxy failure attribution based on lowest individual consistency score.
    Strictly marked as proxy attribution (not human-validated causal attribution).
    """
    scores = {"perception_proxy_failure": s_p, "retrieval_proxy_failure": s_r, "utilization_proxy_failure": s_u}
    return min(scores, key=scores.get)


def build_feature_dataframe(records: List[Dict[str, Any]], ablation: Optional[str] = None) -> pd.DataFrame:
    """Convert score records to feature DataFrame for classifier training / inference."""
    rows = []
    for r in records:
        f = extract_combiner_features(
            s_p=r.get("s_p", 0.5),
            s_r=r.get("s_r", 0.5),
            s_u=r.get("s_u", 0.5),
            evidence_count=r.get("evidence_count", 1),
            claim_count=r.get("claim_count", 1),
            ablation=ablation
        )
        f["id"] = r.get("id", "")
        f["proxy_attribution"] = determine_proxy_attribution(r.get("s_p", 0.5), r.get("s_r", 0.5), r.get("s_u", 0.5))
        if "is_hallucination_gt" in r:
            f["target"] = int(r["is_hallucination_gt"])
        rows.append(f)

    return pd.DataFrame(rows)
