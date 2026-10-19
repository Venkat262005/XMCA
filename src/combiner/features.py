"""Enhanced feature extraction and cause attribution analysis for XMCA combiner."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "s_p",
    "s_r",
    "s_u",
    "diff_sp_sr",
    "diff_sp_su",
    "diff_sr_su",
    "abs_diff_sp_sr",
    "abs_diff_sp_su",
    "abs_diff_sr_su",
    "min_score",
    "max_score",
    "mean_score",
    "harmonic_mean",
    "signal_variance",
    "min_to_mean_ratio",
    "is_perception_lowest",
    "is_retrieval_lowest",
    "is_utilization_lowest",
    "evidence_count",
    "claim_count",
    "geometric_mean",
    "cross_modal_tension",
    "knowledge_visual_gap",
    "deficit_spread",
    "min_to_max_ratio"
]

CAUSE_CLASSES = [
    "perception_failure",
    "retrieval_failure",
    "utilization_failure",
    "compound_failure",
    "faithful"
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
    Extract comprehensive non-linear feature dictionary for a single sample.
    Supports feature masking for ablation studies.
    """
    if ablation != "no_perception" and s_p is None:
        raise ValueError("Cannot extract combiner features: s_p is None/unavailable.")
    if ablation != "no_retrieval" and s_r is None:
        raise ValueError("Cannot extract combiner features: s_r is None/unavailable.")
    if ablation != "no_utilization" and s_u is None:
        raise ValueError("Cannot extract combiner features: s_u is None/unavailable.")

    feat_sp = 0.0 if ablation == "no_perception" else float(np.clip(s_p, 0.0, 1.0))
    feat_sr = 0.0 if ablation == "no_retrieval" else float(np.clip(s_r, 0.0, 1.0))
    feat_su = 0.0 if ablation == "no_utilization" else float(np.clip(s_u, 0.0, 1.0))

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
    variance_s = float(np.var(active_scores)) if len(active_scores) > 1 else 0.0

    # Harmonic mean calculation
    eps = 1e-6
    h_mean = float(len(active_scores) / sum(1.0 / (s + eps) for s in active_scores))
    h_mean = float(np.clip(h_mean, 0.0, 1.0))

    ratio_min_mean = float(min_s / (mean_s + eps))

    # Binary indicator flags for which signal collapsed
    is_p_low = 1.0 if (feat_sp <= min(feat_sr, feat_su) and ablation != "no_perception") else 0.0
    is_r_low = 1.0 if (feat_sr <= min(feat_sp, feat_su) and ablation != "no_retrieval") else 0.0
    is_u_low = 1.0 if (feat_su <= min(feat_sp, feat_sr) and ablation != "no_utilization") else 0.0

    # Shift 1: Conjunctive consistency, cross-modal tension, and deficit spread
    active_deficits = [1.0 - s for s in active_scores]
    geom_mean = float(np.prod(active_scores) ** (1.0 / len(active_scores))) if active_scores else 0.0
    tension_pu = abs(feat_sp - feat_su)
    tension_pr = abs(feat_sp - feat_sr)
    def_spread = float(np.max(active_deficits) - np.min(active_deficits)) if len(active_deficits) > 1 else 0.0
    min_max_ratio = float(min_s / (max_s + eps))

    features = {
        "s_p": feat_sp,
        "s_r": feat_sr,
        "s_u": feat_su,
        "diff_sp_sr": feat_sp - feat_sr,
        "diff_sp_su": feat_sp - feat_su,
        "diff_sr_su": feat_sr - feat_su,
        "abs_diff_sp_sr": abs(feat_sp - feat_sr),
        "abs_diff_sp_su": abs(feat_sp - feat_su),
        "abs_diff_sr_su": abs(feat_sr - feat_su),
        "min_score": min_s,
        "max_score": max_s,
        "mean_score": mean_s,
        "harmonic_mean": h_mean,
        "signal_variance": variance_s,
        "min_to_mean_ratio": ratio_min_mean,
        "is_perception_lowest": is_p_low,
        "is_retrieval_lowest": is_r_low,
        "is_utilization_lowest": is_u_low,
        "evidence_count": float(evidence_count),
        "claim_count": float(claim_count),
        "geometric_mean": float(np.clip(geom_mean, 0.0, 1.0)),
        "cross_modal_tension": float(np.clip(tension_pu, 0.0, 1.0)),
        "knowledge_visual_gap": float(np.clip(tension_pr, 0.0, 1.0)),
        "deficit_spread": float(np.clip(def_spread, 0.0, 1.0)),
        "min_to_max_ratio": float(np.clip(min_max_ratio, 0.0, 1.0))
    }

    return features


def compute_cause_attribution(
    s_p: float,
    s_r: float,
    s_u: float,
    hallucination_prob: float,
    threshold: float = 0.5
) -> Tuple[str, Dict[str, float]]:
    """
    Computes primary cause attribution and soft cause probabilities.
    Returns:
        (primary_cause, cause_probabilities_dict)
    """
    if hallucination_prob < threshold:
        faithful_prob = 1.0 - hallucination_prob
        # Distribute remaining probability among failure modes
        rem = hallucination_prob / 3.0
        return "faithful", {
            "faithful": float(faithful_prob),
            "perception_failure": float(rem),
            "retrieval_failure": float(rem),
            "utilization_failure": float(rem),
            "compound_failure": 0.0
        }

    # Deficits from perfect consistency
    d_p = max(0.0, 1.0 - s_p)
    d_r = max(0.0, 1.0 - s_r)
    d_u = max(0.0, 1.0 - s_u)

    total_deficit = d_p + d_r + d_u + 1e-9
    p_p = d_p / total_deficit
    p_r = d_r / total_deficit
    p_u = d_u / total_deficit

    # Detect compound failure (multiple signals critically low, e.g. difference < 0.15)
    deficits = sorted([d_p, d_r, d_u], reverse=True)
    is_compound = (deficits[0] - deficits[1]) < 0.12 and deficits[1] > 0.45

    if is_compound:
        primary_cause = "compound_failure"
    elif d_p >= max(d_r, d_u):
        primary_cause = "perception_failure"
    elif d_r >= max(d_p, d_u):
        primary_cause = "retrieval_failure"
    else:
        primary_cause = "utilization_failure"

    cause_probs = {
        "perception_failure": float(np.round(p_p * hallucination_prob, 4)),
        "retrieval_failure": float(np.round(p_r * hallucination_prob, 4)),
        "utilization_failure": float(np.round(p_u * hallucination_prob, 4)),
        "compound_failure": float(np.round(0.6 * hallucination_prob if is_compound else 0.05 * hallucination_prob, 4)),
        "faithful": float(np.round(1.0 - hallucination_prob, 4))
    }

    return primary_cause, cause_probs


def determine_proxy_attribution(s_p: float, s_r: float, s_u: float) -> str:
    """
    Proxy failure attribution based on lowest individual consistency score.
    Strictly marked as proxy attribution (not human-validated causal attribution).
    """
    scores = {
        "perception_proxy_failure": s_p,
        "retrieval_proxy_failure": s_r,
        "utilization_proxy_failure": s_u
    }
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
        f["dataset"] = r.get("dataset", "")
        f["proxy_attribution"] = determine_proxy_attribution(r.get("s_p", 0.5), r.get("s_r", 0.5), r.get("s_u", 0.5))
        gt = r.get("target") if r.get("target") is not None else r.get("is_hallucination_gt")
        if gt is not None:
            try:
                f["target"] = int(gt)
            except (ValueError, TypeError):
                f["target"] = np.nan
        else:
            f["target"] = np.nan
        rows.append(f)

    return pd.DataFrame(rows)
