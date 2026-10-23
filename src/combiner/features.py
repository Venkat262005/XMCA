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
    s_p: Optional[float],
    s_r: Optional[float],
    s_u: Optional[float],
    evidence_count: int = 1,
    claim_count: int = 1,
    ablation: Optional[str] = None
) -> Dict[str, float]:
    """
    Extract comprehensive non-linear feature dictionary for a single sample.
    Supports feature masking for ablation studies and principled missingness handling.

    Schema Integrity Notice:
    - Feature #22 ('cross_modal_tension') is an intentional duplicate of Feature #8 ('abs_diff_sp_su').
    - Feature #23 ('knowledge_visual_gap') is an intentional duplicate of Feature #7 ('abs_diff_sp_sr').
    These duplicates are preserved for backward compatibility with the authoritative 25-feature schema
    and will be evaluated in a scheduled deduplication ablation. They are NOT independent dimensions.
    """
    # Fatal check: All signals missing is an invalid row that cannot form a feature vector
    all_missing = (s_p is None or (isinstance(s_p, float) and np.isnan(s_p))) and \
                  (s_r is None or (isinstance(s_r, float) and np.isnan(s_r))) and \
                  (s_u is None or (isinstance(s_u, float) and np.isnan(s_u)))
    if all_missing and ablation is None:
        raise ValueError("Cannot extract combiner features: all consistency signals (s_p, s_r, s_u) are None/unavailable.")

    # Assign active signals or NaN
    feat_sp = np.nan if (s_p is None or (isinstance(s_p, float) and np.isnan(s_p)) or ablation == "no_perception") else float(np.clip(s_p, 0.0, 1.0))
    feat_sr = np.nan if (s_r is None or (isinstance(s_r, float) and np.isnan(s_r)) or ablation == "no_retrieval") else float(np.clip(s_r, 0.0, 1.0))
    feat_su = np.nan if (s_u is None or (isinstance(s_u, float) and np.isnan(s_u)) or ablation == "no_utilization") else float(np.clip(s_u, 0.0, 1.0))

    active_scores = [s for s in [feat_sp, feat_sr, feat_su] if not np.isnan(s)]

    if not active_scores:
        if ablation is not None:
            active_scores = [0.0]
        else:
            raise ValueError("Cannot extract combiner features: active_scores is empty.")

    min_s = float(np.min(active_scores))
    max_s = float(np.max(active_scores))
    mean_s = float(np.mean(active_scores))
    variance_s = float(np.var(active_scores)) if len(active_scores) > 1 else 0.0

    # Harmonic mean calculation
    eps = 1e-6
    h_mean = float(len(active_scores) / sum(1.0 / (s + eps) for s in active_scores))
    h_mean = float(np.clip(h_mean, 0.0, 1.0))

    ratio_min_mean = float(min_s / (mean_s + eps))

    # Binary indicator flags for which signal collapsed (among available signals)
    is_p_low = 1.0 if (not np.isnan(feat_sp) and feat_sp <= min_s and ablation != "no_perception") else 0.0
    is_r_low = 1.0 if (not np.isnan(feat_sr) and feat_sr <= min_s and ablation != "no_retrieval") else 0.0
    is_u_low = 1.0 if (not np.isnan(feat_su) and feat_su <= min_s and ablation != "no_utilization") else 0.0

    # Conjunctive consistency, cross-modal tension, and deficit spread
    active_deficits = [1.0 - s for s in active_scores]
    geom_mean = float(np.prod(active_scores) ** (1.0 / len(active_scores))) if active_scores else 0.0

    # Pairwise differences (NaN if either operand is missing)
    diff_sp_sr = np.nan if (np.isnan(feat_sp) or np.isnan(feat_sr)) else (feat_sp - feat_sr)
    diff_sp_su = np.nan if (np.isnan(feat_sp) or np.isnan(feat_su)) else (feat_sp - feat_su)
    diff_sr_su = np.nan if (np.isnan(feat_sr) or np.isnan(feat_su)) else (feat_sr - feat_su)

    abs_diff_sp_sr = np.nan if (np.isnan(feat_sp) or np.isnan(feat_sr)) else abs(feat_sp - feat_sr)
    abs_diff_sp_su = np.nan if (np.isnan(feat_sp) or np.isnan(feat_su)) else abs(feat_sp - feat_su)
    abs_diff_sr_su = np.nan if (np.isnan(feat_sr) or np.isnan(feat_su)) else abs(feat_sr - feat_su)

    # Cross-modal tension and knowledge visual gap (retained for 25-feature compatibility)
    tension_pu = abs_diff_sp_su
    tension_pr = abs_diff_sp_sr

    def_spread = float(np.max(active_deficits) - np.min(active_deficits)) if len(active_deficits) > 1 else 0.0
    min_max_ratio = float(min_s / (max_s + eps))

    features = {
        "s_p": feat_sp,
        "s_r": feat_sr,
        "s_u": feat_su,
        "diff_sp_sr": diff_sp_sr,
        "diff_sp_su": diff_sp_su,
        "diff_sr_su": diff_sr_su,
        "abs_diff_sp_sr": abs_diff_sp_sr,
        "abs_diff_sp_su": abs_diff_sp_su,
        "abs_diff_sr_su": abs_diff_sr_su,
        "min_score": min_s,
        "max_score": max_s,
        "mean_score": mean_s,
        "harmonic_mean": h_mean,
        "signal_variance": variance_s,
        "min_to_mean_ratio": ratio_min_mean,
        "is_perception_lowest": is_p_low,
        "is_retrieval_lowest": is_r_low,
        "is_utilization_lowest": is_u_low,
        "evidence_count": float(max(0, int(evidence_count))),
        "claim_count": float(max(1, int(claim_count))),
        "geometric_mean": float(np.clip(geom_mean, 0.0, 1.0)),
        "cross_modal_tension": tension_pu,  # Documented duplicate of #8
        "knowledge_visual_gap": tension_pr, # Documented duplicate of #7
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


def determine_proxy_attribution(s_p: Optional[float], s_r: Optional[float], s_u: Optional[float]) -> str:
    """
    Proxy failure attribution based on lowest individual consistency score.
    Strictly marked as proxy attribution (not human-validated causal attribution).
    """
    val_p = float(s_p) if (s_p is not None and not np.isnan(s_p)) else 0.5
    val_r = float(s_r) if (s_r is not None and not np.isnan(s_r)) else 0.5
    val_u = float(s_u) if (s_u is not None and not np.isnan(s_u)) else 0.5
    scores = {
        "perception_proxy_failure": val_p,
        "retrieval_proxy_failure": val_r,
        "utilization_proxy_failure": val_u
    }
    return min(scores, key=scores.get)


def build_feature_dataframe(records: List[Dict[str, Any]], ablation: Optional[str] = None) -> pd.DataFrame:
    """Convert score records to feature DataFrame for classifier training / inference."""
    rows = []
    for r in records:
        sp_raw = r.get("s_p")
        sp = float(sp_raw) if sp_raw is not None else None

        sr_raw = r.get("s_r")
        sr = float(sr_raw) if sr_raw is not None else None

        su_raw = r.get("s_u")
        su = float(su_raw) if su_raw is not None else None

        f = extract_combiner_features(
            s_p=sp,
            s_r=sr,
            s_u=su,
            evidence_count=r.get("evidence_count", 1),
            claim_count=r.get("claim_count", 1),
            ablation=ablation
        )

        f["id"] = r.get("id", "")
        f["dataset"] = r.get("dataset", "")
        f["proxy_attribution"] = determine_proxy_attribution(sp, sr, su)
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
