"""Ablation experiment suite for XMCA-Net-Lite."""

import logging
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from src.combiner.features import build_feature_dataframe
from src.combiner.model import XMCACombiner
from src.combiner.thresholds import tune_decision_threshold
from src.evaluation.bootstrap import compute_bootstrap_confidence_intervals
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def run_ablation_experiment(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    ablation_name: str,
    seeds: List[int] = [42, 101, 202, 303, 404]
) -> Dict[str, Any]:
    """
    Run single ablation condition across 5 seeds.
    Ablations:
    - 'full_model': All signals S_P + S_R + S_U
    - 'no_perception': Remove S_P
    - 'no_retrieval': Remove S_R
    - 'no_utilization': Remove S_U
    - 'text_only_utilization': Evaluated using text-only S_U records
    - 'fixed_weighted_fusion': Evaluated using fixed-weight retrieval fusion records
    - 'min_rule_baseline': 1 - min(S_P, S_R, S_U) heuristic baseline
    - 'cross_vlm_transfer': Tested on LLaVA transfer records
    """
    seed_metrics = []

    # Build dataframes with appropriate feature masks
    feat_ablation = ablation_name if ablation_name in ["no_perception", "no_retrieval", "no_utilization"] else None
    train_df = build_feature_dataframe(train_records, ablation=feat_ablation)
    val_df = build_feature_dataframe(val_records, ablation=feat_ablation)
    test_df = build_feature_dataframe(test_records, ablation=feat_ablation)

    y_test = test_df["target"].values

    for s in seeds:
        combiner = XMCACombiner(model_type="lightgbm", random_state=s)

        if ablation_name == "min_rule_baseline":
            # Heuristic baseline without learned training
            val_probs = combiner.predict_proba(val_df, baseline_min_rule=True)
            thresh, _ = tune_decision_threshold(val_df["target"].values, val_probs, metric="f1")
            test_probs = combiner.predict_proba(test_df, baseline_min_rule=True)
        else:
            combiner.fit(train_df, val_df, target_col="target")
            val_probs = combiner.predict_proba(val_df)
            thresh, _ = tune_decision_threshold(val_df["target"].values, val_probs, metric="f1")
            test_probs = combiner.predict_proba(test_df)

        metrics = compute_all_metrics(y_test, test_probs, threshold=thresh)
        seed_metrics.append(metrics)

    # Aggregate across seeds
    aggregated: Dict[str, Any] = {"ablation": ablation_name, "seeds_evaluated": len(seeds)}
    for m in ["auroc", "auprc", "f1", "precision", "recall", "ece"]:
        vals = [sm[m] for sm in seed_metrics]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        aggregated[f"{m}_mean"] = mean_val
        aggregated[f"{m}_std"] = std_val
        aggregated[f"mean_{m}"] = mean_val
        aggregated[f"std_{m}"] = std_val

    # Compute bootstrap CI on the representative first seed run
    first_seed_probs = test_probs
    ci_results = compute_bootstrap_confidence_intervals(y_test, first_seed_probs, threshold=thresh)
    aggregated["bootstrap_95_ci"] = ci_results

    return aggregated
