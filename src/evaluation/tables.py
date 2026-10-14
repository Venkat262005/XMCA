"""Publication-grade LaTeX and Markdown table generators for research manuscripts."""

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


def generate_main_results_table_latex(
    test_metrics: Dict[str, float],
    ci_metrics: Dict[str, Dict[str, float]],
    output_path: str | Path
) -> str:
    """Generate publication Table 1 (Main Diagnostic Performance) in LaTeX."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{XMCA-Net-Lite Main Diagnostic Performance on Held-Out Test Set.} Evaluating calibrated hallucination detection across all consistency signals with non-parametric bootstrap 95\% confidence intervals.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Value} & \textbf{95\% CI Lower} & \textbf{95\% CI Upper} \\",
        r"\midrule",
    ]

    metric_names = [
        ("AUROC", "auroc"),
        ("AUPRC", "auprc"),
        ("F1 Score", "f1"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("ECE (Expected Calibration Error)", "ece"),
    ]

    for display_name, key in metric_names:
        val = test_metrics.get(key, 0.0)
        ci = ci_metrics.get(key, {})
        lower = ci.get("lower_ci", 0.0)
        upper = ci.get("upper_ci", 0.0)
        lines.append(f"{display_name} & {val:.4f} & {lower:.4f} & {upper:.4f} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_code = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex_code)

    return latex_code


def generate_ablation_table_latex(
    ablation_results: List[Dict[str, Any]],
    output_path: str | Path
) -> str:
    """Generate publication Table 2 (Ablation Study) in LaTeX with DeLong p-values."""
    import numpy as np
    from src.evaluation.delong import compute_delong_test

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    full_model_entry = next((r for r in ablation_results if r.get("ablation") == "full_model"), None)
    has_delong = (
        full_model_entry is not None
        and "test_probs" in full_model_entry
        and "y_test" in full_model_entry
    )

    col_spec = "lcccccc" + ("c" if has_delong else "")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Ablation Study across 5 Random Seeds.} Evaluating marginal contribution of Perception ($S_P$), Retrieval ($S_R$), and Utilization ($S_U$), comparing learned isotonic fusion vs fixed weighting, text-only utilization, and heuristic minimum baseline with DeLong test for paired AUROC significance vs. Full Model.}",
        r"\label{tab:ablations}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\textbf{Ablation Variant} & \textbf{AUROC} $\uparrow$ & \textbf{AUPRC} $\uparrow$ & \textbf{F1} $\uparrow$ & \textbf{Precision} $\uparrow$ & \textbf{Recall} $\uparrow$ & \textbf{ECE} $\downarrow$" + (r" & \textbf{DeLong $p$-val} \\" if has_delong else r" \\"),
        r"\midrule",
    ]

    for r in ablation_results:
        name = r.get("ablation", "").replace("_", " ").title()
        auroc_str = f"{r.get('auroc_mean', 0.0):.3f} $\\pm$ {r.get('auroc_std', 0.0):.3f}"
        auprc_str = f"{r.get('auprc_mean', 0.0):.3f} $\\pm$ {r.get('auprc_std', 0.0):.3f}"
        f1_str = f"{r.get('f1_mean', 0.0):.3f} $\\pm$ {r.get('f1_std', 0.0):.3f}"
        prec_str = f"{r.get('precision_mean', 0.0):.3f} $\\pm$ {r.get('precision_std', 0.0):.3f}"
        rec_str = f"{r.get('recall_mean', 0.0):.3f} $\\pm$ {r.get('recall_std', 0.0):.3f}"
        ece_str = f"{r.get('ece_mean', 0.0):.3f} $\\pm$ {r.get('ece_std', 0.0):.3f}"

        row_str = f"{name} & {auroc_str} & {auprc_str} & {f1_str} & {prec_str} & {rec_str} & {ece_str}"

        if has_delong:
            if r.get("ablation") == "full_model":
                delong_str = "---"
            elif "test_probs" in r and len(r["test_probs"]) == len(full_model_entry["test_probs"]):
                y_true = np.asarray(full_model_entry["y_test"])
                p_full = np.asarray(full_model_entry["test_probs"])
                p_cur = np.asarray(r["test_probs"])
                d_res = compute_delong_test(y_true, p_full, p_cur)
                pval = d_res["p_value"]
                if pval < 0.001:
                    delong_str = "$p < 0.001$*"
                elif pval < 0.05:
                    delong_str = f"${pval:.3f}$*"
                else:
                    delong_str = f"${pval:.3f}$"
            else:
                delong_str = "N/A"
            row_str += f" & {delong_str}"

        lines.append(row_str + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    latex_code = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex_code)

    return latex_code
