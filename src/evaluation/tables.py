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
    """Generate publication Table 2 (Ablation Study) in LaTeX."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Ablation Study across 5 Random Seeds.} Evaluating marginal contribution of Perception ($S_P$), Retrieval ($S_R$), and Utilization ($S_U$), comparing learned isotonic fusion vs fixed weighting, text-only utilization, and heuristic minimum baseline.}",
        r"\label{tab:ablations}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Ablation Variant} & \textbf{AUROC} $\uparrow$ & \textbf{AUPRC} $\uparrow$ & \textbf{F1} $\uparrow$ & \textbf{Precision} $\uparrow$ & \textbf{Recall} $\uparrow$ & \textbf{ECE} $\downarrow$ \\",
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
        lines.append(f"{name} & {auroc_str} & {auprc_str} & {f1_str} & {prec_str} & {rec_str} & {ece_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    latex_code = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex_code)

    return latex_code
