"""Command-Line Interface for XMCA-Net-Lite Multimodal Diagnostic System."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.diagnostic_engine import XMCADiagnosticEngine


def main():
    parser = argparse.ArgumentParser(
        description="XMCA-Net-Lite: Diagnose VLM Hallucination & Attribution Cause"
    )
    parser.add_argument("--image", type=str, default="dummy.jpg", help="Path to input image file")
    parser.add_argument("--question", type=str, required=True, help="Multimodal question asked to the VLM")
    parser.add_argument("--answer", type=str, required=True, help="Generated VLM answer to audit")
    parser.add_argument("--evidence", nargs="*", default=None, help="Retrieved knowledge passages (optional)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Hallucination decision threshold")
    parser.add_argument("--json", action="store_true", help="Output full report as formatted JSON")

    args = parser.parse_args()

    engine = XMCADiagnosticEngine(threshold=args.threshold)
    report = engine.diagnose(
        image=args.image,
        question=args.question,
        answer=args.answer,
        retrieved_passages=args.evidence
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return

    print("\n" + "=" * 65)
    print("        XMCA-Net-Lite Multimodal Diagnostic Report")
    print("=" * 65)
    verdict_badge = "[HALLUCINATION DETECTED]" if report.is_hallucination else "[FAITHFUL / GROUNDED]"
    print(f"Status:                      {verdict_badge}")
    print(f"Hallucination Probability:   {report.hallucination_probability:.1%}")
    print(f"Primary Breakdown Cause:     {report.primary_cause.upper()}")
    print("-" * 65)
    print("Consistency Scores:")
    print(f"  • Perception (S_P):        {report.consistency_scores['s_p']:.4f}")
    print(f"  • Retrieval  (S_R):        {report.consistency_scores['s_r']:.4f}")
    print(f"  • Utilization (S_U):       {report.consistency_scores['s_u']:.4f}")
    print("-" * 65)
    print("Cause Attribution Probabilities:")
    for cause, prob in report.cause_probabilities.items():
        bar = "█" * int(prob * 30)
        print(f"  • {cause:<22} [{prob:5.1%}] {bar}")
    print("-" * 65)
    print("Atomic Claim Audit:")
    for c in report.claims_audited:
        status_icon = "✓" if c.verdict == "grounded" else "✗"
        print(f"  [{status_icon}] Claim #{c.claim_index}: '{c.claim_text}'")
        print(f"      Grounding: P={c.perception_support:.2f} | R={c.retrieval_support:.2f} | U={c.utilization_support:.2f} | Verdict={c.verdict}")
    print("-" * 65)
    print("Diagnostic Summary:")
    print(f"  {report.natural_language_explanation}")
    print("\nActionable Recommendation:")
    print(f"  {report.actionable_recommendation}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
