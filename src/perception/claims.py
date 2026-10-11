"""Claim extraction from generated VLM responses."""

import re
from typing import List


def extract_claims_from_answer(answer: str, max_claims: int = 5) -> List[str]:
    """
    Decompose answer into atomic verifiable factual claim statements.
    Handles sentence-level splitting, comma-clauses, and binary answers.
    """
    clean_ans = answer.strip()
    if not clean_ans:
        return ["The scene contains no notable elements."]

    # Direct short answer cases (e.g. "yes", "no", "a black cat")
    if len(clean_ans.split()) <= 4:
        return [clean_ans]

    # Split on sentence boundaries and bullet points
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+|- |\* ", clean_ans)
    claims: List[str] = []

    for s in raw_sentences:
        s_clean = s.strip().rstrip(".").strip()
        if len(s_clean) > 8:
            claims.append(s_clean)
        if len(claims) >= max_claims:
            break

    if not claims:
        claims = [clean_ans[:100]]

    return claims
