"""Claim extraction from generated VLM responses."""

import re
from typing import List, Optional


def extract_claims_from_answer(
    answer: str,
    max_claims: int = 5,
    question: Optional[str] = None
) -> List[str]:
    """
    Decompose answer into atomic verifiable factual claim statements.
    Extracts grounded visual entities when question context or short answers are provided.
    """
    clean_ans = answer.strip()
    if not clean_ans:
        return ["object"]

    # Strip conversational prefixes (e.g. "The answer is cigarette" -> "cigarette")
    stripped = re.sub(
        r"^(?:the answer is|the correct choice is|the correct answer is|it is|it's|answer:|choice:)\s*",
        "",
        clean_ans,
        flags=re.IGNORECASE
    ).strip()
    stripped = re.sub(r"^\([a-zA-Z0-9]\)\s*", "", stripped).strip().rstrip(".").strip()

    # Binary VQA cases (e.g. "Yes", "No", "Yes, there is...")
    if question and re.match(r"^(?:yes|no)\b", clean_ans.lower()):
        is_affirmative = clean_ans.lower().startswith("yes")
        # Extract visual target object from question
        m = re.search(
            r"(?:is there|are there|do you see)\s+(?:a|an|any)?\s*([a-zA-Z\s]+?)(?:\s+in the image|\s+in this image|\?|$)",
            question,
            re.IGNORECASE
        )
        if m:
            target_entity = m.group(1).strip()
            if is_affirmative:
                return [target_entity]
            else:
                return [f"no {target_entity}"]

    # Direct short answer cases (e.g. "cigarette", "a black cat")
    if len(stripped.split()) <= 4 and stripped:
        return [stripped]

    # Split longer answers on sentence boundaries and bullet points
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+|- |\* ", clean_ans)
    claims: List[str] = []

    for s in raw_sentences:
        s_clean = s.strip().rstrip(".").strip()
        if len(s_clean) > 8:
            claims.append(s_clean)
        if len(claims) >= max_claims:
            break

    if not claims:
        claims = [stripped[:100] if stripped else clean_ans[:100]]

    return claims
