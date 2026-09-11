"""
Automated, non-LLM proxy for grounding failure: flags specific numbers
(dollar amounts, day/percentage figures, dates) that appear in a drafted
reply but do NOT appear in any of the retrieved precedent replies.

Why this exists alongside the LLM judge's "grounded" score: the judge is
itself an LLM and can be fooled by fluent-sounding but unsupported
specifics, or may be systematically lenient on the same failure modes its
own model family tends to produce (see REPORT.md, "what's misleading"
section). This is a dumb, deterministic, fully inspectable second signal —
if the two disagree a lot, that's a finding worth reporting, not a bug to
hide.

This is a proxy, not ground truth: it will also flag legitimate numbers the
model correctly copied from the CUSTOMER's own message (e.g. "order
#12345"), so it over-flags. Report it as "candidate unsupported claims per
reply", not "hallucination rate", and spot check a sample by hand.
"""
import re

_NUMBER_RE = re.compile(
    r"\$\s?\d+(?:\.\d+)?"          # $12, $12.50
    r"|\b\d+\s?%"                    # 20%
    r"|\b\d{1,2}[-\u2013]\d{1,2}\s?(?:day|days|hour|hours|business day|business days)\b"  # 3-5 days
    r"|\b\d+\s?(?:day|days|hour|hours|week|weeks)\b",  # 5 days, 2 weeks
    re.IGNORECASE,
)


def extract_numeric_claims(text: str) -> set[str]:
    return {m.strip().lower() for m in _NUMBER_RE.findall(text)}


def unsupported_numeric_claims(reply: str, message: str, precedent_replies: list[str]) -> list[str]:
    """Numeric claims in `reply` not present in the customer's own message
    and not present in any precedent reply. These are the reply's own
    invented specifics — the ones with no evidence anywhere in context.
    """
    reply_claims = extract_numeric_claims(reply)
    if not reply_claims:
        return []

    allowed = extract_numeric_claims(message)
    for p in precedent_replies:
        allowed |= extract_numeric_claims(p)

    return sorted(reply_claims - allowed)


def hallucination_rate(rows: list[dict]) -> dict:
    """rows: list of {"reply": str, "message": str, "precedent_replies": list[str]}
    Returns the fraction of replies with >=1 unsupported numeric claim, plus
    the flagged examples for spot-checking / the failure-analysis section.
    """
    flagged = []
    for r in rows:
        claims = unsupported_numeric_claims(r["reply"], r["message"], r["precedent_replies"])
        if claims:
            flagged.append({"message": r["message"], "reply": r["reply"], "unsupported_claims": claims})

    return {
        "n_examples": len(rows),
        "n_flagged": len(flagged),
        "flagged_rate": len(flagged) / len(rows) if rows else 0.0,
        "flagged_examples": flagged,
    }
