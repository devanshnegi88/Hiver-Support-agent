"""
Decide whether a message should be auto-handled or escalated to a human.

Deterministic rules on (intent, confidence, retrieval similarity, message
text). False auto-handle (sending a wrong reply on billing/account/anger)
is treated as worse than extra human review.
"""
import re
from dataclasses import dataclass

from config import ALWAYS_ESCALATE_INTENTS, MIN_AUTO_HANDLE_CONFIDENCE, MIN_GROUNDING_SIMILARITY

_ANGER = re.compile(
    r"\b(lawyer|attorney|sue|ftc|bbb|never again|worst|furious|"
    r"unacceptable|scam|fraud|stolen|threat)\b",
    re.I,
)
_REPEAT = re.compile(
    r"\b(again|still waiting|second time|third time|keep doing|"
    r"no (one|body) (replied|answered)|on the phone for)\b",
    re.I,
)
_SECURITY = re.compile(
    r"\b(hacked|stolen (card|account)|locked out|can'?t log ?in|password)\b",
    re.I,
)


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str


def decide(intent: str, confidence: float, top_similarity: float,
           message: str = "") -> EscalationDecision:
    if intent in ALWAYS_ESCALATE_INTENTS:
        return EscalationDecision(
            True,
            f"Escalated because intent '{intent}' is a high-risk class "
            f"(billing, complaint, or account access) and is never auto-handled.",
        )

    if message and _ANGER.search(message):
        return EscalationDecision(
            True,
            "Escalated because the customer used legal, fraud, or strong-anger language.",
        )

    if message and _REPEAT.search(message):
        return EscalationDecision(
            True,
            "Escalated because the customer appears to be reporting a repeated "
            "or still-unresolved contact.",
        )

    if intent == "account_access_issue" or (message and _SECURITY.search(message)):
        return EscalationDecision(
            True,
            "Escalated because the request involves account access or security.",
        )

    if intent == "refund_or_return" and confidence < 0.8:
        return EscalationDecision(
            True,
            "Escalated because refund/return requests are money-moving and "
            f"classification confidence {confidence:.2f} is below 0.80.",
        )

    if confidence < MIN_AUTO_HANDLE_CONFIDENCE:
        return EscalationDecision(
            True,
            f"Escalated because classification confidence {confidence:.2f} is "
            f"below the auto-handle threshold ({MIN_AUTO_HANDLE_CONFIDENCE}).",
        )

    if top_similarity < MIN_GROUNDING_SIMILARITY:
        return EscalationDecision(
            True,
            f"Escalated because no sufficiently similar historical resolution "
            f"was retrieved (best similarity {top_similarity:.2f} < "
            f"{MIN_GROUNDING_SIMILARITY}).",
        )

    return EscalationDecision(
        False,
        f"Auto-handled: intent '{intent}' at confidence {confidence:.2f} with "
        f"a grounded precedent (similarity {top_similarity:.2f}).",
    )
