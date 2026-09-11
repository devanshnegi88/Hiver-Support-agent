"""
Decide whether a message should be auto-handled or escalated to a human.

Design choice: a deterministic rule layered on top of the LLM's own outputs
(intent + confidence + grounding quality) rather than a second free-form LLM
"should I escalate?" call. Escalation is a safety-critical decision — it
should be auditable and reproducible, not subject to the same prompt-
sensitivity as open-ended generation. Every rule below maps to a specific,
inspectable signal. See DECISION_LOG.md.
"""
from dataclasses import dataclass

from config import ALWAYS_ESCALATE_INTENTS, MIN_AUTO_HANDLE_CONFIDENCE, MIN_GROUNDING_SIMILARITY


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str


def decide(intent: str, confidence: float, top_similarity: float) -> EscalationDecision:
    if intent in ALWAYS_ESCALATE_INTENTS:
        return EscalationDecision(
            True, f"Intent '{intent}' is always routed to a human regardless of confidence."
        )

    if confidence < MIN_AUTO_HANDLE_CONFIDENCE:
        return EscalationDecision(
            True,
            f"Intent classification confidence {confidence:.2f} is below the "
            f"auto-handle threshold ({MIN_AUTO_HANDLE_CONFIDENCE}).",
        )

    if top_similarity < MIN_GROUNDING_SIMILARITY:
        return EscalationDecision(
            True,
            f"No sufficiently similar historical precedent was found "
            f"(best similarity {top_similarity:.2f} < {MIN_GROUNDING_SIMILARITY}); "
            "drafting a reply without precedent risks an ungrounded/incorrect answer.",
        )

    return EscalationDecision(
        False,
        f"Intent '{intent}' classified with confidence {confidence:.2f} and a "
        f"grounded precedent (similarity {top_similarity:.2f}) was found.",
    )
