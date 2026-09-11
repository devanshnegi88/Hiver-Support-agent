"""
End-to-end: one customer message in -> full agent decision out.
"""
from dataclasses import dataclass, asdict

from config import BRAND_HANDLE
from intents import classify_intent
from retrieval import PrecedentIndex, RetrievalHit
from reply_gen import draft_reply
from escalation import decide


@dataclass
class AgentResult:
    message: str
    intent: str
    intent_confidence: float
    intent_rationale: str
    top_precedent_similarity: float
    draft_reply: str
    escalate: bool
    escalation_reason: str

    def to_dict(self):
        return asdict(self)


def run_agent(message: str, index: PrecedentIndex, brand: str = BRAND_HANDLE,
              exclude_id: str | None = None) -> AgentResult:
    intent_result = classify_intent(message)

    hits = index.search(message, top_k=3, exclude_id=exclude_id)
    top_sim = hits[0].similarity if hits else 0.0

    decision = decide(
        intent_result["intent"],
        intent_result["confidence"],
        top_sim,
        message=message,
    )

    # Only spend a generation call drafting a reply if we're not immediately
    # escalating for "no grounding" — still draft for other escalation
    # reasons since a human reviewer benefits from a starting draft.
    if decision.reason.startswith("No sufficiently similar"):
        reply = "(Not drafted — no grounded precedent; routed to human.)"
    else:
        reply = draft_reply(message, intent_result["intent"], hits, brand)

    return AgentResult(
        message=message,
        intent=intent_result["intent"],
        intent_confidence=intent_result["confidence"],
        intent_rationale=intent_result["rationale"],
        top_precedent_similarity=top_sim,
        draft_reply=reply,
        escalate=decision.escalate,
        escalation_reason=decision.reason,
    )


if __name__ == "__main__":
    import sys
    idx = PrecedentIndex.load()
    msg = sys.argv[1] if len(sys.argv) > 1 else "My order still hasn't arrived and it's been 2 weeks!"
    result = run_agent(msg, idx)
    import json
    print(json.dumps(result.to_dict(), indent=2))
