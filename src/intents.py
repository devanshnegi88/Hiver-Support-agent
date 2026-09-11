"""
Intent classification for incoming customer messages.

Design choice: single LLM call returning {intent, confidence, rationale}
rather than a trained classifier. This is deliberate for a take-home under
time pressure — see DECISION_LOG.md for the tradeoff (cost/latency per call
vs. no training data / no labeled set needed to bootstrap). The eval harness
measures whether this choice actually holds up against a cheaper baseline.
"""
from config import INTENTS, GEN_MODEL
from llm_client import generate_json

_INTENT_LIST = "\n".join(f"- {i}" for i in INTENTS)

_PROMPT_TMPL = """You are classifying a customer support tweet sent to a brand.
Choose exactly one intent from this fixed list:

{intents}

Rules:
- If the message doesn't clearly fit any specific category, use "other_unclassified" — do not force a fit.
- "complaint_escalation" means the customer is expressing anger, threatening to leave/report, or this is a repeat/unresolved contact — not just a negative event.
- confidence is your genuine calibrated probability (0-1) that this is the correct label, not a fixed high number.

Customer message:
\"\"\"{message}\"\"\"

Respond with ONLY this JSON object, no other text:
{{"intent": "<one of the labels above>", "confidence": <0-1 float>, "rationale": "<one short sentence>"}}
"""


def classify_intent(message: str) -> dict:
    prompt = _PROMPT_TMPL.format(intents=_INTENT_LIST, message=message)
    result = generate_json(prompt, model=GEN_MODEL, temperature=0.0)

    intent = result.get("intent", "other_unclassified")
    if intent not in INTENTS:
        intent = "other_unclassified"
    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "intent": intent,
        "confidence": confidence,
        "rationale": result.get("rationale", ""),
    }
