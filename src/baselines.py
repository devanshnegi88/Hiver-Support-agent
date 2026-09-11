"""
Two baselines, as the report requires "results vs at least two baselines
(a trivial one and a simple one)".

TRIVIAL: majority-class intent + one fixed canned reply for everything,
always escalate nothing (auto-handle everything). This is the "what if we
did the laziest possible thing" floor.

SIMPLE: keyword/regex rule-based intent classifier + template replies keyed
by intent, with a basic escalation rule (escalate if no keyword matched).
This is the "what if we didn't use an LLM at all" comparison — cheap, fast,
zero API cost, and a genuinely reasonable thing a team might ship first.
"""
import re

from config import INTENTS

# ---- Trivial baseline -------------------------------------------------------
TRIVIAL_MAJORITY_INTENT = "delivery_delay_or_missing"  # set from training-split label counts
TRIVIAL_CANNED_REPLY = (
    "Thanks for reaching out! Please DM us your order number and we'll look into this right away."
)


def trivial_predict(message: str) -> dict:
    return {
        "intent": TRIVIAL_MAJORITY_INTENT,
        "confidence": 1.0,
        "reply": TRIVIAL_CANNED_REPLY,
        "escalate": False,
        "reason": "Trivial baseline never escalates.",
    }


# ---- Simple keyword baseline -------------------------------------------------
_KEYWORD_RULES = [
    ("delivery_delay_or_missing", re.compile(r"\b(late|hasn.t arrived|missing|where is my|still waiting|tracking)\b", re.I)),
    ("refund_or_return", re.compile(r"\b(refund|return|money back|reimburse)\b", re.I)),
    ("order_cancellation_change", re.compile(r"\b(cancel|change my order|wrong address)\b", re.I)),
    ("product_defect_or_wrong_item", re.compile(r"\b(broken|defective|damaged|wrong item|doesn.t work)\b", re.I)),
    ("account_access_issue", re.compile(r"\b(can.t log ?in|password|locked out|account access)\b", re.I)),
    ("billing_or_charge_dispute", re.compile(r"\b(charged twice|overcharged|billing|unexpected charge)\b", re.I)),
    ("positive_feedback", re.compile(r"\b(thank you|thanks|great service|love)\b", re.I)),
    ("complaint_escalation", re.compile(r"\b(unacceptable|worst|never again|lawyer|report you|furious)\b", re.I)),
]

_TEMPLATE_REPLIES = {
    "delivery_delay_or_missing": "Sorry for the delay! Please DM your order number so we can check tracking.",
    "refund_or_return": "We can help with that. Please DM your order number to start the refund process.",
    "order_cancellation_change": "Please DM your order number and we'll help update or cancel it.",
    "product_defect_or_wrong_item": "Sorry to hear that. Please DM photos and your order number so we can resolve this.",
    "account_access_issue": "Please DM us and we'll help you regain access to your account.",
    "billing_or_charge_dispute": "We'll look into this. Please DM your order/transaction details.",
    "positive_feedback": "Thank you so much for the kind words! 😊",
    "complaint_escalation": "We're sorry for the frustration — a member of our team will follow up with you shortly.",
    "general_product_question": "Happy to help — could you share a few more details about what you're looking for?",
    "other_unclassified": "Thanks for reaching out! Could you share a bit more detail so we can help?",
}


def simple_predict(message: str) -> dict:
    matched_intent = None
    for intent, pattern in _KEYWORD_RULES:
        if pattern.search(message):
            matched_intent = intent
            break

    if matched_intent is None:
        return {
            "intent": "other_unclassified",
            "confidence": 0.0,
            "reply": _TEMPLATE_REPLIES["other_unclassified"],
            "escalate": True,
            "reason": "No keyword rule matched; routed to human by default.",
        }

    escalate = matched_intent in ("complaint_escalation", "billing_or_charge_dispute")
    return {
        "intent": matched_intent,
        "confidence": 0.6,
        "reply": _TEMPLATE_REPLIES[matched_intent],
        "escalate": escalate,
        "reason": (
            f"Keyword rule matched intent '{matched_intent}'; "
            + ("escalated per fixed rule for this intent." if escalate else "auto-handled.")
        ),
    }
