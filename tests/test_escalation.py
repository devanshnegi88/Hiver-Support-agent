from escalation import decide
from config import ALWAYS_ESCALATE_INTENTS, MIN_AUTO_HANDLE_CONFIDENCE, MIN_GROUNDING_SIMILARITY


def test_always_escalate_intent_overrides_high_confidence():
    intent = next(iter(ALWAYS_ESCALATE_INTENTS))
    decision = decide(intent=intent, confidence=0.99, top_similarity=0.9)
    assert decision.escalate is True
    assert "always routed" in decision.reason.lower()


def test_low_confidence_escalates_even_for_normal_intent():
    decision = decide(
        intent="general_product_question",
        confidence=MIN_AUTO_HANDLE_CONFIDENCE - 0.01,
        top_similarity=0.9,
    )
    assert decision.escalate is True
    assert "confidence" in decision.reason.lower()


def test_low_grounding_similarity_escalates_despite_high_confidence():
    decision = decide(
        intent="general_product_question",
        confidence=0.99,
        top_similarity=MIN_GROUNDING_SIMILARITY - 0.01,
    )
    assert decision.escalate is True
    assert "precedent" in decision.reason.lower()


def test_auto_handles_when_all_conditions_clear():
    decision = decide(
        intent="general_product_question",
        confidence=MIN_AUTO_HANDLE_CONFIDENCE + 0.1,
        top_similarity=MIN_GROUNDING_SIMILARITY + 0.1,
    )
    assert decision.escalate is False


def test_decision_reason_is_never_empty():
    # every branch must produce an auditable reason — this is the whole
    # point of rule-based escalation over a free-form LLM call
    for intent in ["general_product_question", *ALWAYS_ESCALATE_INTENTS]:
        for conf in [0.0, 0.5, 1.0]:
            for sim in [0.0, 0.5, 1.0]:
                decision = decide(intent, conf, sim)
                assert decision.reason.strip() != ""
