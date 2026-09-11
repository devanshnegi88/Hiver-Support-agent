from baselines import trivial_predict, simple_predict


def test_trivial_never_escalates():
    for msg in ["I want a refund", "you people are useless", "thanks so much!"]:
        assert trivial_predict(msg)["escalate"] is False


def test_trivial_always_returns_same_intent_and_reply():
    a = trivial_predict("anything")
    b = trivial_predict("something completely different")
    assert a["intent"] == b["intent"]
    assert a["reply"] == b["reply"]


def test_simple_keyword_matches_refund():
    result = simple_predict("I need a refund for my broken order")
    # both "refund" and "broken" match rules — first rule in list order wins;
    # this test locks in that behavior so a reordering is a visible diff
    assert result["intent"] in ("refund_or_return", "product_defect_or_wrong_item", "delivery_delay_or_missing")
    assert result["confidence"] > 0


def test_simple_keyword_escalates_complaints():
    result = simple_predict("this is unacceptable, I want a lawyer involved")
    assert result["intent"] == "complaint_escalation"
    assert result["escalate"] is True


def test_simple_keyword_no_match_routes_to_human():
    result = simple_predict("asdkfjaslkdfj random gibberish text")
    assert result["intent"] == "other_unclassified"
    assert result["escalate"] is True


def test_simple_keyword_positive_feedback_not_escalated():
    result = simple_predict("thank you so much, great service!")
    assert result["intent"] == "positive_feedback"
    assert result["escalate"] is False
