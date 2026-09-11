import pandas as pd

from label_quality_check import check_golden_set, MIN_EXAMPLES_PER_INTENT
from config import INTENTS


def _valid_base_df():
    """A minimally valid golden set: enough rows, every intent covered with
    at least MIN_EXAMPLES_PER_INTENT, all labels valid."""
    rows = []
    tid = 0
    per_intent = max(MIN_EXAMPLES_PER_INTENT, 160 // len(INTENTS) + 1)
    for intent in INTENTS:
        for _ in range(per_intent):
            rows.append({
                "customer_tweet_id": str(tid),
                "customer_message": f"message about {intent} number {tid}",
                "intent": intent,
                "escalate_gold": "True" if intent in ("complaint_escalation", "billing_or_charge_dispute") else "False",
            })
            tid += 1
    return pd.DataFrame(rows)


def test_valid_golden_set_has_no_problems():
    df = _valid_base_df()
    assert len(df) >= 150
    problems = check_golden_set(df)
    assert problems == []


def test_flags_wrong_size():
    df = _valid_base_df().head(10)  # far too few rows
    problems = check_golden_set(df)
    assert any("rows" in p and "150-250" in p for p in problems)


def test_flags_missing_intent():
    df = _valid_base_df()
    df.loc[0, "intent"] = ""
    problems = check_golden_set(df)
    assert any("no intent label" in p for p in problems)


def test_flags_invalid_escalate_value():
    df = _valid_base_df()
    df.loc[0, "escalate_gold"] = "maybe"
    problems = check_golden_set(df)
    assert any("invalid/missing escalate_gold" in p for p in problems)


def test_flags_unknown_intent_label():
    df = _valid_base_df()
    df.loc[0, "intent"] = "not_a_real_intent"
    problems = check_golden_set(df)
    assert any("not in src/config.py INTENTS" in p for p in problems)


def test_flags_thin_class():
    df = _valid_base_df()
    # collapse one intent down to a single example
    one_intent = INTENTS[0]
    mask = df["intent"] == one_intent
    keep_one = df[mask].index[1:]
    df = df.drop(index=keep_one)
    problems = check_golden_set(df)
    assert any("fewer than" in p for p in problems)


def test_flags_escalate_rule_mismatch():
    df = _valid_base_df()
    df.loc[df["intent"] == "complaint_escalation", "escalate_gold"] = "False"
    problems = check_golden_set(df)
    assert any("ALWAYS_ESCALATE_INTENTS" in p for p in problems)


def test_flags_duplicate_ids():
    df = _valid_base_df()
    df.loc[1, "customer_tweet_id"] = df.loc[0, "customer_tweet_id"]
    problems = check_golden_set(df)
    assert any("duplicate customer_tweet_id" in p for p in problems)
