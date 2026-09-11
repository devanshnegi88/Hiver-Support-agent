import pandas as pd

from llm_judge import heuristic_judge
from run_eval import stratified_subsample


def test_heuristic_judge_returns_all_axes():
    scores = heuristic_judge("where is my order", "Sorry — please DM your order number.", "")
    for ax in ("grounded", "relevant", "tone", "actionable"):
        assert 1 <= scores[ax] <= 5


def test_stratified_subsample_caps_and_keeps_intents():
    rows = []
    for intent in ["delivery_delay_or_missing", "refund_or_return", "positive_feedback"]:
        for i in range(20):
            rows.append({"intent": intent, "customer_tweet_id": f"{intent}-{i}"})
    df = pd.DataFrame(rows)
    out = stratified_subsample(df, n=12)
    assert len(out) == 12
    assert set(out["intent"]) <= set(df["intent"])
    assert out["intent"].nunique() >= 3
