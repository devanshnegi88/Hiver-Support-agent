import pandas as pd

from golden_set_builder import filter_candidates, stratified_sample, _looks_english, _keyword_bucket


def test_looks_english_true_for_plain_text():
    assert _looks_english("my order never arrived and I need help") is True


def test_looks_english_false_for_mostly_non_ascii():
    assert _looks_english("これは日本語のツイートです、英語ではありません") is False


def test_keyword_bucket_matches_known_pattern():
    assert _keyword_bucket("this is broken and defective, wrong item sent") == "product_defect_or_wrong_item"


def test_keyword_bucket_falls_back_when_no_match():
    assert _keyword_bucket("xyz completely unrelated text with no keywords") == "no_keyword_match"


def _make_pool(n=60):
    rows = []
    templates = [
        ("my order is very late and never arrived please help", "delivery_delay_or_missing"),
        ("I need a refund for this broken item now", "refund_or_return"),
        ("can I cancel my order please", "order_cancellation_change"),
        ("thanks so much great service really appreciated", "positive_feedback"),
        ("random filler text about nothing keyword related at all", "other"),
    ]
    for i in range(n):
        text, _ = templates[i % len(templates)]
        rows.append({
            "customer_tweet_id": str(i),
            "customer_message": f"{text} extra {i}",  # keep unique-ish to avoid all-dupe
            "brand_reply": "thanks, we will help",
        })
    return pd.DataFrame(rows)


def test_filter_candidates_removes_short_messages():
    df = pd.DataFrame([
        {"customer_tweet_id": "1", "customer_message": "ok", "brand_reply": "np"},
        {"customer_tweet_id": "2", "customer_message": "this is a sufficiently long message about a real issue", "brand_reply": "np"},
    ])
    filtered = filter_candidates(df)
    assert "1" not in filtered["customer_tweet_id"].values
    assert "2" in filtered["customer_tweet_id"].values


def test_filter_candidates_dedupes_near_identical_messages():
    df = pd.DataFrame([
        {"customer_tweet_id": "1", "customer_message": "My order never arrived, please help!!", "brand_reply": "np"},
        {"customer_tweet_id": "2", "customer_message": "my order never arrived please help", "brand_reply": "np"},
        {"customer_tweet_id": "3", "customer_message": "Completely different message about a refund request here", "brand_reply": "np"},
    ])
    filtered = filter_candidates(df)
    # rows 1 and 2 normalize to the same fingerprint -> only one should remain
    assert len(filtered) == 2


def test_stratified_sample_respects_requested_n():
    pool = _make_pool(60)
    sample = stratified_sample(pool, n=20, seed=1)
    assert len(sample) == 20


def test_stratified_sample_has_no_duplicate_ids():
    pool = _make_pool(60)
    sample = stratified_sample(pool, n=20, seed=1)
    assert sample["customer_tweet_id"].duplicated().sum() == 0
