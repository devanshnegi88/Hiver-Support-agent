import pandas as pd

from retrieval import PrecedentIndex


def _make_index():
    df = pd.DataFrame([
        {"customer_tweet_id": "1", "customer_message": "my order is late and never arrived",
         "brand_reply": "sorry for the delay, please DM your order number"},
        {"customer_tweet_id": "2", "customer_message": "I want a refund for my broken item",
         "brand_reply": "we can help with that refund, please DM details"},
        {"customer_tweet_id": "3", "customer_message": "how do I reset my password",
         "brand_reply": "please DM us and we will help you regain access"},
    ])
    return PrecedentIndex(df)


def test_search_returns_most_similar_first():
    index = _make_index()
    hits = index.search("my package never showed up, it's late", top_k=1)
    assert len(hits) == 1
    assert "delay" in hits[0].brand_reply

def test_search_respects_top_k():
    index = _make_index()
    hits = index.search("random unrelated query about nothing", top_k=2)
    assert len(hits) == 2


def test_search_excludes_given_id_to_prevent_self_match_leakage():
    index = _make_index()
    hits = index.search("my order is late and never arrived", top_k=1, exclude_id="1")
    assert hits[0].customer_message != "my order is late and never arrived"


def test_exclude_id_matches_int_against_string_column():
    """Regression: pandas reads gold IDs as int64, parquet stores them as str."""
    index = _make_index()
    hits = index.search("my order is late and never arrived", top_k=1, exclude_id=1)
    assert hits[0].customer_tweet_id != "1"
    assert hits[0].customer_message != "my order is late and never arrived"


def test_without_ids_drops_gold_rows_from_index():
    index = _make_index().without_ids({1, "1"})
    hits = index.search("my order is late and never arrived", top_k=3)
    assert all(h.customer_tweet_id != "1" for h in hits)


def test_similarity_scores_are_between_zero_and_one():
    index = _make_index()
    hits = index.search("password reset help please", top_k=3)
    for h in hits:
        assert 0.0 <= h.similarity <= 1.0
