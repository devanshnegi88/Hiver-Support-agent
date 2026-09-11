import pandas as pd

from data_prep import clean_text, build_brand_pairs


def test_clean_text_strips_urls_and_mentions():
    raw = "@AmazonHelp please check https://example.com/track?id=123 thanks!!"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "@AmazonHelp" not in cleaned
    assert "thanks" in cleaned


def test_clean_text_collapses_whitespace():
    assert clean_text("hello    \n\n  world") == "hello world"


def test_clean_text_handles_non_string_input():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def _make_raw_df():
    # tweet_id 1: customer complains
    # tweet_id 2: brand replies to 1
    # tweet_id 3: a brand tweet with no valid parent (should be dropped)
    # tweet_id 4: customer message with no brand reply (never appears as parent)
    return pd.DataFrame([
        {"tweet_id": "1", "author_id": "cust1", "inbound": True, "created_at": "t1",
         "text": "@AmazonHelp my order never arrived", "in_response_to_tweet_id": None, "response_tweet_id": "2"},
        {"tweet_id": "2", "author_id": "AmazonHelp", "inbound": False, "created_at": "t2",
         "text": "So sorry! Please DM your order number.", "in_response_to_tweet_id": "1", "response_tweet_id": None},
        {"tweet_id": "3", "author_id": "AmazonHelp", "inbound": False, "created_at": "t3",
         "text": "Orphan reply with no parent", "in_response_to_tweet_id": "999", "response_tweet_id": None},
        {"tweet_id": "4", "author_id": "cust2", "inbound": True, "created_at": "t4",
         "text": "Just browsing, no issue", "in_response_to_tweet_id": None, "response_tweet_id": None},
    ])


def test_build_brand_pairs_links_customer_message_to_correct_reply():
    df = _make_raw_df()
    pairs = build_brand_pairs(df, brand="AmazonHelp")
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert "order never arrived" in row["customer_message"]
    assert "DM your order number" in row["brand_reply"]


def test_build_brand_pairs_drops_orphan_replies():
    df = _make_raw_df()
    pairs = build_brand_pairs(df, brand="AmazonHelp")
    assert "Orphan reply" not in pairs["brand_reply"].to_string()


def test_build_brand_pairs_wrong_brand_raises():
    df = _make_raw_df()
    import pytest
    with pytest.raises(ValueError):
        build_brand_pairs(df, brand="NotARealBrand")
