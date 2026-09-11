import pandas as pd
import pytest

from leakage_check import (
    LeakageError,
    check_index_excludes_gold,
    check_self_retrieval,
    check_split_id_overlap,
)
from retrieval import PrecedentIndex


def test_split_ids_do_not_overlap():
    check_split_id_overlap()


def test_self_retrieval_raises_if_exclude_broken():
    df = pd.DataFrame([
        {"customer_tweet_id": "99", "customer_message": "unique zebra refund please",
         "brand_reply": "secret gold reply"},
        {"customer_tweet_id": "100", "customer_message": "password help",
         "brand_reply": "other"},
    ])
    index = PrecedentIndex(df)
    gold = pd.DataFrame([
        {"customer_tweet_id": 99, "customer_message": "unique zebra refund please"},
    ])
    # after the type-normalization fix this must NOT leak
    check_self_retrieval(index, gold, n_probe=1)


def test_eval_index_must_not_contain_gold_ids():
    df = pd.DataFrame([
        {"customer_tweet_id": "1", "customer_message": "late package", "brand_reply": "a"},
        {"customer_tweet_id": "2", "customer_message": "refund", "brand_reply": "b"},
    ])
    full = PrecedentIndex(df)
    with pytest.raises(LeakageError):
        check_index_excludes_gold(full, ["1"])
    clean = full.without_ids(["1"])
    check_index_excludes_gold(clean, ["1"])
