"""
Fail loudly if eval could retrieve a gold conversation as its own answer.

Gold examples are sampled from threads.parquet. If exclude_id type-mismatches
(int vs str), TF-IDF returns similarity ~1.0 to the same tweet's historical
reply — that is not grounding, it is copying the label.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import GOLDEN_DEV_CSV, GOLDEN_SET_CSV, GOLDEN_TEST_CSV  # noqa: E402
from retrieval import PrecedentIndex, _id_str  # noqa: E402


class LeakageError(RuntimeError):
    pass


def check_split_id_overlap() -> None:
    if not GOLDEN_DEV_CSV.exists() or not GOLDEN_TEST_CSV.exists():
        return
    dev = set(pd.read_csv(GOLDEN_DEV_CSV)["customer_tweet_id"].map(_id_str))
    test = set(pd.read_csv(GOLDEN_TEST_CSV)["customer_tweet_id"].map(_id_str))
    overlap = dev & test
    if overlap:
        raise LeakageError(
            f"dev/test share {len(overlap)} tweet ids (e.g. {next(iter(overlap))})"
        )


def check_self_retrieval(index: PrecedentIndex, gold: pd.DataFrame, n_probe: int = 25) -> None:
    probe = gold.head(min(n_probe, len(gold)))
    leaks = []
    for _, row in probe.iterrows():
        hits = index.search(
            row["customer_message"],
            top_k=1,
            exclude_id=row["customer_tweet_id"],
        )
        gold_id = _id_str(row["customer_tweet_id"])
        if hits and hits[0].customer_tweet_id == gold_id:
            leaks.append(gold_id)
    if leaks:
        raise LeakageError(
            f"self-retrieval leak on gold ids {leaks[:5]} — exclude_id is not working"
        )


def check_index_excludes_gold(index: PrecedentIndex, gold_ids) -> None:
    indexed = set(index.df["customer_tweet_id"].map(_id_str))
    overlap = indexed & {_id_str(i) for i in gold_ids}
    if overlap:
        raise LeakageError(
            f"eval retrieval index still contains {len(overlap)} golden-set ids"
        )


def run_leakage_checks(index: PrecedentIndex, gold: pd.DataFrame) -> None:
    check_split_id_overlap()
    check_self_retrieval(index, gold)
    check_index_excludes_gold(index, gold["customer_tweet_id"])
    print("Leakage checks passed (split IDs disjoint; gold rows not in eval index).")


if __name__ == "__main__":
    gold = pd.read_csv(GOLDEN_SET_CSV)
    idx = PrecedentIndex.load().without_ids(gold["customer_tweet_id"])
    run_leakage_checks(idx, gold)
