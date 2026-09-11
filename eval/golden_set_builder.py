"""
Samples candidate examples for the golden evaluation set and writes them to
a CSV for hand-labeling.

Sampling strategy (see DECISION_LOG.md items 10 and 21 for the "why"):

1. Filter out near-duplicate and very short/low-content messages — real
   support traffic has a lot of boilerplate ("thanks!", retried identical
   complaints) that would otherwise inflate one bucket for free.
2. Filter out probable non-English messages — the intent taxonomy and
   labeling guide are English-only; mixing in un-labelable examples would
   force noisy labels rather than clean gold.
3. Stratify the remaining pool on TWO axes: message-length tercile (a
   proxy for complexity) AND a cheap keyword-heuristic intent guess (a
   proxy for topic). This is a coverage tool only — it decides WHICH real
   messages get shown to you for hand-labeling, it never touches the gold
   label itself (you label independently, see LABELING_GUIDE.md). Without
   this, pure random or length-only sampling would under-represent rare
   but important intents (billing disputes, account access) relative to
   the dominant delivery-complaint traffic, and the golden set would be
   too small in those cells to say anything statistically meaningful about
   them.
4. Guarantee a minimum count per keyword-heuristic bucket (including
   "no keyword matched") so every intent has at least a few representative
   candidates to choose from, even if the true underlying frequency is low.

Usage:
    python eval/golden_set_builder.py --n 200

Writes data/golden_set_candidates.csv with columns:
    customer_tweet_id, customer_message, brand_reply (reference only —
    see LABELING_GUIDE.md for why not to let this anchor your intent label),
    coverage_bucket (which stratification cell this came from — for your
    own tracking, not a suggested label), intent (blank, fill in),
    escalate_gold (blank, fill in: True/False), notes (blank)

After labeling, save as data/golden_set.csv, then run
eval/label_quality_check.py before eval/split_golden_set.py.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import THREADS_PARQUET, DATA_DIR, INTENTS  # noqa: E402
from baselines import _KEYWORD_RULES  # noqa: E402  (reused only for sampling coverage, not as a label)

MIN_MESSAGE_LEN = 15  # below this, messages are usually too low-content to label meaningfully
NON_ASCII_RATIO_THRESHOLD = 0.3  # crude non-English filter


def _dedupe_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse near-identical messages (common in support traffic — same
    complaint retried, copy-pasted templates) to one representative each,
    so the golden set isn't padded with redundant near-copies."""
    fingerprint = (
        df["customer_message"].str.lower()
        .str.replace(r"[^a-z\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df.loc[~fingerprint.duplicated()]


def _looks_english(text: str) -> bool:
    if not text:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / max(len(text), 1)) < NON_ASCII_RATIO_THRESHOLD


def _keyword_bucket(text: str) -> str:
    """Cheap coverage heuristic ONLY — decides which stratification cell a
    candidate falls into for sampling purposes. Never used as the gold
    label; you label from the message itself per LABELING_GUIDE.md."""
    for intent, pattern in _KEYWORD_RULES:
        if pattern.search(text):
            return intent
    return "no_keyword_match"


def filter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["customer_message"].str.len() >= MIN_MESSAGE_LEN]
    df = df[df["customer_message"].apply(_looks_english)]
    df = _dedupe_near_duplicates(df)
    print(f"Filtered {before} -> {len(df)} candidates "
          f"(removed short/non-English/near-duplicate messages)")
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    df["msg_len"] = df["customer_message"].str.len()
    df["length_bucket"] = pd.qcut(df["msg_len"], q=3, labels=["short", "medium", "long"], duplicates="drop")
    df["keyword_bucket"] = df["customer_message"].apply(_keyword_bucket)
    df["coverage_bucket"] = df["length_bucket"].astype(str) + "__" + df["keyword_bucket"]

    cells = df["coverage_bucket"].unique()
    per_cell = max(1, n // len(cells))

    parts = []
    for cell, group in df.groupby("coverage_bucket", observed=True):
        take = min(per_cell, len(group))
        parts.append(group.sample(n=take, random_state=seed))
    sampled = pd.concat(parts)

    # Top up to n from the remaining pool if small/uneven cells left us short.
    if len(sampled) < n:
        remaining = df.drop(sampled.index)
        top_up = remaining.sample(n=min(n - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, top_up])
    elif len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)

    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="Target golden set size. Assignment asks for 150-250.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (150 <= args.n <= 250):
        print(f"WARNING: --n={args.n} is outside the assignment's requested 150-250 range.")

    if not THREADS_PARQUET.exists():
        raise FileNotFoundError("Run src/data_prep.py first to build threads.parquet")

    df = pd.read_parquet(THREADS_PARQUET)
    df = filter_candidates(df)
    sample = stratified_sample(df, args.n, args.seed)

    print("\nCoverage bucket distribution in this sample (for your own visibility —")
    print("expect it roughly even across keyword_bucket, not proportional to real-world frequency):")
    print(sample["coverage_bucket"].value_counts())

    out = sample[["customer_tweet_id", "customer_message", "brand_reply", "coverage_bucket"]].copy()
    out["intent"] = ""          # fill in by hand — see LABELING_GUIDE.md
    out["escalate_gold"] = ""   # fill in by hand: True / False — see LABELING_GUIDE.md
    out["notes"] = ""           # anything ambiguous, edge case, etc.

    out_path = DATA_DIR / "golden_set_candidates.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} candidates to {out_path}")
    print("Label 'intent' and 'escalate_gold' by hand per LABELING_GUIDE.md,")
    print("then save as data/golden_set.csv and run eval/label_quality_check.py.")


if __name__ == "__main__":
    main()
