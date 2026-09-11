"""
Splits data/golden_set.csv (your fully hand-labeled set) into a dev split
and a test split, stratified by intent so both splits have a similar label
distribution.

Why this exists: MIN_AUTO_HANDLE_CONFIDENCE and MIN_GROUNDING_SIMILARITY in
config.py are thresholds you will inevitably want to tune once you see how
the agent performs. If you tune them by looking at performance on the same
examples you then report results on, your headline numbers are optimistic
in a way that won't hold up on new traffic — this is the single most common
way take-home eval numbers end up misleading. Dev/test separation is the
fix: tune on dev (`golden_set_dev.csv`), touch test
(`golden_set_test.csv`) exactly once, at the end, for the numbers that go
in REPORT.md.

Usage:
    python eval/split_golden_set.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import GOLDEN_SET_CSV, GOLDEN_DEV_CSV, GOLDEN_TEST_CSV, DEV_SPLIT_FRACTION  # noqa: E402


def main(seed: int = 13):
    if not GOLDEN_SET_CSV.exists():
        raise FileNotFoundError(f"{GOLDEN_SET_CSV} not found — label your golden set first.")

    df = pd.read_csv(GOLDEN_SET_CSV)

    dev_parts, test_parts = [], []
    for intent, group in df.groupby("intent"):
        group = group.sample(frac=1, random_state=seed)  # shuffle within class
        n_dev = max(1, round(len(group) * DEV_SPLIT_FRACTION)) if len(group) > 1 else 0
        dev_parts.append(group.iloc[:n_dev])
        test_parts.append(group.iloc[n_dev:])

    dev = pd.concat(dev_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    dev.to_csv(GOLDEN_DEV_CSV, index=False)
    test.to_csv(GOLDEN_TEST_CSV, index=False)

    print(f"dev:  {len(dev)} examples -> {GOLDEN_DEV_CSV}")
    print(f"test: {len(test)} examples -> {GOLDEN_TEST_CSV}")
    print("\nIntent distribution check (dev vs test should look similar):")
    comp = pd.DataFrame({
        "dev_pct": dev["intent"].value_counts(normalize=True).round(3),
        "test_pct": test["intent"].value_counts(normalize=True).round(3),
    }).fillna(0.0)
    print(comp)
    print(
        "\nUse golden_set_dev.csv while iterating on thresholds/prompts. "
        "Run eval/run_eval.py against golden_set_test.csv exactly once, "
        "at the end, for the numbers you report."
    )


if __name__ == "__main__":
    main()
