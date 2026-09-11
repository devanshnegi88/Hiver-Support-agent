"""
Run this AFTER hand-labeling data/golden_set.csv and BEFORE
eval/split_golden_set.py. Catches the labeling mistakes that are cheap to
fix now and expensive to discover after you've already run the full eval
(and burned API budget on it).

Usage:
    python eval/label_quality_check.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import GOLDEN_SET_CSV, INTENTS, ALWAYS_ESCALATE_INTENTS  # noqa: E402

MIN_EXAMPLES_PER_INTENT = 5  # below this, macro-F1 on that class is close to meaningless


def check_golden_set(df: pd.DataFrame) -> list[str]:
    """Pure function: given a labeled golden set, return a list of problem
    descriptions (empty list = clean). Kept separate from main() so it's
    unit-testable without touching disk or argv."""
    problems = []

    n = len(df)
    if not (150 <= n <= 250):
        problems.append(f"Golden set has {n} rows — assignment asks for 150-250.")

    missing_intent = df["intent"].isna() | (df["intent"].astype(str).str.strip() == "")
    if missing_intent.any():
        problems.append(f"{missing_intent.sum()} rows have no intent label: "
                         f"{df.loc[missing_intent, 'customer_tweet_id'].tolist()}")

    missing_escalate = ~df["escalate_gold"].astype(str).str.lower().isin(["true", "false"])
    if missing_escalate.any():
        problems.append(f"{missing_escalate.sum()} rows have an invalid/missing escalate_gold "
                         f"(must be exactly 'True' or 'False'): "
                         f"{df.loc[missing_escalate, 'customer_tweet_id'].tolist()}")

    bad_intents = set(df["intent"].dropna()) - set(INTENTS)
    if bad_intents:
        problems.append(f"Found intent labels not in src/config.py INTENTS: {bad_intents}")

    counts = df["intent"].value_counts()
    thin_classes = counts[counts < MIN_EXAMPLES_PER_INTENT]
    if not thin_classes.empty:
        problems.append(
            f"These intents have fewer than {MIN_EXAMPLES_PER_INTENT} examples — "
            f"per-class metrics for them will be noisy, consider targeted re-sampling:\n"
            f"{thin_classes.to_string()}"
        )
    unused_intents = set(INTENTS) - set(counts.index)
    if unused_intents:
        problems.append(f"These intents from the taxonomy have ZERO examples in the golden set: "
                         f"{unused_intents} — you won't be able to say anything about agent "
                         f"behavior on these categories at all.")

    always_escalate_rows = df[df["intent"].isin(ALWAYS_ESCALATE_INTENTS)]
    gold_escalate = always_escalate_rows["escalate_gold"].astype(str).str.lower() == "true"
    if len(always_escalate_rows) and not gold_escalate.all():
        n_mismatch = (~gold_escalate).sum()
        problems.append(
            f"{n_mismatch} examples are labeled with an intent in ALWAYS_ESCALATE_INTENTS "
            f"({ALWAYS_ESCALATE_INTENTS}) but escalate_gold=False. This isn't necessarily wrong "
            f"(your human judgment may reasonably differ from the hard rule on a specific case) "
            f"but review these — a systematic disagreement here means the rule and your own "
            f"labeling philosophy don't actually match, which should be reconciled and written "
            f"up in REPORT.md rather than left implicit."
        )

    dup_ids = df["customer_tweet_id"].duplicated()
    if dup_ids.any():
        problems.append(f"{dup_ids.sum()} duplicate customer_tweet_id rows found.")

    return problems


def main():
    if not GOLDEN_SET_CSV.exists():
        raise FileNotFoundError(f"{GOLDEN_SET_CSV} not found — label your golden set first.")

    df = pd.read_csv(GOLDEN_SET_CSV)
    problems = check_golden_set(df)

    counts = df["intent"].value_counts()
    print(f"Checked {len(df)} labeled examples.\n")
    print("Intent distribution:")
    print(counts.to_string())
    print(f"\nEscalate=True rate: {(df['escalate_gold'].astype(str).str.lower() == 'true').mean():.1%}\n")

    if problems:
        print(f"⚠ {len(problems)} issue(s) found:\n")
        for i, p in enumerate(problems, 1):
            print(f"{i}. {p}\n")
        sys.exit(1)
    else:
        print("✓ No issues found. Safe to run eval/split_golden_set.py.")


if __name__ == "__main__":
    main()
