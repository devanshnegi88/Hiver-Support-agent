"""
Ablation: does grounding retrieval (retrieval.py) actually improve reply
quality, or is the LLM's own general knowledge doing all the work? This is
directly testable and the assignment's report explicitly wants evidence,
not just an architecture diagram — this is that evidence.

Runs the SAME messages through reply_gen.draft_reply twice: once with the
real retrieved precedents, once with an empty precedent list (forcing the
"no precedent found" prompt path), and compares LLM-judge scores plus the
hallucination_check proxy metric between the two conditions.

Usage:
    python eval/ablation.py
(Run against the DEV split — this is a diagnostic to justify a design
choice, not a headline number, so it doesn't need to touch test.)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import GOLDEN_DEV_CSV, BRAND_HANDLE, RESULTS_DIR  # noqa: E402
from intents import classify_intent  # noqa: E402
from retrieval import PrecedentIndex  # noqa: E402
from reply_gen import draft_reply  # noqa: E402
from llm_judge import judge_reply  # noqa: E402
from hallucination_check import hallucination_rate  # noqa: E402
from stats import paired_bootstrap_test  # noqa: E402


def run_ablation():
    golden = pd.read_csv(GOLDEN_DEV_CSV)
    index = PrecedentIndex.load()

    grounded_judge_scores, ungrounded_judge_scores = [], []
    grounded_rows, ungrounded_rows = [], []

    for _, row in golden.iterrows():
        msg = row["customer_message"]
        intent = classify_intent(msg)["intent"]
        hits = index.search(msg, top_k=3, exclude_id=row["customer_tweet_id"])

        grounded_reply = draft_reply(msg, intent, hits, BRAND_HANDLE)
        ungrounded_reply = draft_reply(msg, intent, [], BRAND_HANDLE)  # empty precedents

        precedent_texts = [h.brand_reply for h in hits]

        g_score = judge_reply(msg, grounded_reply, f"(top similarity {hits[0].similarity:.2f})" if hits else "(none)")
        u_score = judge_reply(msg, ungrounded_reply, "(none — ablation: retrieval disabled)")

        grounded_judge_scores.append(g_score)
        ungrounded_judge_scores.append(u_score)
        grounded_rows.append({"message": msg, "reply": grounded_reply, "precedent_replies": precedent_texts})
        ungrounded_rows.append({"message": msg, "reply": ungrounded_reply, "precedent_replies": precedent_texts})

    results = {}
    for axis in ["grounded", "relevant", "tone", "actionable"]:
        g_vals = [s[axis] for s in grounded_judge_scores]
        u_vals = [s[axis] for s in ungrounded_judge_scores]
        results[axis] = paired_bootstrap_test(g_vals, u_vals)

    results["hallucination_with_retrieval"] = hallucination_rate(grounded_rows)
    results["hallucination_without_retrieval"] = hallucination_rate(ungrounded_rows)

    return results


def main():
    results = run_ablation()

    print("Paired bootstrap test: WITH retrieval vs WITHOUT retrieval (dev split)")
    print("(positive mean_diff = grounding helped; CI excluding 0 = likely real effect)\n")
    for axis in ["grounded", "relevant", "tone", "actionable"]:
        r = results[axis]
        sig = "significant" if r["significant_at_0.05"] else "not significant"
        print(f"  {axis:12s} mean_diff={r['mean_diff']:+.2f}  "
              f"95% CI=[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  p={r['p_value']:.3f}  ({sig})")

    print(f"\nUnsupported-numeric-claims rate WITH retrieval:    "
          f"{results['hallucination_with_retrieval']['flagged_rate']:.1%}")
    print(f"Unsupported-numeric-claims rate WITHOUT retrieval: "
          f"{results['hallucination_without_retrieval']['flagged_rate']:.1%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import json
    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {RESULTS_DIR / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
