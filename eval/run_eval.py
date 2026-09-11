"""
Runs the full agent AND both baselines over the golden TEST split (never
dev — thresholds/prompts should already be finalized before this runs),
scores everything with bootstrap CIs and paired significance tests against
each baseline, and writes results/eval_results.json.

This is the script that produces the numbers for REPORT.md. Run it exactly
once per submission — re-running after peeking at results and adjusting the
pipeline defeats the point of a held-out test split (adjust on dev instead,
then re-run test once you're done).

Usage:
    python eval/run_eval.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import GOLDEN_TEST_CSV, RESULTS_DIR, INTENTS, BRAND_HANDLE  # noqa: E402
from pipeline import run_agent  # noqa: E402
from retrieval import PrecedentIndex  # noqa: E402
from baselines import trivial_predict, simple_predict  # noqa: E402
from metrics import intent_metrics, escalation_metrics  # noqa: E402
from llm_judge import judge_reply  # noqa: E402
from hallucination_check import hallucination_rate  # noqa: E402
from stats import bootstrap_ci, paired_bootstrap_test  # noqa: E402


def load_test_split():
    if not GOLDEN_TEST_CSV.exists():
        raise FileNotFoundError(
            f"{GOLDEN_TEST_CSV} not found. Run eval/split_golden_set.py first "
            "to create dev/test splits from your labeled golden_set.csv."
        )
    df = pd.read_csv(GOLDEN_TEST_CSV)
    df["escalate_gold"] = df["escalate_gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def run_system(name: str, predict_fn, golden: pd.DataFrame, index: PrecedentIndex = None):
    intent_preds, escalate_preds, judged, hallu_rows = [], [], [], []
    per_example_intent_correct, per_example_escalate_correct = [], []

    for _, row in golden.iterrows():
        msg = row["customer_message"]

        if name == "agent":
            result = run_agent(msg, index, brand=BRAND_HANDLE, exclude_id=row["customer_tweet_id"])
            intent_pred, escalate_pred, reply = result.intent, result.escalate, result.draft_reply
            hits = index.search(msg, top_k=3, exclude_id=row["customer_tweet_id"])
            precedent_texts = [h.brand_reply for h in hits]
            precedents_text_for_judge = f"(top similarity {result.top_precedent_similarity:.2f})"
        else:
            result = predict_fn(msg)
            intent_pred, escalate_pred, reply = result["intent"], result["escalate"], result["reply"]
            precedent_texts = []
            precedents_text_for_judge = "(baseline — no retrieval)"

        intent_preds.append(intent_pred)
        escalate_preds.append(escalate_pred)
        per_example_intent_correct.append(1.0 if intent_pred == row["intent"] else 0.0)
        per_example_escalate_correct.append(1.0 if escalate_pred == row["escalate_gold"] else 0.0)

        judge_scores = judge_reply(msg, reply, precedents_text_for_judge)
        judged.append(judge_scores)
        hallu_rows.append({"message": msg, "reply": reply, "precedent_replies": precedent_texts})

    im = intent_metrics(golden["intent"].tolist(), intent_preds, labels=INTENTS)
    em = escalation_metrics(golden["escalate_gold"].tolist(), escalate_preds)
    hallu = hallucination_rate(hallu_rows)

    ci_intent_acc = bootstrap_ci(per_example_intent_correct)
    ci_escalate_acc = bootstrap_ci(per_example_escalate_correct)

    per_axis_scores = {
        axis: [j[axis] for j in judged] for axis in ["grounded", "relevant", "tone", "actionable"]
    }
    judge_cis = {axis: bootstrap_ci(vals) for axis, vals in per_axis_scores.items()}

    return {
        "system": name,
        "n_examples": len(golden),
        "intent_accuracy": im["accuracy"],
        "intent_accuracy_ci": [ci_intent_acc["ci_low"], ci_intent_acc["ci_high"]],
        "intent_macro_f1": im["macro_f1"],
        "escalation_precision": em["precision_escalate"],
        "escalation_recall": em["recall_escalate"],
        "escalation_false_negative_rate": em["false_negative_rate"],
        "escalation_decision_accuracy_ci": [ci_escalate_acc["ci_low"], ci_escalate_acc["ci_high"]],
        "judge_scores": {axis: judge_cis[axis]["mean"] for axis in judge_cis},
        "judge_scores_ci": {axis: [judge_cis[axis]["ci_low"], judge_cis[axis]["ci_high"]] for axis in judge_cis},
        "hallucination_flagged_rate": hallu["flagged_rate"],
        "hallucination_flagged_examples_sample": hallu["flagged_examples"][:5],
        "confusion_matrix": im["confusion_matrix"],
        "labels_order": im["labels_order"],
        "_per_example_intent_correct": per_example_intent_correct,   # kept for significance tests below
        "_per_example_judge_scores": per_axis_scores,
    }


def significance_vs_baselines(agent_result: dict, baseline_results: list[dict]) -> dict:
    """Paired bootstrap test: is the agent's intent accuracy and judged
    'relevant' score significantly different from each baseline on the
    SAME test examples (same order, since run_system iterates golden in
    fixed order)."""
    out = {}
    for b in baseline_results:
        out[b["system"]] = {
            "intent_accuracy_vs_agent": paired_bootstrap_test(
                agent_result["_per_example_intent_correct"], b["_per_example_intent_correct"]
            ),
            "judge_relevant_vs_agent": paired_bootstrap_test(
                agent_result["_per_example_judge_scores"]["relevant"],
                b["_per_example_judge_scores"]["relevant"],
            ),
        }
    return out


def main():
    golden = load_test_split()
    index = PrecedentIndex.load()

    print(f"Running on {len(golden)} TEST-split examples (held out from threshold tuning)...")

    results = []
    print("  trivial baseline...")
    results.append(run_system("trivial_baseline", trivial_predict, golden))
    print("  simple keyword baseline...")
    results.append(run_system("simple_keyword_baseline", simple_predict, golden))
    print("  agent (full pipeline)...")
    agent_result = run_system("agent", None, golden, index=index)
    results.append(agent_result)

    sig = significance_vs_baselines(agent_result, [r for r in results if r["system"] != "agent"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    out = {"per_system": clean_results, "significance_vs_baselines": sig}
    with open(RESULTS_DIR / "eval_results.json", "w") as f:
        json.dump(out, f, indent=2)

    df = pd.DataFrame(clean_results).drop(
        columns=["confusion_matrix", "labels_order", "hallucination_flagged_examples_sample",
                 "judge_scores", "judge_scores_ci"]
    )
    print("\n" + df.to_string(index=False))

    print("\nSignificance vs. agent (paired bootstrap, test split):")
    for baseline_name, tests in sig.items():
        ia = tests["intent_accuracy_vs_agent"]
        print(f"  agent vs {baseline_name}: intent-accuracy diff={ia['mean_diff']:+.3f} "
              f"CI=[{ia['ci_low']:+.3f},{ia['ci_high']:+.3f}] p={ia['p_value']:.3f} "
              f"({'significant' if ia['significant_at_0.05'] else 'NOT significant'})")

    print(f"\nFull results written to {RESULTS_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()
