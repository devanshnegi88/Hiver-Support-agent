"""
Evaluation harness.

  python eval/run_eval.py           # FAST / <15 min: 30 test rows, heuristic judge
  python eval/run_eval.py --full    # FULL: all 140 test rows, LLM judge if a backend exists

FAST is a smoke/repro path. It is NOT an LLM-as-judge quality evaluation.
FULL is the quality evaluation. If no LLM is available, FULL still runs but
records judge_type=heuristic and backend=heuristic — it does not pretend
otherwise.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import (  # noqa: E402
    BRAND_HANDLE,
    GOLDEN_SET_CSV,
    GOLDEN_TEST_CSV,
    INTENTS,
    MIN_AUTO_HANDLE_CONFIDENCE,
    MIN_GROUNDING_SIMILARITY,
    RESULTS_DIR,
)
from pipeline import run_agent  # noqa: E402
from retrieval import PrecedentIndex  # noqa: E402
from baselines import trivial_predict, simple_predict  # noqa: E402
from metrics import intent_metrics, escalation_metrics  # noqa: E402
from llm_judge import judge_reply  # noqa: E402
from hallucination_check import hallucination_rate  # noqa: E402
from stats import bootstrap_ci, paired_bootstrap_test  # noqa: E402
from llm_client import get_active_backend  # noqa: E402
from leakage_check import run_leakage_checks  # noqa: E402


def load_test_split():
    if not GOLDEN_TEST_CSV.exists():
        raise FileNotFoundError(
            f"{GOLDEN_TEST_CSV} not found. Run eval/split_golden_set.py first "
            "to create dev/test splits from your labeled golden_set.csv."
        )
    df = pd.read_csv(GOLDEN_TEST_CSV)
    df["escalate_gold"] = df["escalate_gold"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Keep intent mix, cap at n rows. Used by --fast."""
    if n >= len(df):
        return df
    parts = []
    groups = list(df.groupby("intent", sort=False))
    per = max(1, n // max(1, len(groups)))
    for _, g in groups:
        take = min(len(g), per)
        parts.append(g.sample(n=take, random_state=seed) if take < len(g) else g)
    out = pd.concat(parts, ignore_index=False)
    if len(out) < n:
        rest = df.drop(out.index)
        extra_n = min(n - len(out), len(rest))
        if extra_n:
            out = pd.concat([out, rest.sample(n=extra_n, random_state=seed)])
    return out.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)


def run_system(name: str, predict_fn, golden: pd.DataFrame, index: PrecedentIndex = None,
               heuristic_judge: bool = False):
    intent_preds, escalate_preds, judged, hallu_rows = [], [], [], []
    per_example_intent_correct, per_example_escalate_correct = [], []

    for _, row in golden.iterrows():
        msg = row["customer_message"]

        if name.startswith("agent"):
            result = run_agent(msg, index, brand=BRAND_HANDLE, exclude_id=row["customer_tweet_id"])
            intent_pred, escalate_pred, reply = result.intent, result.escalate, result.draft_reply
            hits = index.search(msg, top_k=3, exclude_id=row["customer_tweet_id"])
            precedent_texts = [h.brand_reply for h in hits]
            precedents_text_for_judge = "\n".join(
                f"[sim={h.similarity:.2f}] {h.brand_reply}" for h in hits
            ) or "(no precedent retrieved)"
        else:
            result = predict_fn(msg)
            intent_pred, escalate_pred, reply = result["intent"], result["escalate"], result["reply"]
            precedent_texts = []
            precedents_text_for_judge = "(baseline — no retrieval)"

        intent_preds.append(intent_pred)
        escalate_preds.append(escalate_pred)
        per_example_intent_correct.append(1.0 if intent_pred == row["intent"] else 0.0)
        per_example_escalate_correct.append(1.0 if escalate_pred == row["escalate_gold"] else 0.0)

        judge_scores = judge_reply(
            msg, reply, precedents_text_for_judge, heuristic=heuristic_judge
        )
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
        "escalation_f1": em["f1_escalate"],
        "escalation_false_negative_rate": em["false_negative_rate"],
        "auto_handle_rate": em["auto_handle_rate"],
        "false_auto_handle_rate": em["false_auto_handle_rate"],
        "per_intent": im.get("per_intent", {}),
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="FULL quality eval: all 140 test examples; LLM judge if a backend exists.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="FAST smoke eval: 30 test rows + heuristic judge (default).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Alias for --quick.",
    )
    parser.add_argument("--n", type=int, default=None, help="Cap test examples.")
    parser.add_argument("--heuristic-judge", action="store_true")
    parser.add_argument("--llm-judge", action="store_true")
    args = parser.parse_args()

    golden_all = load_test_split()
    use_full = args.full and not args.fast and not args.quick
    heuristic = (not args.llm_judge) and (args.heuristic_judge or not use_full)
    n = args.n
    if n is None and not use_full:
        n = 30
    golden = stratified_subsample(golden_all, n) if n is not None else golden_all

    all_gold = pd.read_csv(GOLDEN_SET_CSV)
    index = PrecedentIndex.load().without_ids(all_gold["customer_tweet_id"])
    run_leakage_checks(index, all_gold)

    # Probe once so we do not advertise an LLM judge when only keywords exist.
    from llm_client import generate as _probe  # noqa: PLC0415
    try:
        _probe("ping", model="probe", max_retries=1)
    except Exception:
        pass
    probed = get_active_backend()
    if probed == "heuristic":
        heuristic = True

    mode = "FULL" if use_full else "QUICK"
    judge_kind = "heuristic" if heuristic else "llm"
    print(
        f"[{mode}] n={len(golden)} TEST examples | judge={judge_kind} "
        f"| FAST is smoke only; FULL is quality eval."
    )
    if heuristic:
        print("  NOTE: heuristic judge is NOT LLM-as-judge. Do not report these "
              "axis scores as LLM quality.")

    results = []
    print("  trivial baseline...")
    results.append(run_system("trivial_baseline", trivial_predict, golden, heuristic_judge=heuristic))
    print("  simple keyword baseline...")
    results.append(run_system("simple_keyword_baseline", simple_predict, golden, heuristic_judge=heuristic))
    print("  agent (retrieval + LLM-or-fallback)...")
    agent_result = run_system("agent", None, golden, index=index, heuristic_judge=heuristic)
    backend = get_active_backend()
    if backend == "heuristic":
        agent_result["system"] = "agent_keyword_fallback"
        print("  WARNING: no LLM answered; agent used the keyword/heuristic backend. "
              "This is NOT an LLM-agent result.")
    else:
        print(f"  Agent LLM backend: {backend}")
    results.append(agent_result)

    sig = significance_vs_baselines(
        agent_result, [r for r in results if not r["system"].startswith("agent")]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clean_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    out = {
        "mode": mode,
        "n_examples": len(golden),
        "judge_type": judge_kind,
        "agent_backend": backend,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "min_auto_handle_confidence": MIN_AUTO_HANDLE_CONFIDENCE,
            "min_grounding_similarity": MIN_GROUNDING_SIMILARITY,
        },
        "retrieval": "tfidf_cosine_excluding_all_golden_ids",
        "per_system": clean_results,
        "significance_vs_baselines": sig,
    }
    out_name = "full_results.json" if use_full else "quick_results.json"
    with open(RESULTS_DIR / out_name, "w") as f:
        json.dump(out, f, indent=2)

    df = pd.DataFrame(clean_results).drop(
        columns=["confusion_matrix", "labels_order", "hallucination_flagged_examples_sample",
                 "judge_scores", "judge_scores_ci", "per_intent"],
        errors="ignore",
    )
    print("\n" + df.to_string(index=False))

    print("\nSignificance vs. agent (paired bootstrap, test split):")
    for baseline_name, tests in sig.items():
        ia = tests["intent_accuracy_vs_agent"]
        print(f"  agent vs {baseline_name}: intent-accuracy diff={ia['mean_diff']:+.3f} "
              f"CI=[{ia['ci_low']:+.3f},{ia['ci_high']:+.3f}] p={ia['p_value']:.3f} "
              f"({'significant' if ia['significant_at_0.05'] else 'NOT significant'})")

    print(f"\nResults written to {RESULTS_DIR / out_name}")
    if not use_full:
        # Alias used by README / older docs.
        with open(RESULTS_DIR / "eval_results.json", "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
