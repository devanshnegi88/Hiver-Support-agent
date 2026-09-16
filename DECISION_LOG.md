# Decision log

Non-obvious calls, and why. (15 items — assignment asks for 10–15.)

1. **AmazonHelp, not a smaller brand.** Highest volume and the widest mix of delivery / refund / account / billing / complaints. A one-issue brand would make intent and escalation look solved without being interesting.

2. **Ten intents, including `other_unclassified`.** Small on purpose. The extra bucket is so the classifier can refuse a false-precision fit instead of stuffing edge cases into a nearby class and inflating accuracy.

3. **LLM (or fallback) for intent, not a trained classifier.** The only labels we have are the 200-example golden set, and that set is reserved for eval. Training a classifier on it would leak the test set or force a second labeling round we did not have time for.

4. **TF-IDF retrieval, not embeddings.** No extra API, no vector DB, and you can see which words drove a match. Cost: misses paraphrase (“hasn’t shown up” vs “still waiting”). Flagged as next-week work.

5. **Escalation is a rule on (intent, confidence, similarity), not a second LLM call.** That decision is the safety gate. It has to be deterministic and auditable: you can point at the threshold that fired. An LLM “should I escalate?” would inherit the same prompt jitter as generation.

6. **`complaint_escalation` and `billing_or_charge_dispute` always escalate.** Wrong auto-replies here cost money or trust. Giving up auto-handle rate on those two classes is the cost-asymmetry choice, not a metric hack. “Mentions billing” is not this intent (address-change how-tos were relabeled).

7. **One global confidence floor (0.65) and one grounding floor (0.25).** Easier to defend in a first version than per-intent knobs. The 15-minute run’s escalation recall of 1.0 is largely this floor plus (6), not “the model reasoned about risk.”

8. **Do not draft a reply when no precedent clears 0.25 similarity.** An ungrounded draft looks confident to a reviewer and has no evidence behind it. Routing to a human with “(not drafted)” is safer than a fluent guess.

9. **The reply prompt bans invented amounts, timelines, and policies** unless a retrieved precedent supports them. Hallucinated “you’ll get a $50 refund in 3 days” is the failure mode that actually hurts customers; general “be grounded” language is not enough.

10. **Golden set is stratified (length tercile × keyword coverage bucket), then filtered for near-duplicates and <15-char noise, then quality-checked before split.** Random sampling would be almost all short delivery tweets. The keyword bucket is a sampling tool, never the label. `label_quality_check.py` is a gate so gold and `ALWAYS_ESCALATE_INTENTS` cannot silently disagree.

11. **Dev/test split is stratified by intent (60/140); eval never tunes on test.** Gold tweet IDs are `int64` in CSV and strings in parquet — comparing them with `==` silently failed, so TF-IDF retrieved the example's own historical reply (sim≈1). Fix: normalize IDs to `str`, drop **all** golden IDs from the eval index, and fail `eval/leakage_check.py` if a self-hit remains.

12. **Judge scores four axes (grounded / relevant / tone / actionable), not one “quality” number.** Failure analysis needs *why*. Human calibration (`judge_calibration.py`) is a two-step script that makes you score before you see the judge — otherwise “agreement” is just anchoring.

13. **Baselines are majority-class never-escalate, and keyword/regex — not a trained linear model.** Majority class is the honest floor (do nothing). Keywords are what a team could ship in an afternoon with zero ML, which is the actual question “why do we need an LLM?”

14. **Every headline metric has a bootstrap 95% CI; agent vs baseline is a paired bootstrap test; a non-LLM numeric-claim checker runs next to the judge.** n=30 (15-minute path) or n=140 (`--full`) is small. A 6-point “win” without a p-value is noise. The regex hallucination proxy exists because an all-LLM eval stack cannot fully audit itself.

15. **FAST vs FULL eval, and an honest backend label.** `python eval/run_eval.py` is FAST: 30 test rows, heuristic judge, under 15 minutes, **not** LLM-as-judge. `--full` is 140 rows and uses an LLM judge when a backend answers. If Gemini and Ollama both fail, the “agent” is recorded as `agent_keyword_fallback` — never as an LLM agent. Headline numbers must name the backend.
