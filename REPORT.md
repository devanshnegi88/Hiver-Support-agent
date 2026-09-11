# Report — AI Support Agent for AmazonHelp

> This is a skeleton with structure and guidance pre-filled. Run `eval/run_eval.py`
> and fill in the bracketed [TODO] sections with your actual numbers before
> submitting — don't submit this with placeholders left in.

## 1. Problem framing

**What "good" means for this brand:** AmazonHelp fields high-volume, mostly
transactional support requests (order status, refunds, account access). For
this brand, "good" is primarily about **not confidently doing the wrong
thing** — a wrong auto-sent reply on a refund/billing issue has real cost
(money, trust), so precision on the escalation decision for those intents
matters more than raw auto-handle rate. "Good" here is *not* maximizing the
percentage of tickets auto-closed.

**How the golden set was sampled and labeled:** Full methodology in
`LABELING_GUIDE.md` — summary: [TODO — fill in your actual N once labeled,
e.g. "212 examples"] sampled from AmazonHelp's reconstructed
(customer message → historical reply) pairs, after filtering near-duplicates,
very short (<15 char) messages, and probable non-English text. Stratified
across message-length tercile × a keyword-heuristic topic guess (used only
to guarantee sampling coverage of rare intents like billing disputes — never
used as the label itself) so the set isn't dominated by the single most
common issue type. Labeled by a single annotator (me) using the decision
rules in `LABELING_GUIDE.md` (in particular: `complaint_escalation` requires
anger/threat/repeat-contact language, not just an unhappy event; multi-issue
messages are labeled by the most prominent issue with the secondary one
noted). `eval/label_quality_check.py` enforces completeness, valid labels,
minimum-5-examples-per-class, and flags any label ↔ escalation-rule
mismatches before the set is split into dev/test. **Caveat:** single-annotator
labeling has no independent check on label quality itself (only the LLM
judge is validated against a human, via `judge_calibration.py`) — see
section 4.

**What I chose not to build:** multi-turn conversation handling (each
message is scored independently, no thread-level memory), fine-tuning
(pure prompting), neural embedding retrieval (TF-IDF instead — see
`DECISION_LOG.md`), a UI/API layer (batch/CLI only), support for more than
one brand at a time.

## 2. Results vs. baselines

All numbers below are from the **held-out test split** (`golden_set_test.csv`)
— thresholds and prompts were finalized against the dev split only, before
this ran. Run `python eval/split_golden_set.py` once, then
`python eval/run_eval.py`, and paste the printed table + `results/eval_results.json`
values into the table below (CIs are 95% bootstrap intervals over
`n_examples` test examples — small-N here, so treat point estimates with
appropriate caution and lean on the CIs and significance tests, not bare
numbers):

| System | Intent Acc [95% CI] | Intent Macro-F1 | Escalation Precision | Escalation Recall | Judge: Grounded [CI] | Judge: Relevant [CI] | Judge: Tone [CI] | Judge: Actionable [CI] | Hallucination-proxy rate |
|---|---|---|---|---|---|---|---|---|---|
| Trivial baseline | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| Simple keyword baseline | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| **Agent (full pipeline)** | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

**Significance vs. baselines (paired bootstrap, same test examples):**
`run_eval.py` prints agent-vs-each-baseline intent-accuracy and judged-relevance
differences with p-values. [TODO — paste, e.g. "agent vs. keyword baseline:
intent-accuracy diff = +0.18, 95% CI [+0.08, +0.28], p=0.001 (significant)."
If a gap you expected to matter is NOT significant, say so — that's more
credible than only reporting the wins.]

**Ablation — does grounding retrieval actually help?** Run
`python eval/ablation.py` (dev split) and paste the with-vs-without-retrieval
paired comparison here. [TODO — e.g. "grounding retrieval improved the
judge's 'grounded' score by +0.9 (95% CI [+0.5, +1.3], p<0.01) and reduced
the hallucination-proxy flagged rate from X% to Y%, confirming the
retrieval step earns its complexity rather than the LLM's general knowledge
doing the work unaided."]

**Judge calibration (human agreement):** Run
`eval/judge_calibration.py --step sample` then `--step compare` and paste
the exact-match / MAE / within-1-point table here per axis. [TODO — note
explicitly if agreement is weak on any one axis (e.g. "tone" is often the
hardest for a judge to calibrate against a human) since that changes how
much weight that axis's headline number deserves.]

**Headline claim:** [TODO — e.g. "The agent significantly improves macro-F1
intent classification over both baselines (see significance test above) and
achieves Z% escalation recall on safety-critical intents
(complaint_escalation, billing_or_charge_dispute) — which is 100% by
construction given the ALWAYS_ESCALATE_INTENTS rule (see DECISION_LOG.md
item 6); the real test of the agent's own judgment is whether the *intent
classifier* correctly routes those messages into those buckets in the
first place, which is what per-class recall on those two intents from the
confusion matrix shows: [TODO from results/eval_results.json confusion_matrix]."]

## 3. Failure analysis — top 5 failure modes

For each: 1-2 real examples from the golden set (message + what the agent
did + what should have happened), plus your hypothesis for why.

1. **[TODO — e.g. Sarcasm/negation misread as positive_feedback]**
   - Example: [TODO]
   - Hypothesis: [TODO]
2. **[TODO — e.g. TF-IDF retrieval surfaces topically-similar but
   resolution-irrelevant precedents (same keywords, different actual issue)]**
   - Example: [TODO]
   - Hypothesis: [TODO]
3. **[TODO — e.g. Multi-issue messages (delivery delay AND wrong item)
   get force-classified into one intent, reply only addresses one]**
4. **[TODO — e.g. Escalation threshold miscalibrated for one intent —
   check per-intent confidence distributions in results/eval_results.json]**
5. **[TODO — e.g. Judge over-scores generic-but-safe replies as
   "actionable" when they're actually just deflecting to DM]**

## 4. What is misleading about my headline number?

*(Mandatory section — be genuinely self-critical here, this is what
distinguishes a strong submission.)*

- **Golden set is hand-picked by me, not independently audited** — labeling
  ambiguous cases (e.g. is a slightly annoyed delivery question
  `complaint_escalation` or `delivery_delay_or_missing`?) involved judgment
  calls that could shift accuracy numbers by several points either way.
- **Escalation "precision/recall" is partly circular** — the
  `ALWAYS_ESCALATE_INTENTS` rule guarantees 100% recall on those specific
  intents by construction; that's a design choice, not evidence the model
  reasoned well about risk.
- **LLM-as-judge and the generator share a model family (Gemini)** — even
  with human-agreement calibration on a subset, a judge from the same model
  family as the generator may be systematically lenient on the failure
  modes that model family tends to produce.
- **Subsampled dataset** — results are on a subsample per the assignment's
  instructions, not the full 3M-row corpus; precedent coverage (and
  therefore grounding quality) would likely improve with more historical
  data, so current numbers may understate real-world grounding quality.
- **The test split is small (~30% of a 150-250 example golden set, i.e.
  roughly 45-75 examples)** — the bootstrap CIs in section 2 are wide for a
  reason; a 5-10 point swing in intent accuracy on this size of test set is
  well within noise unless the significance test explicitly says otherwise.
  Don't read a single-run point estimate as more precise than the CI says
  it is.
- **The hallucination-proxy metric is a numeric-claim regex, not a real
  factuality checker** (see `eval/hallucination_check.py` docstring) — it
  over-flags legitimate numbers the model correctly copied from the
  customer's own message, and it can't catch non-numeric hallucinations
  (invented policies, wrong process steps) at all. Treat its rate as a
  lower bound on real hallucination, not a complete count.
- **Single-annotator golden set, no inter-annotator agreement measure** — I
  labeled all examples myself. The judge-calibration check
  (`judge_calibration.py`) validates the LLM judge against my scores, but
  nothing validates my gold *labels* against an independent second opinion
  — see `LABELING_GUIDE.md`'s caveat and the "next week" item below.

## 5. What I'd do next with one more week

- [TODO — e.g. swap TF-IDF for a sentence-embedding retrieval index and
  A/B the grounding-quality delta]
- [TODO — e.g. multi-issue message handling — detect and split compound
  tickets before classification]
- [TODO — e.g. expand golden set to 400+ with a second independent labeler
  and report inter-annotator agreement, not just judge-vs-you agreement]
- [TODO — e.g. per-intent escalation thresholds instead of one global
  confidence cutoff]
