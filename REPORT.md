# Report — AI Support Agent for AmazonHelp

**Reproduce these numbers (under 15 minutes, no API key):**

```text
pip install -r requirements.txt
python eval/run_eval.py
```

Writes `results/eval_results.json`. Default eval is 30 stratified test
examples and a heuristic judge. That is the assignment’s 15-minute path.
`--full` (140 examples + LLM judge) can exceed 15 minutes.

---

## 1. Problem framing

**What “good” means for AmazonHelp.** This brand is high-volume transactional
support: where is my package, refund, Prime, account lock, unexpected charge.
A wrong *auto-sent* reply on billing or an angry repeat contact has real cost
(money, trust, a public tweet). So “good” is **not** maximizing auto-handle
rate. It is:

- Route money-risk and rage tickets to a human (high escalation *recall* on
  those classes).
- Only auto-handle when intent is confident **and** a historical AmazonHelp
  reply is similar enough to ground the draft.
- Prefer a generic “please DM your order number” over an invented refund
  amount or delivery date.

**What I chose not to build.** Multi-turn memory (each tweet is scored
alone). Fine-tuning. Neural embeddings (TF-IDF only). A UI. More than one
brand. A second LLM “should I escalate?” call. Those cuts keep the 15-minute
repro path small; they also leave real failures on the table (section 3).

**Golden set.** 200 hand-labeled AmazonHelp (customer → historical reply)
pairs. Filtered near-duplicates, messages under 15 characters, and probable
non-English. Stratified by length tercile × a keyword *coverage* bucket so
billing and account access are not drowned by delivery tweets. The keyword
bucket is never the label. Rules: `complaint_escalation` needs anger, a
threat, or repeat contact — not just a late package; multi-issue tweets take
the dominant issue. `eval/label_quality_check.py` is a gate before split
(valid labels, ≥5 per intent, flag `ALWAYS_ESCALATE` vs `escalate_gold=False`).
Split: 60 dev / 140 test, stratified by intent. Thresholds were not tuned on
test. **Single annotator** — no second-labeler agreement.

---

## 2. Results vs. two baselines

**Trivial baseline:** always predict majority intent
(`delivery_delay_or_missing`), always send one canned “thanks, please DM
your order number,” **never escalate**. Honest floor: do nothing.

**Simple baseline:** first matching keyword/regex → intent + template
reply. Escalate only for `complaint_escalation` / `billing_or_charge_dispute`,
or if no keyword matched (route to human). What a team could ship in an
afternoon with zero ML.

**Agent:** TF-IDF retrieve top-3 historical AmazonHelp replies (excluding
the example’s own id) → classify intent → rule-based escalate
(`ALWAYS_ESCALATE_INTENTS`, confidence &lt; 0.65, or similarity &lt; 0.25) →
draft only if a precedent exists. LLM path is Gemini → xAI → Ollama 3B →
keywords. On this reproduction the cloud keys failed, so **intent is the
keyword path plus a couple of extra delivery patterns**.

All figures: held-out **n = 30** stratified test rows (15-minute command).
95% bootstrap CIs.

| System | Intent acc [95% CI] | Macro-F1 | Esc. precision | Esc. recall | Hallucination-proxy |
|---|---|---|---|---|---|
| Trivial | 0.10 [0.00, 0.23] | 0.02 | 0.00 | 0.00 | 0.00 |
| Simple keyword | **0.63 [0.47, 0.80]** | 0.60 | 0.56 | 0.31 | 0.00 |
| **Agent** | **0.63 [0.47, 0.80]** | 0.60 | 0.53 | **1.00** | 0.00 |

Heuristic judge scores (grounded / relevant / tone / actionable) are ~4.0
on all three systems. They are **not** LLM-as-judge quality. Do not use
them as a quality claim (section 4).

**Paired bootstrap, same 30 rows:**

- Agent vs trivial, intent acc: **+0.53**, CI [+0.30, +0.73], p = 0.000
  (significant).
- Agent vs simple keyword, intent acc: **+0.00**, CI [+0.00, +0.00], p = 1.000
  (**not significant**).

The agent **does not beat keywords on intent**. It **does** escalate more:
recall 1.00 vs 0.31, because keyword confidence 0.6 sits under the 0.65
auto-handle bar, and gold complaints/billing still hit
`ALWAYS_ESCALATE_INTENTS` when the intent string is right. That is a
conservative router, not a better classifier.

**Ablation / LLM judge / human agreement** are implemented
(`eval/ablation.py`, `judge_calibration.py`) but were **not** run in the
15-minute path: they need a live LLM. Hallucination-proxy 0.00 here means
templates almost never introduce a new `$` or “N days” claim — a lower
bound, not “no hallucinations.”

**Headline:** On the required 15-minute reproduction, trust the agent as a
**conservative escalator** (no missed gold-escalate in this slice). Do not
trust it as a better intent model than regex.

---

## 3. Failure analysis — top 5, with real examples

Examples are from `data/golden_set.csv` (tweet id + gold label). Predictions
are the simple keyword path the agent used in this run.

**1. Keywords miss delay language that is not “hasn’t arrived.”**

- Id `2455242`, gold `delivery_delay_or_missing`, escalate gold False:
  *“You know what I hate when two day shipping turns into two weeks later.”*
  Keyword pred: `other_unclassified`, escalate True.
- Id `1054087`, gold `delivery_delay_or_missing`, escalate gold True:
  *“y’all are killing me. I was on the phone for an hour yesterday. I was
  promised that this item would be here today. HELP!!!!”*
  Keyword pred: `other_unclassified`.

Hypothesis: the regex looks for `late|hasn't arrived|missing|where is my|
still waiting|tracking`. Sarcasm (“two day shipping turns into two weeks”)
and “promised … here today” do not match. Extra local patterns (`shown up`,
`2 weeks`) patch one smoke-test sentence; they do not generalize.

**2. The word “billing” is not a charge dispute.**

- Id `179728`, gold `order_cancellation_change`, escalate False:
  *“How do one change the billing address of an already placed order?”*
  A keyword `billing` rule would call this `billing_or_charge_dispute` and
  the always-escalate list would send it to a human.
- Id `2220052`, gold `other_unclassified`, escalate False:
  *“Been trying an hour to get this to update, I’ve re-added cards,
  re-entered billing address & factory reset STILL stick in a loop!”*
  Keyword pred: `billing_or_charge_dispute`, escalate True.

Hypothesis: coverage sampling and the simple baseline both key off
“billing.” Gold labeling treated disputes as *money already moved*. If
runtime intent is wrong, `ALWAYS_ESCALATE_INTENTS` over-escalates calm
how-tos.

**3. “Worst” / “love” steal the intent from the actual issue.**

- Id `1105604`, gold `product_defect_or_wrong_item`, escalate False:
  *“Worst packing job I've ever seen. 20"x20" photograph shipped in a
  plastic bag… Crumpled to Hell.”*
  Keyword pred: `complaint_escalation` (hit `worst`).
- Id `633572`, gold `positive_feedback`, escalate False:
  *“Love the new detail in ‘s tracking info!”*
  Keyword pred: `delivery_delay_or_missing` (hit `tracking` first).

Hypothesis: first-match regex has no notion of *primary* issue. One angry
adjective or one “love” next to “tracking” flips the class. A real LLM
intent prompt can weigh the whole sentence; keywords cannot.

**4. Complaints that do not use the anger lexicon.**

- Id `2703169`, gold `complaint_escalation`, escalate True:
  *“Hey , who can I talk to to report the delivery guy who just *tossed*
  my packages over a fence onto my porch?”*
  Keyword pred: `other_unclassified` (no `unacceptable|lawyer|furious`…).
  Escalate True only because unmatched keywords default to human — luck,
  not understanding.

Hypothesis: gold `complaint_escalation` is defined by report/threat/repeat
contact, not a word list. “Report the delivery guy” is the class; the
regex never saw `report you` as a substring here in the intended way
(`report you` vs `report the delivery guy`).

**5. TF-IDF matches wording, not the resolution that should be reused.**

- Smoke-test delay tweet retrieved a historical AmazonHelp reply that names
  a carrier and a `^HS` agent tag and asks Amazon vs third-party seller.
  On-brand voice; not necessarily the right next step for *this* order.
- Id `557039`, gold `delivery_delay_or_missing`:
  *“what happens when Amazon keeps losing or leaving packages at the wrong
  address?”*
  Keyword pred: `order_cancellation_change` (`wrong address`). Retrieval
  would mix “wrong address” cancellation threads with lost-package threads.

Hypothesis: bag-of-words similarity does not encode *action* (refund vs
reship vs “DM the order number”). Ungrounded drafts are blocked below
similarity 0.25; above that, a fluent but off-policy precedent can still
leak into the draft.

---

## 4. What is misleading about my headline number?

This section is mandatory. The headline to distrust first is **“agent intent
accuracy 0.63, tied with keywords, escalation recall 1.00.”**

- **The 15-minute number is not an LLM result.** Gemini returned
  `API_KEY_SERVICE_BLOCKED`. xAI returned 403 (credits / spend limit).
  Intent therefore used the keyword fallback. Tying the simple baseline
  (p = 1.0) is *the same system competing with itself*, not evidence that
  retrieval + prompting is “as good as regex.”
- **n = 30, CIs are huge.** Intent acc 0.63 lives in [0.47, 0.80]. A 10-point
  swing is noise. The assignment allows a subsample; it does not make a
  point estimate precise.
- **Escalation recall 1.00 is partly by construction.** Gold
  `complaint_escalation` / `billing_or_charge_dispute` always escalate if
  the predicted intent string is one of those two. Unmatched keywords also
  escalate. Recall 1.00 on this slice means “we did not auto-handle a
  gold-escalate ticket,” not “the model understood risk.” Precision 0.53
  means extra volume to humans.
- **Judge scores of 4.0 are a heuristic, not LLM-as-judge.** The 15-minute
  path does not call the rubric LLM. Templates contain “sorry” and “please
  DM,” so tone/actionable saturate. `judge_calibration.py` (score 35 by
  hand *before* seeing the judge) was not completed because the judge LLM
  was down. Do not quote 4.0 as quality.
- **Hallucination-proxy 0.00 is a regex on `$` and day-counts**, not
  factuality. It misses invented process (“reply to that email”) and
  over-flags numbers the customer already said.
- **Gold is one annotator, stratified, not traffic.** I labeled all 200.
  Ambiguous borderlines (annoyed delay vs complaint; “billing address” vs
  dispute) can move accuracy several points. Stratification *over-represents*
  rare intents relative to real AmazonHelp volume, so the 0.63 is not a
  production mix.
- **Same-family LLM judge (when `--full` is used) is lenient by design
  risk.** Generator and judge were pinned to Gemini 3.6 Flash. Even with
  human calibration, that judge may forgive that family’s failure modes.
  That caveat applies to a future `--full` run, not to these 4.0s.
- **Dev/test split is real, but the default eval is a 30-row slice of
  test, not all 140.** `--full` is the honest LLM-judge number; it is also
  the run that blows the 15-minute budget on 8GB Ollama.

**What I would still stand behind:** the *policy* (never auto-handle
billing disputes or explicit complaints; no draft without a precedent;
no invented dollar amounts in the prompt). Those are design choices. The
**0.63 / 1.00 table is not a claim that the LLM agent is production-ready.**

---

## 5. One more week

- Run `--full` with a working LLM (Ollama `llama3.2:3b` or a live Gemini
  key) and replace heuristic judge scores with LLM-as-judge +
  `judge_calibration.py` exact-match / MAE / within-1.
- Ablation on dev (`eval/ablation.py`): with vs without retrieval, paired
  test on the grounded axis.
- Second labeler on a 50-example overlap; report agreement, not just
  judge-vs-me.
- Per-intent confidence floors instead of one 0.65.
- Embeddings instead of TF-IDF, A/B on paraphrase delays like `2455242`.
