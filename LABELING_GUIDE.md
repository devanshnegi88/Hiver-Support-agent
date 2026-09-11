# Golden Set — Sampling & Labeling Guide

**Who labeled:** one annotator (the author), using this guide. There is **no
independent second labeler**. Do not treat these 200 rows as dual-annotated
ground truth. Labels were assigned by reading each customer message (and
notes where needed), not by copying the keyword sampler's guess.

This is the "short note on how you sampled and labeled them" the assignment
asks for, expanded enough to actually defend the choices live.

## Sampling

Source: `data/threads.parquet`, built by `src/data_prep.py` from the raw
Kaggle `twcs.csv`, filtered to `AmazonHelp` (customer message → AmazonHelp's
actual reply, reconstructed by following `in_response_to_tweet_id`).

Before sampling, `eval/golden_set_builder.py` removes:
- Messages under 15 characters (usually too low-content to label meaningfully:
  "ok thanks", "lol")
- Probable non-English messages (crude non-ASCII-ratio heuristic — the intent
  taxonomy and this guide are English-only)
- Near-duplicate messages (same complaint retried, copy-pasted templates —
  collapsed by a lowercased, punctuation-stripped fingerprint so the golden
  set isn't padded with redundant near-copies)

The remaining pool is stratified on two axes before sampling `--n` (default
200) examples:

1. **Message-length tercile** (short/medium/long) — a proxy for complexity;
   short messages skew toward simple templatable issues, long ones toward
   multi-issue or emotionally-charged threads.
2. **A cheap keyword-heuristic intent guess** (the same regex rules as
   `src/baselines.py`'s keyword baseline, reused here purely as a sampling
   tool) — a proxy for topic, so rare-but-important intents (billing
   disputes, account access) get guaranteed representation instead of being
   drowned out by the dominant delivery-complaint traffic.

**Important:** the keyword heuristic decides *which real messages get shown
to you for labeling*. It never supplies or suggests the label itself — you
read each message and assign intent/escalation from the labeling rules
below, independent of what the heuristic guessed. This means:
- The golden set's intent distribution is **stratified, not proportional to
  real-world frequency** — a deliberate choice so rare intents are
  statistically visible at all, disclosed in `REPORT.md`'s "what's
  misleading about my headline number" section (headline accuracy on this
  set is not an estimate of accuracy on true incoming traffic).
- The keyword baseline being evaluated later isn't unfairly advantaged by
  this: gold labels are assigned independent of the heuristic's guess, and
  every system (agent, trivial, keyword) is scored on the exact same
  sampled set.

## Labeling process

Single annotator (me), one pass per example, in the order the CSV was
written (not sorted by anything that could bias attention/fatigue toward
one label). **Caveat this deserves in `REPORT.md`:** single-annotator
labeling has no independent measure of *label* quality — the human-agreement
check in `eval/judge_calibration.py` validates the LLM *judge* against my
scores, not my gold *labels* against a second labeler's. If I had another
week, a second annotator on a 30-40 example overlap subset with a
Cohen's-kappa readout would close that gap (see `REPORT.md` → "what I'd do
next").

Look at `customer_message` and assign:
- `intent` — exactly one label from `src/config.py`'s `INTENTS`
- `escalate_gold` — `True`/`False`: would a competent human support agent
  need to review this before anything is sent, based on the message alone?
- `notes` — anything ambiguous, borderline, or multi-issue (free text, not
  scored, but feeds the failure-analysis section)

`brand_reply` is shown for context but should NOT anchor the intent label —
label what the *customer* is asking, not what category the historical reply
happens to fall into (the brand's own reply is sometimes generic/templated
regardless of the specific issue).

## Intent decision rules (resolves the ambiguous cases)

- **`complaint_escalation` vs. a specific issue intent:** use
  `complaint_escalation` ONLY when the message itself expresses anger, a
  threat to leave/report/get a refund via a bank dispute, or explicitly
  references a repeat/unresolved prior contact. A negative *event*
  described calmly ("my order is late") is the specific issue intent
  (`delivery_delay_or_missing`), not `complaint_escalation`, even though
  it's an unhappy customer — anger/threat/repeat-contact is the signal, not
  sentiment alone.
- **Multi-issue messages:** label the intent of the issue mentioned FIRST /
  most prominently, and note the secondary issue in `notes`. This is a
  scope cut (see `REPORT.md` "what I chose not to build" — no compound-intent
  handling) — flag it, don't silently pick one and hide the ambiguity.
- **Pre-purchase questions with no order yet:** `general_product_question`,
  even if phrased urgently — no account/order is at risk, so it's
  categorically different from a post-purchase problem regardless of tone.
- **Ambiguous refund vs. cancellation** ("I don't want this anymore"):
  `order_cancellation_change` if the order hasn't shipped/arrived yet per
  the message; `refund_or_return` if it has already arrived. If genuinely
  unclear from the message alone, use `refund_or_return` (the more common
  and typically correct resolution path) and note the ambiguity.
- **`billing_or_charge_dispute` vs. messages that merely mention billing:**
  use this intent only when money has already moved (or is alleged to have)
  in a way the customer is contesting — double charge, unexpected charge,
  unauthorized charge. A how-to about changing a billing address, a
  hypothetical "how do I contact billing support?", or a mid-thread
  confirmation that someone asked for billing details is **not** this
  intent, even if the sampling bucket was `*__billing_or_charge_dispute`
  (the keyword heuristic is a coverage tool, not the label).
- **Genuinely doesn't fit:** use `other_unclassified` rather than forcing a
  fit. A classifier (or a human labeler) that never uses this bucket is
  probably overfitting labels to the taxonomy rather than to the message.

## Escalation decision rules

`escalate_gold=True` when ANY of:
- The message expresses anger, a threat, or repeat/unresolved contact
  (i.e., `complaint_escalation` or a message that reads that way regardless
  of its assigned intent)
- Any billing/charge dispute (money already moved, not just a future
  refund request)
- The message is ambiguous enough that a reasonable reply requires
  information not present in the message (e.g., references a prior DM
  exchange we can't see)
- Legal, safety, or health-and-safety language of any kind

Otherwise `False`. This mirrors — but is independently applied to, not
copied from — the `ALWAYS_ESCALATE_INTENTS` rule in `src/config.py`;
`eval/label_quality_check.py` flags cases where they disagree so a
systematic mismatch gets caught and reconciled rather than left implicit.

## After labeling

Run `python eval/label_quality_check.py` — it checks the set is 150-250
rows, every row has a valid intent from the taxonomy and a valid
True/False escalation label, every intent has at least 5 examples (below
that, per-class metrics are close to meaningless), no intent from the
taxonomy has zero examples, and flags any escalation-rule mismatches
described above. Fix anything it flags before running
`eval/split_golden_set.py`.
