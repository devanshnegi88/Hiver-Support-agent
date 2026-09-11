# Decision Log

Plain list of non-obvious calls made while building this, and why.

1. **Chose AmazonHelp as the brand.** Highest tweet volume and most diverse
   issue mix in the dataset — a low-volume or single-issue-type brand would
   make the intent taxonomy and escalation logic trivially easy to nail,
   which defeats the point of the exercise.

2. **10-way intent taxonomy, not more.** Kept small deliberately per the
   assignment's ask. Includes an explicit `other_unclassified` bucket so the
   classifier isn't forced into a false-precision fit for edge cases —
   forcing a fit would inflate apparent accuracy while hiding real failures.

3. **One LLM call per message for intent, not a trained classifier.** No
   labeled training data exists up front (the golden set IS the only labeled
   data, and it's needed for eval, not training). A trained classifier would
   need its own separate labeled set, doubling labeling effort for a
   take-home under time pressure. Tradeoff: higher per-message latency/cost
   than a trained model at inference time — acceptable for a support agent,
   not necessarily for a real-time UI.

4. **TF-IDF retrieval instead of neural embeddings for grounding.** Zero
   external dependency (no embedding API calls, no vector DB), fully
   inspectable (can literally show which words drove a match), and cheap to
   run over the full historical corpus. Tradeoff: misses paraphrase-level
   similarity (different wording, same issue) that embeddings would catch —
   flagged as a "next week" item.

5. **Escalation is rule-based over LLM outputs, not a second free-form LLM
   call.** Escalation is the safety-critical decision in this system. A rule
   layered on (intent, confidence, grounding similarity) is deterministic,
   auditable, and reproducible — you can point to exactly which threshold
   fired. An LLM "should I escalate?" call would be subject to the same
   prompt sensitivity as generation, which is the wrong property for a
   safety gate.

6. **`ALWAYS_ESCALATE_INTENTS` hardcodes complaint_escalation and
   billing_or_charge_dispute to always route to a human**, regardless of
   confidence. These are exactly the cases where a wrong auto-reply is most
   costly (angry customer, money dispute) — the cost asymmetry justifies
   giving up potential auto-handle rate on these categories entirely.

7. **Confidence threshold (0.65) and grounding-similarity threshold (0.25)
   are global constants, not per-intent.** Simpler to reason about and
   defend for a first version; flagged explicitly in REPORT.md's "next week"
   section as something that should be per-intent (some intents are
   inherently harder to classify confidently even when correct).

8. **Reply generation is skipped (not drafted) when no precedent clears the
   grounding threshold**, rather than letting the model draft ungrounded.
   An ungrounded reply is worse than no reply — it looks confident to a
   human reviewer while carrying no evidence behind it.

9. **The grounding prompt explicitly tells the model not to invent specific
   numbers/timelines/policies** not backed by a retrieved precedent, rather
   than relying on general instruction-following. Hallucinated policy
   specifics (a refund amount, a delivery timeline) are the single most
   damaging failure mode for a support agent — worth a dedicated prompt
   instruction rather than hoping general grounding language covers it.

10. **Golden set sampling is stratified by message length, not random.**
    Real support traffic is skewed toward short, simple, high-frequency
    issues. Pure random sampling would produce a golden set — and therefore
    an eval — dominated by the easy cases, overstating how good the system
    looks.

11. **LLM-judge scores 4 separate axes (grounded/relevant/tone/actionable)
    instead of one blended quality score.** A single number can't tell you
    *why* a reply failed, which is exactly what the failure-analysis section
    needs. Decomposing into axes also makes the judge itself easier to
    calibrate against humans one dimension at a time.

12. **Human-agreement calibration is a separate manual two-step script
    (`judge_calibration.py`)**, not automated end-to-end, and explicitly
    instructs labeling before viewing judge scores. This is deliberate —
    an automated pipeline that shows you the judge's score right before you
    label would anchor the human label, making the "agreement" number
    meaningless.

13. **Trivial baseline predicts the single majority-class intent for
    everything and never escalates**, rather than being a random guesser.
    A majority-class baseline is the more honest "floor" — it's what you'd
    get by doing zero work, and it's a stronger baseline than random,
    making any improvement claim more meaningful.

14. **Simple baseline is keyword/regex rules, not a lightweight trained
    model (e.g. logistic regression on TF-IDF).** Chosen specifically
    because it represents what a team would plausibly ship in an afternoon
    with zero ML — the most realistic "why do we need an LLM at all"
    comparison point, which is the actual question this baseline needs to
    answer.

15. **`exclude_id` parameter threads through eval to prevent a golden-set
    example's own historical reply from being retrieved as its own
    precedent.** Without this, evaluating on examples that are themselves
    part of the historical corpus would let the retrieval step "cheat" by
    finding the exact answer, inflating grounding/judge scores in a way
    that wouldn't hold for genuinely new incoming messages.

16. **Golden set is split into dev/test (stratified by intent, 70/30) and
    `run_eval.py` only ever touches test.** Thresholds
    (`MIN_AUTO_HANDLE_CONFIDENCE`, `MIN_GROUNDING_SIMILARITY`) and prompts
    get iterated against dev. Reporting numbers on the same examples used
    to tune those thresholds would make the headline numbers optimistic in
    a way that wouldn't generalize to new traffic — this is the most common
    way take-home eval results end up misleading, so it gets a dedicated
    script (`eval/split_golden_set.py`) rather than being left to
    discipline alone.

17. **All headline metrics are reported with bootstrap 95% confidence
    intervals, and the agent is compared to each baseline with a paired
    bootstrap significance test, not just point-estimate deltas**
    (`eval/stats.py`). A 150-250 example golden set (and a ~30% test slice
    of it) is small enough that a "we beat the baseline by 6 points" claim
    needs a p-value or CI next to it to mean anything — otherwise it may
    just be noise from which specific examples landed in the set.

18. **A rule-based, non-LLM "hallucination-proxy" metric
    (`eval/hallucination_check.py`) runs alongside the LLM judge**,
    flagging numeric claims (dollar amounts, day/percent figures) in a
    reply that don't appear in the customer's message or any retrieved
    precedent. This is a second, independent, fully deterministic signal
    specifically because the LLM judge shares a model family with the
    generator and could be systematically lenient on that family's own
    failure modes — an all-LLM evaluation stack can't fully audit itself.

19. **A dedicated ablation (`eval/ablation.py`) runs every dev-split
    message through reply generation twice — with and without retrieved
    precedents — and compares LLM-judge scores and hallucination-proxy
    rates between the two with a paired significance test.** This turns
    "grounding retrieval improves reply quality" from an assumption baked
    into the architecture into a falsifiable, tested claim with a number
    attached, which is what the report's "results" section needs to be
    convincing rather than just descriptive.

20. **All pure-logic components (escalation rules, TF-IDF retrieval,
    thread reconstruction, keyword baseline, hallucination checker,
    bootstrap stats) have unit tests that require no API key and no
    dataset download** (`tests/`, run via `make test`). Given the
    assignment states graders will ask to "explain and modify your own
    code live," tests that run in under 2 seconds with zero setup are the
    fastest way to demonstrate the logic is actually correct — and, in
    practice, writing `test_paired_bootstrap_no_difference_not_significant`
    caught a real off-by-sign edge case in the significance test's p-value
    calculation (it returned p=0 instead of p=1 for two identical inputs),
    which is exactly the kind of bug that's invisible until you write the
    test for it.

21. **Golden-set sampling stratifies on a keyword-heuristic topic guess in
    addition to message length**, reusing the keyword baseline's regex
    rules purely as a coverage tool (never as the label itself — see
    `LABELING_GUIDE.md`). Length-only stratification would still leave
    low-frequency intents (billing disputes, account access) statistically
    invisible in a 150-250 example set; this guarantees enough examples per
    topic bucket to say something meaningful about each intent, at the
    acknowledged cost of the golden set no longer reflecting real-world
    intent frequency (disclosed explicitly in REPORT.md section 4).

22. **`eval/label_quality_check.py` runs as a mandatory gate between
    labeling and evaluation**, not just a nice-to-have. It's cheap
    (no API calls, pure pandas) and catches exactly the mistakes that are
    expensive to discover after burning eval budget: missing labels,
    invalid intent strings, intents with fewer than 5 examples (too few
    for a meaningful per-class metric), intents with zero examples, and
    cases where a hand-assigned escalation label contradicts the hardcoded
    `ALWAYS_ESCALATE_INTENTS` rule (a signal that either the rule or the
    labeling philosophy needs reconciling, not something to leave silently
    inconsistent).

23. **Near-duplicate and very-short messages are filtered out before
    sampling, not after.** Real support Twitter traffic has a lot of
    boilerplate (retried identical complaints, "thanks!" replies) — leaving
    these in would let one or two template messages consume a
    disproportionate share of a 150-250 example budget for zero labeling
    value.

24. **Pinned `GEN_MODEL` / `JUDGE_MODEL` to `gemini-3.6-flash` after
    `gemini-2.0-flash` returned 404 (model retired).** The Gemini API's
    own error named `models/gemini-3.6-flash` as the replacement; swapping
    only the model id in `src/config.py` (not the `google.generativeai`
    client) was enough — the Interactions API the error also mentioned is
    a recommendation, not a requirement for `generate_content` to work.
    Generator and judge stay on the same model family, so the
    same-family-leniency caveat in REPORT.md section 4 still applies.

25. **Gemini client fails over across a pool of API keys** rather than
    dying the eval when the first key expires or hits quota. Keys are
    read from `GOOGLE_API_KEY` / `GEMINI_API_KEY`, `GOOGLE_API_KEY_2`..`_N`,
    and/or comma-separated `GOOGLE_API_KEYS`. Expired/invalid keys are
    retired for the rest of the process; 429/quota errors rotate to the
    next key but stay in the pool (per-minute limits recover). A 404
    "model no longer available" is *not* treated as a key failure — that
    is a config pin, not a credential problem. Failover lives in
    `src/llm_client.py` so every Gemini call (intent, reply, judge)
    inherits it.

26. **Gemini calls go through the Interactions REST API (`urllib`), not
    `google.generativeai` / `google-genai`.** Both SDKs produced 401
    `ACCESS_TOKEN_TYPE_UNSUPPORTED` by attaching an OAuth Authorization
    header. For `AQ.` auth keys we try Bearer then `x-goog-api-key` and
    cache the winner. If Google still rejects every Gemini key — a known
    AI Studio project bug — `generate()` falls back to xAI, then to a
    local Ollama model sized for 8GB RAM (`llama3.2:3b`, `num_ctx=2048`),
    then to a keyword heuristic. Documented in README.md; this is an
    infrastructure workaround, not a modeling claim.
