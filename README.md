# Hiver SDE Intern Assignment — AI Support Agent for AmazonHelp

An AI support agent built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, targeting the `AmazonHelp` brand. Given an incoming customer tweet,
the agent:

1. Classifies intent into one of 10 brand-derived categories
2. Drafts a reply grounded in how AmazonHelp has historically resolved similar issues (TF-IDF retrieval over ~historical resolved threads + LLM generation)
3. Decides auto-handle vs. escalate, with a stated reason

See `REPORT.md` for problem framing, results, and failure analysis, and
`DECISION_LOG.md` for the non-obvious calls made along the way.

## Why AmazonHelp

Highest-volume single brand in the dataset, wide variety of issue types
(delivery, refunds, account, billing) — a brand with only 1-2 issue types
would make the intent taxonomy trivial and the escalation logic uninteresting.

## Setup (should take ~5 min)

```bash
git clone <this-repo>
cd hiver-support-agent
pip install -r requirements.txt
export GOOGLE_API_KEY="your-gemini-api-key"
```

See **Gemini API keys (automatic failover)** below for backups and PowerShell.

### 0. Run the test suite first (no API key, no dataset needed)

```bash
make test
# or: python -m pytest tests/ -v
```

68 unit tests cover the pure-logic pieces — escalation rules, TF-IDF
retrieval (including the self-match leakage guard used during eval),
data cleaning/thread reconstruction, the keyword baseline, the automated
hallucination checker, the bootstrap statistics helpers, and Gemini API-key
failover. None of them call an LLM or touch the raw dataset, so this is the
fastest way to confirm the logic is correct before spending time on API keys
or a 350MB download — and it's what I'd point to first in a live code
walkthrough.

### 1. Get the data

Download `twcs.csv` from Kaggle:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Place it at `data/twcs.csv`. (Not committed to the repo — it's ~350MB.)

### 2. Build the brand's historical precedent set

```bash
python src/data_prep.py
```

This reconstructs (customer message → AmazonHelp's reply) pairs from the raw
tweet table and writes `data/threads.parquet`. Runs in under a minute even on
the full 3M-row file (it's a groupby/join, not an LLM call).

### 3. Build + label the golden evaluation set

```bash
python eval/golden_set_builder.py --n 200
```

Writes `data/golden_set_candidates.csv` — pre-filtered (near-duplicates,
very short messages, probable non-English text removed) and stratified
across message-length × a keyword-heuristic topic guess so rare intents
(billing disputes, account access) are actually represented rather than
drowned out by delivery complaints. **See `LABELING_GUIDE.md` for the full
sampling methodology and the exact decision rules to use while labeling**
(what makes something `complaint_escalation` vs. just an unhappy customer,
how to handle multi-issue messages, escalation criteria, etc.) — hand-label
the `intent` and `escalate_gold` columns per that guide, save as
`data/golden_set.csv`.

Then sanity-check the labels before spending eval budget on them:

```bash
python eval/label_quality_check.py
```

Checks the set is 150-250 rows, every label is valid and non-empty, every
intent has at least 5 examples (below that, per-class metrics are close to
meaningless), no intent has zero examples, and flags any case where a
label's assigned intent is in `ALWAYS_ESCALATE_INTENTS` but you marked
`escalate_gold=False` (worth a second look — not necessarily wrong, but
worth reconciling explicitly). **A pre-labeled `data/golden_set.csv` is
included in this repo** so graders can reproduce results without
relabeling from scratch.

### 4. Split into dev / test — do this before touching thresholds

```bash
python eval/split_golden_set.py
```

Writes `data/golden_set_dev.csv` and `data/golden_set_test.csv`, stratified
by intent (70/30 split by default — `DEV_SPLIT_FRACTION` in `src/config.py`).
**Use dev while iterating on prompts/thresholds
(`MIN_AUTO_HANDLE_CONFIDENCE`, `MIN_GROUNDING_SIMILARITY` in
`src/config.py`); only run against test once, at the end, for the numbers
in `REPORT.md`.** Reporting numbers on the same data you tuned thresholds
against is the single most common way take-home eval results end up
misleading — this split exists specifically to avoid that.

### 5. Run the agent on a single message (sanity check)

```bash
python src/pipeline.py "My package still hasn't shown up and it's been 2 weeks, this is ridiculous"
```

### 6. (Recommended) Ablation — does grounding retrieval actually help?

```bash
python eval/ablation.py
```

Runs the same dev-split messages through reply generation twice — once with
real retrieved precedents, once with retrieval disabled — and reports a
paired bootstrap significance test on LLM-judge scores per axis, plus the
automated hallucination-proxy rate in both conditions. This is the evidence
behind the retrieval-grounding design choice, not just an architecture
diagram asserting it helps.

### 7. Run the full evaluation (headline results)

```bash
python eval/run_eval.py
```

Runs the agent + both baselines over the **test** split only, scores intent
accuracy/F1 with bootstrap 95% confidence intervals, escalation
precision/recall, LLM-judged reply quality (with CIs), the automated
non-LLM hallucination-proxy rate, and a paired bootstrap significance test
of the agent vs. each baseline. Prints a comparison table and writes
`results/eval_results.json`. **This is the ~15-minute step** — budget for
it based on test-split size × Gemini latency (~1-2s/call, 3 calls per
example per system ≈ 10-15 min at n≈60 test examples out of a 200-example
golden set at default rate limits; reduce `--n` in step 3 to run faster).

### 8. (Optional) Judge calibration — human agreement check

```bash
python eval/judge_calibration.py --step sample --n 35
# hand-score the human_* columns in data/judge_calibration_sample.csv
python eval/judge_calibration.py --step compare
```

Steps 4, 6, 7, 8 are also available as `make split`, `make dev-ablation`,
`make eval`, `make calibrate-sample` / `make calibrate-compare`, or
`make all` to chain data → split → ablation → eval end to end.

## Gemini API keys (automatic failover)

`eval/run_eval.py` makes hundreds of Gemini calls (intent + reply + judge
across 140 test examples × 3 systems). One expired or quota-exhausted key
would otherwise kill the run mid-way. `src/llm_client.py` keeps a key pool
and fails over on the **same call** so you do not have to restart.

**Primary + numbered backups** (first-listed is tried first):

```bash
export GOOGLE_API_KEY="key-1"
export GOOGLE_API_KEY_2="key-2"
export GOOGLE_API_KEY_3="key-3"
```

PowerShell (this session only — `export` does nothing here):

```powershell
$env:GOOGLE_API_KEY = "key-1"
$env:GOOGLE_API_KEY_2 = "key-2"
$env:GOOGLE_API_KEY_3 = "key-3"
```

**Or one comma-separated list:**

```bash
export GOOGLE_API_KEYS="key-1,key-2,key-3"
```

```powershell
$env:GOOGLE_API_KEYS = "key-1,key-2,key-3"
```

Duplicates are dropped; order is preserved. Intent classification, reply
drafting, and the LLM judge all go through this client, so failover is
inherited everywhere.

| Error | What happens |
|---|---|
| Expired / invalid API key (`API_KEY_INVALID`, "API key expired") | That key is **retired** for the rest of the process; the next live key is used immediately |
| 429 / quota exhausted | Rotates to the next key but **keeps** the old one in the pool (per-minute limits recover) |
| Model 404 (retired model id) | **Not** treated as a key failure — fix `GEN_MODEL` / `JUDGE_MODEL` in `src/config.py` |

On failover the process prints (keys themselves are never logged):

```text
Gemini API key 1/3 failed (expired/invalid); switching to key 2/3.
```

If every cloud key is dead, the client falls through to **Ollama**, then a
keyword heuristic. Get Gemini keys from
[Google AI Studio](https://aistudio.google.com/apikey). Current model pin is
`gemini-3.6-flash` (`gemini-2.0-flash` was retired).

Calls go to the Gemini **Interactions** REST API with only the
`x-goog-api-key` header (no Gemini Python SDK). The `google.generativeai`
and `google-genai` clients send an OAuth `Authorization` header that
Gemini 3.x rejects with `401 ACCESS_TOKEN_TYPE_UNSUPPORTED`.

A valid key from [Google AI Studio](https://aistudio.google.com/apikey)
starts with `AIza` (standard) or `AQ.` (auth; this is what new keys are).
Not a `gcloud` OAuth token (`ya29...`) and not a key from another provider.
If you have an older **Unrestricted** `AIza` key, restrict it to Gemini API
only in AI Studio — unrestricted standard keys are now rejected.
`GEMINI_API_KEY` is accepted as an alias for `GOOGLE_API_KEY`.

On startup the client prints the key *kind* (prefix only), e.g.
`Gemini client: 1 key(s) loaded (key 1=auth AQ.).` so you can confirm the
env var is the right type of secret without leaking it.

**If Google still returns `ACCESS_TOKEN_TYPE_UNSUPPORTED` for an `AQ.` key:**
that is a known Gemini API issue on some AI Studio projects (the same 401
happens with raw curl). The client will try `Authorization: Bearer` then `x-goog-api-key` before
giving up. To still run eval, set an
xAI key and the client falls back automatically:

```powershell
$env:XAI_API_KEY = "your-xai-key"   # https://console.x.ai
```

## Local Ollama (8GB RAM)

When Gemini/xAI keys are missing, blocked, or out of credits, the client
calls a local [Ollama](https://ollama.com) model. Default is **`llama3.2:3b`**
(~2GB download, ~3–4GB RAM) so it fits an 8GB machine. Do not pull 7B/8B
weights on 8GB RAM — they will swap.

```powershell
winget install Ollama.Ollama
ollama pull llama3.2:3b
# Ollama usually starts its own background service on http://127.0.0.1:11434
```

Then:

```powershell
py src/pipeline.py "My package still hasn't shown up and it's been 2 weeks, this is ridiculous"
```

You should see `Using Ollama model llama3.2:3b (8GB RAM profile, num_ctx=2048).`

| Env var | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model tag (keep 1B–3B on 8GB RAM) |
| `OLLAMA_NUM_CTX` | `2048` | Context length (lower = less RAM) |
| `LLM_PROVIDER` | (auto) | Force `ollama` or `local` (keyword heuristic) |

If Ollama is not running, the client prints `ollama pull llama3.2:3b` and
falls back to the keyword heuristic so eval still completes.

## Repo structure

```
src/
  config.py       brand, model, intent taxonomy, thresholds — the one file to edit to retarget
  data_prep.py    raw twcs.csv -> reconstructed (customer, brand_reply) pairs
  llm_client.py   Gemini Interactions REST + retry + key failover
  intents.py      LLM-based intent classification
  retrieval.py    TF-IDF precedent search (the "grounding" step)
  reply_gen.py    grounded reply drafting
  escalation.py   auto-handle vs escalate decision logic
  pipeline.py     wires the above into one end-to-end call
  baselines.py    trivial + simple-keyword baselines
eval/
  golden_set_builder.py   filtered + coverage-stratified sampling for hand-labeling
  label_quality_check.py  post-labeling sanity checks (balance, validity, consistency)
  split_golden_set.py     dev/test split (stratified by intent)
  metrics.py              intent/escalation automated metrics
  stats.py                bootstrap CIs + paired significance testing
  hallucination_check.py  automated (non-LLM) unsupported-claim detector
  ablation.py              with-retrieval vs without-retrieval comparison
  llm_judge.py            LLM-as-judge rubric + human-agreement scoring
  judge_calibration.py    workflow script for the human-agreement check
  run_eval.py             runs everything on test split, produces headline numbers
tests/
  test_escalation.py, test_baselines.py, test_data_prep.py,
  test_retrieval.py, test_hallucination_check.py, test_stats.py,
  test_llm_client.py
  — 68 tests, no API key or dataset required (`make test`)
Makefile            one command per pipeline stage
REPORT.md           problem framing, results, failure analysis (assignment deliverable)
DECISION_LOG.md     non-obvious decisions and why (assignment deliverable)
LABELING_GUIDE.md   golden-set sampling methodology + labeling decision rules (assignment deliverable)
```

## What this does NOT do (scope cuts — see REPORT.md for the full "what I chose not to build")

- No multi-turn conversation state (each message scored independently)
- No neural embedding retrieval (TF-IDF only — justified with an ablation, see DECISION_LOG.md)
- No fine-tuning — pure prompting against Gemini
- Not run against the full 3M-row dataset — subsampled per assignment instructions
- The automated hallucination checker is a numeric-claim proxy, not a full factuality checker — see its docstring for what it does and doesn't catch
#   H i v e r - S u p p o r t - a g e n t  
 