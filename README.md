# Hiver SDE Intern Assignment — AI Support Agent for AmazonHelp

An AI support agent built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, targeting the `AmazonHelp` brand. Given an incoming customer tweet,
the agent:

1. Classifies intent into one of 10 brand-derived categories
2. Drafts a reply grounded in how AmazonHelp has historically resolved similar issues (TF-IDF retrieval over historical resolved threads + LLM generation)
3. Decides auto-handle vs. escalate, with a stated reason

See `REPORT.md` for problem framing, results, and failure analysis, and
`DECISION_LOG.md` for the non-obvious calls made along the way.

## Why AmazonHelp

Highest-volume single brand in the dataset, wide variety of issue types
(delivery, refunds, account, billing). A brand with only 1–2 issue types
would make the intent taxonomy trivial and the escalation logic uninteresting.

## LLM backends (automatic)

Every LLM call (intent, reply, judge) goes through `src/llm_client.py`.
It tries, in order:

1. **Gemini** (`GOOGLE_API_KEY` / `GEMINI_API_KEY`) — preferred
2. **xAI Grok** (`XAI_API_KEY`) — if Gemini is missing, blocked, or exhausted
3. **Ollama** local model (`llama3.2:3b`, sized for **8GB RAM**)
4. **Keyword heuristic** — last resort so eval never crashes

You only need **one** of Gemini, xAI, or Ollama to run the agent. Ollama is
the path that works with no cloud credits.

## Assignment deliverables

| Required | Where |
|---|---|
| Runnable pipeline, headline results in **under 15 minutes** | This README, `py eval/run_eval.py` |
| Golden set, 150–250 hand-labeled, sampling note | `data/golden_set.csv` (200 rows); how: `LABELING_GUIDE.md` + Report §1 |
| Eval harness: automated metrics + LLM-as-judge rubric + **human agreement** | `eval/run_eval.py`, `eval/llm_judge.py`; agreement: `results/judge_agreement.json` (n=35, `eval/judge_calibration.py`) |
| Report (≤6 pages) | `REPORT.md` — framing, two baselines, 5 failures, misleading headline, next week |
| Decision log (10–15 bullets) | `DECISION_LOG.md` (15 items) |

---

## Reproduce headline results (under 15 minutes)

Graders: this is the path the assignment asks for. No Kaggle download and
no API key required. Uses the checked-in golden set + `threads.parquet`.

```powershell
cd hiver-support-agent
py -m pip install -r requirements.txt
py -m pytest tests/ -q
py eval/run_eval.py
```

That eval is **30 stratified test examples** and a **heuristic judge**, so it
finishes in well under 15 minutes (typically 1–3 minutes; ~30–60s with
`LLM_PROVIDER=local`). It prints the comparison table and writes
`results/eval_results.json` — those are the headline numbers in `REPORT.md`.

```powershell
py src/pipeline.py "My package still hasn't shown up and it's been 2 weeks, this is ridiculous"
```

Optional, still inside 15 minutes if Ollama is already pulled:

```powershell
ollama pull llama3.2:3b
$env:LLM_PROVIDER = "ollama"
py eval/run_eval.py
```

Judge–human agreement is already computed (`results/judge_agreement.json`).
To regenerate: fill `data/judge_calibration_sample.csv` (already filled)
then `py eval/judge_calibration.py --step compare`.

**Not in the 15-minute budget:** `py eval/run_eval.py --full` (all 140
examples + LLM-as-judge) and `eval/ablation.py`.

---

## How to run the whole project (from scratch)

Commands below use **PowerShell** (`py`). On bash/macOS/Linux, use
`python` instead of `py` and `export VAR=value` instead of
`$env:VAR = "value"`.

### Step 0 — Prerequisites

- Python 3.11+
- ~1 GB free disk for the dataset + Ollama model
- One of:
  - a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey), **or**
  - an xAI key from [console.x.ai](https://console.x.ai) with credits, **or**
  - [Ollama](https://ollama.com) with `llama3.2:3b` (recommended on 8GB RAM)

### Step 1 — Clone and install

```powershell
cd C:\Users\DELL\Downloads\hiver-support-agent\hiver-support-agent
py -m pip install -r requirements.txt
```

Optional virtualenv:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

### Step 2 — Unit tests (no API key, no dataset)

```powershell
py -m pytest tests/ -v
```

70 tests. This is the fastest check that escalation, retrieval, baselines,
stats, hallucination checks, and key-failover logic are correct.

### Step 3 — Choose an LLM

**Option A — Gemini (preferred for reported numbers)**

```powershell
$env:GOOGLE_API_KEY = "your-gemini-api-key"
```

Optional backups (auto-switch if the first key dies):

```powershell
$env:GOOGLE_API_KEY_2 = "backup-key-2"
$env:GOOGLE_API_KEY_3 = "backup-key-3"
```

**Option B — xAI (if Gemini is blocked or out of quota)**

```powershell
$env:XAI_API_KEY = "your-xai-key"
```

**Option C — Ollama on 8GB RAM (no cloud key)**

```powershell
winget install Ollama.Ollama
```

Close and reopen the terminal, then:

```powershell
ollama pull llama3.2:3b
```

Ollama serves `http://127.0.0.1:11434` in the background. Do **not** pull
7B/8B models on 8GB RAM.

Force a backend:

```powershell
$env:LLM_PROVIDER = "ollama"   # skip cloud, use Ollama
$env:LLM_PROVIDER = "local"    # keyword heuristic only (no LLM)
```

`$env:...` lasts for **this PowerShell window only**. Keep using the same
window for later steps. `export` does nothing in PowerShell.

### Step 4 — Get the data

Download `twcs.csv` from Kaggle:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Place it at `data/twcs.csv` (~350MB, not committed).

**Skip this step** if `data/threads.parquet` already exists (it does in
this working copy).

### Step 5 — Build historical precedents

```powershell
py src/data_prep.py
```

Reconstructs (customer message → AmazonHelp reply) pairs and writes
`data/threads.parquet`. Under a minute; no LLM.

**Skip** if `data/threads.parquet` is already present.

### Step 6 — Golden set (label + quality check)

This repo already includes a labeled `data/golden_set.csv` (200 examples).
To rebuild candidates from scratch:

```powershell
py eval/golden_set_builder.py --n 200
```

Hand-label `intent` and `escalate_gold` using `LABELING_GUIDE.md`, save as
`data/golden_set.csv`, then:

```powershell
py eval/label_quality_check.py
```

Must print `No issues found` before you split. **Skip labeling** if you are
reproducing with the checked-in golden set; still run the quality check:

```powershell
py eval/label_quality_check.py
```

### Step 7 — Dev / test split (before touching thresholds)

```powershell
py eval/split_golden_set.py
```

Writes `data/golden_set_dev.csv` (60) and `data/golden_set_test.csv` (140),
stratified by intent. Tune prompts/thresholds on **dev** only. Run test
**once**, at the end, for `REPORT.md`.

**Skip** if those two CSVs already exist and you have not relabeled.

### Step 8 — Smoke-test one message

```powershell
py src/pipeline.py "My package still hasn't shown up and it's been 2 weeks, this is ridiculous"
```

Expect JSON with `intent`, `draft_reply`, `escalate`, `escalation_reason`.

| Log line | Meaning |
|---|---|
| `Gemini auth working: ...` | Cloud Gemini is live |
| `falling back to xAI` | Gemini failed; Grok is live |
| `Using Ollama model llama3.2:3b` | Local 8GB model is live |
| `using keyword heuristic backend` | No LLM available; eval can still run |

First Ollama call can take 30–60s while the model loads.

### Step 9 — (Recommended) Ablation on dev

Does retrieval actually help replies?

```powershell
py eval/ablation.py
```

Dev split only. Uses the LLM (or Ollama). Do not use these numbers as the
headline test results.

### Step 10 — Evaluation (default is under 15 minutes)

```powershell
py eval/run_eval.py
```

Default = **30** stratified test examples + heuristic judge. Writes
`results/eval_results.json`. This is the assignment reproduction command.

```powershell
py eval/run_eval.py --full
```

All **140** test examples + LLM-as-judge. Writes `results/eval_results_full.json`.
Can exceed 15 minutes (Gemini ~20–40 min; Ollama 3B on 8GB CPU often 30–60+ min).

| Command | n | Judge | Typical time |
|---|---|---|---|
| `py eval/run_eval.py` | 30 | heuristic | **<15 min** (often 1–3 min) |
| `py eval/run_eval.py --full` | 140 | LLM | 20–60+ min |
| `$env:LLM_PROVIDER="local"; py eval/run_eval.py` | 30 | heuristic | ~30–60 sec |

Paste the default run's table into `REPORT.md`. If the agent fell back to
keywords because Gemini/xAI/Ollama were unavailable, say so — that is a
real limitation, not a hidden one.

### Step 11 — (Optional) Judge calibration

```powershell
py eval/judge_calibration.py --step sample --n 35
```

Hand-score the `human_*` columns in `data/judge_calibration_sample.csv`,
then:

```powershell
py eval/judge_calibration.py --step compare
```

### Step 12 — Fill the deliverables

- `REPORT.md` — problem, metrics, failure analysis
- `DECISION_LOG.md` — non-obvious calls
- `LABELING_GUIDE.md` — already written; used in step 6

---

## Already-done checklist (this working copy)

You can start at **step 2** then **step 8** if these files exist:

| File | Role |
|---|---|
| `data/threads.parquet` | Precedent index |
| `data/golden_set.csv` | 200 labeled examples |
| `data/golden_set_dev.csv` / `golden_set_test.csv` | 60 / 140 split |
| `results/eval_results.json` | Last eval run (re-run step 10 after a real LLM is up) |

---

## Make shortcuts

```text
make test            py -m pytest tests/ -v
make data            py src/data_prep.py
make golden-candidates
make label-check
make split
make dev-ablation
make eval            py eval/run_eval.py          (<15 min headline run)
make eval-full       py eval/run_eval.py --full
make all             data → split → ablation → eval
```

On Windows without `make`, use the `py ...` commands in the steps above.

---

## Gemini keys (detail)

`eval/run_eval.py` makes hundreds of LLM calls. `src/llm_client.py` fails
over on the **same call** so you do not restart.

```powershell
$env:GOOGLE_API_KEY = "key-1"
$env:GOOGLE_API_KEY_2 = "key-2"
$env:GOOGLE_API_KEYS = "key-1,key-2,key-3"
```

| Error | What happens |
|---|---|
| Expired / invalid key | Retired for this process; next key used immediately |
| 429 / quota | Rotate to next key; keep the old one (limits recover) |
| `API_KEY_SERVICE_BLOCKED` / `ACCESS_TOKEN_TYPE_UNSUPPORTED` | Skip Gemini; try xAI, then Ollama |
| xAI 403 out of credits | Skip xAI; try Ollama |
| Ollama not running | Keyword heuristic so eval still finishes |
| Model 404 | Not a key failure — pin `GEN_MODEL` in `src/config.py` (currently `gemini-3.6-flash`) |

Keys themselves are never logged. Startup prints the *kind* only, e.g.
`Gemini client: 1 key(s) loaded (key 1=auth AQ.).`

A valid Gemini key from [AI Studio](https://aistudio.google.com/apikey)
starts with `AIza` or `AQ.` — not a `gcloud` token (`ya29...`).
`GEMINI_API_KEY` is an alias for `GOOGLE_API_KEY`.

Calls use the Gemini **Interactions** REST API (`x-goog-api-key`). The
legacy `google.generativeai` SDK is not used.

---

## Ollama (8GB RAM) — detail

Default model **`llama3.2:3b`**: ~2GB download, ~3–4GB RAM, `num_ctx=2048`.

```powershell
winget install Ollama.Ollama
ollama pull llama3.2:3b
py src/pipeline.py "My package still hasn't shown up and it's been 2 weeks, this is ridiculous"
```

Success looks like:

```text
Using Ollama model llama3.2:3b (8GB RAM profile, num_ctx=2048).
```

| Env var | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.2:3b` | Keep 1B–3B on 8GB RAM |
| `OLLAMA_NUM_CTX` | `2048` | Lower = less RAM |
| `LLM_PROVIDER` | (auto) | `ollama` or `local` |

---

## Repo structure

```
src/
  config.py       brand, model, intent taxonomy, thresholds
  data_prep.py    raw twcs.csv → (customer, brand_reply) pairs
  llm_client.py   Gemini → xAI → Ollama → heuristic
  intents.py      intent classification
  retrieval.py    TF-IDF precedent search
  reply_gen.py    grounded reply drafting
  escalation.py   auto-handle vs escalate
  pipeline.py     one end-to-end call
  baselines.py    trivial + simple-keyword baselines
eval/
  golden_set_builder.py   stratified sampling for labeling
  label_quality_check.py  post-labeling sanity checks
  split_golden_set.py     dev/test split
  metrics.py / stats.py
  hallucination_check.py
  ablation.py
  llm_judge.py / judge_calibration.py
  run_eval.py             default <15 min headline run; --full for 140+LLM judge
tests/                    70 tests, no API key or dataset (`py -m pytest tests/ -v`)
Makefile
REPORT.md / DECISION_LOG.md / LABELING_GUIDE.md
```

## What this does NOT do

- No multi-turn conversation state (each message scored independently)
- No neural embedding retrieval (TF-IDF only — see ablation / DECISION_LOG.md)
- No fine-tuning
- Not run against the full 3M-row dataset
- The automated hallucination checker is a numeric-claim proxy, not full factuality
