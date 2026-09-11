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

## FAST evaluation (under 15 minutes) — smoke / reproducibility

This is **not** LLM-as-judge quality evaluation. It proves the pipeline
runs, leakage checks pass, and automated intent/escalation metrics exist.

No Kaggle download. No API key. Uses checked-in `data/threads.parquet`.

```bash
git clone https://github.com/devanshnegi88/Hiver-Support-agent.git
cd Hiver-Support-agent
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python eval/leakage_check.py
python eval/run_eval.py
```

Writes `results/quick_results.json`. Judge type is **heuristic**. If no LLM
is up, the agent row is named `agent_keyword_fallback`.

Timed locally at ~30 seconds with `LLM_PROVIDER=local`.

---

## Run with Ollama (no cloud API key, 8GB RAM)

Use this when Gemini/xAI keys are missing, blocked, or out of credits.
Model: **`llama3.2:3b`** (~2GB download, ~3–4GB RAM). Do **not** pull 7B/8B
models on 8GB RAM.

### 1. Install Ollama (Windows)

```powershell
winget install Ollama.Ollama
```

Close **all** PowerShell/terminal windows, then open a new one.

macOS: `brew install ollama`  
Linux: see https://ollama.com/download

### 2. Start Ollama and pull the 3B model

```powershell
ollama serve
```

Leave that window open (or let the Windows service run). In a **second**
terminal:

```powershell
ollama pull llama3.2:3b
ollama list
```

You should see `llama3.2:3b`. Probe the server:

```powershell
curl http://127.0.0.1:11434/api/tags
```

If that fails, Ollama is not running — go back to `ollama serve`.

### 3. Smoke-test the agent on one message

```powershell
cd path\to\Hiver-Support-agent
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "llama3.2:3b"
py src/pipeline.py "My package still hasn't shown up and it's been 2 weeks, this is ridiculous"
```

Linux/mac:

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2:3b
python src/pipeline.py "My package still hasn't shown up and it's been 2 weeks, this is ridiculous"
```

Success looks like:

```text
Using Ollama model llama3.2:3b (8GB RAM profile, num_ctx=2048).
```

then JSON with `intent`, `draft_reply`, `escalate`, `escalation_reason`.
The first call can take 30–60s while the model loads.

If you see `agent_keyword_fallback` or `using keyword heuristic backend`,
Ollama was not reached — check `ollama serve` and `ollama list`.

### 4. FAST eval with Ollama (still under 15 minutes if already pulled)

```powershell
$env:LLM_PROVIDER = "ollama"
py eval/run_eval.py
```

Check `results/quick_results.json` → `"agent_backend": "ollama"`.
If it says `"heuristic"`, Ollama did not answer.

### 5. FULL eval with Ollama (quality; 20–60+ min on CPU)

```powershell
$env:LLM_PROVIDER = "ollama"
py eval/run_eval.py --full
```

Writes `results/full_results.json`. Expect `"agent_backend": "ollama"` and
`"judge_type": "llm"`.

| Env var | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | (auto) | Set to `ollama` to skip cloud APIs |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.2:3b` | Keep 1B–3B on 8GB RAM |
| `OLLAMA_NUM_CTX` | `2048` | Lower = less RAM |

---

## Run with Gemini or xAI API keys

Use this when you have a working cloud key. `$env:...` lasts **only in this
PowerShell window**. `export` does nothing in PowerShell.

### Gemini (preferred)

1. Create a key at https://aistudio.google.com/apikey  
   Valid keys start with `AIza` or `AQ.` — not a `gcloud` token (`ya29`).
2. In the **same** terminal you will run eval:

```powershell
cd path\to\Hiver-Support-agent
$env:GOOGLE_API_KEY = "paste-your-gemini-key-here"
# optional backups if the first key dies:
$env:GOOGLE_API_KEY_2 = "backup-key-2"
```

Linux/mac: `export GOOGLE_API_KEY="paste-your-gemini-key-here"`

3. Smoke-test:

```powershell
py src/pipeline.py "My package still hasn't shown up and it's been 2 weeks, this is ridiculous"
```

You want a log line like `Gemini auth working` or `Gemini client: 1 key(s) loaded`.
If you see `API_KEY_SERVICE_BLOCKED` or `ACCESS_TOKEN_TYPE_UNSUPPORTED`,
that key cannot call Gemini — use Ollama or xAI instead.

4. FAST then FULL:

```powershell
py eval/run_eval.py
py eval/run_eval.py --full
```

### xAI / Grok (if Gemini is blocked)

1. Create a key at https://console.x.ai (needs credits).
2.

```powershell
$env:XAI_API_KEY = "paste-your-xai-key-here"
py src/pipeline.py "My package still hasn't shown up and it's been 2 weeks, this is ridiculous"
py eval/run_eval.py --full
```

Linux/mac: `export XAI_API_KEY="paste-your-xai-key-here"`

If xAI returns 403 out of credits, the client falls through to Ollama, then
keywords. Check `"agent_backend"` in the results JSON — do not report
keyword fallback as an LLM agent.

### Auto order (if you do not set `LLM_PROVIDER`)

1. Gemini, if `GOOGLE_API_KEY` / `GEMINI_API_KEY` is set  
2. xAI, if `XAI_API_KEY` is set  
3. Ollama on `127.0.0.1:11434`  
4. Keyword heuristic (labeled `agent_keyword_fallback`)

---

Judge–human agreement (already computed): `results/judge_agreement.json`.
Regenerate after a real LLM run:

```powershell
py eval/judge_calibration.py --step compare --llm-judge
```

(`--llm-judge` needs a live Gemini/xAI/Ollama backend.)

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

78 tests. This is the fastest check that escalation, retrieval, baselines,
stats, hallucination checks, and key-failover logic are correct.

### Step 3 — Choose an LLM

Follow one of the full write-ups above:

- **Ollama:** “Run with Ollama (no cloud API key, 8GB RAM)”
- **Gemini / xAI:** “Run with Gemini or xAI API keys”

Keep using the **same** terminal after you set `$env:...` or `export`.

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

## LLM troubleshooting

Install and run steps: **Run with Ollama** and **Run with Gemini or xAI API keys** above.

| Error | What happens |
|---|---|
| Expired / invalid Gemini key | Retired; next `GOOGLE_API_KEY_2`… is used |
| 429 / quota | Rotate keys; old key stays in the pool |
| `API_KEY_SERVICE_BLOCKED` / `ACCESS_TOKEN_TYPE_UNSUPPORTED` | Skip Gemini; try xAI, then Ollama |
| xAI 403 out of credits | Skip xAI; try Ollama |
| Ollama not running | Keyword heuristic; results JSON says `agent_keyword_fallback` |
| Model 404 | Pin `GEN_MODEL` in `src/config.py` (`gemini-3.6-flash`) |

Always read `"agent_backend"` and `"judge_type"` in the results JSON before quoting numbers.

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
