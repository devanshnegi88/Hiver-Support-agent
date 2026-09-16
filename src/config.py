"""
Central config for the support agent.

Swapping brands / models should only require editing this file.
Keys are loaded from a project-root `.env` file (see `.env.example`).
"""
import os
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def load_env_file(path: Path | None = None, override: bool = False) -> None:
    """Load KEY=VALUE pairs from `.env` into os.environ.

    Does not override variables already set in the shell (so `$env:GOOGLE_API_KEY`
    still wins). Missing file is a no-op.
    """
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key or not val:
            continue
        if override or key not in os.environ:
            os.environ[key] = val


load_env_file()
RAW_CSV = DATA_DIR / "twcs.csv"                 # the Kaggle twcs.csv you download
THREADS_PARQUET = DATA_DIR / "threads.parquet"  # built by data_prep.py
GOLDEN_SET_CSV = DATA_DIR / "golden_set.csv"
GOLDEN_DEV_CSV = DATA_DIR / "golden_set_dev.csv"    # for threshold tuning / prompt iteration
GOLDEN_TEST_CSV = DATA_DIR / "golden_set_test.csv"  # untouched until final reported numbers
RESULTS_DIR = ROOT / "results"

# Fraction of the golden set held out as dev (used to pick
# MIN_AUTO_HANDLE_CONFIDENCE / MIN_GROUNDING_SIMILARITY below). The
# remainder is test and should only be run once, at the end, for the
# numbers that go in REPORT.md. See DECISION_LOG.md item 16.
DEV_SPLIT_FRACTION = 0.3

# ---- Brand -------------------------------------------------------------
# The Twitter handle in the dataset (author_id column) for the brand we're
# building the agent for. Swap this to target a different brand — everything
# downstream (threads, intents, eval) is derived from this one constant.
BRAND_HANDLE = os.environ.get("BRAND_HANDLE", "AmazonHelp")

# ---- LLM ---------------------------------------------------------------
def load_google_api_keys(env=None) -> list[str]:
    """Single Gemini key: GOOGLE_API_KEY, or GEMINI_API_KEY if that is unset."""
    env = os.environ if env is None else env
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        raw = (env.get(name) or "").strip().strip('"').strip("'").replace("\n", "").replace("\r", "")
        if raw:
            return [raw]
    return []


GOOGLE_API_KEYS = load_google_api_keys()
GOOGLE_API_KEY = GOOGLE_API_KEYS[0] if GOOGLE_API_KEYS else ""
GEN_MODEL = os.environ.get("GEN_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")

# ---- Intent taxonomy -----------------------------------------------------
# Defined by inspecting a sample of real threads for BRAND_HANDLE (see
# notebooks / DECISION_LOG.md for how these were derived). Keep this small —
# the assignment explicitly wants "a small set of intents you define from
# the data", not an exhaustive ontology.
INTENTS = [
    "delivery_delay_or_missing",   # order not arrived / tracking shows stuck
    "order_cancellation_change",   # wants to cancel / modify an order
    "refund_or_return",            # refund status, return request
    "product_defect_or_wrong_item",# item broken, wrong item received
    "account_access_issue",        # login, password, locked account
    "billing_or_charge_dispute",   # double charge, unexpected charge
    "general_product_question",    # pre-purchase / how-to question
    "complaint_escalation",        # angry, threatening to leave, repeat contact
    "positive_feedback",           # thanks / praise, no action needed
    "other_unclassified",          # doesn't fit cleanly — forces honesty
]

# ---- Escalation ------------------------------------------------------------
# Intents that should never be auto-closed regardless of model confidence.
ALWAYS_ESCALATE_INTENTS = {
    "complaint_escalation",
    "billing_or_charge_dispute",
}

# Below this confidence, escalate regardless of intent.
MIN_AUTO_HANDLE_CONFIDENCE = 0.65

# If the retrieval step can't find a historical precedent above this
# similarity, we don't trust the agent to draft a grounded reply.
MIN_GROUNDING_SIMILARITY = 0.25
