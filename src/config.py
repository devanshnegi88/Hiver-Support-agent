"""
Central config for the support agent.

Swapping brands / models should only require editing this file.
"""
import os
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
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
    """Collect Gemini keys from the environment, first-listed first.

    Accepted (any combination; duplicates dropped, order preserved):
      GOOGLE_API_KEY          first/primary key
      GEMINI_API_KEY          alias used by the current google-genai SDK
      GOOGLE_API_KEY_2 .. _N  backups (also accepts _1)
      GOOGLE_API_KEYS         comma- or semicolon-separated list
    """
    env = os.environ if env is None else env
    keys: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        if not raw:
            return
        for part in str(raw).replace(";", ",").split(","):
            k = part.strip().strip('"').strip("'").replace("\n", "").replace("\r", "")
            if k and k not in seen:
                seen.add(k)
                keys.append(k)

    add(env.get("GOOGLE_API_KEY", ""))
    add(env.get("GEMINI_API_KEY", ""))
    add(env.get("GOOGLE_API_KEYS", ""))
    consecutive_empty = 0
    for i in range(1, 21):
        v = env.get(f"GOOGLE_API_KEY_{i}", "")
        if v:
            add(v)
            consecutive_empty = 0
        else:
            consecutive_empty += 1
            if i >= 2 and consecutive_empty >= 3:
                break
    return keys


GOOGLE_API_KEYS = load_google_api_keys()
GOOGLE_API_KEY = GOOGLE_API_KEYS[0] if GOOGLE_API_KEYS else ""
GEN_MODEL = "gemini-3.6-flash"      # classification + reply drafting
# Used with the Interactions REST API (see src/llm_client.py).
JUDGE_MODEL = "gemini-3.6-flash"    # LLM-as-judge (kept separate constant so
                                     # you can deliberately use a *different*
                                     # model than the generator to reduce
                                     # self-preference bias if you want)
                                     # 2.0-flash was retired by the Gemini API
                                     # (404); 3.6-flash is the replacement
                                     # the API itself instructed us to use.

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
