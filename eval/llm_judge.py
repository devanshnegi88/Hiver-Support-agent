"""
LLM-as-judge for reply quality, plus the human-agreement check the
assignment explicitly requires ("evidence of how well your judge agrees
with a human").

Rubric is scored 1-5 on four axes rather than one blended score — a single
"quality" number hides *why* something scored low, which you need for the
failure analysis section of the report.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import JUDGE_MODEL  # noqa: E402
from llm_client import generate_json  # noqa: E402


_JUDGE_PROMPT = """You are an expert customer support QA reviewer. Score this draft reply on 4 axes, 1-5 each (5 = best):

- grounded: Does it avoid inventing specific policy details (amounts, timelines, promises) not supported by the given precedents?
- relevant: Does it actually address what the customer asked?
- tone: Is it appropriately empathetic/professional for the brand?
- actionable: Does it give the customer a clear next step?

Customer message:
\"\"\"{message}\"\"\"

Historical precedents the reply was supposed to be grounded in:
{precedents}

Draft reply:
\"\"\"{reply}\"\"\"

Respond with ONLY this JSON:
{{"grounded": <1-5>, "relevant": <1-5>, "tone": <1-5>, "actionable": <1-5>, "overall_notes": "<one sentence, mention the single biggest issue if any>"}}
"""


def judge_reply(message: str, reply: str, precedents_text: str) -> dict:
    prompt = _JUDGE_PROMPT.format(message=message, reply=reply, precedents=precedents_text)
    return generate_json(prompt, model=JUDGE_MODEL, temperature=0.0)


def human_agreement(judge_scores: list[dict], human_scores: list[dict], axis: str = "overall") -> dict:
    """Compare judge vs. human on a hand-scored subset (do this on ~30-40
    of your golden set examples, scored by you independently before looking
    at the judge's output — that ordering matters, don't anchor yourself).

    Reports exact-match rate and mean absolute difference on a 1-5 scale,
    per axis. Both matter: exact match is strict, MAE tells you if
    disagreements are near-misses (judge says 4, human says 3) vs. wild
    (judge says 5, human says 1).
    """
    axes = ["grounded", "relevant", "tone", "actionable"]
    out = {}
    for ax in axes:
        j = [s[ax] for s in judge_scores]
        h = [s[ax] for s in human_scores]
        exact = sum(1 for a, b in zip(j, h) if a == b) / len(j)
        mae = sum(abs(a - b) for a, b in zip(j, h)) / len(j)
        within_1 = sum(1 for a, b in zip(j, h) if abs(a - b) <= 1) / len(j)
        out[ax] = {"exact_match_rate": exact, "mae": mae, "within_1_rate": within_1}
    return out
