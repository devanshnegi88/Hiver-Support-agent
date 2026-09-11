"""
Drafts a reply to a customer message, grounded in the brand's historical
precedent replies to similar issues (from retrieval.py).

The prompt explicitly instructs the model to only use info it can support
from the precedents or the message itself — not to invent policy details
(refund amounts, timelines, etc.) that aren't backed by a precedent. This
is the single biggest hallucination risk in this task and it's called out
directly in the prompt rather than hoped away.
"""
from config import GEN_MODEL
from llm_client import generate
from retrieval import RetrievalHit


_PROMPT_TMPL = """You are drafting a customer support reply for {brand} on Twitter (280 char limit, brand's real voice).

Customer's message:
\"\"\"{message}\"\"\"

Their detected intent: {intent}

Here is how {brand} has replied to similar past issues (most similar first).
Use these ONLY to match tone, structure, and any policy specifics (refund windows, next steps, where to DM). Do NOT invent specific numbers, timelines, or policies that aren't supported by these precedents or general common sense — if the precedents don't cover something specific, keep the reply generic and directive (e.g. "please DM your order number").

{precedents}

Write ONLY the reply text, under 280 characters, no hashtags, no quotation marks around it.
"""


def _format_precedents(hits: list[RetrievalHit]) -> str:
    if not hits:
        return "(No sufficiently similar precedent found.)"
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(
            f"[Precedent {i}, similarity={h.similarity:.2f}]\n"
            f"Customer said: {h.customer_message}\n"
            f"Brand replied: {h.brand_reply}\n"
        )
    return "\n".join(lines)


def draft_reply(message: str, intent: str, precedents: list[RetrievalHit], brand: str) -> str:
    prompt = _PROMPT_TMPL.format(
        brand=brand,
        message=message,
        intent=intent,
        precedents=_format_precedents(precedents),
    )
    reply = generate(prompt, model=GEN_MODEL, temperature=0.4)
    # Hard-enforce the length constraint regardless of what the model does.
    return reply[:280]
