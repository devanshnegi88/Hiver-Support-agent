"""
Single choke point for all LLM calls (intent, reply, judge).

Order:
    Gemini (primary) → Ollama (local 8GB) → keyword heuristic.

Gemini uses the standard generateContent REST API with
x-goog-api-key authentication.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from config import load_google_api_keys


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
)

OLLAMA_HOST = os.environ.get(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
).rstrip("/")

# Suitable for an 8 GB RAM machine.
OLLAMA_DEFAULT_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "llama3.2:3b",
)

OLLAMA_NUM_CTX = int(
    os.environ.get("OLLAMA_NUM_CTX", "2048")
)

_8GB_OLLAMA_MODELS = (
    "llama3.2:3b",
    "llama3.2:1b",
    "phi3:mini",
    "phi3:3.8b",
    "qwen2.5:3b",
    "qwen2.5:1.5b",
    "gemma2:2b",
    "tinyllama",
)


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------

_configured = False

_keys: list[str] = []

_active_backend: str | None = None
_ollama_model: str | None = None

_gemini_unusable = False


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def _short_err(err: Exception, limit: int = 160) -> str:
    text = str(err).replace("\n", " ")

    if "API_KEY_SERVICE_BLOCKED" in text:
        return (
            "API_KEY_SERVICE_BLOCKED "
            "(this Gemini key cannot call the API)"
        )

    if "ACCESS_TOKEN_TYPE_UNSUPPORTED" in text:
        return (
            "ACCESS_TOKEN_TYPE_UNSUPPORTED "
            "(wrong Gemini credential type)"
        )

    return text if len(text) <= limit else text[:limit] + "…"


def _error_text(err: Exception) -> str:
    return f"{type(err).__name__} {err}".lower()


_PERMANENT_KEY_MARKERS = (
    "api_key_invalid",
    "api key expired",
    "api key not valid",
    "invalid api key",
    "api_key_service_blocked",
    "unauthenticated",
    "access_token_type_unsupported",
    "expired",
)


_ROTATE_KEY_MARKERS = (
    "permission_denied",
    "resource_exhausted",
    "exceeded your current quota",
    "quota",
    "rate limit",
    "rate_limit",
    "429",
    "403",
    "401",
)


def is_permanent_key_failure(err: Exception) -> bool:
    text = _error_text(err)

    if "404" in text and "model" in text:
        return False

    return any(
        marker in text
        for marker in _PERMANENT_KEY_MARKERS
    )


def is_gemini_auth_blocked(err: Exception) -> bool:
    """
    True when Gemini will not accept this credential.
    """
    text = _error_text(err)

    return any(
        marker in text
        for marker in (
            "access_token_type_unsupported",
            "api_key_service_blocked",
            "unauthenticated",
            "api_key_invalid",
            "invalid api key",
            "401",
        )
    )


def is_rotatable_key_failure(err: Exception) -> bool:
    if is_permanent_key_failure(err):
        return True

    text = _error_text(err)

    if "404" in text and "model" in text:
        return False

    return any(
        marker in text
        for marker in _ROTATE_KEY_MARKERS
    )


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def _key_kind(api_key: str) -> str:
    if api_key.startswith("AIza"):
        return "standard AIza"

    if api_key.startswith("AQ."):
        return "auth AQ."

    if api_key.startswith("ya29"):
        return (
            "OAuth token (ya29) — "
            "will not work as a Gemini API key"
        )

    prefix = api_key[:4] if api_key else ""

    return f"unknown prefix {prefix!r}"


def reset_client_state() -> None:
    """Test helper: drop cached LLM state so each test starts clean."""
    global _configured, _keys, _active_backend, _ollama_model, _gemini_unusable
    _configured = False
    _keys = []
    _active_backend = None
    _ollama_model = None
    _gemini_unusable = False


def _ensure_configured() -> None:
    global _configured
    global _keys

    if _configured:
        return

    _keys = load_google_api_keys()[:1]

    if _keys:
        print(
            f"Gemini client: 1 key loaded "
            f"({_key_kind(_keys[0])})."
        )

    _configured = True


def _wrong_credential_hint() -> str:
    kind = _key_kind(_keys[0]) if _keys else "no key"
    return (
        "Gemini rejected the credential. "
        f"Key in use looks like: {kind}. "
        "Use a valid Gemini API key, or Ollama as fallback."
    )


# ---------------------------------------------------------------------------
# Generic HTTP JSON helper
# ---------------------------------------------------------------------------

def _http_json(
    url: str,
    payload: dict,
    api_key: str,
    auth_mode: str = "api_key_header",
    timeout: int = 90,
) -> dict:

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    if auth_mode == "api_key_header":
        headers["x-goog-api-key"] = api_key

    elif auth_mode == "bearer":
        headers["Authorization"] = f"Bearer {api_key}"

    elif auth_mode == "none":
        pass

    else:
        raise ValueError(
            f"unknown auth_mode {auth_mode}"
        )

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=headers,
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:

            raw = response.read().decode("utf-8")

    except urllib.error.HTTPError as error:

        raw = error.read().decode(
            "utf-8",
            errors="replace",
        )

        lowered = raw.lower()

        if error.code in (401, 403) and (
            "credits" in lowered
            or "spending limit" in lowered
        ):
            raise RuntimeError(
                f"{error.code} LLM provider is out of "
                "credits or at its spending limit. "
                f"{raw}"
            ) from error

        raise RuntimeError(
            f"{error.code} {raw}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"HTTP error: {error}"
        ) from error

    if not raw.strip():
        raise RuntimeError(
            "LLM returned an empty HTTP body"
        )

    try:
        return json.loads(raw)

    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"LLM returned non-JSON: {raw[:300]}"
        ) from error


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _text_from_response(body: dict) -> str:

    # Generic output_text format.
    if (
        isinstance(body.get("output_text"), str)
        and body["output_text"].strip()
    ):
        return body["output_text"].strip()

    # OpenAI/xAI style.
    try:
        content = body["choices"][0]["message"]["content"]

        if isinstance(content, str) and content.strip():
            return content.strip()

    except (
        KeyError,
        IndexError,
        TypeError,
    ):
        pass

    # Generic recursive extraction.
    texts: list[str] = []

    def walk(node) -> None:

        if isinstance(node, str):

            if node.strip():
                texts.append(node)

            return

        if isinstance(node, list):

            for item in node:
                walk(item)

            return

        if isinstance(node, dict):

            if (
                isinstance(node.get("text"), str)
                and node["text"].strip()
            ):
                texts.append(node["text"])

            for key in (
                "outputs",
                "content",
                "parts",
                "steps",
            ):
                if key in node:
                    walk(node[key])

    walk(body)

    return "".join(texts).strip()


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _call_gemini(
    prompt: str,
    model: str,
    temperature: float,
    response_mime_type: str | None,
) -> str:

    global _active_backend

    if not _keys:
        raise RuntimeError(
            "No Gemini API key configured."
        )

    key = _keys[0]

    # Standard Gemini generateContent REST endpoint.
    url = (
        f"{GEMINI_BASE_URL}/"
        f"{model}:generateContent"
    )

    payload: dict = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }

    if response_mime_type:
        payload["generationConfig"][
            "responseMimeType"
        ] = response_mime_type

    body = _http_json(
        url,
        payload,
        api_key=key,
        auth_mode="api_key_header",
        timeout=90,
    )

    # Normal Gemini response:
    #
    # candidates
    #   -> content
    #       -> parts
    #           -> text
    #
    try:

        text = (
            body["candidates"][0]
            ["content"]
            ["parts"][0]
            ["text"]
            .strip()
        )

    except (
        KeyError,
        IndexError,
        TypeError,
    ):

        text = _text_from_response(body)

    if not text:

        raise RuntimeError(
            "Gemini returned no text "
            f"(keys={list(body)[:8]})"
        )

    print(
        f"Gemini auth working: "
        f"x-goog-api-key on {url}"
    )

    _active_backend = "gemini"

    return text


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _ollama_is_up() -> bool:

    try:

        request = urllib.request.Request(
            f"{OLLAMA_HOST}/api/tags",
            method="GET",
        )

        with urllib.request.urlopen(
            request,
            timeout=2,
        ) as response:

            return 200 <= response.status < 300

    except Exception:
        return False


def _ollama_installed_models() -> list[str]:

    try:

        request = urllib.request.Request(
            f"{OLLAMA_HOST}/api/tags",
            method="GET",
        )

        with urllib.request.urlopen(
            request,
            timeout=5,
        ) as response:

            body = json.loads(
                response.read().decode("utf-8")
            )

    except Exception:
        return []

    names = []

    for model in body.get("models") or []:

        name = (
            model.get("name")
            or model.get("model")
        )

        if name:
            names.append(name)

    return names


def _pick_ollama_model(
    installed: list[str],
) -> str:

    wanted = OLLAMA_DEFAULT_MODEL

    for candidate in (
        wanted,
        *_8GB_OLLAMA_MODELS,
    ):

        for name in installed:

            if (
                name == candidate
                or name.startswith(candidate)
            ):
                return name

    return wanted


def _call_ollama(
    prompt: str,
    temperature: float,
    response_mime_type: str | None,
) -> str:

    global _ollama_model

    if not _ollama_is_up():

        raise RuntimeError(
            f"Ollama is not running at "
            f"{OLLAMA_HOST}. "
            "Install from https://ollama.com/download "
            "then run: ollama pull llama3.2:3b"
        )

    if _ollama_model is None:

        installed = _ollama_installed_models()

        _ollama_model = _pick_ollama_model(
            installed
        )

        print(
            f"Using Ollama model "
            f"{_ollama_model} "
            f"(8GB RAM profile, "
            f"num_ctx={OLLAMA_NUM_CTX})."
        )

        if (
            installed
            and _ollama_model not in installed
            and not any(
                name.startswith(_ollama_model)
                for name in installed
            )
        ):

            print(
                f"Model {_ollama_model} "
                "is not installed. "
                f"Run: ollama pull {_ollama_model}"
            )

    payload: dict = {
        "model": _ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }

    if response_mime_type == "application/json":
        payload["format"] = "json"

    body = _http_json(
        f"{OLLAMA_HOST}/api/generate",
        payload,
        api_key="",
        auth_mode="none",
        timeout=180,
    )

    text = (
        body.get("response") or ""
    ).strip()

    if not text:

        raise RuntimeError(
            f"Ollama returned no text "
            f"(keys={list(body)[:8]})"
        )

    return text


# ---------------------------------------------------------------------------
# Local heuristic fallback
# ---------------------------------------------------------------------------

def _quoted_blocks(prompt: str) -> list[str]:

    return [
        block.strip()
        for block in prompt.split('"""')[1::2]
        if block.strip()
    ]


_LOCAL_INTENT_EXTRA = (

    (
        "delivery_delay_or_missing",
        re.compile(
            r"\b("
            r"package|parcel|shown up|"
            r"hasn.?t (arrived|shown|come)|"
            r"2 weeks|two weeks|"
            r"still hasn|"
            r"not (here|arrived|come)"
            r")\b",
            re.I,
        ),
    ),

    (
        "complaint_escalation",
        re.compile(
            r"\b("
            r"ridiculous|"
            r"furious|"
            r"disgusting|"
            r"scam"
            r")\b",
            re.I,
        ),
    ),
)


def _local_intent(
    customer: str,
) -> dict:

    from baselines import simple_predict

    pred = simple_predict(customer)

    if pred["confidence"] > 0:
        return pred

    from baselines import _TEMPLATE_REPLIES

    for intent, pattern in _LOCAL_INTENT_EXTRA:

        if pattern.search(customer):

            return {
                "intent": intent,
                "confidence": 0.7,
                "reply": _TEMPLATE_REPLIES.get(
                    intent,
                    pred["reply"],
                ),
                "escalate": intent in (
                    "complaint_escalation",
                    "billing_or_charge_dispute",
                ),
                "reason": (
                    "Local extra keyword match."
                ),
            }

    return pred


def _call_local(
    prompt: str,
    response_mime_type: str | None,
) -> str:

    from baselines import simple_predict

    blocks = _quoted_blocks(prompt)

    customer = (
        blocks[0]
        if blocks
        else prompt[-400:]
    )

    pred = _local_intent(customer)

    if (
        response_mime_type == "application/json"
        or '"intent"' in prompt
    ):

        # Local judge.
        if (
            "grounded" in prompt
            and "relevant" in prompt
        ):

            reply = (
                blocks[-1]
                if blocks
                else ""
            )

            grounded = (
                3
                if any(
                    ch.isdigit()
                    for ch in reply
                )
                else 4
            )

            tone = (
                4
                if any(
                    word in reply.lower()
                    for word in (
                        "sorry",
                        "thank",
                        "please",
                    )
                )
                else 3
            )

            actionable = (
                4
                if any(
                    word in reply.lower()
                    for word in (
                        "dm",
                        "order number",
                        "follow",
                    )
                )
                else 3
            )

            return json.dumps({
                "grounded": grounded,
                "relevant": 4,
                "tone": tone,
                "actionable": actionable,
                "overall_notes": (
                    "Local heuristic judge "
                    "(remote LLM unavailable)."
                ),
            })

        # Local intent classifier.
        return json.dumps({
            "intent": pred["intent"],
            "confidence": pred["confidence"],
            "rationale": (
                "Local keyword fallback "
                "(remote LLM unavailable)."
            ),
        })

    # Reply generation.
    if "Brand replied:" in prompt:

        # Prefer first retrieved precedent.
        for line in prompt.splitlines():

            if line.startswith(
                "Brand replied:"
            ):

                historical = line.split(
                    "Brand replied:",
                    1,
                )[1].strip()

                if historical:
                    return historical[:280]

    return pred["reply"][:280]


# ---------------------------------------------------------------------------
# Heuristic backend
# ---------------------------------------------------------------------------

def _use_heuristic(
    reason: str,
    prompt: str,
    response_mime_type: str | None,
) -> str:

    global _active_backend

    if _active_backend != "heuristic":

        print(
            f"{reason}; "
            "using keyword heuristic backend."
        )

    _active_backend = "heuristic"

    return _call_local(
        prompt,
        response_mime_type,
    )


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

def _fallback_after_gemini(
    reason: str,
    prompt: str,
    temperature: float,
    response_mime_type: str | None,
) -> str:

    """Gemini failed or is unset: Ollama, then keyword heuristic."""

    global _active_backend

    if _active_backend != "ollama":
        print(f"{reason}; falling back to Ollama.")

    try:
        text = _call_ollama(
            prompt,
            temperature,
            response_mime_type,
        )
        _active_backend = "ollama"
        return text
    except Exception as ollama_err:
        return _use_heuristic(
            f"{reason}; Ollama unavailable ({ollama_err})",
            prompt,
            response_mime_type,
        )


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate(
    prompt: str,
    model: str,
    temperature: float = 0.2,
    max_retries: int = 3,
    response_mime_type: str | None = None,
) -> str:

    """
    Gemini primary.

    If Gemini fails: Ollama → keyword heuristic.
    """

    global _active_backend
    global _gemini_unusable

    _ensure_configured()

    provider = (
        os.environ
        .get("LLM_PROVIDER", "")
        .lower()
        .strip()
    )

    # Explicit local mode.
    if (
        provider == "local"
        or _active_backend == "heuristic"
    ):
        return _call_local(
            prompt,
            response_mime_type,
        )

    # Explicit Ollama mode.
    if (
        provider == "ollama"
        or _active_backend == "ollama"
    ):

        return _fallback_after_gemini(
            "Using Ollama",
            prompt,
            temperature,
            response_mime_type,
        )

    # -------------------------------------------------------
    # Gemini primary
    # -------------------------------------------------------

    last_err = None

    if (
        _keys
        and not _gemini_unusable
        and provider in (
            "",
            "auto",
            "gemini",
        )
    ):

        try:

            text = _call_gemini(
                prompt,
                model,
                temperature,
                response_mime_type,
            )

            _active_backend = "gemini"

            return text

        except Exception as error:

            last_err = error
            _gemini_unusable = True

        return _fallback_after_gemini(
            "Gemini failed "
            f"({_short_err(last_err)})",
            prompt,
            temperature,
            response_mime_type,
        )

    # -------------------------------------------------------
    # No Gemini key / Gemini already unusable
    # -------------------------------------------------------

    return _fallback_after_gemini(
        (
            "Gemini unavailable"
            if _gemini_unusable
            else "No Gemini key"
        ),
        prompt,
        temperature,
        response_mime_type,
    )


# ---------------------------------------------------------------------------
# JSON generation helper
# ---------------------------------------------------------------------------

def generate_json(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> dict:

    raw = generate(
        prompt,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        response_mime_type="application/json",
    )

    text = raw.strip()

    # Remove Markdown code fences if the model adds them.
    if text.startswith("```"):

        text = text.strip("`")

        if "\n" in text:
            text = text.split(
                "\n",
                1,
            )[-1]

        if text.lower().startswith("json"):
            text = text[4:]

    # Normal JSON.
    try:

        return json.loads(text)

    except json.JSONDecodeError:

        # Try extracting the JSON object.
        start = text.find("{")
        end = text.rfind("}")

        if (
            start != -1
            and end != -1
            and end > start
        ):

            return json.loads(
                text[start:end + 1]
            )

        raise