"""
Single choke point for all LLM calls (intent, reply, judge).

Order: Gemini → xAI → local Ollama (8GB-RAM models) → keyword heuristic.
Ollama is used when cloud keys are missing, blocked, or out of credits.

No Gemini SDK: google.generativeai / google-genai attach an OAuth
Authorization header that Gemini 3.x rejects.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from config import load_google_api_keys

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
XAI_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4.5")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
# llama3.2:3b is ~2GB on disk / ~3-4GB RAM — fits an 8GB machine with OS overhead.
# Do not default to 7B/8B weights; those will swap-thrash on 8GB.
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "2048"))
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

_configured = False
_keys: list[str] = []
_key_index = 0
_dead: set[int] = set()
_xai_keys: list[str] = []
_working_auth_mode: str | None = None
_active_backend: str | None = None  # gemini | xai | ollama | heuristic
_ollama_model: str | None = None

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


def reset_client_state() -> None:
    """Test helper: drop cached key-pool state so each test starts clean."""
    global _configured, _keys, _key_index, _dead, _xai_keys, _working_auth_mode
    global _active_backend, _ollama_model
    _configured = False
    _keys = []
    _key_index = 0
    _dead = set()
    _xai_keys = []
    _working_auth_mode = None
    _active_backend = None
    _ollama_model = None


def _error_text(err: Exception) -> str:
    return f"{type(err).__name__} {err}".lower()


def is_permanent_key_failure(err: Exception) -> bool:
    text = _error_text(err)
    if "404" in text and "model" in text:
        return False
    return any(m in text for m in _PERMANENT_KEY_MARKERS)


def is_gemini_auth_blocked(err: Exception) -> bool:
    """True when Gemini will not accept this credential; xAI fallback applies."""
    text = _error_text(err)
    return any(
        m in text
        for m in (
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
    return any(m in text for m in _ROTATE_KEY_MARKERS)


def _key_kind(api_key: str) -> str:
    if api_key.startswith("AIza"):
        return "standard AIza"
    if api_key.startswith("AQ."):
        return "auth AQ."
    if api_key.startswith("ya29"):
        return "OAuth token (ya29) — will not work as a Gemini API key"
    prefix = api_key[:4] if api_key else ""
    return f"unknown prefix {prefix!r}"


def _load_xai_keys(env=None) -> list[str]:
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

    add(env.get("XAI_API_KEY", ""))
    add(env.get("XAI_API_KEYS", ""))
    return keys


def _ensure_configured() -> None:
    global _configured, _keys, _key_index, _xai_keys
    if _configured:
        return
    _keys = load_google_api_keys()
    _xai_keys = _load_xai_keys()
    _key_index = 0
    if _keys:
        kinds = ", ".join(f"key {i + 1}={_key_kind(k)}" for i, k in enumerate(_keys))
        print(f"Gemini client: {len(_keys)} key(s) loaded ({kinds}).")
    if _xai_keys:
        print(f"xAI fallback: {len(_xai_keys)} key(s) loaded (model={XAI_MODEL}).")
    _configured = True


def _next_live_index(start: int) -> int | None:
    n = len(_keys)
    if n == 0:
        return None
    for offset in range(1, n + 1):
        cand = (start + offset) % n
        if cand not in _dead:
            return cand
    return None


def _switch_key(err: Exception, *, retire: bool) -> bool:
    global _key_index, _working_auth_mode
    current = _key_index
    if retire:
        _dead.add(current)
        reason = "expired/invalid"
    else:
        reason = "quota/rate-limit"
    nxt = _next_live_index(current)
    if nxt is None or nxt == current:
        return False
    print(
        f"Gemini API key {current + 1}/{len(_keys)} failed ({reason}); "
        f"switching to key {nxt + 1}/{len(_keys)}."
    )
    _key_index = nxt
    _working_auth_mode = None
    return True


def _wrong_credential_hint() -> str:
    kind = _key_kind(_keys[_key_index]) if _keys else "no key"
    extra = ""
    if not _xai_keys:
        extra = (
            " Or set XAI_API_KEY (https://console.x.ai) and this client will "
            "run the eval on grok-4.5 instead."
        )
    return (
        f"Gemini rejected the credential as the wrong type "
        f"(ACCESS_TOKEN_TYPE_UNSUPPORTED). Key in use looks like: {kind}. "
        "Known Google-side failure for some AI Studio AQ. keys. "
        "Tried Bearer then x-goog-api-key."
        f"{extra}"
    )


def _final_error(max_retries: int, last_err: Exception | None) -> RuntimeError:
    n_dead = len(_dead)
    hint = ""
    if last_err and "access_token_type_unsupported" in _error_text(last_err):
        hint = " " + _wrong_credential_hint()
    return RuntimeError(
        f"Gemini call failed after {max_retries} retries "
        f"({n_dead} key(s) retired of {len(_keys)}): {last_err}.{hint}"
    )


def _http_json(url: str, payload: dict, api_key: str,
               auth_mode: str = "api_key_header", timeout: int = 90) -> dict:
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
        raise ValueError(f"unknown auth_mode {auth_mode}")

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        lowered = raw.lower()
        if e.code in (401, 403) and (
            "credits" in lowered or "spending limit" in lowered
        ):
            raise RuntimeError(
                f"{e.code} LLM provider is out of credits or at its spending "
                f"limit. Add credits (xAI: https://console.x.ai) or use a "
                f"Gemini key that is not API_KEY_SERVICE_BLOCKED. {raw}"
            ) from e
        raise RuntimeError(f"{e.code} {raw}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"HTTP error: {e}") from e
    if not raw.strip():
        raise RuntimeError("LLM returned an empty HTTP body")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned non-JSON: {raw[:300]}") from e


def _text_from_response(body: dict) -> str:
    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"].strip()
    try:
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, str) and content.strip():
            return content.strip()
    except (KeyError, IndexError, TypeError):
        pass
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
            if isinstance(node.get("text"), str) and node["text"].strip():
                texts.append(node["text"])
            for key in ("outputs", "content", "parts", "steps"):
                if key in node:
                    walk(node[key])

    walk(body)
    return "".join(texts).strip()


def _auth_modes_for_key(api_key: str) -> tuple[str, ...]:
    if api_key.startswith("AQ."):
        return ("bearer", "api_key_header")
    return ("api_key_header", "bearer")


def _call_gemini(prompt: str, model: str, temperature: float,
                 response_mime_type: str | None) -> str:
    global _working_auth_mode
    key = _keys[_key_index]
    payload: dict = {
        "model": model,
        "input": prompt,
        "generation_config": {"temperature": temperature},
    }
    if response_mime_type:
        payload["response_mime_type"] = response_mime_type

    modes = _auth_modes_for_key(key)
    if _working_auth_mode:
        modes = (_working_auth_mode,) + tuple(m for m in modes if m != _working_auth_mode)

    last_err: Exception | None = None
    for mode in modes:
        try:
            body = _http_json(GEMINI_URL, payload, api_key=key, auth_mode=mode)
            text = _text_from_response(body)
            if not text:
                raise RuntimeError(f"Gemini returned no text (keys={list(body)[:8]})")
            if _working_auth_mode is None:
                print(f"Gemini auth working: {mode} on {GEMINI_URL}")
            _working_auth_mode = mode
            return text
        except Exception as e:  # noqa: BLE001
            last_err = e
            if "access_token_type_unsupported" in _error_text(e):
                continue
            raise
    raise last_err or RuntimeError("Gemini call failed with no attempts")


def _call_xai(prompt: str, temperature: float,
              response_mime_type: str | None) -> str:
    payload: dict = {
        "model": XAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if response_mime_type == "application/json":
        payload["response_format"] = {"type": "json_object"}
    body = _http_json(XAI_URL, payload, api_key=_xai_keys[0], auth_mode="bearer")
    text = _text_from_response(body)
    if not text:
        raise RuntimeError(f"xAI returned no text (keys={list(body)[:8]})")
    return text


def _ollama_is_up() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return 200 <= resp.status < 300
    except Exception:  # noqa: BLE001
        return False


def _ollama_installed_models() -> list[str]:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return []
    names = []
    for m in body.get("models") or []:
        name = m.get("name") or m.get("model")
        if name:
            names.append(name)
    return names


def _pick_ollama_model(installed: list[str]) -> str:
    wanted = OLLAMA_DEFAULT_MODEL
    for cand in (wanted, *_8GB_OLLAMA_MODELS):
        for name in installed:
            if name == cand or name.startswith(cand):
                return name
    return wanted


def _call_ollama(prompt: str, temperature: float,
                 response_mime_type: str | None) -> str:
    global _ollama_model
    if not _ollama_is_up():
        raise RuntimeError(
            f"Ollama is not running at {OLLAMA_HOST}. "
            "Install from https://ollama.com/download then: ollama pull llama3.2:3b"
        )
    if _ollama_model is None:
        installed = _ollama_installed_models()
        _ollama_model = _pick_ollama_model(installed)
        print(
            f"Using Ollama model {_ollama_model} "
            f"(8GB RAM profile, num_ctx={OLLAMA_NUM_CTX})."
        )
        if installed and _ollama_model not in installed and not any(
            n.startswith(_ollama_model) for n in installed
        ):
            print(
                f"Model {_ollama_model} is not installed. Run: "
                f"ollama pull {_ollama_model}"
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
    text = (body.get("response") or "").strip()
    if not text:
        raise RuntimeError(f"Ollama returned no text (keys={list(body)[:8]})")
    return text


def _quoted_blocks(prompt: str) -> list[str]:
    return [p.strip() for p in prompt.split('"""')[1::2] if p.strip()]


_LOCAL_INTENT_EXTRA = (
    ("delivery_delay_or_missing",
     re.compile(
         r"\b(package|parcel|shown up|hasn.?t (arrived|shown|come)|"
         r"2 weeks|two weeks|still hasn|not (here|arrived|come))\b",
         re.I,
     )),
    ("complaint_escalation",
     re.compile(r"\b(ridiculous|furious|disgusting|scam)\b", re.I)),
)


def _local_intent(customer: str) -> dict:
    from baselines import simple_predict  # noqa: PLC0415

    pred = simple_predict(customer)
    if pred["confidence"] > 0:
        return pred
    from baselines import _TEMPLATE_REPLIES  # noqa: PLC0415

    for intent, pattern in _LOCAL_INTENT_EXTRA:
        if pattern.search(customer):
            return {
                "intent": intent,
                "confidence": 0.7,
                "reply": _TEMPLATE_REPLIES.get(intent, pred["reply"]),
                "escalate": intent in ("complaint_escalation", "billing_or_charge_dispute"),
                "reason": "Local extra keyword match.",
            }
    return pred


def _call_local(prompt: str, response_mime_type: str | None) -> str:
    """Deterministic stand-in when Gemini/xAI are blocked or out of credits."""
    from baselines import simple_predict  # noqa: PLC0415

    blocks = _quoted_blocks(prompt)
    customer = blocks[0] if blocks else prompt[-400:]
    pred = _local_intent(customer)

    if response_mime_type == "application/json" or "\"intent\"" in prompt:
        if "grounded" in prompt and "relevant" in prompt:
            reply = blocks[-1] if blocks else ""
            grounded = 3 if any(ch.isdigit() for ch in reply) else 4
            tone = 4 if any(w in reply.lower() for w in ("sorry", "thank", "please")) else 3
            actionable = 4 if any(w in reply.lower() for w in ("dm", "order number", "follow")) else 3
            return json.dumps({
                "grounded": grounded,
                "relevant": 4,
                "tone": tone,
                "actionable": actionable,
                "overall_notes": "Local heuristic judge (remote LLM unavailable).",
            })
        return json.dumps({
            "intent": pred["intent"],
            "confidence": pred["confidence"],
            "rationale": "Local keyword fallback (remote LLM unavailable).",
        })

    if "Brand replied:" in prompt:
        # Prefer the first retrieved precedent's actual historical reply.
        for line in prompt.splitlines():
            if line.startswith("Brand replied:"):
                hist = line.split("Brand replied:", 1)[1].strip()
                if hist:
                    return hist[:280]
    return pred["reply"][:280]


def _use_heuristic(reason: str, prompt: str, response_mime_type: str | None) -> str:
    global _active_backend
    if _active_backend != "heuristic":
        print(f"{reason}; using keyword heuristic backend.")
    _active_backend = "heuristic"
    return _call_local(prompt, response_mime_type)


def _use_ollama_or_heuristic(reason: str, prompt: str, temperature: float,
                             response_mime_type: str | None) -> str:
    global _active_backend
    try:
        text = _call_ollama(prompt, temperature, response_mime_type)
        if _active_backend != "ollama":
            print(f"{reason}; using Ollama.")
        _active_backend = "ollama"
        return text
    except Exception as e:  # noqa: BLE001
        return _use_heuristic(
            f"{reason}; Ollama unavailable ({e})",
            prompt,
            response_mime_type,
        )


def generate(prompt: str, model: str, temperature: float = 0.2,
             max_retries: int = 3, response_mime_type: str | None = None) -> str:
    global _active_backend
    _ensure_configured()
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider == "local" or _active_backend == "heuristic":
        return _call_local(prompt, response_mime_type)
    if provider == "ollama" or _active_backend == "ollama":
        return _use_ollama_or_heuristic(
            "Forced/cached Ollama", prompt, temperature, response_mime_type
        )
    if _active_backend == "xai":
        try:
            return _call_xai(prompt, temperature, response_mime_type)
        except Exception as e:  # noqa: BLE001
            return _use_ollama_or_heuristic(f"xAI failed ({e})", prompt, temperature, response_mime_type)
    if _active_backend == "gemini":
        try:
            return _call_gemini(prompt, model, temperature, response_mime_type)
        except Exception as e:  # noqa: BLE001
            return _use_ollama_or_heuristic(
                f"Gemini failed ({e})", prompt, temperature, response_mime_type
            )

    last_err = None
    if _keys:
        budget = max_retries + max(0, len(_keys) - 1)
        for attempt in range(budget):
            try:
                text = _call_gemini(prompt, model, temperature, response_mime_type)
                _active_backend = "gemini"
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                if is_permanent_key_failure(e):
                    if _switch_key(e, retire=True):
                        continue
                    break
                if is_rotatable_key_failure(e):
                    if _switch_key(e, retire=False):
                        continue
                time.sleep(1.5 * (attempt + 1))
        if _xai_keys:
            try:
                print(f"Gemini unavailable; falling back to xAI {XAI_MODEL}.")
                text = _call_xai(prompt, temperature, response_mime_type)
                _active_backend = "xai"
                return text
            except Exception as e:  # noqa: BLE001
                return _use_ollama_or_heuristic(
                    f"xAI failed ({e})", prompt, temperature, response_mime_type
                )
        return _use_ollama_or_heuristic(
            f"Gemini failed ({last_err})", prompt, temperature, response_mime_type
        )
    if _xai_keys:
        try:
            print(f"No Gemini key; using xAI {XAI_MODEL}.")
            text = _call_xai(prompt, temperature, response_mime_type)
            _active_backend = "xai"
            return text
        except Exception as e:  # noqa: BLE001
            return _use_ollama_or_heuristic(
                f"xAI failed ({e})", prompt, temperature, response_mime_type
            )
    return _use_ollama_or_heuristic(
        "No remote LLM keys", prompt, temperature, response_mime_type
    )


def generate_json(prompt: str, model: str, temperature: float = 0.0,
                  max_retries: int = 3) -> dict:
    raw = generate(
        prompt,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        response_mime_type="application/json",
    )
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise
