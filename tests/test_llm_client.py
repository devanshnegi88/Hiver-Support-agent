from unittest.mock import patch

import pytest

from config import load_google_api_keys
from llm_client import (
    _key_kind,
    generate,
    is_gemini_auth_blocked,
    is_permanent_key_failure,
    is_rotatable_key_failure,
    reset_client_state,
)


@pytest.fixture(autouse=True)
def _disable_ollama_unless_named(request):
    if "ollama" in request.node.name:
        yield
        return
    with patch("llm_client._ollama_is_up", return_value=False):
        yield


def test_load_keys_primary_numbered_and_csv():
    env = {
        "GOOGLE_API_KEY": "aaa",
        "GOOGLE_API_KEY_2": "bbb",
        "GOOGLE_API_KEYS": "aaa, ccc",
    }
    assert load_google_api_keys(env) == ["aaa", "ccc", "bbb"]


def test_load_keys_semicolon_separated():
    assert load_google_api_keys({"GOOGLE_API_KEYS": "k1; k2"}) == ["k1", "k2"]


def test_load_keys_empty():
    assert load_google_api_keys({}) == []


def test_expired_and_invalid_are_permanent():
    assert is_permanent_key_failure(RuntimeError("API key expired"))
    assert is_permanent_key_failure(RuntimeError("API_KEY_INVALID"))
    assert is_permanent_key_failure(RuntimeError("API key not valid. Please pass a valid API key."))


def test_model_404_is_not_a_key_failure():
    err = RuntimeError("404 This model models/gemini-2.0-flash is no longer available")
    assert not is_permanent_key_failure(err)
    assert not is_rotatable_key_failure(err)


def test_quota_is_rotatable_not_permanent():
    err = RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
    assert not is_permanent_key_failure(err)
    assert is_rotatable_key_failure(err)


@patch("llm_client.time.sleep", lambda *a, **k: None)
@patch("llm_client._http_json")
def test_generate_switches_to_next_key_on_expired(mock_http):
    reset_client_state()
    mock_http.side_effect = [
        RuntimeError("API key expired"),
        {"output_text": "ok"},
    ]

    with patch("llm_client.load_google_api_keys", return_value=["dead-key", "live-key"]):
        with patch("llm_client._load_xai_keys", return_value=[]):
            assert generate("hi", model="gemini-3.6-flash") == "ok"

    keys_used = [c.kwargs["api_key"] for c in mock_http.call_args_list]
    assert keys_used[0] == "dead-key"
    assert keys_used[-1] == "live-key"


@patch("llm_client.time.sleep", lambda *a, **k: None)
@patch("llm_client._http_json")
def test_generate_falls_back_to_local_when_every_key_is_expired(mock_http):
    reset_client_state()
    mock_http.side_effect = RuntimeError("API_KEY_INVALID")

    with patch("llm_client.load_google_api_keys", return_value=["k1", "k2"]):
        with patch("llm_client._load_xai_keys", return_value=[]):
            text = generate("Write ONLY the reply text. Thanks!", model="gemini-3.6-flash")
            assert isinstance(text, str) and text


def test_generate_uses_local_backend_when_no_keys_configured():
    reset_client_state()
    with patch("llm_client.load_google_api_keys", return_value=[]):
        with patch("llm_client._load_xai_keys", return_value=[]):
            text = generate(
                'Choose exactly one intent\nCustomer message:\n"""where is my package"""\n'
                'Respond with ONLY this JSON object:\n{"intent": "..."}',
                model="gemini-3.6-flash",
                response_mime_type="application/json",
            )
            data = __import__("json").loads(text)
            assert data["intent"] == "delivery_delay_or_missing"


@patch("llm_client._call_xai", side_effect=RuntimeError("403 out of credits"))
@patch("llm_client._call_gemini", side_effect=RuntimeError("401 API_KEY_SERVICE_BLOCKED"))
def test_falls_back_to_local_when_xai_is_out_of_credits(_gemini, _xai):
    reset_client_state()
    with patch("llm_client.load_google_api_keys", return_value=["AQ.dummy"]):
        with patch("llm_client._load_xai_keys", return_value=["xai-key"]):
            reply = generate("Write ONLY the reply text\nThanks!", model="x")
            assert isinstance(reply, str) and len(reply) > 0


@patch("llm_client.time.sleep", lambda *a, **k: None)
@patch("llm_client._http_json")
def test_aq_key_retries_next_auth_mode_on_access_token_type(mock_http):
    reset_client_state()
    mock_http.side_effect = [
        RuntimeError("401 ACCESS_TOKEN_TYPE_UNSUPPORTED"),
        {"output_text": "ok"},
    ]
    with patch("llm_client.load_google_api_keys", return_value=["AQ.dummy-key"]):
        with patch("llm_client._load_xai_keys", return_value=[]):
            assert generate("hi", model="gemini-3.6-flash") == "ok"
    assert mock_http.call_args_list[0].kwargs["auth_mode"] == "bearer"
    assert mock_http.call_args_list[1].kwargs["auth_mode"] == "api_key_header"


@patch("llm_client._call_xai", return_value="from-xai")
@patch("llm_client._call_gemini", side_effect=RuntimeError(
    "401 ACCESS_TOKEN_TYPE_UNSUPPORTED"
))
def test_falls_back_to_xai_when_gemini_rejects_auth_type(_gemini, _xai):
    reset_client_state()
    with patch("llm_client.load_google_api_keys", return_value=["AQ.dummy-key"]):
        with patch("llm_client._load_xai_keys", return_value=["xai-key"]):
            assert generate("hi", model="gemini-3.6-flash") == "from-xai"


@patch("llm_client._call_xai", return_value="from-xai")
@patch("llm_client._call_gemini", side_effect=RuntimeError(
    '401 API_KEY_SERVICE_BLOCKED generativelanguage.googleapis.com'
))
def test_falls_back_to_xai_when_gemini_service_is_blocked(_gemini, _xai):
    reset_client_state()
    with patch("llm_client.load_google_api_keys", return_value=["AQ.dummy-key"]):
        with patch("llm_client._load_xai_keys", return_value=["xai-key"]):
            assert generate("hi", model="gemini-3.6-flash") == "from-xai"


def test_service_blocked_counts_as_gemini_auth_blocked():
    err = RuntimeError("401 API_KEY_SERVICE_BLOCKED")
    assert is_permanent_key_failure(err)
    assert is_gemini_auth_blocked(err)


def test_access_token_type_unsupported_is_permanent():
    err = RuntimeError(
        '401 ACCESS_TOKEN_TYPE_UNSUPPORTED Expected OAuth 2 access token'
    )
    assert is_permanent_key_failure(err)


def test_load_keys_accepts_gemini_api_key_alias_and_strips_quotes():
    env = {"GEMINI_API_KEY": '"aaa"', "GOOGLE_API_KEY_2": "'bbb'"}
    assert load_google_api_keys(env) == ["aaa", "bbb"]


def test_key_kind_from_prefix():
    assert "AIza" in _key_kind("AIzaSyDummy")
    assert "AQ." in _key_kind("AQ.dummy")
    assert "OAuth" in _key_kind("ya29.dummy")


@patch("llm_client._call_ollama", return_value="from-ollama")
@patch("llm_client._ollama_is_up", return_value=True)
@patch("llm_client._call_gemini", side_effect=RuntimeError("401 API_KEY_SERVICE_BLOCKED"))
def test_uses_ollama_when_cloud_keys_fail(_gemini, _up, _ollama):
    reset_client_state()
    with patch("llm_client.load_google_api_keys", return_value=["AQ.dummy"]):
        with patch("llm_client._load_xai_keys", return_value=[]):
            assert generate("hello", model="gemini-3.6-flash") == "from-ollama"


@patch("llm_client._call_ollama", return_value="from-ollama")
@patch("llm_client._ollama_is_up", return_value=True)
def test_uses_ollama_when_no_api_keys(_up, _ollama):
    reset_client_state()
    with patch("llm_client.load_google_api_keys", return_value=[]):
        with patch("llm_client._load_xai_keys", return_value=[]):
            assert generate("hello", model="gemini-3.6-flash") == "from-ollama"
