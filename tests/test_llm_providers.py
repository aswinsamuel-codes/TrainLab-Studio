import json

import pytest

from trainlab.generator import llm_client as lc

KEYS = [info.key_env for info in lc.PROVIDERS.values() if info.key_env]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setattr(lc, "load_env", lambda: None)  # ignore the developer's real .env
    for k in KEYS + ["TRAINLAB_LLM_PROVIDER", "TRAINLAB_LLM_MODEL", "CUSTOM_LLM_BASE_URL", "OLLAMA_BASE_URL"]:
        monkeypatch.delenv(k, raising=False)


SCENARIO = {k: "" for k in lc.SCHEMA["required"]} | {"name": "x_y", "controls": []}


def test_parse_json_plain_and_fenced():
    text = json.dumps(SCENARIO)
    assert lc.parse_json_loose(text)["name"] == "x_y"
    assert lc.parse_json_loose(f"Sure! Here it is:\n```json\n{text}\n```\nEnjoy")["name"] == "x_y"
    assert lc.parse_json_loose(f"blah {text} blah")["name"] == "x_y"


def test_parse_json_rejects_garbage_and_missing_fields():
    with pytest.raises(lc.GenerationError):
        lc.parse_json_loose("no json here")
    with pytest.raises(lc.GenerationError):
        lc.parse_json_loose('{"name": "only_name"}')


def test_default_provider_follows_available_key(monkeypatch):
    assert lc.default_provider() == "anthropic"
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    assert lc.default_provider() == "gemini"
    monkeypatch.setenv("TRAINLAB_LLM_PROVIDER", "ollama")
    assert lc.default_provider() == "ollama"


def test_placeholder_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", lc.PLACEHOLDER)
    assert lc.key_status()["openai"] == "missing"
    with pytest.raises(lc.MissingKeyError, match="platform.openai.com"):
        lc.get_provider("openai")


@pytest.mark.parametrize("provider", ["openai", "gemini", "groq", "openrouter", "deepseek", "mistral"])
def test_openai_compatible_providers_build(provider):
    p = lc.get_provider(provider, api_key="test-key")
    assert p.name == provider and p.model
    expected = lc.PROVIDERS[provider].base_url
    if expected:
        assert str(p.client.base_url).rstrip("/") == expected.rstrip("/")


def test_ollama_needs_no_key():
    p = lc.get_provider("ollama")
    assert "11434" in str(p.client.base_url)


def test_custom_needs_base_url():
    with pytest.raises(lc.MissingKeyError):
        lc.get_provider("custom", model="m")
    p = lc.get_provider("custom", model="m", base_url="http://127.0.0.1:1234/v1")
    assert "1234" in str(p.client.base_url)


def test_anthropic_builds_with_key():
    p = lc.get_provider("anthropic", api_key="test-key")
    assert p.model == "claude-opus-5-5"


def test_unknown_provider():
    with pytest.raises(lc.GenerationError):
        lc.get_provider("nope", api_key="k")


def test_json_format_fallback(monkeypatch):
    """Servers that reject json_schema should be retried with looser formats."""
    import httpx
    import openai

    p = lc.get_provider("groq", api_key="k")
    calls = []

    class Msg:
        content = json.dumps(SCENARIO)

    class Choice:
        message, finish_reason = Msg(), "stop"

    class Resp:
        choices = [Choice()]

    def fake_create(**kwargs):
        calls.append(kwargs.get("response_format", {}).get("type") if kwargs.get("response_format") else None)
        if len(calls) < 3:
            req = httpx.Request("POST", "http://x")
            raise openai.BadRequestError("response_format not supported", response=httpx.Response(400, request=req),
                                         body=None)
        return Resp()

    monkeypatch.setattr(p.client.chat.completions, "create", fake_create)
    data, text = p.generate("sys", [{"role": "user", "content": "hi"}])
    assert data["name"] == "x_y" and calls == ["json_schema", "json_object", None]
