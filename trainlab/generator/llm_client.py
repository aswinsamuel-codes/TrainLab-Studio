"""LLM access for scenario generation: bring your own provider and API key.

Supported out of the box:
    anthropic   Claude, via the official Anthropic SDK
    openai      GPT models
    gemini      Google Gemini (free tier key from Google AI Studio)
    groq        fast open models (free tier)
    openrouter  one key for hundreds of models (some free)
    deepseek, mistral
    ollama      models running on your own PC, free, no key
    custom      any other OpenAI-compatible endpoint (set a base URL)

Everything except Anthropic goes through the OpenAI-compatible Chat Completions API,
which all of these providers expose. Keys come from TrainLab/.env (see .env.example) or
are typed into Studio > Settings for one session. Note: a chat subscription (ChatGPT
Plus, Claude Pro, Gemini Advanced...) is not an API key; keys come from each provider's
developer console.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PLACEHOLDER = "your-key-here"


@dataclass(frozen=True)
class ProviderInfo:
    label: str
    key_env: str | None  # None = no key needed
    base_url: str | None  # None = SDK default
    models: list[str] = field(default_factory=list)  # suggestions; "Load models" fetches the live list
    key_url: str = ""
    note: str = ""


PROVIDERS: dict[str, ProviderInfo] = {
    "anthropic": ProviderInfo("Anthropic (Claude)", "ANTHROPIC_API_KEY", None,
                              ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1"],
                              "https://console.anthropic.com"),
    "openai": ProviderInfo("OpenAI (GPT)", "OPENAI_API_KEY", None, ["gpt-5", "gpt-5-mini"],
                           "https://platform.openai.com/api-keys"),
    "gemini": ProviderInfo("Google Gemini", "GEMINI_API_KEY",
                           "https://generativelanguage.googleapis.com/v1beta/openai/",
                           ["gemini-2.5-pro", "gemini-2.5-flash"], "https://aistudio.google.com/apikey",
                           "Has a free tier."),
    "groq": ProviderInfo("Groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1",
                         ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"], "https://console.groq.com/keys",
                         "Has a free tier."),
    "openrouter": ProviderInfo("OpenRouter (many models)", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
                               ["openrouter/auto"], "https://openrouter.ai/keys",
                               "One key for most models; some are free."),
    "deepseek": ProviderInfo("DeepSeek", "DEEPSEEK_API_KEY", "https://api.deepseek.com",
                             ["deepseek-chat", "deepseek-reasoner"], "https://platform.deepseek.com/api_keys"),
    "mistral": ProviderInfo("Mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1",
                            ["mistral-large-latest", "codestral-latest"], "https://console.mistral.ai/api-keys"),
    "ollama": ProviderInfo("Ollama (local, free)", None, "http://localhost:11434/v1",
                           ["qwen2.5-coder:14b", "llama3.1:8b"], "https://ollama.com/download",
                           "Runs on your PC, no key, no cost. Small models may need several repair rounds."),
    "custom": ProviderInfo("Custom OpenAI-compatible", "CUSTOM_LLM_API_KEY", None, [], "",
                           "Set the base URL (CUSTOM_LLM_BASE_URL) of any OpenAI-compatible server."),
}
DEFAULT_PROVIDER = "anthropic"

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "mjcf_xml": {"type": "string"},
        "scenario_py": {"type": "string"},
        "config_yaml": {"type": "string"},
        "reward_explanation": {"type": "string"},
        "controls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "meaning": {"type": "string"}},
                "required": ["name", "meaning"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["name", "title", "summary", "mjcf_xml", "scenario_py", "config_yaml",
                 "reward_explanation", "controls"],
    "additionalProperties": False,
}


class MissingKeyError(RuntimeError):
    pass


class GenerationError(RuntimeError):
    pass


# ------------------------------------------------------------------ settings
def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env", override=False)


def _real(value: str | None) -> str:
    return value if value and value != PLACEHOLDER else ""


def env_key(provider: str) -> str:
    load_env()
    info = PROVIDERS[provider]
    return _real(os.environ.get(info.key_env, "")) if info.key_env else ""


def env_base_url(provider: str) -> str | None:
    load_env()
    if provider == "custom":
        return os.environ.get("CUSTOM_LLM_BASE_URL") or None
    if provider == "ollama":
        return os.environ.get("OLLAMA_BASE_URL") or PROVIDERS["ollama"].base_url
    return PROVIDERS[provider].base_url


def default_provider() -> str:
    """TRAINLAB_LLM_PROVIDER if set, else the first provider that has a key, else Anthropic."""
    load_env()
    chosen = os.environ.get("TRAINLAB_LLM_PROVIDER", "").strip().lower()
    if chosen in PROVIDERS:
        return chosen
    for name, info in PROVIDERS.items():
        if info.key_env and env_key(name):
            return name
    return DEFAULT_PROVIDER


def default_model(provider: str) -> str:
    load_env()
    if os.environ.get("TRAINLAB_LLM_PROVIDER", "").strip().lower() == provider and os.environ.get("TRAINLAB_LLM_MODEL"):
        return os.environ["TRAINLAB_LLM_MODEL"]
    models = PROVIDERS[provider].models
    return models[0] if models else ""


def key_status() -> dict[str, str]:
    return {name: ("not needed" if info.key_env is None else ("set" if env_key(name) else "missing"))
            for name, info in PROVIDERS.items()}


def save_settings_to_env(provider: str, key: str = "", model: str = "", base_url: str = "") -> Path:
    """Write the chosen provider (and its key/model/base URL) into TrainLab/.env, keeping other lines."""
    path = ROOT / ".env"
    lines = path.read_text().splitlines() if path.exists() else []
    updates = {"TRAINLAB_LLM_PROVIDER": provider}
    if model:
        updates["TRAINLAB_LLM_MODEL"] = model
    if key and PROVIDERS[provider].key_env:
        updates[PROVIDERS[provider].key_env] = key
    if base_url:
        updates["CUSTOM_LLM_BASE_URL" if provider == "custom" else "OLLAMA_BASE_URL"] = base_url
    lines = [ln for ln in lines if ln.split("=", 1)[0] not in updates]
    lines = [f"{k}={v}" for k, v in updates.items()] + lines
    path.write_text("\n".join(lines) + "\n")
    for k, v in updates.items():
        os.environ[k] = v
    return path


def parse_json_loose(text: str) -> dict:
    """Parse JSON even if a model wrapped it in ``` fences or added chatter around it."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise GenerationError(f"The model did not return valid JSON ({e}). Try again or pick a stronger model.") from e
    missing = [k for k in SCHEMA["required"] if k not in data]
    if missing:
        raise GenerationError(f"The model's JSON is missing fields: {', '.join(missing)}")
    return data


# ----------------------------------------------------------------- providers
class LLMProvider:
    name = "base"
    model = ""

    def generate(self, system: str, messages: list[dict]) -> tuple[dict, object]:
        """Return (parsed JSON, assistant content to append to the conversation)."""
        raise NotImplementedError

    def list_models(self) -> list[str]:
        return []


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None, effort: str = "high"):
        import anthropic

        key = api_key or env_key("anthropic")
        if not key:
            raise MissingKeyError(_missing_key_msg("anthropic"))
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model or default_model("anthropic")
        self.effort = effort

    def generate(self, system: str, messages: list[dict]):
        import anthropic

        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=64000,
                system=system,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
                # If a safety classifier declines, retry server-side on Anthropic's recommended model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                response = stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise MissingKeyError(f"The Anthropic API key was rejected: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise GenerationError(f"Model '{self.model}' not found. Click 'Load models' to see yours.") from e
        except anthropic.RateLimitError as e:
            raise GenerationError("Rate limited by the API - wait a minute and try again.") from e
        except anthropic.APIStatusError as e:
            raise GenerationError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise GenerationError("Could not reach the API - check your internet connection.") from e

        if response.stop_reason == "refusal":
            reason = getattr(response.stop_details, "explanation", None) if response.stop_details else None
            raise GenerationError(f"The model declined this request. {reason or ''}".strip())
        if response.stop_reason == "max_tokens":
            raise GenerationError("The response was cut off (too long). Try a simpler world.")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return parse_json_loose(text), response.content

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI, Gemini, Groq, OpenRouter, DeepSeek, Mistral, Ollama, or any compatible server."""

    def __init__(self, provider: str, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None):
        from openai import OpenAI

        self.name = provider
        info = PROVIDERS[provider]
        key = api_key or env_key(provider)
        if info.key_env and not key and provider != "custom":  # local custom servers often need no key
            raise MissingKeyError(_missing_key_msg(provider))
        base_url = base_url or env_base_url(provider)
        if provider == "custom" and not base_url:
            raise MissingKeyError("Custom provider needs a base URL (Settings, or CUSTOM_LLM_BASE_URL in .env).")
        self.client = OpenAI(api_key=key or "not-needed", base_url=base_url, timeout=900.0, max_retries=2)
        self.model = model or default_model(provider)
        if not self.model:
            raise MissingKeyError(f"Pick a model for {info.label} (click 'Load models' in Settings).")
        self.max_tokens = int(os.environ.get("TRAINLAB_LLM_MAX_TOKENS", "0")) or None

    def _formats(self):
        # Strictest first; fall back for servers that don't support JSON schemas.
        yield {"type": "json_schema", "json_schema": {"name": "trainlab_scenario", "schema": SCHEMA, "strict": True}}
        yield {"type": "json_object"}
        yield None

    def generate(self, system: str, messages: list[dict]):
        import openai

        chat = [{"role": "system", "content": system}] + [
            {"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])}
            for m in messages]
        last_error = None
        for fmt in self._formats():
            kwargs = {"model": self.model, "messages": chat}
            if fmt is not None:
                kwargs["response_format"] = fmt
            if self.max_tokens:
                kwargs["max_tokens"] = self.max_tokens
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except openai.BadRequestError as e:  # likely "response_format not supported": try a looser format
                last_error = e
                continue
            except openai.AuthenticationError as e:
                raise MissingKeyError(f"The {PROVIDERS[self.name].label} API key was rejected.") from e
            except openai.NotFoundError as e:
                raise GenerationError(f"Model '{self.model}' not found on {PROVIDERS[self.name].label}. "
                                      "Click 'Load models' in Settings.") from e
            except openai.RateLimitError as e:
                raise GenerationError("Rate limited (or out of credit/free quota) - wait and try again.") from e
            except openai.APIConnectionError as e:
                hint = " Is Ollama running? (ollama serve)" if self.name == "ollama" else ""
                raise GenerationError(f"Could not reach {PROVIDERS[self.name].label}.{hint}") from e
            except openai.APIStatusError as e:
                raise GenerationError(f"API error {e.status_code}: {e.message}") from e
        else:
            raise GenerationError(f"{PROVIDERS[self.name].label} rejected the request: {last_error}")

        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise GenerationError("The response was cut off. Set TRAINLAB_LLM_MAX_TOKENS higher in .env, "
                                  "or use a model with a larger output limit.")
        text = choice.message.content or ""
        return parse_json_loose(text), text

    def list_models(self) -> list[str]:
        # Fail fast here (e.g. Ollama not running) instead of the long generation timeout.
        return sorted(m.id for m in self.client.with_options(timeout=20.0, max_retries=0).models.list())


class OllamaProvider(OpenAICompatibleProvider):
    """Local models through Ollama's native /api/chat.

    The OpenAI-compatible endpoint can't change the context window, and Ollama's default is
    too small for TrainLab's instructions plus a full scenario, so generation uses the native
    API (context size, JSON-schema output, thinking off for CPU speed). Model listing still
    uses the compatible endpoint.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None):
        super().__init__("ollama", model=model, base_url=base_url)
        self.root = str(self.client.base_url).rstrip("/").removesuffix("/v1")
        self.num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "12288"))
        self.think = os.environ.get("OLLAMA_THINK", "false").lower() == "true"

    def memory_warning(self) -> str | None:
        """Warn when the model probably doesn't fit in RAM (it then swaps and the laptop crawls)."""
        import httpx
        import psutil

        try:
            tags = httpx.get(f"{self.root}/api/tags", timeout=5).json().get("models", [])
        except Exception:
            return None
        size = next((m.get("size", 0) for m in tags if m.get("name") == self.model), 0) / 1024 ** 3
        total = psutil.virtual_memory().total / 1024 ** 3
        if size and size + 3.5 > total:  # Windows + browser need ~3.5 GB of their own
            small = sorted((m["size"] / 1024 ** 3, m["name"]) for m in tags if m["size"] / 1024 ** 3 + 3.5 <= total)
            hint = f" Smaller models you have: {', '.join(n for _, n in small[:3])}." if small else \
                " Try a 3-4B model (e.g. `ollama pull qwen2.5-coder:3b`) or a free cloud key (Gemini, Groq)."
            return (f"{self.model} needs ~{size:.1f} GB plus working memory, but this PC has {total:.0f} GB RAM: "
                    f"it will swap and be very slow, and the laptop may lag.{hint}")
        return None

    def generate(self, system: str, messages: list[dict]):
        import httpx

        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
                        + [{"role": m["role"], "content": m["content"]} for m in messages],
            "stream": False,
            "format": SCHEMA,
            "think": self.think,
            "keep_alive": "2m",  # free the RAM soon after generating (repair rounds come within seconds)
            "options": {"num_ctx": self.num_ctx, "temperature": 0.3, "num_predict": self.max_tokens or 12000},
        }
        try:
            resp = httpx.post(f"{self.root}/api/chat", json=body, timeout=httpx.Timeout(3600.0, connect=10.0))
        except httpx.ConnectError as e:
            raise GenerationError("Could not reach Ollama. Is it running? (ollama serve)") from e
        except httpx.TimeoutException as e:
            raise GenerationError("Ollama took longer than an hour; try a smaller model.") from e
        if resp.status_code == 404:
            raise GenerationError(f"Ollama has no model '{self.model}'. Run: ollama pull {self.model}")
        if resp.status_code != 200:
            raise GenerationError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if data.get("done_reason") == "length":
            raise GenerationError("The local model hit its output limit. Set TRAINLAB_LLM_MAX_TOKENS higher.")
        text = data.get("message", {}).get("content", "")
        return parse_json_loose(text), text


def _missing_key_msg(provider: str) -> str:
    info = PROVIDERS[provider]
    where = f" Get one at {info.key_url}." if info.key_url else ""
    return (f"No API key for {info.label}. Paste it in Studio > Settings, or put {info.key_env}=... in "
            f"TrainLab/.env (see .env.example).{where}")


def get_provider(provider: str | None = None, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None) -> LLMProvider:
    provider = (provider or default_provider()).lower()
    if provider not in PROVIDERS:
        raise GenerationError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    if provider == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key)
    if provider == "ollama":
        return OllamaProvider(model=model, base_url=base_url)
    return OpenAICompatibleProvider(provider, model=model, api_key=api_key, base_url=base_url)
