# src/llm/backends.py

import json
import os
from pathlib import Path
from typing import Any, Optional
from urllib import error, parse, request

from src.core.retry import call_with_backoff, classify_retryable_error, is_request_too_large_error
from src.core.run_logging import log_line, record_model_switch, record_model_usage

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AZURE_FOUNDRY_API_VERSION = "2024-05-01-preview"
DEFAULT_AZURE_FOUNDRY_TIMEOUT_SECONDS = 300
GROQ_MULTI_MODEL_SENTINEL = "multi"
MAX_GROQ_MULTI_MODELS = 5
DEFAULT_GROQ_MULTI_SAME_MODEL_RETRIES = 2
DEFAULT_GROQ_COMPLETION_TOKEN_RESERVE = 2500
DEFAULT_GROQ_MULTI_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
]
GROQ_MODEL_REQUEST_TOKEN_LIMITS = {
    "llama-3.3-70b-versatile": 12000,
    "openai/gpt-oss-120b": 8000,
    "openai/gpt-oss-20b": 8000,
    "llama-3.1-8b-instant": 6000,
    "qwen/qwen3-32b": 6000,
}


def _load_local_dotenv() -> None:
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


# Load .env if present.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_REPO_ROOT / ".env")
except Exception:
    _load_local_dotenv()

# Azure OpenAI
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

# OpenAI client (used for OpenAI-compatible providers)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Mistral
try:
    from mistralai import Mistral
except Exception:
    Mistral = None

# Google Gen AI SDK (Gemini API / AI Studio)
try:
    from google import genai
except Exception:
    genai = None

def _extract_json_block(text: str) -> Optional[str]:
    """Robustly extract a top-level JSON object from a text response."""
    if not text:
        return None

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    stack = []
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start != -1:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        continue
    return None


def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _normalize_foundry_chat_url(raw_endpoint: str, api_version: str) -> str:
    endpoint = (raw_endpoint or "").strip()
    if not endpoint:
        raise RuntimeError("Azure Foundry endpoint is empty")

    parsed = parse.urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError("Azure Foundry endpoint must be a full HTTPS URL")

    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        final_path = path
    elif path.endswith("/models"):
        final_path = f"{path}/chat/completions"
    else:
        final_path = f"{path}/models/chat/completions"

    query = parse.parse_qs(parsed.query, keep_blank_values=True)
    query.setdefault("api-version", [api_version])
    final_query = parse.urlencode(query, doseq=True)
    return parse.urlunparse((parsed.scheme, parsed.netloc, final_path, "", final_query, ""))


def _build_foundry_payload(
    model_name: str,
    system_message: str,
    user_prompt: str,
    temperature: float,
    max_tokens: Optional[int] = None,
    force_json: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if force_json:
        payload["response_format"] = {"type": "json_object"}
    return payload


def request_azure_foundry_chat(
    *,
    endpoint: str,
    api_key: str,
    api_version: str,
    model_name: str,
    system_message: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    force_json: bool = False,
    timeout_seconds: int = DEFAULT_AZURE_FOUNDRY_TIMEOUT_SECONDS,
) -> tuple[str, dict[str, Any]]:
    url = _normalize_foundry_chat_url(endpoint, api_version)
    payload = _build_foundry_payload(
        model_name=model_name,
        system_message=system_message,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        force_json=force_json,
    )
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "api-key": api_key,
        },
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Azure Foundry request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Azure Foundry request failed: {exc}") from exc

    return url, json.loads(raw)


_GROQ_MODEL_ALIASES = {
    "llama-3.1-70b-versatile": "llama-3.3-70b-versatile",
    "llama-3.1-70b-specdec": "llama-3.3-70b-specdec",
    "llama3-70b-8192": "llama-3.3-70b-versatile",
    "llama3-8b-8192": "llama-3.1-8b-instant",
}


def _resolve_groq_model_name(model_name: str) -> str:
    name = (model_name or "").strip()
    resolved = _GROQ_MODEL_ALIASES.get(name, name)
    if resolved != name:
        print(f"[groq] Model '{name}' is deprecated; using '{resolved}' instead.", flush=True)
    return resolved


class BaseBackend:
    provider: str
    model_name: str

    def name(self) -> str:
        return f"{self.provider}:{self.model_name}"

    def model_pool(self) -> list[str]:
        return [self.model_name]

    def active_model_name(self) -> str:
        return self.model_name

    def multi_model_mode(self) -> bool:
        return False

    def failover_events(self) -> list[dict[str, Any]]:
        return []

    def chat_json(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float = 0.0,
        force_json: bool = True,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        record_model_usage(provider=self.provider, model=self.active_model_name())
        text = self._chat_raw(
            system_message,
            user_prompt,
            temperature,
            force_json,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        if not force_json:
            return text or ""
        block = _extract_json_block(text or "")
        return block if block is not None else "{}"

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        raise NotImplementedError()


def _parse_model_list(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    models: list[str] = []
    for part in str(raw).split(","):
        name = part.strip()
        if name and name not in models:
            models.append(name)
    return models


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except Exception:
        return default
    return max(minimum, value)


def _groq_model_request_limit(model_name: str) -> int | None:
    return GROQ_MODEL_REQUEST_TOKEN_LIMITS.get(_resolve_groq_model_name(model_name))


def _prioritize_groq_models(models: list[str]) -> list[str]:
    indexed = list(enumerate(models))
    indexed.sort(key=lambda item: (-(_groq_model_request_limit(item[1]) or 0), item[0]))
    return [model for _idx, model in indexed]


def _estimate_groq_request_tokens(system_message: str, user_prompt: str, max_tokens: Optional[int]) -> int:
    prompt = f"{system_message or ''}\n\n{user_prompt or ''}".strip()
    prompt_tokens = max(1, len(prompt) // 4) if prompt else 0
    if max_tokens is None:
        completion_reserve = _env_int(
            "GROQ_MULTI_COMPLETION_TOKEN_RESERVE",
            DEFAULT_GROQ_COMPLETION_TOKEN_RESERVE,
            minimum=0,
        )
    else:
        try:
            completion_reserve = max(0, int(max_tokens))
        except Exception:
            completion_reserve = DEFAULT_GROQ_COMPLETION_TOKEN_RESERVE
    return prompt_tokens + completion_reserve


def _groq_multi_same_model_retries() -> int:
    return _env_int(
        "GROQ_MULTI_SAME_MODEL_RETRIES",
        DEFAULT_GROQ_MULTI_SAME_MODEL_RETRIES,
        minimum=0,
    )


def groq_experiment_model_pool() -> list[str]:
    configured = _parse_model_list(os.getenv("GROQ_MULTI_MODELS"))
    if not configured:
        configured = list(DEFAULT_GROQ_MULTI_MODELS)

    resolved: list[str] = []
    for model_name in configured:
        normalized = _resolve_groq_model_name(model_name)
        if normalized and normalized not in resolved:
            resolved.append(normalized)
    return _prioritize_groq_models(resolved)[:MAX_GROQ_MULTI_MODELS]


def use_groq_multi_model_mode(model: Optional[str]) -> bool:
    explicit_model = str(model or "").strip()
    if explicit_model.lower() == GROQ_MULTI_MODEL_SENTINEL:
        return True
    if explicit_model:
        return False
    raw = str(os.getenv("GROQ_MULTI_MODEL_MODE", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


class FailoverBackend(BaseBackend):
    def __init__(self, *, provider: str, backends: list[BaseBackend], label: str = GROQ_MULTI_MODEL_SENTINEL):
        if not backends:
            raise RuntimeError("FailoverBackend requires at least one backend")
        self.provider = provider
        self._backends = backends
        self._label = label
        self._active_index = 0
        self._switch_events: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self.active_model_name()

    def name(self) -> str:
        return f"{self.provider}:{self._label}"

    def model_pool(self) -> list[str]:
        return [backend.model_name for backend in self._backends]

    def active_model_name(self) -> str:
        return self._backends[self._active_index].model_name

    def multi_model_mode(self) -> bool:
        return True

    def failover_events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._switch_events]

    def _candidate_indices_for_request(
        self,
        system_message: str,
        user_prompt: str,
        max_tokens: Optional[int],
    ) -> list[int]:
        indices = list(range(len(self._backends)))
        if self.provider != "groq":
            return indices

        estimated_tokens = _estimate_groq_request_tokens(system_message, user_prompt, max_tokens)
        viable: list[int] = []
        skipped: list[tuple[str, int]] = []
        for index in indices:
            backend = self._backends[index]
            limit = _groq_model_request_limit(backend.model_name)
            if limit is not None and estimated_tokens > limit:
                skipped.append((backend.model_name, limit))
                continue
            viable.append(index)

        for model_name, limit in skipped:
            log_line(
                f"[llm_failover] provider={self.provider} skipping model {model_name}: "
                f"estimated_request_tokens={estimated_tokens} exceeds configured_limit={limit}"
            )

        if viable:
            return viable

        largest_index = max(
            indices,
            key=lambda idx: _groq_model_request_limit(self._backends[idx].model_name) or 0,
        )
        log_line(
            f"[llm_failover] provider={self.provider} no configured model appears to fit "
            f"estimated_request_tokens={estimated_tokens}; trying largest model "
            f"{self._backends[largest_index].model_name} for provider-confirmed handling"
        )
        return [largest_index]

    def _record_switch(self, *, from_index: int, to_index: int, reason: str, error: str) -> None:
        old_model = self._backends[from_index].model_name
        new_model = self._backends[to_index].model_name
        event = {
            "reason": reason,
            "from_model": old_model,
            "to_model": new_model,
            "error": error,
        }
        self._switch_events.append(event)
        record_model_switch(
            provider=self.provider,
            from_model=old_model,
            to_model=new_model,
            reason=reason,
            error=error,
        )
        log_line(
            f"[llm_failover] provider={self.provider} switching model "
            f"{old_model} -> {new_model} after {reason}: {error}"
        )

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        raise NotImplementedError("FailoverBackend uses chat_json directly")

    def chat_json(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float = 0.0,
        force_json: bool = True,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        last_error: Exception | None = None
        candidate_indices = self._candidate_indices_for_request(system_message, user_prompt, max_tokens)
        same_model_retries = _groq_multi_same_model_retries() if self.provider == "groq" else 0
        for position, backend_index in enumerate(candidate_indices):
            self._active_index = backend_index
            backend = self._backends[backend_index]
            try:
                return call_with_backoff(
                    lambda: backend.chat_json(
                        system_message=system_message,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        force_json=force_json,
                        max_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                    ),
                    max_retries=same_model_retries,
                    name=f"{self.provider}:{backend.model_name}",
                )
            except Exception as exc:
                last_error = exc
                next_index = candidate_indices[position + 1] if position + 1 < len(candidate_indices) else None

                if self.provider == "groq" and is_request_too_large_error(exc):
                    if next_index is None:
                        raise RuntimeError(
                            f"groq_prompt_too_large: model {backend.model_name} rejected the request; {exc}"
                        ) from exc
                    self._record_switch(
                        from_index=backend_index,
                        to_index=next_index,
                        reason="prompt_too_large",
                        error=str(exc),
                    )
                    continue

                should_retry, reason = classify_retryable_error(exc)
                if reason != "rate_limit" or not should_retry or next_index is None:
                    raise

                self._record_switch(
                    from_index=backend_index,
                    to_index=next_index,
                    reason=reason,
                    error=str(exc),
                )
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failover backend exhausted without a concrete error")


class AzureBackend(BaseBackend):
    provider = "azure"

    def __init__(self, deployment: Optional[str] = None):
        if AzureOpenAI is None:
            raise RuntimeError("openai package is not installed")

        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-dspy")

        if not api_key or not endpoint:
            raise RuntimeError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT missing")

        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        self.model_name = deployment

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        kwargs = dict(
            model=self.model_name,  # deployment name
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        )
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout_seconds is not None:
            kwargs["timeout"] = timeout_seconds
        r = self._client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""


class MistralBackend(BaseBackend):
    provider = "mistral"

    def __init__(self, model: Optional[str] = None):
        if Mistral is None:
            raise RuntimeError("mistralai package is not installed")

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY missing")

        self.model_name = model or os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        self._client = Mistral(api_key=api_key)

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        del max_tokens, timeout_seconds
        messages = [
            {"role": "system", "content": system_message + " Always return a single JSON object."},
            {"role": "user", "content": user_prompt},
        ]
        resp = self._client.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class GroqBackend(BaseBackend):
    provider = "groq"

    def __init__(self, model: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY missing")

        configured_model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.model_name = _resolve_groq_model_name(configured_model)
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        # Groq is OpenAI-compatible, but JSON enforcement can vary by model.
        messages = [
            {"role": "system", "content": system_message + " Always return a single JSON object."},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout_seconds is not None:
            kwargs["timeout"] = timeout_seconds
        r = self._client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""


class TogetherBackend(BaseBackend):
    provider = "together"

    def __init__(self, model: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")

        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY missing")

        base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1").rstrip("/")
        self.model_name = model or os.getenv("TOGETHER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_message + " Always return a single JSON object."},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout_seconds is not None:
            kwargs["timeout"] = timeout_seconds
        r = self._client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""


class FireworksBackend(BaseBackend):
    provider = "fireworks"

    def __init__(self, model: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")

        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY missing")

        base_url = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1").rstrip("/")
        self.model_name = model or os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct")
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_message + " Always return a single JSON object."},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout_seconds is not None:
            kwargs["timeout"] = timeout_seconds
        r = self._client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""


class GeminiBackend(BaseBackend):
    provider = "gemini"

    def __init__(self, model: Optional[str] = None):
        if genai is None:
            raise RuntimeError("google-genai package is not installed")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY missing")

        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._client = genai.Client(api_key=api_key)

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        del temperature, force_json, max_tokens, timeout_seconds
        # Gemini SDK uses generate_content; we'll combine system + user text.
        # Keep it simple and rely on your JSON extraction.
        combined = f"SYSTEM:\n{system_message}\n\nUSER:\n{user_prompt}\n\nReturn a single JSON object only."
        resp = self._client.models.generate_content(
            model=self.model_name,
            contents=combined,
        )
        return (resp.text or "").strip()


class AzureFoundryBackend(BaseBackend):
    """
    Azure AI Foundry Model Inference endpoint backend.
    Endpoint form: https://<resource-name>.services.ai.azure.com/models
    Route: POST /chat/completions?api-version=2024-05-01-preview
    """

    provider = "azure_foundry"

    def __init__(self, model: Optional[str] = None):
        api_key = _env_first("AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDARY_API_KEY")
        endpoint = _env_first("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDARY_ENDPOINT")
        api_version = _env_first(
            "AZURE_FOUNDRY_API_VERSION",
            "AZURE_FOUNDARY_API_VERSION",
            default=DEFAULT_AZURE_FOUNDRY_API_VERSION,
        )
        timeout_raw = _env_first(
            "AZURE_FOUNDRY_TIMEOUT_SECONDS",
            "AZURE_FOUNDARY_TIMEOUT_SECONDS",
            default=str(DEFAULT_AZURE_FOUNDRY_TIMEOUT_SECONDS),
        )
        model_name = model or _env_first("AZURE_FOUNDRY_MODEL", "AZURE_FOUNDARY_MODEL", default="DeepSeek-R1-0528")

        if not api_key or not endpoint:
            raise RuntimeError(
                "AZURE_FOUNDRY_API_KEY or AZURE_FOUNDRY_ENDPOINT missing "
                "(legacy AZURE_FOUNDARY_* names are also supported)"
            )

        self._api_key = api_key
        self._url = _normalize_foundry_chat_url(endpoint, api_version)
        self._api_version = api_version
        try:
            self._timeout_seconds = max(1, int(str(timeout_raw).strip()))
        except Exception:
            self._timeout_seconds = DEFAULT_AZURE_FOUNDRY_TIMEOUT_SECONDS
        self.model_name = model_name

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        _url, data = request_azure_foundry_chat(
            endpoint=self._url,
            api_key=self._api_key,
            api_version=self._api_version,
            model_name=self.model_name,
            system_message=system_message,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            force_json=force_json,
            timeout_seconds=timeout_seconds or self._timeout_seconds,
        )
        return data["choices"][0]["message"].get("content", "") or ""


def make_backend(provider: Optional[str] = None, model: Optional[str] = None) -> BaseBackend:
    provider = (provider or os.getenv("LLM_PROVIDER", "azure")).lower()

    if provider == "azure":
        return AzureBackend(deployment=model)
    if provider == "mistral":
        return MistralBackend(model=model)
    if provider == "groq":
        if use_groq_multi_model_mode(model):
            return FailoverBackend(
                provider="groq",
                backends=[GroqBackend(model=model_name) for model_name in groq_experiment_model_pool()],
            )
        return GroqBackend(model=model)
    if provider in ("fireworks", "fireworks_ai"):
        return FireworksBackend(model=model)
    if provider in ("together", "together_ai"):
        return TogetherBackend(model=model)
    if provider in ("google", "gemini"):
        return GeminiBackend(model=model)
    if provider in ("azure_foundry", "foundry", "azure-deepseek", "deepseek"):
        return AzureFoundryBackend(model=model)
    if provider in ("lmstudio", "local"):
        return LMStudioBackend(model=model)
    if provider in ("lmstudio_qwen", "lmstudio_native", "local_qwen"):
        return LMStudioNativeChatBackend(model=model)

    raise RuntimeError(f"Unknown LLM_PROVIDER: {provider}")


class LMStudioBackend(BaseBackend):
    provider = "lmstudio"

    def __init__(self, model: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")

        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
        self.model_name = model or os.getenv("LMSTUDIO_MODEL", "meta-llama-3.1-8b-instruct")

        # LM Studio does not require a real key, but OpenAI client expects a string
        self._client = OpenAI(api_key=os.getenv("LMSTUDIO_API_KEY", "lmstudio"), base_url=base_url)

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_message + " Return a single JSON object only."},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout_seconds is not None:
            kwargs["timeout"] = timeout_seconds
        r = self._client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""


def _extract_lmstudio_native_text(data: Any) -> str:
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return "" if data is None else str(data)

    for key in ("output", "response", "text", "content"):
        value = data.get(key)
        if isinstance(value, str):
            return value

    message = data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message.get("content") or ""

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message.get("content") or ""
            if isinstance(first.get("text"), str):
                return first.get("text") or ""

    return json.dumps(data, ensure_ascii=False)


class LMStudioNativeChatBackend(BaseBackend):
    provider = "lmstudio_qwen"

    def __init__(self, model: Optional[str] = None):
        self.model_name = model or os.getenv("LMSTUDIO_QWEN_MODEL", "qwen2.5-3b-instruct.gguf")
        self._url = os.getenv("LMSTUDIO_QWEN_CHAT_URL", "http://localhost:1234/api/v1/chat").strip()
        if not self._url:
            raise RuntimeError("LMSTUDIO_QWEN_CHAT_URL is empty")

    def _chat_raw(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float,
        force_json: bool,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> str:
        # LM Studio's native /api/v1/chat endpoint rejects OpenAI-style
        # max_tokens, so callers may pass it but this backend must omit it.
        del max_tokens
        system_prompt = system_message
        if force_json:
            system_prompt = system_prompt + " Return a single JSON object only."
        payload: dict[str, Any] = {
            "model": self.model_name,
            "system_prompt": system_prompt,
            "input": user_prompt,
            "temperature": temperature,
        }

        req = request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=timeout_seconds or DEFAULT_AZURE_FOUNDRY_TIMEOUT_SECONDS) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LM Studio native chat request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LM Studio native chat request failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except Exception:
            return raw
        return _extract_lmstudio_native_text(data)
