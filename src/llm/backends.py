# src/llm/backends.py

import json
import os
from pathlib import Path
from typing import Any, Optional
from urllib import error, parse, request

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AZURE_FOUNDRY_API_VERSION = "2024-05-01-preview"
DEFAULT_AZURE_FOUNDRY_TIMEOUT_SECONDS = 300


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

# OpenAI client (used for Groq too)
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

    def chat_json(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float = 0.0,
        force_json: bool = True,
    ) -> str:
        text = self._chat_raw(system_message, user_prompt, temperature, force_json)
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
    ) -> str:
        raise NotImplementedError()


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

    def _chat_raw(self, system_message: str, user_prompt: str, temperature: float, force_json: bool) -> str:
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

    def _chat_raw(self, system_message: str, user_prompt: str, temperature: float, force_json: bool) -> str:
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

    def _chat_raw(self, system_message: str, user_prompt: str, temperature: float, force_json: bool) -> str:
        # Groq is OpenAI-compatible, but JSON enforcement can vary by model.
        messages = [
            {"role": "system", "content": system_message + " Always return a single JSON object."},
            {"role": "user", "content": user_prompt},
        ]
        r = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
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

    def _chat_raw(self, system_message: str, user_prompt: str, temperature: float, force_json: bool) -> str:
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

    def _chat_raw(self, system_message: str, user_prompt: str, temperature: float, force_json: bool) -> str:
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

    def _chat_raw(self, system_message: str, user_prompt: str, temperature: float, force_json: bool) -> str:
        _url, data = request_azure_foundry_chat(
            endpoint=self._url,
            api_key=self._api_key,
            api_version=self._api_version,
            model_name=self.model_name,
            system_message=system_message,
            user_prompt=user_prompt,
            temperature=temperature,
            force_json=force_json,
            timeout_seconds=self._timeout_seconds,
        )
        return data["choices"][0]["message"].get("content", "") or ""


def make_backend(provider: Optional[str] = None, model: Optional[str] = None) -> BaseBackend:
    provider = (provider or os.getenv("LLM_PROVIDER", "azure")).lower()

    if provider == "azure":
        return AzureBackend(deployment=model)
    if provider == "mistral":
        return MistralBackend(model=model)
    if provider == "groq":
        return GroqBackend(model=model)
    if provider in ("together", "together_ai"):
        return TogetherBackend(model=model)
    if provider in ("google", "gemini"):
        return GeminiBackend(model=model)
    if provider in ("azure_foundry", "foundry", "azure-deepseek", "deepseek"):
        return AzureFoundryBackend(model=model)
    if provider in ("lmstudio", "local"):
        return LMStudioBackend(model=model)

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

    def _chat_raw(self, system_message: str, user_prompt: str, temperature: float, force_json: bool) -> str:
        messages = [
            {"role": "system", "content": system_message + " Return a single JSON object only."},
            {"role": "user", "content": user_prompt},
        ]
        r = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        return r.choices[0].message.content or ""
