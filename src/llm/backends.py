# src/llm/backends.py

import os
import json
from typing import Optional

# Load .env if present (safe to call even if python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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

# Requests for Azure Foundry Model Inference endpoint
try:
    import requests
except Exception:
    requests = None


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

        self.model_name = model or os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
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
        if requests is None:
            raise RuntimeError("requests package is not installed")

        api_key = os.getenv("AZURE_FOUNDARY_API_KEY")
        endpoint = os.getenv("AZURE_FOUNDARY_ENDPOINT")
        api_version = os.getenv("AZURE_FOUNDARY_API_VERSION", "2024-05-01-preview")
        model_name = model or os.getenv("AZURE_FOUNDARY_MODEL", "deepseek-r1")

        if not api_key or not endpoint:
            raise RuntimeError("AZURE_FOUNDARY_API_KEY or AZURE_FOUNDARY_ENDPOINT missing")

        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._api_version = api_version
        self.model_name = model_name

    def _chat_raw(self, system_message: str, user_prompt: str, temperature: float, force_json: bool) -> str:
        url = f"{self._endpoint}/models/chat/completions"
        params = {"api-version": self._api_version}
        headers = {
            "Content-Type": "application/json",
            "api-key": self._api_key,
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        }

        if force_json:
            payload["response_format"] = {"type": "json_object"}

        r = requests.post(url, params=params, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"].get("content", "") or ""


def make_backend(provider: Optional[str] = None, model: Optional[str] = None) -> BaseBackend:
    provider = (provider or os.getenv("LLM_PROVIDER", "azure")).lower()

    if provider == "azure":
        return AzureBackend(deployment=model)
    if provider == "mistral":
        return MistralBackend(model=model)
    if provider == "groq":
        return GroqBackend(model=model)
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

