from __future__ import annotations

import time
from typing import Any

from src.config import CONFIG
from src.core.run_logging import finish_llm_trace, log_line, start_llm_trace
from src.llm.autogen_runner import run_autogen_agent
from src.llm.backends import BaseBackend

_LOCAL_TIMEOUT_PROVIDERS = {"lmstudio", "lmstudio_qwen"}


def _active_model_name(backend: BaseBackend) -> str:
    try:
        return str(backend.active_model_name())
    except Exception:
        return str(getattr(backend, "model_name", "") or "")


def _log_with_logger(logger: Any, message: str, *, is_error: bool = False) -> None:
    if logger is None:
        return
    try:
        if is_error and hasattr(logger, "error"):
            logger.error(message)
        elif hasattr(logger, "info"):
            logger.info(message)
        elif callable(logger):
            logger(message)
    except Exception:
        return


def call_autogen_gateway(
    *,
    role_name: str,
    system_message: str,
    user_prompt: str,
    backend: BaseBackend,
    temperature: float = 0.0,
    timeout_s: int | float | None = None,
    force_json: bool = True,
    logger: Any = None,
    metadata: dict | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Central AutoGen-backed entry point for AutoLLMCompose LLM reasoning calls.

    Stage code should call this gateway, or an llm_call wrapper backed by it.
    Direct provider/backend calls stay in the backend and AutoGen model-client
    adapter layers.
    """
    provider = str(getattr(backend, "provider", "") or "")
    model = _active_model_name(backend)
    prompt_chars = len(user_prompt or "")
    timeout_seconds = timeout_s
    if timeout_seconds is None and provider not in _LOCAL_TIMEOUT_PROVIDERS:
        configured_timeout = getattr(CONFIG, "remote_llm_timeout_seconds", None)
        if configured_timeout and configured_timeout > 0:
            timeout_seconds = configured_timeout
    trace_id: str | None = None
    started_at = time.perf_counter()
    trace_metadata = dict(metadata or {})
    trace_metadata.update(
        {
            "gateway": "autogen_gateway",
            "provider": provider,
            "model": model,
        }
    )

    if CONFIG.llm_debug_enabled:
        trace_id = start_llm_trace(
            role_name=role_name,
            provider=provider,
            model=model,
            system_message=system_message,
            prompt=user_prompt,
            temperature=temperature,
            force_json=force_json,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            metadata=trace_metadata,
        )

    try:
        response = run_autogen_agent(
            backend=backend,
            role_name=role_name,
            system_message=system_message,
            prompt=user_prompt,
            temperature=temperature,
            force_json=force_json,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            trace_enabled=False,
        )
        duration_seconds = round(time.perf_counter() - started_at, 3)
        if CONFIG.llm_debug_enabled:
            finish_llm_trace(trace_id, response_text=response, duration_seconds=duration_seconds)
        _log_with_logger(
            logger,
            (
                f"[llm_gateway] success role={role_name} provider={provider} model={model} "
                f"prompt_chars={prompt_chars} response_chars={len(response or '')} "
                f"duration_s={duration_seconds} timeout_s={timeout_seconds} "
                f"temperature={temperature} force_json={force_json}"
            ),
        )
        return response
    except Exception as exc:
        duration_seconds = round(time.perf_counter() - started_at, 3)
        error_message = str(exc)
        if CONFIG.llm_debug_enabled:
            finish_llm_trace(trace_id, error=error_message, duration_seconds=duration_seconds)
        _log_with_logger(
            logger,
            (
                f"[llm_gateway] failed role={role_name} provider={provider} model={model} "
                f"prompt_chars={prompt_chars} duration_s={duration_seconds} "
                f"timeout_s={timeout_seconds} temperature={temperature} "
                f"force_json={force_json} error={error_message}"
            ),
            is_error=True,
        )
        if logger is None and not CONFIG.llm_debug_enabled:
            log_line(
                f"[llm_gateway] failed role={role_name} provider={provider} model={model} "
                f"duration_s={duration_seconds} error={error_message}"
            )
        raise
