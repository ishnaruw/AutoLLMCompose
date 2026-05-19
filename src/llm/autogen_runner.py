from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, AsyncGenerator, Mapping, Sequence

from src.config import CONFIG
from src.core.run_logging import finish_llm_trace, start_llm_trace
from src.llm.backends import BaseBackend

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core import CancellationToken
    from autogen_core.models import (
        AssistantMessage,
        ChatCompletionClient,
        CreateResult,
        LLMMessage,
        ModelFamily,
        ModelInfo,
        RequestUsage,
        SystemMessage,
        UserMessage,
    )
except Exception:
    AssistantAgent = None
    CancellationToken = None
    ChatCompletionClient = object
    CreateResult = None
    LLMMessage = Any
    ModelFamily = None
    ModelInfo = dict
    RequestUsage = None
    SystemMessage = None
    UserMessage = None
    AssistantMessage = None


def _require_autogen() -> None:
    if AssistantAgent is None or CreateResult is None or RequestUsage is None or ModelFamily is None:
        raise RuntimeError(
            "AutoGen packages are not available in the active interpreter. "
            "Run the pipeline with the project virtualenv, for example: AutoLLMCompose/.venv/bin/python ..."
        )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _messages_to_backend_payload(messages: Sequence[LLMMessage]) -> tuple[str, str]:
    system_parts: list[str] = []
    convo_parts: list[tuple[str, str]] = []

    for message in messages:
        content = _content_to_text(getattr(message, "content", ""))
        if not content:
            continue

        if SystemMessage is not None and isinstance(message, SystemMessage):
            system_parts.append(content)
            continue

        role = "user"
        if UserMessage is not None and isinstance(message, UserMessage):
            role = "user"
        elif AssistantMessage is not None and isinstance(message, AssistantMessage):
            role = "assistant"
        else:
            role = str(getattr(message, "source", type(message).__name__)).strip().lower() or "user"
        convo_parts.append((role, content))

    system_message = "\n\n".join(part for part in system_parts if part).strip()
    if len(convo_parts) == 1 and convo_parts[0][0] == "user":
        user_prompt = convo_parts[0][1]
    else:
        user_prompt = "\n\n".join(f"{role.upper()}:\n{content}" for role, content in convo_parts if content).strip()

    return system_message, user_prompt


def _estimate_token_count(text: str) -> int:
    text = text or ""
    return max(1, len(text) // 4) if text else 0


def _usage(prompt_text: str, output_text: str) -> RequestUsage:
    return RequestUsage(
        prompt_tokens=_estimate_token_count(prompt_text),
        completion_tokens=_estimate_token_count(output_text),
    )


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            error["value"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "value" in error:
        raise error["value"]
    return result.get("value")


def _extract_task_result_text(task_result: Any) -> str:
    messages = list(getattr(task_result, "messages", []) or [])
    for message in reversed(messages):
        content = _content_to_text(getattr(message, "content", ""))
        if content:
            return content
    return ""


class BackendChatCompletionClient(ChatCompletionClient):
    def __init__(
        self,
        backend: BaseBackend,
        *,
        temperature: float = 0.0,
        force_json: bool = True,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ):
        _require_autogen()
        self._backend = backend
        self._temperature = temperature
        self._force_json = force_json
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0
        self._model_info: ModelInfo = {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": ModelFamily.ANY,
            "structured_output": False,
            "multiple_system_messages": True,
        }

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        del tools, tool_choice, cancellation_token

        system_message, user_prompt = _messages_to_backend_payload(messages)
        temperature = float(extra_create_args.get("temperature", self._temperature) or 0.0)
        force_json = self._force_json if json_output is None else bool(json_output)
        max_tokens = extra_create_args.get("max_tokens", self._max_tokens)
        timeout_seconds = extra_create_args.get("timeout_seconds", self._timeout_seconds)
        output_text = self._backend.chat_json(
            system_message=system_message,
            user_prompt=user_prompt,
            temperature=temperature,
            force_json=force_json,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        usage = _usage(f"{system_message}\n\n{user_prompt}".strip(), output_text)
        self._prompt_tokens_total += usage.prompt_tokens
        self._completion_tokens_total += usage.completion_tokens
        return CreateResult(
            finish_reason="stop",
            content=output_text or "",
            usage=usage,
            cached=False,
            logprobs=None,
            thought=None,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Any = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[str | CreateResult, None]:
        result = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield result

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._prompt_tokens_total,
            completion_tokens=self._completion_tokens_total,
        )

    def total_usage(self) -> RequestUsage:
        return self.actual_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        del tools
        system_message, user_prompt = _messages_to_backend_payload(messages)
        return _estimate_token_count(f"{system_message}\n\n{user_prompt}".strip())

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        del tools
        return max(0, 1_000_000 - self.count_tokens(messages))

    @property
    def capabilities(self) -> Any:
        return self._model_info

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info


def run_autogen_agent(
    *,
    backend: BaseBackend,
    role_name: str,
    system_message: str,
    prompt: str,
    temperature: float = 0.0,
    force_json: bool = True,
    max_tokens: int | None = None,
    timeout_seconds: float | None = None,
    trace_enabled: bool = True,
) -> str:
    _require_autogen()
    trace_id: str | None = None
    started_at = time.perf_counter()
    if trace_enabled and CONFIG.llm_debug_enabled:
        trace_id = start_llm_trace(
            role_name=role_name,
            provider=getattr(backend, "provider", ""),
            model=getattr(backend, "model_name", ""),
            system_message=system_message,
            prompt=prompt,
            temperature=temperature,
            force_json=force_json,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
    model_client = BackendChatCompletionClient(
        backend=backend,
        temperature=temperature,
        force_json=force_json,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )
    agent = AssistantAgent(
        name=(role_name or "agent").replace(" ", "_"),
        description=f"AutoGen wrapper for {role_name or 'llm'}",
        model_client=model_client,
        system_message=system_message,
        reflect_on_tool_use=False,
        max_tool_iterations=1,
    )
    try:
        task_result = _run_coro_sync(agent.run(task=prompt, output_task_messages=False))
        response_text = _extract_task_result_text(task_result)
        if trace_enabled and CONFIG.llm_debug_enabled:
            finish_llm_trace(
                trace_id,
                response_text=response_text,
                duration_seconds=round(time.perf_counter() - started_at, 3),
            )
        return response_text
    except Exception as exc:
        if trace_enabled and CONFIG.llm_debug_enabled:
            finish_llm_trace(
                trace_id,
                error=str(exc),
                duration_seconds=round(time.perf_counter() - started_at, 3),
            )
        raise
