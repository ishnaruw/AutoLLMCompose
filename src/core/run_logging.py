from __future__ import annotations

import json
import re
from threading import Lock
from datetime import datetime
from pathlib import Path

_RUN_LOG_PATH: Path | None = None
_LLM_DEBUG_DIR: Path | None = None
_MODEL_USAGE_PATH: Path | None = None
_WARNING_LOG_PATH: Path | None = None
_ERROR_LOG_PATH: Path | None = None
_INVALID_CASES_LOG_PATH: Path | None = None
_LLM_DEBUG_COUNTER = 0
_LLM_DEBUG_LOCK = Lock()
_MODEL_USAGE_LOCK = Lock()
_STRUCTURED_LOG_LOCK = Lock()


def configure_run_log(path: str | Path | None) -> Path | None:
    global _RUN_LOG_PATH, _LLM_DEBUG_DIR, _MODEL_USAGE_PATH, _WARNING_LOG_PATH, _ERROR_LOG_PATH, _INVALID_CASES_LOG_PATH, _LLM_DEBUG_COUNTER
    if path is None:
        _RUN_LOG_PATH = None
        _LLM_DEBUG_DIR = None
        _MODEL_USAGE_PATH = None
        _WARNING_LOG_PATH = None
        _ERROR_LOG_PATH = None
        _INVALID_CASES_LOG_PATH = None
        _LLM_DEBUG_COUNTER = 0
        return None
    _RUN_LOG_PATH = Path(path)
    _RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LLM_DEBUG_DIR = _RUN_LOG_PATH.parent / "llm_debug"
    _LLM_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    _MODEL_USAGE_PATH = _RUN_LOG_PATH.parent / "model_usage.json"
    _WARNING_LOG_PATH = _RUN_LOG_PATH.parent / "warnings.log"
    _ERROR_LOG_PATH = _RUN_LOG_PATH.parent / "errors.log"
    _INVALID_CASES_LOG_PATH = _RUN_LOG_PATH.parent / "invalid_cases.log"
    _LLM_DEBUG_COUNTER = 0
    return _RUN_LOG_PATH


def clear_run_log() -> None:
    global _RUN_LOG_PATH, _LLM_DEBUG_DIR, _MODEL_USAGE_PATH, _WARNING_LOG_PATH, _ERROR_LOG_PATH, _INVALID_CASES_LOG_PATH, _LLM_DEBUG_COUNTER
    _RUN_LOG_PATH = None
    _LLM_DEBUG_DIR = None
    _MODEL_USAGE_PATH = None
    _WARNING_LOG_PATH = None
    _ERROR_LOG_PATH = None
    _INVALID_CASES_LOG_PATH = None
    _LLM_DEBUG_COUNTER = 0


def current_run_log_path() -> Path | None:
    return _RUN_LOG_PATH


def current_model_usage_path() -> Path | None:
    return _MODEL_USAGE_PATH


def current_warning_log_path() -> Path | None:
    return _WARNING_LOG_PATH


def current_error_log_path() -> Path | None:
    return _ERROR_LOG_PATH


def current_invalid_cases_log_path() -> Path | None:
    return _INVALID_CASES_LOG_PATH


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_line(message: str, *, echo: bool = True) -> None:
    if echo:
        print(message)
    if _RUN_LOG_PATH is None:
        return
    with _RUN_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{_timestamp()} {message}\n")


def _structured_log_path(level: str) -> Path:
    if level == "warning" and _WARNING_LOG_PATH is not None:
        return _WARNING_LOG_PATH
    if level == "error" and _ERROR_LOG_PATH is not None:
        return _ERROR_LOG_PATH
    return Path("results/logs") / f"{level}s.log"


def _invalid_cases_log_path() -> Path:
    if _INVALID_CASES_LOG_PATH is not None:
        return _INVALID_CASES_LOG_PATH
    return Path("results/logs") / "invalid_cases.log"


def _write_structured_log(level: str, payload: dict) -> None:
    event = dict(payload or {})
    event.setdefault("level", level)
    event.setdefault("timestamp", _timestamp())
    path = _structured_log_path(level)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _STRUCTURED_LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def log_warning_event(payload: dict) -> None:
    _write_structured_log("warning", payload)


def log_error_event(payload: dict) -> None:
    _write_structured_log("error", payload)


def log_invalid_case_event(payload: dict) -> None:
    event = dict(payload or {})
    event.setdefault("level", "warning")
    event.setdefault("event_type", "invalid_evaluation_case")
    event.setdefault("exclude_from_ranking_eval", True)
    event.setdefault("timestamp", _timestamp())
    path = _invalid_cases_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _STRUCTURED_LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def _safe_name(text: str) -> str:
    text = (text or "llm").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or "llm"


def _debug_path(call_id: str, suffix: str) -> Path | None:
    if _LLM_DEBUG_DIR is None:
        return None
    return _LLM_DEBUG_DIR / f"{call_id}_{suffix}"


def _load_model_usage() -> dict:
    if _MODEL_USAGE_PATH is None or not _MODEL_USAGE_PATH.exists():
        return {}
    try:
        data = json.loads(_MODEL_USAGE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def configure_model_usage(
    *,
    provider: str,
    model_tag: str,
    multi_model_mode: bool,
    model_pool: list[str],
) -> Path | None:
    if _MODEL_USAGE_PATH is None:
        return None
    payload = {
        "provider": provider,
        "model_tag": model_tag,
        "multi_model_mode": multi_model_mode,
        "model_pool": list(model_pool),
        "active_model": model_pool[0] if model_pool else None,
        "models_seen": [],
        "switch_events": [],
        "call_counts": {},
        "last_used_at": {},
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
    }
    _MODEL_USAGE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return _MODEL_USAGE_PATH


def record_model_usage(*, provider: str, model: str, role_name: str | None = None) -> None:
    if _MODEL_USAGE_PATH is None or not model:
        return
    with _MODEL_USAGE_LOCK:
        payload = _load_model_usage()
        if not payload:
            payload = {
                "provider": provider,
                "model_tag": f"{provider}:{model}",
                "multi_model_mode": False,
                "model_pool": [model],
                "active_model": model,
                "models_seen": [],
                "switch_events": [],
                "call_counts": {},
                "last_used_at": {},
                "created_at": _timestamp(),
            }
        payload["provider"] = provider
        payload["active_model"] = model
        models_seen = list(payload.get("models_seen", []))
        if model not in models_seen:
            models_seen.append(model)
        payload["models_seen"] = models_seen

        call_counts = payload.get("call_counts", {})
        if not isinstance(call_counts, dict):
            call_counts = {}
        entry = call_counts.get(model, {"total": 0, "by_role": {}})
        entry["total"] = int(entry.get("total", 0)) + 1
        by_role = entry.get("by_role", {})
        if not isinstance(by_role, dict):
            by_role = {}
        if role_name:
            by_role[role_name] = int(by_role.get(role_name, 0)) + 1
        entry["by_role"] = by_role
        call_counts[model] = entry
        payload["call_counts"] = call_counts

        last_used_at = payload.get("last_used_at", {})
        if not isinstance(last_used_at, dict):
            last_used_at = {}
        last_used_at[model] = _timestamp()
        payload["last_used_at"] = last_used_at
        payload["updated_at"] = _timestamp()
        _MODEL_USAGE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def record_model_switch(*, provider: str, from_model: str, to_model: str, reason: str, error: str | None = None) -> None:
    if _MODEL_USAGE_PATH is None:
        return
    with _MODEL_USAGE_LOCK:
        payload = _load_model_usage()
        if not payload:
            payload = {
                "provider": provider,
                "model_tag": f"{provider}:multi",
                "multi_model_mode": True,
                "model_pool": [from_model, to_model],
                "models_seen": [],
                "switch_events": [],
                "call_counts": {},
                "last_used_at": {},
                "created_at": _timestamp(),
            }
        payload["provider"] = provider
        payload["active_model"] = to_model
        switch_events = payload.get("switch_events", [])
        if not isinstance(switch_events, list):
            switch_events = []
        switch_events.append(
            {
                "timestamp": _timestamp(),
                "from_model": from_model,
                "to_model": to_model,
                "reason": reason,
                "error": error,
            }
        )
        payload["switch_events"] = switch_events
        models_seen = list(payload.get("models_seen", []))
        for model in (from_model, to_model):
            if model and model not in models_seen:
                models_seen.append(model)
        payload["models_seen"] = models_seen
        payload["updated_at"] = _timestamp()
        _MODEL_USAGE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def start_llm_trace(
    *,
    role_name: str,
    provider: str,
    model: str,
    system_message: str,
    prompt: str,
    temperature: float,
    force_json: bool,
    max_tokens: int | None,
    timeout_seconds: float | None,
) -> str | None:
    global _LLM_DEBUG_COUNTER
    if _LLM_DEBUG_DIR is None:
        return None

    with _LLM_DEBUG_LOCK:
        _LLM_DEBUG_COUNTER += 1
        call_id = f"{_LLM_DEBUG_COUNTER:04d}_{_safe_name(role_name)}"

    meta = {
        "call_id": call_id,
        "started_at": _timestamp(),
        "status": "running",
        "role_name": role_name,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "force_json": force_json,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
        "system_chars": len(system_message or ""),
        "prompt_chars": len(prompt or ""),
    }
    meta_path = _debug_path(call_id, "meta.json")
    system_path = _debug_path(call_id, "system.txt")
    prompt_path = _debug_path(call_id, "prompt.txt")
    if meta_path is not None:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if system_path is not None:
        system_path.write_text(system_message or "", encoding="utf-8")
    if prompt_path is not None:
        prompt_path.write_text(prompt or "", encoding="utf-8")

    log_line(
        f"[llm_debug] {call_id} start role={role_name} provider={provider} model={model} "
        f"prompt_chars={len(prompt or '')} max_tokens={max_tokens} timeout_s={timeout_seconds}"
    )
    return call_id


def finish_llm_trace(
    call_id: str | None,
    *,
    response_text: str | None = None,
    error: str | None = None,
    duration_seconds: float | None = None,
) -> None:
    if not call_id or _LLM_DEBUG_DIR is None:
        return

    meta_path = _debug_path(call_id, "meta.json")
    response_path = _debug_path(call_id, "response.txt")
    error_path = _debug_path(call_id, "error.txt")
    meta: dict = {}
    if meta_path is not None and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    if response_text is not None and response_path is not None:
        response_path.write_text(response_text, encoding="utf-8")
        meta["response_chars"] = len(response_text)
    if error is not None and error_path is not None:
        error_path.write_text(error, encoding="utf-8")

    meta["ended_at"] = _timestamp()
    meta["status"] = "failed" if error is not None else "completed"
    if duration_seconds is not None:
        meta["duration_seconds"] = duration_seconds
    if error is not None:
        meta["error"] = error

    if meta_path is not None:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if error is not None:
        log_line(f"[llm_debug] {call_id} failed duration_s={duration_seconds} error={error}")
    else:
        log_line(f"[llm_debug] {call_id} completed duration_s={duration_seconds} response_chars={len(response_text or '')}")
