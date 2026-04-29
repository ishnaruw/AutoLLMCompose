from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DESCRIPTION_MAX_CHARS = 400
TOOL_DESCRIPTION_MAX_CHARS = 250
PARAMETER_DESCRIPTION_MAX_CHARS = 160
MAX_PARAMETERS_PER_API = 6

EXCLUDED_PARAMS = {
    "api_key",
    "apikey",
    "token",
    "access_token",
    "authorization",
    "auth",
    "key",
    "secret",
    "host",
    "x-rapidapi-key",
    "x-rapidapi-host",
}

TOOLBENCH_TOOLS_ROOT = Path(
    os.getenv(
        "TOOLBENCH_TOOLS_ROOT",
        "/Users/ishwaryapns/Documents/Thesis/ToolBench/data/toolenv/tools",
    )
)

_GENERIC_PARAMETER_DESCRIPTIONS = {
    "body",
    "boolean",
    "default",
    "integer",
    "n/a",
    "none",
    "null",
    "number",
    "optional",
    "parameter",
    "query",
    "required",
    "string",
    "text",
    "type",
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "get",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "use",
    "with",
}


def truncate_text(text: Any, max_chars: int) -> str:
    """Return text normalized to one line and deterministically truncated."""
    value = "" if text is None else str(text)
    value = " ".join(value.strip().split())
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."


def normalize_api_for_ranking(
    api: Dict[str, Any],
    subtask_text: str | None = None,
    include_qos_rank: bool = False,
) -> Dict[str, Any]:
    """Build compact deterministic API evidence for functional and ranking prompts."""
    if not isinstance(api, dict):
        api = {}

    compressed = _as_dict(api.get("compressed"))
    service = _as_dict(api.get("service"))
    endpoint_detail = _find_endpoint_detail(api, compressed, service)

    api_id = _first_value(
        api.get("api_id"),
        compressed.get("api_id"),
        service.get("api_id"),
        api.get("id"),
        compressed.get("id"),
        service.get("id"),
    )
    name = _first_value(
        compressed.get("name"),
        service.get("name"),
        api.get("name"),
        compressed.get("operation"),
        service.get("operation"),
        api.get("operation"),
        compressed.get("title"),
        service.get("title"),
        api.get("title"),
    )
    category = _first_value(compressed.get("category"), service.get("category"), api.get("category"))
    tool_name = _first_value(
        compressed.get("tool_name"),
        service.get("tool_name"),
        api.get("tool_name"),
        endpoint_detail.get("tool_name"),
        service.get("_tool"),
        api.get("_tool"),
    )
    tool_description = _first_value(
        compressed.get("tool_description"),
        service.get("tool_description"),
        api.get("tool_description"),
        endpoint_detail.get("tool_description"),
    )
    description = _first_value(
        compressed.get("summary"),
        compressed.get("description"),
        compressed.get("desc"),
        service.get("description"),
        service.get("summary"),
        service.get("desc"),
        api.get("description"),
        api.get("summary"),
        api.get("desc"),
        endpoint_detail.get("description"),
    )
    method = _first_value(compressed.get("method"), service.get("method"), api.get("method"))

    compact: Dict[str, Any] = {
        "api_id": _clean_text(api_id),
        "name": _clean_text(name),
        "category": _clean_text(category),
        "tool_name": _clean_text(tool_name),
        "tool_description": truncate_text(tool_description, TOOL_DESCRIPTION_MAX_CHARS),
        "description": truncate_text(description, DESCRIPTION_MAX_CHARS),
        "method": _clean_text(method),
        "parameters": _select_parameters(api, compressed, service, endpoint_detail, subtask_text),
    }

    if include_qos_rank:
        qos_rank = _first_value(
            api.get("qos_llm_rank"),
            _as_dict(api.get("qos")).get("qos_llm_rank"),
            service.get("qos_llm_rank"),
            _as_dict(service.get("qos")).get("qos_llm_rank"),
        )
        if qos_rank is not None:
            compact["qos_llm_rank"] = qos_rank

    return compact


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return ""


def _clean_text(value: Any) -> str:
    return truncate_text(value, 10_000)


def _candidate_field(
    api: Dict[str, Any],
    compressed: Dict[str, Any],
    service: Dict[str, Any],
    key: str,
) -> Any:
    return _first_value(compressed.get(key), service.get(key), api.get(key))


def _find_endpoint_detail(
    api: Dict[str, Any],
    compressed: Dict[str, Any],
    service: Dict[str, Any],
) -> Dict[str, Any]:
    existing = _as_dict(_first_value(api.get("endpoint_details"), compressed.get("endpoint_details"), service.get("endpoint_details")))
    category = _candidate_field(api, compressed, service, "category")
    file_name = _first_value(
        service.get("_file"),
        api.get("_file"),
        compressed.get("_file"),
        service.get("file_name"),
        api.get("file_name"),
    )
    tool_json = _load_tool_json(str(category or ""), str(file_name or ""))
    if not tool_json:
        return {"endpoint_details": existing} if existing else {}

    endpoint = _match_endpoint(tool_json, api, compressed, service)
    detail: Dict[str, Any] = {
        "tool_name": _first_value(tool_json.get("tool_name"), tool_json.get("name"), tool_json.get("title")),
        "tool_description": tool_json.get("tool_description"),
    }
    if endpoint:
        detail["description"] = endpoint.get("description")
        detail["endpoint_details"] = {
            "required_parameters": endpoint.get("required_parameters") or [],
            "optional_parameters": endpoint.get("optional_parameters") or [],
        }
    elif existing:
        detail["endpoint_details"] = existing
    return detail


@lru_cache(maxsize=1024)
def _load_tool_json(category: str, file_name: str) -> Dict[str, Any]:
    if not category or not file_name:
        return {}
    path = TOOLBENCH_TOOLS_ROOT / category / file_name
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _match_endpoint(
    tool_json: Dict[str, Any],
    api: Dict[str, Any],
    compressed: Dict[str, Any],
    service: Dict[str, Any],
) -> Dict[str, Any]:
    api_list = tool_json.get("api_list")
    if not isinstance(api_list, list):
        return {}

    name = str(_candidate_field(api, compressed, service, "name") or "")
    method = str(_candidate_field(api, compressed, service, "method") or "").upper()
    url = str(
        _first_value(
            compressed.get("url"),
            service.get("url"),
            api.get("url"),
            compressed.get("endpoint"),
            service.get("endpoint"),
            api.get("endpoint"),
            compressed.get("path"),
            service.get("path"),
            api.get("path"),
        )
        or ""
    )

    candidates = [ep for ep in api_list if isinstance(ep, dict)]
    if name:
        name_matches = [ep for ep in candidates if str(ep.get("name") or "") == name]
        if name_matches:
            candidates = name_matches
    if method:
        method_matches = [ep for ep in candidates if str(ep.get("method") or "").upper() == method]
        if method_matches:
            candidates = method_matches
    if url:
        url_matches = [ep for ep in candidates if str(ep.get("url") or "") == url]
        if url_matches:
            candidates = url_matches
    return candidates[0] if candidates else {}


def _select_parameters(
    api: Dict[str, Any],
    compressed: Dict[str, Any],
    service: Dict[str, Any],
    endpoint_detail: Dict[str, Any],
    subtask_text: str | None,
) -> List[Dict[str, str]]:
    candidates = _collect_parameter_candidates(api, compressed, service, endpoint_detail)
    subtask_words = _word_set(subtask_text or "")
    scored: List[Tuple[int, int, str, str]] = []

    for index, param in enumerate(candidates):
        name = _clean_text(param.get("name"))
        description = truncate_text(param.get("description"), PARAMETER_DESCRIPTION_MAX_CHARS)
        if not name or _is_excluded_parameter_name(name):
            continue
        if not _has_meaningful_parameter_description(description):
            continue

        score = 0
        if param.get("required"):
            score += 3
        score += 2
        param_words = _word_set(f"{name} {description}")
        if subtask_words and param_words.intersection(subtask_words):
            score += 1

        scored.append((score, index, name, description))

    scored.sort(key=lambda item: (-item[0], item[1], item[2].lower()))
    return [
        {"name": name, "description": description}
        for _, _, name, description in scored[:MAX_PARAMETERS_PER_API]
    ]


def _collect_parameter_candidates(
    api: Dict[str, Any],
    compressed: Dict[str, Any],
    service: Dict[str, Any],
    endpoint_detail: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()

    def add_many(value: Any, *, required: bool) -> None:
        for item in _iter_parameters(value):
            name = _clean_text(item.get("name"))
            description = truncate_text(item.get("description"), 10_000)
            if not name:
                continue
            key = (name.lower(), description.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": name, "description": description, "required": required})

    for source in (
        endpoint_detail.get("endpoint_details"),
        api.get("endpoint_details"),
        compressed.get("endpoint_details"),
        service.get("endpoint_details"),
    ):
        details = _as_dict(source)
        add_many(details.get("required_parameters"), required=True)
        add_many(details.get("optional_parameters"), required=False)

    for source in (api, compressed, service):
        add_many(source.get("required_parameters"), required=True)
        add_many(source.get("optional_parameters"), required=False)

    for source in (api, compressed, service):
        add_many(source.get("parameters"), required=False)
        add_many(source.get("params"), required=False)

    return out


def _iter_parameters(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
    elif isinstance(value, dict):
        for name, item in value.items():
            if isinstance(item, dict):
                merged = dict(item)
                merged.setdefault("name", name)
                yield merged
            else:
                yield {"name": name, "description": item}


def _is_excluded_parameter_name(name: str) -> bool:
    lowered = name.strip().lower()
    variants = {
        lowered,
        lowered.replace("-", "_"),
        lowered.replace("_", "-"),
        re.sub(r"[^a-z0-9]+", "_", lowered).strip("_"),
        re.sub(r"[^a-z0-9]+", "-", lowered).strip("-"),
    }
    return any(variant in EXCLUDED_PARAMS for variant in variants)


def _has_meaningful_parameter_description(description: str) -> bool:
    cleaned = truncate_text(description, 10_000).lower()
    if not cleaned:
        return False
    compact = re.sub(r"[^a-z0-9]+", " ", cleaned).strip()
    if not compact:
        return False
    if compact in _GENERIC_PARAMETER_DESCRIPTIONS:
        return False
    if compact.startswith("type ") and len(compact.split()) <= 2:
        return False
    return True


def _word_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in _STOPWORDS
    }
