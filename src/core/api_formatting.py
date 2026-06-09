from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.core.run_logging import log_warning_event


DESCRIPTION_MAX_CHARS = 400
TOOL_DESCRIPTION_MAX_CHARS = 250
PARAMETER_DESCRIPTION_MAX_CHARS = 160
MAX_PARAMETERS_PER_API = 6
SHORT_DESCRIPTION_MIN_CHARS = 25

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
    include_functional_match_label: bool = False,
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
        endpoint_detail.get("tool_name"),
        compressed.get("tool_name"),
        service.get("tool_name"),
        api.get("tool_name"),
        service.get("_tool"),
        api.get("_tool"),
    )
    tool_description = _first_value(
        endpoint_detail.get("tool_description"),
        compressed.get("tool_description"),
        service.get("tool_description"),
        api.get("tool_description"),
    )
    description = _first_value(
        endpoint_detail.get("description"),
        service.get("description"),
        api.get("description"),
        compressed.get("description"),
        compressed.get("desc"),
        service.get("summary"),
        api.get("summary"),
        compressed.get("summary"),
    )
    method = _first_value(compressed.get("method"), service.get("method"), api.get("method"))
    normalized_description = truncate_text(description, DESCRIPTION_MAX_CHARS)
    raw_parameter_count = _raw_parameter_count(api, compressed, service, endpoint_detail)
    parameter_candidates = _collect_parameter_candidates(api, compressed, service, endpoint_detail)
    parameters = _select_parameter_candidates(parameter_candidates, subtask_text)

    compact: Dict[str, Any] = {
        "api_id": _clean_text(api_id),
        "name": _clean_text(name),
        "category": _clean_text(category),
        "tool_name": _clean_text(tool_name),
        "tool_description": truncate_text(tool_description, TOOL_DESCRIPTION_MAX_CHARS),
        "description": normalized_description,
        "method": _clean_text(method),
        "parameters": parameters,
    }

    _log_formatting_anomalies(
        compact=compact,
        api=api,
        compressed=compressed,
        service=service,
        endpoint_detail=endpoint_detail,
        raw_parameter_count=raw_parameter_count,
        normalized_parameter_count=len(parameters),
    )

    if include_qos_rank:
        for key in ("qos_llm_rank", "qos_llm_score", "rt_s", "tp_kbps", "availability"):
            compact[key] = _qos_evidence_value(api, service, key)

    if include_functional_match_label:
        label = _functional_match_label(api)
        if label is not None:
            compact["functional_match_label"] = label
            reason = _functional_match_reason(api)
            if reason:
                compact["functional_match_reason"] = truncate_text(reason, 160)

    return compact


def _functional_match_label(api: Dict[str, Any]) -> int | None:
    for key in ("Functional Match Label", "Functional Match (0/1)", "functional_match_label", "functional_match", "relevant"):
        value = api.get(key)
        if isinstance(value, bool):
            return 1 if value else 0
        if value in (0, 1):
            return int(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "relevant", "match"}:
                return 1
            if text in {"0", "false", "no", "irrelevant", "nonmatch", "non-match", "not_match"}:
                return 0
    return None


def _functional_match_reason(api: Dict[str, Any]) -> str:
    return _clean_text(
        _first_value(
            api.get("functional_refiner_reason"),
            api.get("Functional Match Reason"),
            api.get("Comments"),
            api.get("comment"),
        )
    )


def _log_formatting_anomalies(
    *,
    compact: Dict[str, Any],
    api: Dict[str, Any],
    compressed: Dict[str, Any],
    service: Dict[str, Any],
    endpoint_detail: Dict[str, Any],
    raw_parameter_count: int,
    normalized_parameter_count: int,
) -> None:
    api_id = compact.get("api_id", "")
    base = {
        "api_id": api_id,
        "name": compact.get("name", ""),
        "category": compact.get("category", ""),
        "source": "api_formatting",
    }

    if not api_id:
        log_warning_event(
            {
                **base,
                "warning_type": "missing_api_id",
            }
        )

    raw_endpoint_details = _collect_endpoint_detail_dicts(api, compressed, service, endpoint_detail)
    endpoint_description = _endpoint_details_description(raw_endpoint_details)
    normalized_description = str(compact.get("description") or "")
    if endpoint_description and not normalized_description:
        log_warning_event(
            {
                **base,
                "warning_type": "endpoint_details_description_not_used",
                "has_endpoint_details_description": True,
                "normalized_description": normalized_description,
            }
        )
    elif raw_endpoint_details and not endpoint_detail.get("description") and not normalized_description:
        log_warning_event(
            {
                **base,
                "warning_type": "endpoint_details_not_used",
                "has_endpoint_details": True,
                "normalized_description": normalized_description,
            }
        )

    description_length = len(normalized_description)
    if description_length < SHORT_DESCRIPTION_MIN_CHARS:
        log_warning_event(
            {
                **base,
                "warning_type": "missing_or_short_description",
                "description_length": description_length,
            }
        )

    if raw_parameter_count > 0 and normalized_parameter_count == 0:
        log_warning_event(
            {
                **base,
                "warning_type": "parameters_filtered_out",
                "raw_parameter_count": raw_parameter_count,
                "normalized_parameter_count": normalized_parameter_count,
            }
        )


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


def _first_present_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _qos_evidence_value(api: Dict[str, Any], service: Dict[str, Any], key: str) -> Any:
    row_qos = _as_dict(api.get("qos"))
    service_qos = _as_dict(service.get("qos"))
    return _first_present_value(
        api.get(key),
        row_qos.get(key),
        service_qos.get(key),
        service.get(key),
    )


def _clean_text(value: Any) -> str:
    return truncate_text(value, 10_000)


def _collect_endpoint_detail_dicts(
    api: Dict[str, Any],
    compressed: Dict[str, Any],
    service: Dict[str, Any],
    endpoint_detail: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for source in (
        api.get("endpoint_details"),
        compressed.get("endpoint_details"),
        service.get("endpoint_details"),
        endpoint_detail.get("endpoint_details"),
    ):
        detail = _as_dict(source)
        if detail:
            out.append(detail)
    return out


def _endpoint_details_description(details: List[Dict[str, Any]]) -> str:
    values: List[Any] = []
    for detail in details:
        values.extend([detail.get("description"), detail.get("desc"), detail.get("summary")])
    return truncate_text(_first_value(*values), DESCRIPTION_MAX_CHARS)


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
    precomputed, has_precomputed = _precomputed_endpoint_detail(api, compressed, service)
    if has_precomputed:
        return precomputed

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


def _precomputed_endpoint_detail(
    api: Dict[str, Any],
    compressed: Dict[str, Any],
    service: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    has_precomputed = False
    detail: Dict[str, Any] = {}

    for source in (api, compressed, service):
        if not isinstance(source, dict):
            continue
        enrichment = _as_dict(source.get("toolbench_enrichment"))
        source_has_precomputed = bool(enrichment) or any(
            key in source
            for key in (
                "toolbench_tool_name",
                "toolbench_tool_description",
                "toolbench_endpoint_description",
                "toolbench_endpoint_details",
            )
        )
        if not source_has_precomputed:
            continue
        has_precomputed = True

        endpoint_details = _as_dict(
            _first_value(
                source.get("toolbench_endpoint_details"),
                enrichment.get("endpoint_details"),
                source.get("endpoint_details"),
            )
        )
        description = _first_value(
            source.get("toolbench_endpoint_description"),
            enrichment.get("endpoint_description"),
            endpoint_details.get("description"),
        )
        tool_name = _first_value(
            source.get("toolbench_tool_name"),
            enrichment.get("tool_name"),
        )
        tool_description = _first_value(
            source.get("toolbench_tool_description"),
            enrichment.get("tool_description"),
        )

        if tool_name and not detail.get("tool_name"):
            detail["tool_name"] = tool_name
        if tool_description and not detail.get("tool_description"):
            detail["tool_description"] = tool_description
        if description and not detail.get("description"):
            detail["description"] = description
        if endpoint_details and not detail.get("endpoint_details"):
            detail["endpoint_details"] = endpoint_details

    return detail, has_precomputed


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
    return _select_parameter_candidates(candidates, subtask_text)


def _select_parameter_candidates(
    candidates: List[Dict[str, Any]],
    subtask_text: str | None,
) -> List[Dict[str, str]]:
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


def _raw_parameter_count(
    api: Dict[str, Any],
    compressed: Dict[str, Any],
    service: Dict[str, Any],
    endpoint_detail: Dict[str, Any],
) -> int:
    count = 0
    seen_detail_ids: set[int] = set()
    for source in (
        endpoint_detail.get("endpoint_details"),
        api.get("endpoint_details"),
        compressed.get("endpoint_details"),
        service.get("endpoint_details"),
    ):
        details = _as_dict(source)
        if not details or id(details) in seen_detail_ids:
            continue
        seen_detail_ids.add(id(details))
        count += sum(1 for _ in _iter_parameters(details.get("required_parameters")))
        count += sum(1 for _ in _iter_parameters(details.get("optional_parameters")))

    for source in (api, compressed, service):
        count += sum(1 for _ in _iter_parameters(source.get("required_parameters")))
        count += sum(1 for _ in _iter_parameters(source.get("optional_parameters")))

    for source in (api, compressed, service):
        count += sum(1 for _ in _iter_parameters(source.get("parameters")))
        count += sum(1 for _ in _iter_parameters(source.get("params")))
    return count


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
