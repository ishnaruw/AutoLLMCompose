from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.core.candidate_ids import assign_candidate_ids
from src.core.api_formatting import normalize_api_for_ranking
from src.core.json_parsing import normalize_llm_payload, parse_llm_json, validate_expected_ids
from src.core.run_logging import log_line, log_retry_outcome_event, log_warning_event


class InvalidRankingOutput(RuntimeError):
    def __init__(self, metadata: Dict[str, Any]) -> None:
        self.metadata = metadata
        super().__init__(str(metadata.get("failure_reason") or "invalid_ranking_output"))


def _truncate(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def _slim_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    comp = c.get("compressed") or {}
    if not isinstance(comp, dict):
        comp = {}

    service = c.get("service") or {}
    if not isinstance(service, dict):
        service = {}

    name = comp.get("name") or service.get("name") or comp.get("operation") or comp.get("title")
    summary = comp.get("summary") or comp.get("description") or comp.get("desc") or service.get("description")
    method = comp.get("method") or service.get("method")
    path = comp.get("path") or comp.get("endpoint") or comp.get("url") or service.get("url")
    category = comp.get("category") or service.get("category") or c.get("category")
    tool_name = comp.get("tool_name") or service.get("tool_name") or service.get("_tool")
    tool_description = comp.get("tool_description") or service.get("tool_description")

    params = comp.get("params") or comp.get("parameters")
    param_names: List[str] = []
    if isinstance(params, list):
        for p in params[:20]:
            if isinstance(p, dict) and p.get("name"):
                param_names.append(str(p.get("name")))
            elif isinstance(p, str):
                param_names.append(p)
    elif isinstance(params, dict):
        param_names = [str(k) for k in list(params.keys())[:20]]

    qos = c.get("qos") if isinstance(c.get("qos"), dict) else {}
    service_qos = service.get("qos") if isinstance(service.get("qos"), dict) else {}

    slim: Dict[str, Any] = {
        "api_id": c.get("api_id"),
        "retrieved_rank": c.get("retrieved_rank"),
        "rag_score": c.get("rag_score"),
        "category": category,
        "tool_name": _truncate(tool_name, 120),
        "tool_description": _truncate(tool_description, 220),
        "name": _truncate(name, 120),
        "summary": _truncate(summary, 220),
        "method": _truncate(method, 16),
        "path": _truncate(path, 140),
    }
    if param_names:
        slim["param_names"] = param_names

    for key in ["rt_ms", "tp_rps", "availability", "qos_score", "qos_rank", "topsis_score", "topsis_rank", "qos_llm_score", "qos_llm_rank"]:
        val = c.get(key)
        if val is None:
            val = qos.get(key)
        if val is None:
            val = service.get(key)
        if val is None:
            val = service_qos.get(key)
        if val is not None:
            slim[key] = val

    return slim


def _sort_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _candidate_prompt_sort_key(c: Dict[str, Any]) -> tuple[str, str, str]:
    if not isinstance(c, dict):
        return ("", "", "")
    comp = c.get("compressed") if isinstance(c.get("compressed"), dict) else {}
    service = c.get("service") if isinstance(c.get("service"), dict) else {}
    api_id = c.get("api_id") or comp.get("api_id") or service.get("api_id")
    name = c.get("name") or comp.get("name") or service.get("name") or comp.get("operation") or service.get("operation")
    tool_name = c.get("tool_name") or comp.get("tool_name") or service.get("tool_name") or service.get("_tool")
    primary = api_id or name or tool_name or ""
    return (_sort_text(primary), _sort_text(name), _sort_text(tool_name))


def _retrieved_rank_sort_key(c: Dict[str, Any]) -> int:
    try:
        return int(c.get("retrieved_rank") or 10**9)
    except Exception:
        return 10**9


def _warn_api_id_quality(candidates: List[Dict[str, Any]], *, context: str) -> None:
    ids = [str(c.get("api_id") or "").strip() for c in candidates if isinstance(c, dict)]
    missing = sum(1 for api_id in ids if not api_id)
    duplicates = sorted(api_id for api_id, count in Counter(api_id for api_id in ids if api_id).items() if count > 1)
    if not missing and not duplicates:
        return

    parts = []
    if missing:
        parts.append(f"{missing} missing api_id")
    if duplicates:
        preview = ", ".join(duplicates[:5])
        suffix = "..." if len(duplicates) > 5 else ""
        parts.append(f"{len(duplicates)} duplicate api_id value(s): {preview}{suffix}")
    log_line(f"[ranker] warning: {context} has " + "; ".join(parts))
    if missing:
        log_warning_event(
            {
                "warning_type": "missing_api_id",
                "missing_api_id_count": missing,
                "context": context,
                "source": "ranker_prompt_payload",
            }
        )
    if duplicates:
        log_warning_event(
            {
                "warning_type": "duplicate_api_id_in_prompt_payload",
                "duplicate_api_ids": duplicates,
                "duplicate_api_id_count": len(duplicates),
                "context": context,
                "source": "ranker_prompt_payload",
            }
        )


def _write_ranker_debug(debug_raw_path: str | None, raw: str, attempt: int) -> None:
    if not debug_raw_path:
        return
    path = Path(debug_raw_path)
    if attempt > 1:
        path = path.parent / f"{path.stem}_retry{attempt - 1}{path.suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw or "", encoding="utf-8")


def _exception_reason(exc: Exception) -> str:
    text = str(exc).lower()
    if "groq_prompt_too_large" in text or "request too large for model" in text:
        return "groq_prompt_too_large"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "json" in text:
        return "invalid_json"
    return "parse_error"


def _parse_ranked_output(
    raw: str,
    expected_ids: List[str],
    candidate_id_to_api_id: Dict[str, str] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    candidate_id_to_api_id = {
        str(candidate_id).strip(): str(api_id).strip()
        for candidate_id, api_id in (candidate_id_to_api_id or {}).items()
        if str(candidate_id).strip() and str(api_id).strip()
    }
    api_id_to_candidate_id: Dict[str, str] = {}
    for candidate_id, api_id in candidate_id_to_api_id.items():
        api_id_to_candidate_id.setdefault(api_id, candidate_id)
    expected_candidate_ids = list(candidate_id_to_api_id.keys())

    parsed_json = parse_llm_json(raw)
    if parsed_json.error:
        issue = {
            **parsed_json.error,
            "expected_api_count": len(expected_ids),
            "actual_api_count": 0,
        }
        if expected_candidate_ids:
            issue.update({"expected_candidate_count": len(expected_candidate_ids), "actual_candidate_count": 0})
        return [], issue

    ranked_raw, key_issue = normalize_llm_payload(
        parsed_json.value,
        "ranked",
        aliases={"ranked_apis": "ranked", "ranking": "ranked"},
        allow_list=True,
    )
    if key_issue:
        issue = {
            **key_issue,
            "expected_api_count": len(expected_ids),
            "actual_api_count": 0,
        }
        if expected_candidate_ids:
            issue.update({"expected_candidate_count": len(expected_candidate_ids), "actual_candidate_count": 0})
        return [], issue

    contains_candidate_ids = any(
        isinstance(item, dict) and str(item.get("candidate_id") or "").strip()
        for item in ranked_raw
    )
    if candidate_id_to_api_id and contains_candidate_ids:
        ranked, issue = _parse_ranked_output_by_candidate_id(
            ranked_raw,
            expected_ids,
            candidate_id_to_api_id,
            api_id_to_candidate_id,
        )
    else:
        ranked, issue = _parse_ranked_output_by_api_id(
            ranked_raw,
            expected_ids,
            expected_candidate_ids,
            api_id_to_candidate_id,
        )

    if issue is not None:
        return ranked, issue
    _assign_order_ranks_when_compact(ranked)
    return _validate_and_sort_by_reported_rank(
        ranked,
        expected_api_count=len(expected_ids),
        expected_candidate_count=len(expected_candidate_ids) or None,
    )


def _ranked_item_reason_fields(item: Dict[str, Any], parsed_item: Dict[str, Any]) -> None:
    if item.get("rank") is not None:
        parsed_item["llm_reported_rank"] = item.get("rank")
    if item.get("functional_reason") is not None:
        parsed_item["functional_reason"] = str(item.get("functional_reason") or "")[:160]
    if item.get("qos_reason") is not None:
        parsed_item["qos_reason"] = str(item.get("qos_reason") or "")[:160]


def _assign_order_ranks_when_compact(ranked: List[Dict[str, Any]]) -> None:
    if not ranked or any("llm_reported_rank" in item for item in ranked):
        return
    for idx, item in enumerate(ranked, start=1):
        item["llm_reported_rank"] = idx


def _ranker_output_contract(include_reasons: bool) -> str:
    base = (
        "LLM output contract:\n"
        "- Return one compact JSON object only.\n"
        '- Preferred compact schema: {"ranked": [{"candidate_id": "C01"}]}.\n'
        "- Return one item for every input candidate_id exactly once, in ranked order.\n"
        "- Do not include rank values when using the compact schema; list order is the rank.\n"
        "- Use candidate_id in output and do not output api_id."
    )
    if include_reasons:
        return base + "\n- Optional: each item may include a short reason string."
    return base + "\n- Do not include reason, functional_reason, qos_reason, explanation, or any prose."


def _apply_ranker_output_contract(prompt: str, include_reasons: bool) -> str:
    contract = _ranker_output_contract(include_reasons)
    if "{llm_output_contract}" in prompt:
        return prompt.replace("{llm_output_contract}", contract)
    return prompt + "\n\n" + contract


def _coerce_rank_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
    return None


def _is_missing_rank_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _rank_issue_base(
    ranked: List[Dict[str, Any]],
    *,
    expected_api_count: int,
    expected_candidate_count: int | None,
    coerced_ranks: List[int],
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "expected_api_count": expected_api_count,
        "actual_api_count": len(ranked),
        "expected_rank_count": expected_api_count,
        "actual_rank_count": len(set(coerced_ranks)),
        "returned_rank_count": len(coerced_ranks),
    }
    if expected_candidate_count is not None:
        base.update(
            {
                "expected_candidate_count": expected_candidate_count,
                "actual_candidate_count": len(ranked),
            }
        )
    return base


def _validate_and_sort_by_reported_rank(
    ranked: List[Dict[str, Any]],
    *,
    expected_api_count: int,
    expected_candidate_count: int | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    coerced_ranks: List[int] = []
    missing_rank_items: List[Dict[str, Any]] = []
    non_integer_rank_items: List[Dict[str, Any]] = []
    non_integer_rank_values: List[str] = []

    for item in ranked:
        raw_rank = item.get("llm_reported_rank")
        if "llm_reported_rank" not in item or _is_missing_rank_value(raw_rank):
            missing_rank_items.append(item)
            continue
        rank = _coerce_rank_value(raw_rank)
        if rank is None:
            non_integer_rank_items.append(item)
            non_integer_rank_values.append(str(raw_rank))
            continue
        item["llm_reported_rank"] = rank
        coerced_ranks.append(rank)

    issue_base = _rank_issue_base(
        ranked,
        expected_api_count=expected_api_count,
        expected_candidate_count=expected_candidate_count,
        coerced_ranks=coerced_ranks,
    )
    if missing_rank_items:
        return ranked, {
            **issue_base,
            "reason": "missing_rank_values",
            "missing_rank_candidate_ids": [
                str(item.get("candidate_id"))
                for item in missing_rank_items
                if str(item.get("candidate_id") or "").strip()
            ],
            "missing_rank_api_ids": [
                str(item.get("api_id"))
                for item in missing_rank_items
                if str(item.get("api_id") or "").strip()
            ],
        }
    if non_integer_rank_items:
        return ranked, {
            **issue_base,
            "reason": "non_integer_rank_values",
            "non_integer_rank_values": non_integer_rank_values,
            "non_integer_rank_candidate_ids": [
                str(item.get("candidate_id"))
                for item in non_integer_rank_items
                if str(item.get("candidate_id") or "").strip()
            ],
            "non_integer_rank_api_ids": [
                str(item.get("api_id"))
                for item in non_integer_rank_items
                if str(item.get("api_id") or "").strip()
            ],
        }

    rank_counts = Counter(coerced_ranks)
    duplicate_rank_values = sorted(rank for rank, count in rank_counts.items() if count > 1)
    if duplicate_rank_values:
        return ranked, {
            **issue_base,
            "reason": "duplicate_rank_values",
            "duplicate_rank_values": duplicate_rank_values,
        }

    expected_rank_values = set(range(1, expected_api_count + 1))
    returned_rank_values = set(coerced_ranks)
    out_of_range_rank_values = sorted(rank for rank in returned_rank_values if rank not in expected_rank_values)
    missing_rank_values = sorted(expected_rank_values - returned_rank_values)
    if out_of_range_rank_values:
        return ranked, {
            **issue_base,
            "reason": "rank_values_out_of_range",
            "rank_values_out_of_range": out_of_range_rank_values,
            "missing_rank_values": missing_rank_values,
        }
    if missing_rank_values:
        return ranked, {
            **issue_base,
            "reason": "incomplete_rank_sequence",
            "missing_rank_values": missing_rank_values,
            "returned_rank_values": sorted(returned_rank_values),
        }

    return sorted(ranked, key=lambda item: int(item["llm_reported_rank"])), None


def _parse_ranked_output_by_api_id(
    ranked_raw: List[Any],
    expected_ids: List[str],
    expected_candidate_ids: List[str] | None = None,
    api_id_to_candidate_id: Dict[str, str] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    expected_candidate_ids = expected_candidate_ids or []
    api_id_to_candidate_id = api_id_to_candidate_id or {}
    expected_set = set(expected_ids)
    returned_ids: List[str] = []
    ranked: List[Dict[str, Any]] = []
    malformed_items = 0
    for item in ranked_raw:
        if not isinstance(item, dict):
            malformed_items += 1
            continue
        api_id = str(item.get("api_id") or "").strip()
        if not api_id:
            malformed_items += 1
            continue
        returned_ids.append(api_id)
        parsed_item = {
            "api_id": api_id,
            "reason": (item.get("reason", "") or "")[:160],
            "is_unknown_api_id": api_id not in expected_set,
        }
        if api_id in api_id_to_candidate_id:
            parsed_item["candidate_id"] = api_id_to_candidate_id[api_id]
        _ranked_item_reason_fields(item, parsed_item)
        ranked.append(parsed_item)

    returned_expected_ids = {api_id for api_id in returned_ids if api_id in expected_set}
    issue_base = {
        "expected_api_count": len(expected_ids),
        "actual_api_count": len(returned_expected_ids),
        "returned_api_count": len(returned_ids),
    }
    if expected_candidate_ids:
        issue_base.update(
            {
                "expected_candidate_count": len(expected_candidate_ids),
                "actual_candidate_count": len(returned_expected_ids),
            }
        )

    if malformed_items:
        return ranked, {
            **issue_base,
            "reason": "parse_error",
            "malformed_ranked_item_count": malformed_items,
        }
    id_issue = validate_expected_ids(
        returned_ids,
        expected_ids,
        duplicate_reason="duplicate_ranked_apis",
        unknown_reason="unknown_api_ids",
        missing_reason="incomplete_ranked_api_list",
        id_label="api",
    )
    if id_issue:
        return ranked, {**issue_base, **id_issue}

    return ranked, None


def _parse_ranked_output_by_candidate_id(
    ranked_raw: List[Any],
    expected_ids: List[str],
    candidate_id_to_api_id: Dict[str, str],
    api_id_to_candidate_id: Dict[str, str],
) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    expected_candidate_ids = list(candidate_id_to_api_id.keys())
    expected_candidate_set = set(expected_candidate_ids)
    returned_candidate_ids: List[str] = []
    unknown_legacy_api_ids: List[str] = []
    ranked: List[Dict[str, Any]] = []
    malformed_items = 0
    for item in ranked_raw:
        if not isinstance(item, dict):
            malformed_items += 1
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        raw_api_id = str(item.get("api_id") or "").strip()
        if candidate_id:
            api_id = candidate_id_to_api_id.get(candidate_id, raw_api_id)
            returned_candidate_ids.append(candidate_id)
        elif raw_api_id:
            api_id = raw_api_id
            candidate_id = api_id_to_candidate_id.get(raw_api_id, "")
            if candidate_id:
                returned_candidate_ids.append(candidate_id)
            else:
                unknown_legacy_api_ids.append(raw_api_id)
        else:
            malformed_items += 1
            continue

        parsed_item = {
            "api_id": api_id,
            "candidate_id": candidate_id,
            "reason": (item.get("reason", "") or "")[:160],
            "is_unknown_api_id": api_id not in set(expected_ids),
            "is_unknown_candidate_id": candidate_id not in expected_candidate_set,
        }
        _ranked_item_reason_fields(item, parsed_item)
        ranked.append(parsed_item)

    returned_counts = Counter(returned_candidate_ids)
    duplicate_candidate_ids = sorted(candidate_id for candidate_id, count in returned_counts.items() if count > 1)
    unknown_candidate_ids = sorted(candidate_id for candidate_id in returned_counts if candidate_id not in expected_candidate_set)
    valid_returned_candidate_ids = {
        candidate_id for candidate_id in returned_candidate_ids if candidate_id in expected_candidate_set
    }
    missing_candidate_ids = [
        candidate_id for candidate_id in expected_candidate_ids if candidate_id not in valid_returned_candidate_ids
    ]
    duplicate_api_ids = sorted(
        {
            candidate_id_to_api_id[candidate_id]
            for candidate_id in duplicate_candidate_ids
            if candidate_id in candidate_id_to_api_id
        }
    )
    missing_api_ids = [
        candidate_id_to_api_id[candidate_id]
        for candidate_id in missing_candidate_ids
        if candidate_id in candidate_id_to_api_id
    ]
    unknown_api_ids = sorted(set(unknown_legacy_api_ids))
    issue_base = {
        "expected_api_count": len(expected_ids),
        "actual_api_count": len(valid_returned_candidate_ids),
        "returned_api_count": len(returned_candidate_ids) + len(unknown_legacy_api_ids),
        "expected_candidate_count": len(expected_candidate_ids),
        "actual_candidate_count": len(valid_returned_candidate_ids),
        "returned_candidate_count": len(returned_candidate_ids),
    }

    if malformed_items:
        return ranked, {
            **issue_base,
            "reason": "parse_error",
            "malformed_ranked_item_count": malformed_items,
        }
    if duplicate_candidate_ids:
        return ranked, {
            **issue_base,
            "reason": "duplicate_candidate_ids",
            "duplicate_candidate_ids": duplicate_candidate_ids,
            "duplicate_api_ids": duplicate_api_ids,
        }
    if unknown_candidate_ids:
        return ranked, {
            **issue_base,
            "reason": "unknown_candidate_ids",
            "unknown_candidate_ids": unknown_candidate_ids,
            "missing_candidate_ids": missing_candidate_ids,
            "missing_api_ids": missing_api_ids,
        }
    if unknown_api_ids:
        return ranked, {
            **issue_base,
            "reason": "unknown_api_ids",
            "unknown_api_ids": unknown_api_ids,
            "missing_candidate_ids": missing_candidate_ids,
            "missing_api_ids": missing_api_ids,
        }
    if missing_candidate_ids:
        return ranked, {
            **issue_base,
            "reason": "incomplete_candidate_id_list",
            "missing_candidate_ids": missing_candidate_ids,
            "missing_api_ids": missing_api_ids,
        }
    if len(returned_candidate_ids) != len(expected_candidate_ids):
        return ranked, {
            **issue_base,
            "reason": "missing_candidate_ids",
            "missing_candidate_ids": missing_candidate_ids,
            "missing_api_ids": missing_api_ids,
        }

    return ranked, None


def _ranker_retry_prompt(prompt: str, issue: Dict[str, Any]) -> str:
    reason = issue.get("reason", "invalid_ranking_output")
    expected = issue.get("expected_candidate_count") or issue.get("expected_api_count", "")
    expected_ranks = issue.get("expected_rank_count")
    rank_rule = (
        f" Include integer rank values 1 through {expected_ranks} exactly once."
        if expected_ranks
        else ""
    )
    if issue.get("expected_candidate_count"):
        return (
            prompt
            + "\n\nIMPORTANT: The previous ranking output was invalid "
            + f"({reason}). Return JSON only using the output key requested by the prompt, containing every one of the {expected} candidate_id values exactly once. "
            + "Use candidate_id only. Do not output api_id. Do not omit candidates, duplicate candidates, or invent candidate_id values."
            + rank_rule
        )
    return (
        prompt
        + "\n\nIMPORTANT: The previous ranking output was invalid "
        + f"({reason}). Return JSON only using the output key requested by the prompt, containing every one of the {expected} candidate api_id values exactly once. "
        + "Do not omit candidates, duplicate candidates, or invent api_id values."
        + rank_rule
    )


def _failure_metadata(issue: Dict[str, Any], *, after_retries: bool) -> Dict[str, Any]:
    reason = str(issue.get("reason") or "parse_error")
    failure_reason = reason if reason == "timeout" or not after_retries else f"{reason}_after_retries"
    return {
        "failure_flag": True,
        "failure_stage": "llm_ranking",
        "failure_reason": failure_reason,
        "exclude_from_ranking_eval": True,
        **issue,
    }


def rank_subtask(
    llm_call: Callable[[str], str],
    *,
    user_query: str,
    subtask: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    prompt_path: str,
    debug_raw_path: str | None = None,
    use_compact_api_evidence: bool = False,
    include_qos_rank: bool = False,
    max_validation_retries: int = 2,
) -> List[Dict[str, Any]]:
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    from src.config import CONFIG

    max_rank_candidates = CONFIG.ranker_max_candidates
    ranker_pool_n = CONFIG.ranker_pool_n
    retrieved_pool = sorted(
        (c for c in candidates if isinstance(c, dict)),
        key=_retrieved_rank_sort_key,
    )[:max_rank_candidates]
    _warn_api_id_quality(retrieved_pool, context=f"{prompt_path} top-{max_rank_candidates} prompt pool")
    cand_trimmed = [c for c in retrieved_pool if c.get("api_id")]
    if use_compact_api_evidence:
        subtask_text = str(subtask.get("description") or subtask.get("summary") or subtask.get("task") or "")
        cand_slim = [
            normalize_api_for_ranking(c, subtask_text=subtask_text, include_qos_rank=include_qos_rank)
            for c in cand_trimmed
        ]
    else:
        cand_slim = [_slim_candidate(c) for c in cand_trimmed]
    cand_slim = sorted(cand_slim, key=_candidate_prompt_sort_key)
    cand_slim, candidate_id_to_api_id, _api_id_to_candidate_id = assign_candidate_ids(cand_slim)
    expected_ids = list(candidate_id_to_api_id.values())
    if not expected_ids:
        return []

    prompt = (
        template
        .replace("{user_query}", user_query)
        .replace("{subtask_json}", json.dumps(subtask, ensure_ascii=False))
        .replace("{candidates_json}", json.dumps(cand_slim, ensure_ascii=False))
    )
    prompt = _apply_ranker_output_contract(prompt, bool(CONFIG.include_llm_reasons))

    attempts = max(0, int(max_validation_retries or 0)) + 1
    last_issue: Dict[str, Any] = {
        "reason": "parse_error",
        "expected_api_count": len(expected_ids),
        "expected_candidate_count": len(candidate_id_to_api_id),
        "actual_api_count": 0,
        "actual_candidate_count": 0,
    }
    last_ranked: List[Dict[str, Any]] = []
    invalid_attempts = 0
    for attempt in range(1, attempts + 1):
        prompt_for_attempt = prompt if attempt == 1 else _ranker_retry_prompt(prompt, last_issue)
        try:
            resp_raw = llm_call(prompt_for_attempt)
        except Exception as exc:
            issue = {
                "reason": _exception_reason(exc),
                "expected_api_count": len(expected_ids),
                "actual_api_count": 0,
                "error": str(exc),
            }
            raise InvalidRankingOutput(_failure_metadata(issue, after_retries=False)) from exc

        _write_ranker_debug(debug_raw_path, resp_raw, attempt)
        ranked, issue = _parse_ranked_output(resp_raw, expected_ids, candidate_id_to_api_id)
        if issue is None:
            if invalid_attempts:
                log_line(
                    f"[ranker] retry validation succeeded after {invalid_attempts} invalid attempt(s); "
                    f"final_attempt={attempt}/{attempts}"
                )
                log_retry_outcome_event(
                    {
                        "stage": "llm_ranking",
                        "role": "ranker",
                        "outcome": "success",
                        "invalid_attempts": invalid_attempts,
                        "final_attempt": attempt,
                        "max_attempts": attempts,
                        "last_failure_reason": last_issue.get("reason"),
                        "expected_api_count": last_issue.get("expected_api_count"),
                        "actual_api_count": last_issue.get("actual_api_count"),
                        "expected_candidate_count": last_issue.get("expected_candidate_count"),
                        "actual_candidate_count": last_issue.get("actual_candidate_count"),
                    }
                )
            return ranked[:ranker_pool_n]

        last_issue = issue
        last_ranked = ranked
        invalid_attempts += 1
        log_line(
            f"[ranker] invalid LLM ranking output ({issue.get('reason')}) "
            f"attempt {attempt}/{attempts}; expected={issue.get('expected_candidate_count', issue.get('expected_api_count'))} "
            f"actual={issue.get('actual_candidate_count', issue.get('actual_api_count'))}"
        )

    metadata = _failure_metadata(last_issue, after_retries=True)
    metadata["retry_invalid_attempts"] = invalid_attempts
    metadata["retry_max_attempts"] = attempts
    if last_ranked:
        metadata["_invalid_ranked_items"] = last_ranked
    log_retry_outcome_event(
        {
            "stage": "llm_ranking",
            "role": "ranker",
            "outcome": "failed",
            "invalid_attempts": invalid_attempts,
            "final_attempt": attempts,
            "max_attempts": attempts,
            "last_failure_reason": last_issue.get("reason"),
            "failure_reason": metadata.get("failure_reason"),
            "expected_api_count": last_issue.get("expected_api_count"),
            "actual_api_count": last_issue.get("actual_api_count"),
            "expected_candidate_count": last_issue.get("expected_candidate_count"),
            "actual_candidate_count": last_issue.get("actual_candidate_count"),
        }
    )
    raise InvalidRankingOutput(metadata)
