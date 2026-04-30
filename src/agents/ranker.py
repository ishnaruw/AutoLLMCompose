from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.core.api_formatting import normalize_api_for_ranking
from src.core.run_logging import log_line, log_ranking_anomaly_event


class InvalidRankingOutput(RuntimeError):
    def __init__(self, metadata: Dict[str, Any]) -> None:
        self.metadata = metadata
        super().__init__(str(metadata.get("failure_reason") or "invalid_ranking_output"))

FATAL_RANKING_REASONS = {"empty_response", "invalid_json", "parse_error", "timeout"}
RECOVERABLE_RANKING_ANOMALIES = {
    "duplicate_ranked_apis",
    "unknown_api_ids",
    "incomplete_ranked_api_list",
    "missing_ranked_apis",
}


def _base_reason(reason: Any) -> str:
    text = str(reason or "parse_error")
    return text[:-len("_after_retries")] if text.endswith("_after_retries") else text


def _is_recoverable_ranking_anomaly(issue: Dict[str, Any] | None) -> bool:
    return _base_reason((issue or {}).get("reason")) in RECOVERABLE_RANKING_ANOMALIES


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


def _coerce_json(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "{}"
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    return m.group(1) if m else "{}"


def _extract_json_text(s: str) -> tuple[str, str | None]:
    text = (s or "").strip()
    if not text:
        return "", "empty_response"
    try:
        json.loads(text)
        return text, None
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        return "", "invalid_json"
    return m.group(1), None


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
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "json" in text:
        return "invalid_json"
    return "parse_error"


def _parse_ranked_output(raw: str, expected_ids: List[str]) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    json_text, json_error = _extract_json_text(raw)
    if json_error:
        return [], {
            "reason": json_error,
            "expected_api_count": len(expected_ids),
            "actual_api_count": 0,
        }

    try:
        data = json.loads(json_text)
    except Exception as exc:
        return [], {
            "reason": "invalid_json",
            "expected_api_count": len(expected_ids),
            "actual_api_count": 0,
            "parse_error": str(exc),
        }

    if isinstance(data, dict):
        ranked_raw = data.get("ranked_apis")
        if not isinstance(ranked_raw, list):
            ranked_raw = data.get("ranked")
    else:
        ranked_raw = None
    if not isinstance(ranked_raw, list):
        return [], {
            "reason": "parse_error",
            "expected_api_count": len(expected_ids),
            "actual_api_count": 0,
            "detail": "missing_ranked_or_ranked_apis_list",
        }

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
        if item.get("rank") is not None:
            parsed_item["llm_reported_rank"] = item.get("rank")
        if item.get("functional_reason") is not None:
            parsed_item["functional_reason"] = str(item.get("functional_reason") or "")[:160]
        if item.get("qos_reason") is not None:
            parsed_item["qos_reason"] = str(item.get("qos_reason") or "")[:160]
        ranked.append(parsed_item)

    returned_counts = Counter(returned_ids)
    duplicate_ids = sorted(api_id for api_id, count in returned_counts.items() if count > 1)
    unknown_ids = sorted(api_id for api_id in returned_counts if api_id not in expected_set)
    returned_expected_ids = {api_id for api_id in returned_ids if api_id in expected_set}
    missing_ids = [api_id for api_id in expected_ids if api_id not in returned_expected_ids]
    issue_base = {
        "expected_api_count": len(expected_ids),
        "actual_api_count": len(returned_expected_ids),
        "returned_api_count": len(returned_ids),
    }

    if malformed_items:
        return ranked, {
            **issue_base,
            "reason": "parse_error",
            "malformed_ranked_item_count": malformed_items,
        }
    if duplicate_ids:
        return ranked, {
            **issue_base,
            "reason": "duplicate_ranked_apis",
            "duplicate_api_ids": duplicate_ids,
        }
    if unknown_ids:
        return ranked, {
            **issue_base,
            "reason": "unknown_api_ids",
            "unknown_api_ids": unknown_ids,
            "missing_api_ids": missing_ids,
        }
    if missing_ids:
        return ranked, {
            **issue_base,
            "reason": "incomplete_ranked_api_list",
            "missing_api_ids": missing_ids,
        }
    if len(returned_ids) != len(expected_ids):
        return ranked, {
            **issue_base,
            "reason": "missing_ranked_apis",
        }

    return ranked, None


def _ranker_retry_prompt(prompt: str, issue: Dict[str, Any]) -> str:
    reason = issue.get("reason", "invalid_ranking_output")
    expected = issue.get("expected_api_count", "")
    return (
        prompt
        + "\n\nIMPORTANT: The previous ranking output was invalid "
        + f"({reason}). Return JSON only using the output key requested by the prompt, containing every one of the {expected} candidate api_id values exactly once. "
        + "Do not omit candidates, duplicate candidates, or invent api_id values."
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


def _ranking_anomaly_metadata(issue: Dict[str, Any], *, after_retries: bool) -> Dict[str, Any]:
    reason = str(issue.get("reason") or "ranking_anomaly")
    anomaly_reason = reason if not after_retries else f"{reason}_after_retries"
    return {
        "ranking_anomaly": True,
        "ranking_anomaly_stage": "llm_ranking",
        "ranking_anomaly_reason": anomaly_reason,
        "failure_flag": False,
        "exclude_from_ranking_eval": False,
        **issue,
    }


def _attach_ranking_anomaly(ranked: List[Dict[str, Any]], issue: Dict[str, Any], *, after_retries: bool) -> List[Dict[str, Any]]:
    metadata = _ranking_anomaly_metadata(issue, after_retries=after_retries)
    log_ranking_anomaly_event(metadata)
    annotated: List[Dict[str, Any]] = []
    for item in ranked:
        row = dict(item)
        row.update(metadata)
        annotated.append(row)
    return annotated


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
    expected_ids = [str(c.get("api_id")) for c in cand_slim if c.get("api_id")]
    if not expected_ids:
        return []

    prompt = (
        template
        .replace("{user_query}", user_query)
        .replace("{subtask_json}", json.dumps(subtask, ensure_ascii=False))
        .replace("{candidates_json}", json.dumps(cand_slim, ensure_ascii=False))
    )

    attempts = max(0, int(max_validation_retries or 0)) + 1
    last_issue: Dict[str, Any] = {
        "reason": "parse_error",
        "expected_api_count": len(expected_ids),
        "actual_api_count": 0,
    }
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
        ranked, issue = _parse_ranked_output(resp_raw, expected_ids)
        if issue is None:
            return ranked[:ranker_pool_n]

        last_issue = issue
        if _is_recoverable_ranking_anomaly(issue):
            log_line(
                f"[ranker] recoverable LLM ranking anomaly ({issue.get('reason')}) "
                f"attempt {attempt}/{attempts}; expected={issue.get('expected_api_count')} actual={issue.get('actual_api_count')}"
            )
        else:
            log_line(
                f"[ranker] invalid LLM ranking output ({issue.get('reason')}) "
                f"attempt {attempt}/{attempts}; expected={issue.get('expected_api_count')} actual={issue.get('actual_api_count')}"
            )

        if _is_recoverable_ranking_anomaly(issue) and attempt == attempts:
            return _attach_ranking_anomaly(ranked[:ranker_pool_n], issue, after_retries=True)

    raise InvalidRankingOutput(_failure_metadata(last_issue, after_retries=True))
