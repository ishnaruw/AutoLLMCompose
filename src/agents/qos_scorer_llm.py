from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.core.candidate_ids import assign_candidate_ids
from src.core.run_logging import log_line, log_retry_outcome_event


class InvalidQosScoringOutput(RuntimeError):
    def __init__(self, metadata: Dict[str, Any]) -> None:
        self.metadata = metadata
        super().__init__(str(metadata.get("failure_reason") or "invalid_qos_scoring_output"))


def _extract_json_text(raw: str) -> tuple[str, str | None]:
    text = (raw or "").strip()
    if not text:
        return "", "empty_response"
    try:
        json.loads(text)
        return text, None
    except Exception:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not match:
        return "", "invalid_json"
    return match.group(1), None


def _parse_llm_qos_scores_no_rank(raw: str, expected_ids: List[str] | None = None) -> Dict[str, float]:
    """
    Extract QoS scores from LLM output without ranking.
    Used for batched processing where global ranking happens after all batches.
    Returns api_id -> score.
    """
    scores, issue = _parse_qos_score_output(raw, expected_ids or [])
    if expected_ids is None:
        return scores
    return scores if issue is None else {}


def _parse_llm_qos_scores(raw: str, expected_ids: List[str] | None = None) -> Dict[str, Dict[str, Any]]:
    """
    Extract QoS scores from LLM output and assign ranks from LLM-provided scores.
    Returns api_id -> {qos_llm_score, qos_llm_rank}.
    """
    scores, issue = _parse_qos_score_output(raw, expected_ids or [])
    if expected_ids is not None and issue is not None:
        return {}
    return _rank_qos_scores(scores)


def _parse_qos_score_output(
    raw: str,
    expected_ids: List[str],
    candidate_id_to_api_id: Dict[str, str] | None = None,
) -> tuple[Dict[str, float], Dict[str, Any] | None]:
    json_text, json_error = _extract_json_text(raw)
    candidate_id_to_api_id = {
        str(candidate_id).strip(): str(api_id).strip()
        for candidate_id, api_id in (candidate_id_to_api_id or {}).items()
        if str(candidate_id).strip() and str(api_id).strip()
    }
    api_id_to_candidate_id: Dict[str, str] = {}
    for candidate_id, api_id in candidate_id_to_api_id.items():
        api_id_to_candidate_id.setdefault(api_id, candidate_id)
    expected_candidate_ids = list(candidate_id_to_api_id.keys())

    if json_error:
        issue = {
            "reason": json_error,
            "expected_api_count": len(expected_ids),
            "actual_api_count": 0,
        }
        if expected_candidate_ids:
            issue.update({"expected_candidate_count": len(expected_candidate_ids), "actual_candidate_count": 0})
        return {}, issue

    try:
        data = json.loads(json_text)
    except Exception as exc:
        issue = {
            "reason": "invalid_json",
            "expected_api_count": len(expected_ids),
            "actual_api_count": 0,
            "parse_error": str(exc),
        }
        if expected_candidate_ids:
            issue.update({"expected_candidate_count": len(expected_candidate_ids), "actual_candidate_count": 0})
        return {}, issue

    items = data.get("qos_scored") if isinstance(data, dict) else None
    if not isinstance(items, list):
        issue = {
            "reason": "parse_error",
            "expected_api_count": len(expected_ids),
            "actual_api_count": 0,
            "detail": "missing_qos_scored_list",
        }
        if expected_candidate_ids:
            issue.update({"expected_candidate_count": len(expected_candidate_ids), "actual_candidate_count": 0})
        return {}, issue

    contains_candidate_ids = any(
        isinstance(item, dict) and str(item.get("candidate_id") or "").strip()
        for item in items
    )
    if candidate_id_to_api_id and contains_candidate_ids:
        return _parse_qos_score_output_by_candidate_id(items, expected_ids, candidate_id_to_api_id, api_id_to_candidate_id)

    return _parse_qos_score_output_by_api_id(items, expected_ids, expected_candidate_ids)


def _parse_qos_score_output_by_api_id(
    items: List[Any],
    expected_ids: List[str],
    expected_candidate_ids: List[str] | None = None,
) -> tuple[Dict[str, float], Dict[str, Any] | None]:
    expected_candidate_ids = expected_candidate_ids or []
    expected_set = set(expected_ids)
    validate_expected_ids = bool(expected_ids)
    returned_ids: List[str] = []
    scores: Dict[str, float] = {}
    malformed_items = 0
    for item in items:
        if not isinstance(item, dict):
            malformed_items += 1
            continue
        api_id = str(item.get("api_id") or "").strip()
        if not api_id or item.get("qos_score") is None:
            malformed_items += 1
            continue
        returned_ids.append(api_id)
        if validate_expected_ids and api_id not in expected_set:
            continue
        try:
            scores[api_id] = float(item.get("qos_score"))
        except Exception:
            malformed_items += 1

    returned_counts = Counter(returned_ids)
    duplicate_ids = sorted(api_id for api_id, count in returned_counts.items() if count > 1)
    unknown_ids = sorted(api_id for api_id in returned_counts if validate_expected_ids and api_id not in expected_set)
    missing_ids = [api_id for api_id in expected_ids if api_id not in scores] if validate_expected_ids else []
    issue_base = {
        "expected_api_count": len(expected_ids),
        "actual_api_count": len(scores),
        "returned_api_count": len(returned_ids),
    }
    if expected_candidate_ids:
        issue_base.update(
            {
                "expected_candidate_count": len(expected_candidate_ids),
                "actual_candidate_count": len(scores),
            }
        )

    if malformed_items:
        return scores, {
            **issue_base,
            "reason": "parse_error",
            "malformed_qos_item_count": malformed_items,
        }
    if duplicate_ids:
        return scores, {
            **issue_base,
            "reason": "duplicate_api_ids",
            "duplicate_api_ids": duplicate_ids,
        }
    if unknown_ids:
        return scores, {
            **issue_base,
            "reason": "unknown_api_ids",
            "unknown_api_ids": unknown_ids,
            "missing_api_ids": missing_ids,
        }
    if missing_ids:
        reason = "missing_api_scores" if not scores else "incomplete_qos_scores"
        return scores, {
            **issue_base,
            "reason": reason,
            "missing_api_ids": missing_ids,
        }

    return scores, None


def _parse_qos_score_output_by_candidate_id(
    items: List[Any],
    expected_ids: List[str],
    candidate_id_to_api_id: Dict[str, str],
    api_id_to_candidate_id: Dict[str, str],
) -> tuple[Dict[str, float], Dict[str, Any] | None]:
    expected_candidate_ids = list(candidate_id_to_api_id.keys())
    expected_candidate_set = set(expected_candidate_ids)
    returned_candidate_ids: List[str] = []
    unknown_legacy_api_ids: List[str] = []
    scores: Dict[str, float] = {}
    malformed_items = 0
    for item in items:
        if not isinstance(item, dict):
            malformed_items += 1
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        raw_api_id = str(item.get("api_id") or "").strip()
        if not candidate_id and not raw_api_id:
            malformed_items += 1
            continue
        if item.get("qos_score") is None:
            malformed_items += 1
            continue

        if candidate_id:
            api_id = candidate_id_to_api_id.get(candidate_id, raw_api_id)
            returned_candidate_ids.append(candidate_id)
        else:
            api_id = raw_api_id
            candidate_id = api_id_to_candidate_id.get(raw_api_id, "")
            if candidate_id:
                returned_candidate_ids.append(candidate_id)
            else:
                unknown_legacy_api_ids.append(raw_api_id)

        if candidate_id not in expected_candidate_set:
            continue
        try:
            scores[api_id] = float(item.get("qos_score"))
        except Exception:
            malformed_items += 1

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
        "actual_api_count": len(scores),
        "returned_api_count": len(returned_candidate_ids) + len(unknown_legacy_api_ids),
        "expected_candidate_count": len(expected_candidate_ids),
        "actual_candidate_count": len(scores),
        "returned_candidate_count": len(returned_candidate_ids),
    }

    if malformed_items:
        return scores, {
            **issue_base,
            "reason": "parse_error",
            "malformed_qos_item_count": malformed_items,
        }
    if duplicate_candidate_ids:
        return scores, {
            **issue_base,
            "reason": "duplicate_candidate_ids",
            "duplicate_candidate_ids": duplicate_candidate_ids,
            "duplicate_api_ids": duplicate_api_ids,
        }
    if unknown_candidate_ids:
        return scores, {
            **issue_base,
            "reason": "unknown_candidate_ids",
            "unknown_candidate_ids": unknown_candidate_ids,
            "missing_candidate_ids": missing_candidate_ids,
            "missing_api_ids": missing_api_ids,
        }
    if unknown_api_ids:
        return scores, {
            **issue_base,
            "reason": "unknown_api_ids",
            "unknown_api_ids": unknown_api_ids,
            "missing_candidate_ids": missing_candidate_ids,
            "missing_api_ids": missing_api_ids,
        }
    if missing_candidate_ids:
        reason = "missing_api_scores" if not scores else "incomplete_qos_scores"
        return scores, {
            **issue_base,
            "reason": reason,
            "missing_candidate_ids": missing_candidate_ids,
            "missing_api_ids": missing_api_ids,
        }

    return scores, None


def _rank_qos_scores(scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    sorted_items = sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
    return {
        api_id: {
            "qos_llm_score": score,
            "qos_llm_rank": idx,
        }
        for idx, (api_id, score) in enumerate(sorted_items, start=1)
    }


def _write_qos_debug(debug_raw_path: str | None, raw: str, attempt: int, batch_idx: int | None) -> None:
    if not debug_raw_path:
        return
    path = Path(debug_raw_path)
    stem = path.stem
    if batch_idx is not None:
        stem = f"{stem}_batch{batch_idx}"
    if attempt > 1:
        stem = f"{stem}_retry{attempt - 1}"
    path = path.parent / f"{stem}{path.suffix}"
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


def _qos_retry_prompt(prompt: str, issue: Dict[str, Any]) -> str:
    reason = issue.get("reason", "invalid_qos_scores")
    expected = issue.get("expected_candidate_count") or issue.get("expected_api_count", "")
    if issue.get("expected_candidate_count"):
        return (
            prompt
            + "\n\nIMPORTANT: The previous QoS scoring output was invalid "
            + f"({reason}). Return JSON only with qos_scored containing every one of the {expected} input candidate_id values exactly once. "
            + "Use candidate_id only. Do not output api_id. Do not omit APIs, duplicate APIs, or invent candidate_id values."
        )
    return (
        prompt
        + "\n\nIMPORTANT: The previous QoS scoring output was invalid "
        + f"({reason}). Return JSON only with qos_scored containing every one of the {expected} input api_id values exactly once. "
        + "Do not omit APIs, duplicate APIs, or invent api_id values."
    )


def _failure_metadata(issue: Dict[str, Any], *, after_retries: bool) -> Dict[str, Any]:
    reason = str(issue.get("reason") or "parse_error")
    failure_reason = reason if reason == "timeout" or not after_retries else f"{reason}_after_retries"
    return {
        "failure_flag": True,
        "failure_stage": "qos_llm_scoring",
        "failure_reason": failure_reason,
        "exclude_from_ranking_eval": True,
        **issue,
    }


def _score_payload_with_retries(
    llm_call: Callable[[str], str],
    *,
    template: str,
    payload: List[Dict[str, Any]],
    candidate_id_to_api_id: Dict[str, str] | None,
    debug_raw_path: str | None,
    batch_idx: int | None,
    max_validation_retries: int,
) -> Dict[str, float]:
    candidate_id_to_api_id = candidate_id_to_api_id or {}
    expected_ids = list(candidate_id_to_api_id.values()) if candidate_id_to_api_id else [
        str(item.get("api_id") or "").strip() for item in payload if str(item.get("api_id") or "").strip()
    ]
    if not expected_ids:
        return {}

    prompt = template.replace("{candidates_json}", json.dumps(payload, ensure_ascii=False))
    attempts = max(0, int(max_validation_retries or 0)) + 1
    last_issue: Dict[str, Any] = {
        "reason": "parse_error",
        "expected_api_count": len(expected_ids),
        "actual_api_count": 0,
    }
    if candidate_id_to_api_id:
        last_issue.update(
            {
                "expected_candidate_count": len(candidate_id_to_api_id),
                "actual_candidate_count": 0,
            }
        )

    invalid_attempts = 0
    for attempt in range(1, attempts + 1):
        prompt_for_attempt = prompt if attempt == 1 else _qos_retry_prompt(prompt, last_issue)
        try:
            raw = llm_call(prompt_for_attempt)
        except Exception as exc:
            issue = {
                "reason": _exception_reason(exc),
                "expected_api_count": len(expected_ids),
                "actual_api_count": 0,
                "error": str(exc),
            }
            raise InvalidQosScoringOutput(_failure_metadata(issue, after_retries=False)) from exc

        _write_qos_debug(debug_raw_path, raw, attempt, batch_idx)
        scores, issue = _parse_qos_score_output(raw, expected_ids, candidate_id_to_api_id)
        if issue is None:
            if invalid_attempts:
                log_line(
                    f"[qos_scorer] retry validation succeeded after {invalid_attempts} invalid attempt(s); "
                    f"final_attempt={attempt}/{attempts}"
                )
                log_retry_outcome_event(
                    {
                        "stage": "qos_llm_scoring",
                        "role": "qos_scorer",
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
            return scores

        last_issue = issue
        invalid_attempts += 1
        log_line(
            f"[qos_scorer] invalid LLM QoS output ({issue.get('reason')}) "
            f"attempt {attempt}/{attempts}; expected={issue.get('expected_candidate_count', issue.get('expected_api_count'))} "
            f"actual={issue.get('actual_candidate_count', issue.get('actual_api_count'))}"
        )

    metadata = _failure_metadata(last_issue, after_retries=True)
    metadata["retry_invalid_attempts"] = invalid_attempts
    metadata["retry_max_attempts"] = attempts
    log_retry_outcome_event(
        {
            "stage": "qos_llm_scoring",
            "role": "qos_scorer",
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
    raise InvalidQosScoringOutput(metadata)


def score_qos_llm(
    llm_call: Callable[[str], str],
    *,
    candidates: List[Dict[str, Any]],
    prompt_path: str = "prompts/qos_score_llm.md",
    debug_raw_path: str | None = None,
    batch_size: int | None = 15,
    max_validation_retries: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """
    LLM-only QoS scoring with optional adaptive batching.

    Returns api_id -> {qos_llm_score, qos_llm_rank}.
    Invalid, empty, or incomplete LLM score outputs raise InvalidQosScoringOutput
    instead of manufacturing fallback scores or ranks.
    """
    template = Path(prompt_path).read_text(encoding="utf-8")
    try:
        effective_batch_size = 0 if batch_size is None else int(batch_size)
    except Exception:
        effective_batch_size = 0

    if not candidates:
        return {}

    candidate_rows = []
    for c in candidates:
        api_id = str(c.get("api_id") or "").strip()
        if not api_id:
            continue
        candidate_rows.append(
            {
                "api_id": api_id,
                "rt_ms": c.get("rt_ms"),
                "tp_rps": c.get("tp_rps"),
                "availability": c.get("availability"),
            }
        )
    candidate_rows = sorted(candidate_rows, key=lambda item: str(item.get("api_id") or "").lower())
    candidate_rows_with_ids, candidate_id_to_api_id, _api_id_to_candidate_id = assign_candidate_ids(candidate_rows)
    prompt_payload = [
        {
            "candidate_id": item.get("candidate_id"),
            "rt_ms": item.get("rt_ms"),
            "tp_rps": item.get("tp_rps"),
            "availability": item.get("availability"),
        }
        for item in candidate_rows_with_ids
    ]

    if effective_batch_size <= 0:
        scores = _score_payload_with_retries(
            llm_call,
            template=template,
            payload=prompt_payload,
            candidate_id_to_api_id=candidate_id_to_api_id,
            debug_raw_path=debug_raw_path,
            batch_idx=None,
            max_validation_retries=max_validation_retries,
        )
        return _rank_qos_scores(scores)

    all_scores: Dict[str, float] = {}
    for batch_idx, i in enumerate(range(0, len(prompt_payload), effective_batch_size)):
        batch_payload = prompt_payload[i : i + effective_batch_size]
        batch_mapping = {
            str(item.get("candidate_id")): candidate_id_to_api_id[str(item.get("candidate_id"))]
            for item in batch_payload
            if str(item.get("candidate_id")) in candidate_id_to_api_id
        }
        scores = _score_payload_with_retries(
            llm_call,
            template=template,
            payload=batch_payload,
            candidate_id_to_api_id=batch_mapping,
            debug_raw_path=debug_raw_path,
            batch_idx=batch_idx,
            max_validation_retries=max_validation_retries,
        )
        all_scores.update(scores)

    expected_ids = list(candidate_id_to_api_id.values())
    missing_ids = [api_id for api_id in expected_ids if api_id not in all_scores]
    if missing_ids:
        missing_id_set = set(missing_ids)
        missing_candidate_ids = [
            candidate_id
            for candidate_id, api_id in candidate_id_to_api_id.items()
            if api_id in missing_id_set
        ]
        raise InvalidQosScoringOutput(
            _failure_metadata(
                {
                    "reason": "incomplete_qos_scores",
                    "expected_api_count": len(expected_ids),
                    "expected_candidate_count": len(candidate_id_to_api_id),
                    "actual_api_count": len(all_scores),
                    "actual_candidate_count": len(all_scores),
                    "missing_candidate_ids": missing_candidate_ids,
                    "missing_api_ids": missing_ids,
                },
                after_retries=True,
            )
        )

    return _rank_qos_scores(all_scores)
