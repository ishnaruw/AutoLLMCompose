from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.core.candidate_ids import assign_candidate_ids
from src.core.json_parsing import (
    coerce_finite_score,
    normalize_llm_payload,
    parse_llm_json,
    recover_scores_key_from_single_list,
    validate_expected_ids,
)
from src.core.output_schemas import QoSScoreOutput, validate_output_schema
from src.core.run_logging import log_line, log_retry_outcome_event

QOS_FORMULA_TOLERANCE = 0.001


class InvalidQosScoringOutput(RuntimeError):
    def __init__(self, metadata: Dict[str, Any]) -> None:
        self.metadata = metadata
        super().__init__(str(metadata.get("failure_reason") or "invalid_qos_scoring_output"))


def _parse_qos_score_output(
    raw: str,
    expected_ids: List[str],
    candidate_id_to_api_id: Dict[str, str] | None = None,
) -> tuple[Dict[str, float], Dict[str, Any] | None]:
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
        return {}, issue

    payload_value = parsed_json.value
    items, key_issue = normalize_llm_payload(
        payload_value,
        "scores",
        aliases={"qos_scored": "scores", "qos_scores": "scores"},
        allow_list=False,
    )
    if key_issue:
        recovered_payload, recovery_issue = recover_scores_key_from_single_list(payload_value)
        if recovered_payload is not None:
            payload_value = recovered_payload
            items, key_issue = normalize_llm_payload(
                payload_value,
                "scores",
                aliases={"qos_scored": "scores", "qos_scores": "scores"},
                allow_list=False,
            )
            if key_issue is None:
                log_line(
                    "[qos_scorer] normalized unexpected QoS scores key "
                    f"original_key={recovery_issue.get('original_key') if recovery_issue else ''}"
                )
        elif recovery_issue is not None:
            key_issue = recovery_issue

    if key_issue:
        issue = {
            **key_issue,
            "expected_api_count": len(expected_ids),
            "actual_api_count": 0,
        }
        if expected_candidate_ids:
            issue.update({"expected_candidate_count": len(expected_candidate_ids), "actual_candidate_count": 0})
        return {}, issue

    _schema, schema_issue = validate_output_schema(QoSScoreOutput, {"scores": items})
    if schema_issue:
        issue = {
            **schema_issue,
            "expected_api_count": len(expected_ids),
            "actual_api_count": 0,
            "returned_api_count": len(items) if isinstance(items, list) else 0,
        }
        if expected_candidate_ids:
            issue.update(
                {
                    "expected_candidate_count": len(expected_candidate_ids),
                    "actual_candidate_count": 0,
                    "returned_candidate_count": len(items) if isinstance(items, list) else 0,
                }
            )
        return {}, issue

    contains_candidate_ids = any(
        isinstance(item, dict) and str(item.get("candidate_id") or "").strip()
        for item in items
    )
    if candidate_id_to_api_id and contains_candidate_ids:
        return _parse_qos_score_output_by_candidate_id(items, expected_ids, candidate_id_to_api_id, api_id_to_candidate_id)

    return _parse_qos_score_output_by_api_id(items, expected_ids, expected_candidate_ids)


def _score_value(item: Dict[str, Any]) -> Any:
    if "score" in item:
        return item.get("score")
    return item.get("qos_score")


def _qos_output_contract(include_reasons: bool) -> str:
    base = (
        "LLM output contract:\n"
        '- Return one compact JSON object only, with exactly one top-level key: "scores".\n'
        '- Preferred compact schema: {"scores": [{"candidate_id": "C01", "score": 0.75}]}.\n'
        '- Never return a bare candidate object like {"candidate_id": "C01", "score": 0.75}.\n'
        "- score must be a finite number from 0.0 to 1.0.\n"
        "- Return one item for every input candidate_id exactly once.\n"
        "- Use candidate_id in output and do not output api_id."
    )
    if include_reasons:
        return base + "\n- Optional: each item may include a short reason or explanation string."
    return base + "\n- Do not include reason, explanation, comments, or any prose."


def _apply_qos_output_contract(prompt: str, include_reasons: bool) -> str:
    contract = _qos_output_contract(include_reasons)
    if "{llm_output_contract}" in prompt:
        return prompt.replace("{llm_output_contract}", contract)
    return prompt + "\n\n" + contract


def _parse_qos_score_output_by_api_id(
    items: List[Any],
    expected_ids: List[str],
    expected_candidate_ids: List[str] | None = None,
) -> tuple[Dict[str, float], Dict[str, Any] | None]:
    expected_candidate_ids = expected_candidate_ids or []
    expected_set = set(expected_ids)
    should_validate_expected_ids = bool(expected_ids)
    returned_ids: List[str] = []
    scores: Dict[str, float] = {}
    malformed_items = 0
    score_errors: List[str] = []
    score_error_api_ids: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            malformed_items += 1
            continue
        api_id = str(item.get("api_id") or "").strip()
        if not api_id:
            malformed_items += 1
            continue
        score, score_error = coerce_finite_score(_score_value(item))
        if score_error:
            score_errors.append(score_error)
            score_error_api_ids.append(api_id)
            returned_ids.append(api_id)
            continue
        returned_ids.append(api_id)
        if should_validate_expected_ids and api_id not in expected_set:
            continue
        scores[api_id] = score

    missing_ids = [api_id for api_id in expected_ids if api_id not in scores] if should_validate_expected_ids else []
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
    id_issue = validate_expected_ids(
        returned_ids,
        expected_ids,
        duplicate_reason="duplicate_api_ids",
        unknown_reason="unknown_api_ids",
        missing_reason="incomplete_qos_scores",
        id_label="api",
    ) if should_validate_expected_ids else None
    if id_issue:
        return scores, {**issue_base, **id_issue}
    if score_errors:
        reason = score_errors[0]
        return scores, {
            **issue_base,
            "reason": reason,
            "invalid_score_api_ids": score_error_api_ids,
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
    score_errors: List[str] = []
    score_error_candidate_ids: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            malformed_items += 1
            continue
        candidate_id = str(item.get("candidate_id") or "").strip()
        raw_api_id = str(item.get("api_id") or "").strip()
        if not candidate_id and not raw_api_id:
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

        score, score_error = coerce_finite_score(_score_value(item))
        if score_error:
            score_errors.append(score_error)
            score_error_candidate_ids.append(candidate_id or raw_api_id)
            continue
        if candidate_id not in expected_candidate_set:
            continue
        scores[api_id] = score

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
    if score_errors:
        return scores, {
            **issue_base,
            "reason": score_errors[0],
            "invalid_score_candidate_ids": score_error_candidate_ids,
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
    sorted_items = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
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


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _qos_preference_context() -> Dict[str, Any]:
    return {
        "weights_provided": False,
        "preference": "Treat response time, throughput, and availability as equally important.",
        "weights": None,
        "fixed_formula": None,
    }


def _qos_formula_context(payload: List[Dict[str, Any]]) -> Dict[str, Any]:
    rt_values = [
        value
        for value in (_finite_float(item.get("rt_s")) for item in payload)
        if value is not None
    ]
    tp_values = [
        value
        for value in (_finite_float(item.get("tp_kbps")) for item in payload)
        if value is not None
    ]
    return {
        "max_rt_s": max(rt_values) if rt_values else 0.0,
        "max_tp_kbps": max(tp_values) if tp_values else 0.0,
        "total_candidate_count": len(payload),
    }


def _apply_normalization_context(prompt: str, context: Dict[str, Any]) -> str:
    context_json = json.dumps(context, ensure_ascii=False)
    if "{normalization_context}" in prompt:
        return prompt.replace("{normalization_context}", context_json)
    return prompt


def _score_from_formula(item: Dict[str, Any], context: Dict[str, Any]) -> float:
    rt_s = _finite_float(item.get("rt_s"))
    tp_kbps = _finite_float(item.get("tp_kbps"))
    availability = _finite_float(item.get("availability"))
    if rt_s is None or tp_kbps is None or availability is None:
        return 0.0

    max_rt = _finite_float(context.get("max_rt_s")) or 0.0
    max_tp = _finite_float(context.get("max_tp_kbps")) or 0.0
    norm_rt = 0.0 if max_rt <= 0 else max(0.0, min(1.0, 1.0 - (rt_s / max_rt)))
    norm_tp = 0.0 if max_tp <= 0 else max(0.0, min(1.0, tp_kbps / max_tp))
    norm_availability = max(0.0, min(1.0, availability))
    return round((norm_rt + norm_tp + norm_availability) / 3.0, 4)


def _expected_formula_scores_by_api_id(
    payload: List[Dict[str, Any]],
    candidate_id_to_api_id: Dict[str, str],
    context: Dict[str, Any],
) -> Dict[str, float]:
    expected: Dict[str, float] = {}
    for item in payload:
        candidate_id = str(item.get("candidate_id") or "").strip()
        api_id = candidate_id_to_api_id.get(candidate_id) or str(item.get("api_id") or "").strip()
        if not api_id:
            continue
        expected[api_id] = _score_from_formula(item, context)
    return expected


def _validate_qos_formula_scores(
    scores: Dict[str, float],
    payload: List[Dict[str, Any]],
    candidate_id_to_api_id: Dict[str, str],
    context: Dict[str, Any],
) -> Dict[str, Any] | None:
    expected_scores = _expected_formula_scores_by_api_id(payload, candidate_id_to_api_id, context)
    mismatches = []
    for api_id, expected_score in expected_scores.items():
        if api_id not in scores:
            continue
        actual_score = round(float(scores[api_id]), 4)
        delta = abs(actual_score - expected_score)
        if delta > QOS_FORMULA_TOLERANCE:
            mismatches.append((api_id, expected_score, actual_score, delta))

    if not mismatches:
        return None

    api_id_to_candidate_id = {api_id: candidate_id for candidate_id, api_id in candidate_id_to_api_id.items()}
    mismatches.sort(key=lambda item: item[3], reverse=True)
    issue = {
        "reason": "qos_score_formula_mismatch",
        "expected_api_count": len(expected_scores),
        "actual_api_count": len(scores),
        "mismatched_api_ids": [api_id for api_id, _expected, _actual, _delta in mismatches],
        "mismatched_candidate_ids": [
            api_id_to_candidate_id.get(api_id, "")
            for api_id, _expected, _actual, _delta in mismatches
            if api_id_to_candidate_id.get(api_id, "")
        ],
        "max_score_delta": round(mismatches[0][3], 6),
        "score_mismatches": [
            {
                "api_id": api_id,
                "candidate_id": api_id_to_candidate_id.get(api_id),
                "expected_score": expected_score,
                "actual_score": actual_score,
                "delta": round(delta, 6),
            }
            for api_id, expected_score, actual_score, delta in mismatches[:10]
        ],
    }
    if candidate_id_to_api_id:
        issue.update(
            {
                "expected_candidate_count": len(expected_scores),
                "actual_candidate_count": len(scores),
            }
        )
    return issue


def _log_qos_formula_audit(issue: Dict[str, Any], *, batch_idx: int | None) -> None:
    mismatch_count = len(issue.get("mismatched_api_ids") or [])
    if not mismatch_count:
        return
    batch_label = "all" if batch_idx is None else str(batch_idx)
    log_line(
        "[qos_scorer] formula audit mismatch "
        f"batch={batch_label} mismatched={mismatch_count} max_delta={issue.get('max_score_delta')}"
    )


def _qos_retry_prompt(prompt: str, issue: Dict[str, Any], payload: List[Dict[str, Any]] | None = None) -> str:
    reason = issue.get("reason", "invalid_qos_scores")
    expected = issue.get("expected_candidate_count") or issue.get("expected_api_count", "")
    candidate_ids = [
        str(item.get("candidate_id") or "").strip()
        for item in (payload or [])
        if str(item.get("candidate_id") or "").strip()
    ]
    candidate_list = ""
    if candidate_ids:
        candidate_list = "\nRequired candidate_id values for this attempt: " + json.dumps(candidate_ids, ensure_ascii=False)
    if issue.get("expected_candidate_count"):
        return (
            prompt
            + "\n\nIMPORTANT: The previous QoS scoring output was invalid "
            + f"({reason}). Return JSON only with scores containing every one of the {expected} input candidate_id values exactly once. "
            + "Use candidate_id only. Do not output api_id. Do not omit APIs, duplicate APIs, or invent candidate_id values."
            + ' The top-level object must be {"scores": [...]}; never return a bare {"candidate_id": ..., "score": ...} object.'
            + candidate_list
        )
    return (
        prompt
        + "\n\nIMPORTANT: The previous QoS scoring output was invalid "
        + f"({reason}). Return JSON only with scores containing every one of the {expected} input api_id values exactly once. "
        + "Do not omit APIs, duplicate APIs, or invent api_id values."
        + ' The top-level object must be {"scores": [...]}.'
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
    preference_context: Dict[str, Any] | None = None,
    formula_context: Dict[str, Any] | None = None,
    validate_formula: bool = False,
    formula_audit: bool = False,
) -> Dict[str, float]:
    candidate_id_to_api_id = candidate_id_to_api_id or {}
    expected_ids = list(candidate_id_to_api_id.values()) if candidate_id_to_api_id else [
        str(item.get("api_id") or "").strip() for item in payload if str(item.get("api_id") or "").strip()
    ]
    if not expected_ids:
        return {}

    from src.config import CONFIG

    preference_context = preference_context or _qos_preference_context()
    formula_context = formula_context or _qos_formula_context(payload)
    prompt = template.replace("{candidates_json}", json.dumps(payload, ensure_ascii=False))
    prompt = _apply_normalization_context(prompt, preference_context)
    prompt = _apply_qos_output_contract(prompt, bool(CONFIG.include_llm_reasons))
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
        prompt_for_attempt = prompt if attempt == 1 else _qos_retry_prompt(prompt, last_issue, payload)
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
            if validate_formula or formula_audit:
                formula_issue = _validate_qos_formula_scores(scores, payload, candidate_id_to_api_id, formula_context)
                if formula_issue and formula_audit:
                    _log_qos_formula_audit(formula_issue, batch_idx=batch_idx)
                if formula_issue and validate_formula:
                    raise InvalidQosScoringOutput(_failure_metadata(formula_issue, after_retries=False))
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
    batch_size: int | None = 0,
    max_validation_retries: int = 2,
    validate_formula: bool = False,
    formula_audit: bool = False,
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
                "rt_s": c.get("rt_s"),
                "tp_kbps": c.get("tp_kbps"),
                "availability": c.get("availability"),
            }
        )
    candidate_rows = sorted(candidate_rows, key=lambda item: str(item.get("api_id") or "").lower())
    candidate_rows_with_ids, candidate_id_to_api_id, _api_id_to_candidate_id = assign_candidate_ids(candidate_rows)
    prompt_payload = [
        {
            "candidate_id": item.get("candidate_id"),
            "rt_s": item.get("rt_s"),
            "tp_kbps": item.get("tp_kbps"),
            "availability": item.get("availability"),
        }
        for item in candidate_rows_with_ids
    ]
    preference_context = _qos_preference_context()
    formula_context = _qos_formula_context(prompt_payload)

    if effective_batch_size <= 0:
        scores = _score_payload_with_retries(
            llm_call,
            template=template,
            payload=prompt_payload,
            candidate_id_to_api_id=candidate_id_to_api_id,
            debug_raw_path=debug_raw_path,
            batch_idx=None,
            max_validation_retries=max_validation_retries,
            preference_context=preference_context,
            formula_context=formula_context,
            validate_formula=validate_formula,
            formula_audit=formula_audit,
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
            preference_context=preference_context,
            formula_context=formula_context,
            validate_formula=validate_formula,
            formula_audit=formula_audit,
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
