from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class JsonParseResult:
    value: Any | None
    error: Dict[str, Any] | None = None
    json_text: str = ""


def strip_markdown_fences(text: str) -> str:
    text = (text or "").strip()
    fence_match = re.fullmatch(r"```\s*(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def extract_first_json_value(text: str) -> JsonParseResult:
    stripped = strip_markdown_fences(text)
    if not stripped:
        return JsonParseResult(None, {"reason": "empty_response"})

    decoder = json.JSONDecoder()
    starts = [idx for idx, char in enumerate(stripped) if char in "{["]
    if not starts:
        return JsonParseResult(None, {"reason": "invalid_json", "detail": "no_json_value_found"})

    first_error: str | None = None
    for start in starts:
        try:
            value, end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError as exc:
            if first_error is None:
                first_error = str(exc)
            continue
        json_text = stripped[start : start + end]
        return JsonParseResult(value, None, json_text)

    error: Dict[str, Any] = {"reason": "invalid_json"}
    if first_error:
        error["parse_error"] = first_error
    return JsonParseResult(None, error)


def parse_llm_json(text: str) -> JsonParseResult:
    return extract_first_json_value(text)


def normalize_llm_payload(
    payload: Any,
    expected_key: str,
    aliases: Mapping[str, str] | Iterable[str] | None = None,
    *,
    allow_list: bool = True,
) -> tuple[List[Any], Dict[str, Any] | None]:
    if isinstance(aliases, Mapping):
        alias_map = {str(k): str(v) for k, v in aliases.items()}
    else:
        alias_map = {str(alias): expected_key for alias in (aliases or [])}
    alias_map.setdefault(expected_key, expected_key)

    if isinstance(payload, list):
        if allow_list:
            return payload, None
        return [], {"reason": "wrong_json_type", "expected_type": "object", "actual_type": "list"}

    if not isinstance(payload, dict):
        return [], {
            "reason": "wrong_json_type",
            "expected_type": "object",
            "actual_type": type(payload).__name__,
        }

    for key, normalized_key in alias_map.items():
        if normalized_key != expected_key:
            continue
        value = payload.get(key)
        if value is not None:
            if isinstance(value, list):
                return value, None
            return [], {
                "reason": "wrong_json_type",
                "expected_key": expected_key,
                "actual_key": key,
                "expected_type": "list",
                "actual_type": type(value).__name__,
            }

    return [], {
        "reason": "missing_required_key",
        "expected_key": expected_key,
        "aliases": sorted(key for key, value in alias_map.items() if value == expected_key and key != expected_key),
    }


def _has_supported_score_identifier(item: Mapping[str, Any]) -> bool:
    return any(str(item.get(key) or "").strip() for key in ("candidate_id", "api_id"))


def _has_supported_score_field(item: Mapping[str, Any]) -> bool:
    return any(key in item for key in ("score", "qos_score"))


def recover_scores_key_from_single_list(
    payload: Any,
    *,
    expected_key: str = "scores",
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    """
    Recover QoS score rows when a model uses one unexpected top-level list key.

    This is intentionally narrow: it only rewrites object payloads with exactly
    one list-valued field, and only when the list is shaped like QoS score rows.
    The caller should still run schema and candidate-id validation afterwards.
    """
    if not isinstance(payload, dict) or expected_key in payload:
        return None, None

    list_fields = [(str(key), value) for key, value in payload.items() if isinstance(value, list)]
    if not list_fields:
        return None, None
    if len(list_fields) > 1:
        return None, {
            "reason": "ambiguous_score_list_key",
            "expected_key": expected_key,
            "candidate_keys": sorted(key for key, _value in list_fields),
        }

    original_key, items = list_fields[0]
    if not items:
        return None, None
    if not all(isinstance(item, Mapping) for item in items):
        return None, None
    if not all(_has_supported_score_identifier(item) for item in items):
        return None, None
    if not any(_has_supported_score_field(item) for item in items):
        return None, None

    normalized = dict(payload)
    normalized[expected_key] = items
    normalized.pop(original_key, None)
    return normalized, {
        "reason": "normalized_unexpected_scores_key",
        "expected_key": expected_key,
        "original_key": original_key,
    }


def validate_expected_ids(
    returned_ids: Sequence[str],
    expected_ids: Sequence[str],
    *,
    duplicate_reason: str = "duplicate_api_id",
    unknown_reason: str = "unknown_api_id",
    missing_reason: str = "incomplete_api_list",
    id_label: str = "api",
) -> Dict[str, Any] | None:
    expected_set = set(expected_ids)
    returned_counts = Counter(str(item).strip() for item in returned_ids if str(item).strip())
    duplicate_ids = sorted(item_id for item_id, count in returned_counts.items() if count > 1)
    unknown_ids = sorted(item_id for item_id in returned_counts if item_id not in expected_set)
    returned_expected_ids = {item_id for item_id in returned_counts if item_id in expected_set}
    missing_ids = [item_id for item_id in expected_ids if item_id not in returned_expected_ids]

    base = {
        f"expected_{id_label}_count": len(expected_ids),
        f"actual_{id_label}_count": len(returned_expected_ids),
        f"returned_{id_label}_count": len(returned_ids),
    }
    if duplicate_ids:
        return {**base, "reason": duplicate_reason, f"duplicate_{id_label}_ids": duplicate_ids}
    if unknown_ids:
        return {
            **base,
            "reason": unknown_reason,
            f"unknown_{id_label}_ids": unknown_ids,
            f"missing_{id_label}_ids": missing_ids,
        }
    if missing_ids:
        return {**base, "reason": missing_reason, f"missing_{id_label}_ids": missing_ids}
    if len(returned_ids) != len(expected_ids):
        return {**base, "reason": missing_reason, f"missing_{id_label}_ids": missing_ids}
    return None


def coerce_finite_score(value: Any) -> tuple[float | None, str | None]:
    if isinstance(value, bool) or value is None:
        return None, "missing_score"
    try:
        score = float(value)
    except Exception:
        return None, "invalid_score_value"
    if not math.isfinite(score):
        return None, "invalid_score_range"
    if score < 0.0 or score > 1.0:
        return None, "invalid_score_range"
    return score, None


def normalize_binary_label(value: Any) -> tuple[int | None, str | None]:
    if isinstance(value, bool):
        return (1 if value else 0), None
    if value in (0, 1):
        return int(value), None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "relevant"}:
            return 1, None
        if text in {"0", "false", "no", "irrelevant", "not relevant"}:
            return 0, None
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "missing_label"
    return None, "invalid_label_value"
