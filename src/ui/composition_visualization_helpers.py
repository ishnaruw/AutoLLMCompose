from __future__ import annotations

import json
import math
import re
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
NA = "N/A"


RESPONSE_TIME_LABEL = "Response Time (s)"
TOTAL_RESPONSE_TIME_LABEL = "Total Response Time (s)"
THROUGHPUT_LABEL = "Throughput (kbps)"
BOTTLENECK_THROUGHPUT_LABEL = "Bottleneck Throughput (kbps)"
WORKFLOW_AVAILABILITY_LABEL = "Average Workflow Availability"
WORKFLOW_AVAILABILITY_HELP = "Average of selected API availability values."
API_HEALTH_HELP = (
    "API QoS Health is normalized within each subtask's candidate pool. This means each selected "
    "API is compared only against alternative APIs for the same subtask, not against APIs from "
    "other workflow steps. Lower response time is better, while higher throughput and availability "
    "are better. This score is used only for visualization and explanation. It does not modify the "
    "original ranking, selection, planning, or evaluation results."
)
API_HEALTH_POOL_WARNING = "Subtask-level candidate pool unavailable; API QoS Health could not be computed reliably."

HEALTH_COLORS = {
    "green": "#C6EFCE",
    "orange": "#FCE4D6",
    "red": "#F4CCCC",
    "gray": "#E5E7EB",
    "blue": "#D9EAF7",
    "subtask": "#F8FAFC",
    "final": "#E2F0D9",
}

HEALTH_BORDER_COLORS = {
    "green": "#2F855A",
    "orange": "#B7791F",
    "red": "#C53030",
    "gray": "#64748B",
    "blue": "#0284C7",
    "subtask": "#94A3B8",
    "final": "#548235",
}

COMPOSITION_STATUS_COLORS = {
    "recommended": HEALTH_COLORS["green"],
    "valid_alternative": "#FFF2CC",
    "qos_risk": HEALTH_COLORS["orange"],
    "risk": HEALTH_COLORS["red"],
    "missing": HEALTH_COLORS["gray"],
}

COMPOSITION_STATUS_BORDERS = {
    "recommended": HEALTH_BORDER_COLORS["green"],
    "valid_alternative": "#B7791F",
    "qos_risk": HEALTH_BORDER_COLORS["orange"],
    "risk": HEALTH_BORDER_COLORS["red"],
    "missing": HEALTH_BORDER_COLORS["gray"],
}
COMPLETENESS_GATE_THRESHOLD = 0.999
BEST_MODE_TOLERANCE = 1e-9


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return False
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    try:
        return bool(result)
    except Exception:
        return False


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def _as_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _as_match_label(value: Any) -> int | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "relevant", "match", "functional match"}:
            return 1
        if text in {"0", "false", "no", "not relevant", "not a functional match"}:
            return 0
    return None


def _normalize_id(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _first_value(row: dict[str, Any] | pd.Series | None, *names: str) -> Any:
    if row is None:
        return None
    for name in names:
        if isinstance(row, pd.Series):
            if name in row and not _is_missing(row.get(name)):
                return row.get(name)
        elif isinstance(row, dict):
            if name in row and not _is_missing(row.get(name)):
                return row.get(name)
    return None


def _shorten(text: Any, max_len: int = 48) -> str:
    if _is_missing(text):
        return NA
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."


def _wrap_text(text: Any, width: int = 28, max_lines: int = 3) -> str:
    if _is_missing(text):
        return NA
    raw_words = re.sub(r"\s+", " ", str(text)).strip().split(" ")
    words: list[str] = []
    for raw_word in raw_words:
        word = raw_word
        while len(word) > width:
            words.append(word[:width])
            word = word[width:]
        if word:
            words.append(word)
    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + len(word) + 1 <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    remaining = " ".join(words)
    rendered = " ".join(lines)
    if len(rendered) < len(remaining):
        lines[-1] = lines[-1].rstrip(".") + "..."
    return "<br>".join(escape(line) for line in lines)


def _plain_label(text: Any, width: int = 32, max_lines: int = 2) -> str:
    return _wrap_text(text, width=width, max_lines=max_lines).replace("<br>", "\n")


def wrap_label(text: Any, width: int = 28, max_lines: int = 3) -> str:
    if _is_missing(text):
        return NA
    raw_words = re.sub(r"\s+", " ", str(text)).strip().split(" ")
    words: list[str] = []
    for raw_word in raw_words:
        word = raw_word
        while len(word) > width:
            words.append(word[:width])
            word = word[width:]
        if word:
            words.append(word)

    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + len(word) + 1 <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)

    rendered = " ".join(lines)
    full = " ".join(words)
    if lines and len(rendered) < len(full):
        lines[-1] = lines[-1].rstrip(".") + "..."
    return "\n".join(lines) or NA


def _line_fill(status: Any) -> str:
    status_key = str(status or "gray").lower()
    return HEALTH_COLORS.get(status_key, HEALTH_COLORS["gray"])


def _line_border(status: Any) -> str:
    status_key = str(status or "gray").lower()
    return HEALTH_BORDER_COLORS.get(status_key, HEALTH_BORDER_COLORS["gray"])


def _selection_status(row: pd.Series) -> Any:
    return row.get("API_Health_Status") or row.get("Selection_Quality_Status") or row.get("Health_Status")


def _selection_fill(row: pd.Series) -> str:
    return _line_fill(_selection_status(row))


def _selection_border(row: pd.Series) -> str:
    return _line_border(_selection_status(row))


def _selection_color(row: pd.Series) -> str:
    return row.get("API_Health_Color") or row.get("Selection_Quality_Color") or row.get("Health_Color") or HEALTH_COLORS["gray"]


def workflow_availability_value(eval_row: dict[str, Any] | pd.Series | None, workflow: pd.DataFrame | None = None) -> Any:
    value = _first_value(eval_row, "Average_Workflow_Availability", "Workflow_Availability")
    if _is_missing(value) and workflow is not None and not workflow.empty and "availability" in workflow:
        availability_values = pd.to_numeric(workflow["availability"], errors="coerce").dropna()
        if not availability_values.empty:
            value = float(availability_values.mean())
    return value


def _normalize_qos_component(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    present = numeric.dropna()
    out = pd.Series([None] * len(numeric), index=numeric.index, dtype=object)
    if present.empty:
        return out
    min_value = present.min()
    max_value = present.max()
    if max_value == min_value:
        out.loc[present.index] = 1.0
        return out
    if higher_is_better:
        out.loc[present.index] = (present - min_value) / (max_value - min_value)
    else:
        out.loc[present.index] = (max_value - present) / (max_value - min_value)
    return out


def classify_api_health(score: Any) -> tuple[str, str, str, str]:
    parsed = _as_float(score)
    if parsed is None:
        return "gray", "Unknown", "Unknown", HEALTH_COLORS["gray"]
    if parsed >= 0.75:
        return "green", "Strong QoS / Low Risk", "Low", HEALTH_COLORS["green"]
    if parsed >= 0.40:
        return "orange", "Moderate QoS / Medium Risk", "Medium", HEALTH_COLORS["orange"]
    return "red", "Weak QoS / High Risk", "High", HEALTH_COLORS["red"]


def functional_risk_label(value: Any) -> str:
    label = _as_match_label(value)
    if label == 1:
        return "Low"
    if label == 0:
        return "High"
    return "Unknown"


def qos_risk_label(row: pd.Series | dict[str, Any]) -> str:
    explicit = str(_first_value(row, "API_Risk_Label", "QoS_Risk") or "").strip().title()
    if explicit in {"Low", "Medium", "High", "Unknown"}:
        return explicit
    _, _, risk, _ = classify_api_health(_first_value(row, "API_QoS_Health"))
    return risk


def _composition_validity(eval_row: dict[str, Any] | pd.Series | None) -> int | None:
    return _as_match_label(_first_value(eval_row, "Composition_Validity", "Valid"))


def _composition_completeness_gate(eval_row: dict[str, Any] | pd.Series | None) -> float | None:
    gate = _as_float(_first_value(eval_row, "Composition_Completeness_Gate"))
    if gate is not None:
        return 1.0 if gate >= 1.0 else 0.0
    completeness = _as_float(_first_value(eval_row, "Composition_Completeness"))
    if completeness is None:
        return None
    return 1.0 if completeness >= COMPLETENESS_GATE_THRESHOLD else 0.0


def format_api_health(score: Any) -> str:
    return format_value(score, decimals=2)


def normalize_api_qos_scores(rows: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    out = pd.DataFrame(rows).copy()
    if out.empty:
        return out

    metric_specs = [
        ("rt_s", "Normalized_Response_Time_Score", False),
        ("tp_kbps", "Normalized_Throughput_Score", True),
        ("availability", "Normalized_Availability_Score", True),
    ]
    for raw_col, norm_col, higher_is_better in metric_specs:
        if raw_col in out:
            out[norm_col] = _normalize_qos_component(out[raw_col], higher_is_better=higher_is_better)
        else:
            out[norm_col] = None

    health_scores: list[float | None] = []
    selection_scores: list[float | None] = []
    statuses: list[str] = []
    labels: list[str] = []
    risks: list[str] = []
    colors: list[str] = []
    for _, row in out.iterrows():
        components = [
            _as_float(row.get("Normalized_Response_Time_Score")),
            _as_float(row.get("Normalized_Throughput_Score")),
            _as_float(row.get("Normalized_Availability_Score")),
        ]
        available_components = [value for value in components if value is not None]
        health = sum(available_components) / len(available_components) if available_components else None
        functional = _as_match_label(row.get("Functional_Match"))
        selection = (0.5 * float(functional) + 0.5 * health) if functional is not None and health is not None else None
        status, label, risk, color = classify_api_health(health)
        health_scores.append(health)
        selection_scores.append(selection)
        statuses.append(status)
        labels.append(label)
        risks.append(risk)
        colors.append(color)

    out["API_QoS_Health"] = health_scores
    out["API_Selection_Health"] = selection_scores
    out["API_Health_Status"] = statuses
    out["API_Health_Label"] = labels
    out["API_Risk_Label"] = risks
    out["API_Health_Color"] = colors
    return out


def _filter_qos_context_for_row(context: pd.DataFrame, row: pd.Series | dict[str, Any]) -> pd.DataFrame:
    if context.empty:
        return context
    filtered = context.copy()
    query_id = _first_value(row, "Query_ID", "query_id")
    if not _is_missing(query_id) and "Query_ID" in filtered:
        query_matches = filtered["Query_ID"].astype(str) == str(query_id)
        if query_matches.any():
            filtered = filtered[query_matches]
    mode = _first_value(row, "Mode", "mode")
    if not _is_missing(mode) and "Mode" in filtered:
        mode_matches = filtered["Mode"].astype(str) == str(mode)
        if mode_matches.any():
            filtered = filtered[mode_matches]
    subtask_id = _normalize_id(_first_value(row, "Subtask_ID", "Sub Task", "subtask_id"))
    if subtask_id and "Subtask_ID" in filtered:
        filtered = filtered[filtered["Subtask_ID"].astype(str).map(_normalize_id) == subtask_id]
    elif subtask_id and "Sub Task" in filtered:
        filtered = filtered[filtered["Sub Task"].astype(str).map(_normalize_id) == subtask_id]
    return filtered


def _fallback_api_qos_health(row: pd.Series | dict[str, Any]) -> tuple[float | None, str]:
    fallback_specs = [
        ("TOPSIS_Score", "TOPSIS score"),
        ("topsis_score", "TOPSIS score"),
        ("QoS_Score", "existing QoS score"),
        ("qos_score", "existing QoS score"),
        ("QoS_LLM_Score", "QoS LLM score"),
        ("qos_llm_score", "QoS LLM score"),
    ]
    for col, label in fallback_specs:
        value = _as_float(_first_value(row, col))
        if value is not None:
            return value, f"Fallback: {label}"
    return None, "Unavailable"


def compute_api_qos_health(row: pd.Series | dict[str, Any], context_rows: pd.DataFrame | list[dict[str, Any]]) -> float | None:
    context = _filter_qos_context_for_row(pd.DataFrame(context_rows).copy(), row)
    if context.empty:
        fallback, _ = _fallback_api_qos_health(row)
        return fallback
    normalized = normalize_api_qos_scores(context)
    if isinstance(row, pd.Series) and row.name in normalized.index:
        return _as_float(normalized.loc[row.name].get("API_QoS_Health"))
    api_id = _first_value(row, "API_ID", "api_id")
    subtask_id = _normalize_id(_first_value(row, "Subtask_ID", "Sub Task", "subtask_id"))
    if not _is_missing(api_id) and "API_ID" in normalized:
        matches = normalized[normalized["API_ID"].astype(str) == str(api_id)]
        if subtask_id and "Subtask_ID" in matches:
            matches = matches[matches["Subtask_ID"].astype(str).map(_normalize_id) == subtask_id]
        if not matches.empty:
            return _as_float(matches.iloc[0].get("API_QoS_Health"))
    fallback, _ = _fallback_api_qos_health(row)
    return fallback


def _candidate_api_id(row: dict[str, Any]) -> str:
    return str(row.get("Selected_API") or row.get("api_id") or row.get("API_ID") or "").strip()


def _candidate_subtask_id(row: dict[str, Any]) -> str:
    return _normalize_id(row.get("Sub Task", row.get("Subtask_ID", row.get("subtask_id"))))


def _candidate_qos_row(
    candidate: dict[str, Any],
    ranked_row: dict[str, Any] | None,
    *,
    query_id: str,
    mode: str,
    subtask_id: str,
) -> dict[str, Any]:
    ranked_row = ranked_row or {}
    service = ranked_row.get("service") if isinstance(ranked_row, dict) else {}
    service = service if isinstance(service, dict) else {}
    qos = service.get("qos") if isinstance(service.get("qos"), dict) else {}
    topsis_score = _first_value(candidate, "TOPSIS_Score", "topsis_score")
    if _is_missing(topsis_score):
        topsis_score = _first_value(ranked_row, "topsis_score")
    qos_llm_score = _first_value(candidate, "QoS_LLM_Score", "qos_llm_score")
    if _is_missing(qos_llm_score):
        qos_llm_score = _first_value(ranked_row, "qos_llm_score")
    return {
        "Query_ID": _first_value(candidate, "Query_ID", "query_id") or query_id,
        "Mode": _first_value(candidate, "Mode", "mode") or mode,
        "Subtask_ID": _candidate_subtask_id(candidate) or subtask_id,
        "API_ID": _candidate_api_id(candidate),
        "Functional_Match": _as_match_label(candidate.get("Functional Match (0/1)", candidate.get("Functional_Match"))),
        "rt_s": _as_float(candidate.get("QoS_RT_s")) if _as_float(candidate.get("QoS_RT_s")) is not None else _as_float(qos.get("rt_s")),
        "tp_kbps": _as_float(candidate.get("QoS_TP_kbps")) if _as_float(candidate.get("QoS_TP_kbps")) is not None else _as_float(qos.get("tp_kbps")),
        "availability": _as_float(candidate.get("QoS Availability")) if _as_float(candidate.get("QoS Availability")) is not None else _as_float(qos.get("availability")),
        "TOPSIS_Score": topsis_score,
        "QoS_LLM_Score": qos_llm_score,
    }


def _selected_row_for_qos_context(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    return {
        "Query_ID": _first_value(row, "Query_ID", "query_id"),
        "Mode": _first_value(row, "Mode", "mode"),
        "Subtask_ID": _first_value(row, "Subtask_ID", "Sub Task", "subtask_id"),
        "API_ID": _first_value(row, "API_ID", "Selected_API", "api_id"),
        "Functional_Match": _first_value(row, "Functional_Match", "Functional Match (0/1)"),
        "rt_s": _first_value(row, "rt_s", "QoS_RT_s"),
        "tp_kbps": _first_value(row, "tp_kbps", "QoS_TP_kbps"),
        "availability": _first_value(row, "availability", "QoS Availability"),
        "TOPSIS_Score": _first_value(row, "TOPSIS_Score", "topsis_score"),
        "QoS_LLM_Score": _first_value(row, "QoS_LLM_Score", "qos_llm_score"),
    }


def _append_api_health_columns(
    out: pd.DataFrame,
    *,
    index: Any,
    normalized_row: pd.Series | None,
    source: str,
    warning: str = "",
) -> None:
    health = _as_float(normalized_row.get("API_QoS_Health")) if normalized_row is not None else None
    functional = _as_match_label(out.at[index, "Functional_Match"]) if "Functional_Match" in out else None
    selection = (0.5 * float(functional) + 0.5 * health) if functional is not None and health is not None else None
    status, label, risk, color = classify_api_health(health)
    for col in ["Normalized_Response_Time_Score", "Normalized_Throughput_Score", "Normalized_Availability_Score"]:
        out.at[index, col] = normalized_row.get(col) if normalized_row is not None and col in normalized_row else None
    out.at[index, "API_QoS_Health"] = health
    out.at[index, "API_Selection_Health"] = selection
    out.at[index, "API_Health_Status"] = status
    out.at[index, "API_Health_Label"] = label
    out.at[index, "API_Risk_Label"] = risk
    out.at[index, "API_Health_Color"] = color
    out.at[index, "API_QoS_Health_Source"] = source
    out.at[index, "API_QoS_Health_Warning"] = warning


def _append_composition_risk_columns(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty:
        return out
    out = out.copy()
    out["Functional_Risk"] = [functional_risk_label(row.get("Functional_Match")) for _, row in out.iterrows()]
    out["QoS_Risk"] = [qos_risk_label(row) for _, row in out.iterrows()]
    out["Composition_Risk"] = [
        "Functional Risk" if row.get("Functional_Risk") == "High" else "QoS Risk" if row.get("QoS_Risk") == "High" else "Low"
        for _, row in out.iterrows()
    ]
    return out


def normalize_workflow_api_qos_scores_by_subtask_candidates(
    rows: pd.DataFrame | list[dict[str, Any]],
    *,
    run_dir: Path,
    query_id: str,
    mode: str,
) -> pd.DataFrame:
    out = pd.DataFrame(rows).copy()
    if out.empty:
        return out

    health_cols = [
        "Normalized_Response_Time_Score",
        "Normalized_Throughput_Score",
        "Normalized_Availability_Score",
        "API_QoS_Health",
        "API_Selection_Health",
        "API_Health_Status",
        "API_Health_Label",
        "API_Risk_Label",
        "API_Health_Color",
        "API_QoS_Health_Source",
        "API_QoS_Health_Warning",
    ]
    for col in health_cols:
        if col not in out:
            out[col] = None

    candidate_cache: dict[str, list[dict[str, Any]]] = {}
    ranked_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for idx, row in out.iterrows():
        sid = _normalize_id(_first_value(row, "Subtask_ID", "Sub Task", "subtask_id"))
        api_id = str(_first_value(row, "API_ID", "Selected_API", "api_id") or "").strip()
        candidate_rows = candidate_cache.setdefault(sid, load_candidate_rows(run_dir, query_id, mode=mode, subtask_id=sid))
        if not candidate_rows:
            fallback, source = _fallback_api_qos_health(row)
            fallback_row = pd.Series({"API_QoS_Health": fallback}) if fallback is not None else None
            _append_api_health_columns(out, index=idx, normalized_row=fallback_row, source=source, warning=API_HEALTH_POOL_WARNING)
            continue

        ranked_lookup = ranked_cache.setdefault(sid, load_ranked_candidate_lookup(run_dir, mode, sid))
        pool_rows = [
            _candidate_qos_row(candidate, ranked_lookup.get(_candidate_api_id(candidate)), query_id=query_id, mode=mode, subtask_id=sid)
            for candidate in candidate_rows
            if _candidate_api_id(candidate)
        ]
        if not pool_rows:
            fallback, source = _fallback_api_qos_health(row)
            fallback_row = pd.Series({"API_QoS_Health": fallback}) if fallback is not None else None
            _append_api_health_columns(out, index=idx, normalized_row=fallback_row, source=source, warning=API_HEALTH_POOL_WARNING)
            continue
        if not any(str(pool.get("API_ID") or "") == api_id for pool in pool_rows):
            pool_rows.append(_selected_row_for_qos_context(row))
        normalized_pool = normalize_api_qos_scores(pd.DataFrame(pool_rows))
        matches = normalized_pool[normalized_pool["API_ID"].astype(str) == api_id] if "API_ID" in normalized_pool else pd.DataFrame()
        normalized_row = matches.iloc[0] if not matches.empty else None
        _append_api_health_columns(out, index=idx, normalized_row=normalized_row, source="Subtask candidate pool")
    return out


def _hover_for_step(row: pd.Series, *, title: str | None = None) -> str:
    title_text = title or row.get("API_Name") or row.get("API_ID") or "Workflow step"
    parts = [
        f"<b>{escape(str(title_text))}</b>",
        f"Subtask: {escape(str(row.get('Subtask_ID') or NA))}",
        f"API ID: {escape(str(row.get('API_ID') or NA))}",
        f"Functional fit: {format_functional_fit(row.get('Functional_Match'))}",
        f"Mode rank: {format_value(row.get('Mode_Rank'), 0)}",
        f"{RESPONSE_TIME_LABEL}: {format_response_time(row.get('rt_s'))}",
        f"{THROUGHPUT_LABEL}: {format_throughput(row.get('tp_kbps'))}",
        f"Availability: {format_value(row.get('availability'))}",
        f"API QoS Health: {format_api_health(row.get('API_QoS_Health'))}",
        f"API QoS Health source: {escape(str(row.get('API_QoS_Health_Source') or NA))}",
        f"API Selection Health: {format_api_health(row.get('API_Selection_Health'))}",
        f"Risk: {escape(str(row.get('API_Risk_Label') or NA))}",
        f"QoS LLM score: {format_value(row.get('QoS_LLM_Score'))}",
        f"TOPSIS score: {format_value(row.get('TOPSIS_Score'))}",
        f"API health: {escape(str(row.get('API_Health_Label') or row.get('Selection_Quality_Reason') or row.get('Health_Reason') or NA))}",
    ]
    if not _is_missing(row.get("Bottleneck_Dimensions")):
        parts.append(f"QoS Signal: {escape(str(row.get('Bottleneck_Dimensions')))}")
    if not _is_missing(row.get("API_QoS_Health_Warning")):
        parts.append(escape(str(row.get("API_QoS_Health_Warning"))))
    if not _is_missing(row.get("Subtask")):
        parts.insert(2, f"Subtask purpose: {escape(str(row.get('Subtask')))}")
    if not _is_missing(row.get("Action")):
        parts.append(f"Action: {escape(str(row.get('Action')))}")
    return "<br>".join(parts)


def _dot_escape(text: Any) -> str:
    value = "" if _is_missing(text) else str(text)
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _sort_key(value: Any) -> tuple[int, str]:
    text = _normalize_id(value)
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)


def format_value(value: Any, decimals: int = 3, *, percent: bool = False) -> str:
    parsed = _as_float(value)
    if parsed is None:
        return NA
    if percent:
        return f"{parsed * 100:.1f}%"
    if abs(parsed) >= 100:
        return f"{parsed:.1f}"
    return f"{parsed:.{decimals}f}"


def format_response_time(value: Any) -> str:
    formatted = format_value(value)
    return NA if formatted == NA else f"{formatted} s"


def format_throughput(value: Any) -> str:
    formatted = format_value(value)
    return NA if formatted == NA else f"{formatted} kbps"


def format_flag(value: Any) -> str:
    label = _as_match_label(value)
    return NA if label is None else str(label)


def format_functional_fit(value: Any) -> str:
    label = _as_match_label(value)
    if label == 1:
        return "Yes"
    if label == 0:
        return "No"
    return NA


def _subtask_id_from_stage_path(path: Path) -> str:
    match = re.search(r"(?:2_ranked|3_selected)_s([^./]+)", path.name)
    return _normalize_id(match.group(1)) if match else ""


def _subtask_id_from_selected_path(path: Path) -> str:
    return _subtask_id_from_stage_path(path)


def load_query_context(run_dir: Path, query_id: str, query_lookup: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    meta = _read_json(run_dir / "meta.json", {})
    query_lookup = query_lookup or {}
    fallback = query_lookup.get(str(query_id), {})
    title = str(meta.get("query_title") or fallback.get("title") or "").strip()
    goal = str(meta.get("user_goal") or fallback.get("goal") or "").strip()
    label = f"{query_id} | {title}" if title else str(query_id)
    return {"query_id": str(query_id), "title": title, "goal": goal, "label": label}


def load_subtask_descriptions(run_dir: Path) -> dict[str, str]:
    payload = _read_json(run_dir / "0_decomposer.json", [])
    subtasks = payload.get("subtasks", []) if isinstance(payload, dict) else payload
    out: dict[str, str] = {}
    if not isinstance(subtasks, list):
        return out
    for idx, item in enumerate(subtasks, start=1):
        if not isinstance(item, dict):
            continue
        sid = _normalize_id(item.get("id") or idx)
        description = str(item.get("description") or item.get("subtask") or item.get("goal") or "").strip()
        if sid:
            out[sid] = description
    return out


def _candidate_rows_path(run_dir: Path, query_id: str) -> Path | None:
    direct = run_dir / "evaluation" / f"query_{query_id}_candidate_api_rankings_rows.json"
    if direct.exists():
        return direct
    matches = sorted((run_dir / "evaluation").glob("query_*_candidate_api_rankings_rows.json"))
    return matches[0] if matches else None


def load_candidate_lookup(run_dir: Path, query_id: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = _candidate_rows_path(run_dir, query_id)
    payload = _read_json(path, []) if path else []
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not isinstance(payload, list):
        return lookup
    for row in payload:
        if not isinstance(row, dict):
            continue
        mode = str(row.get("Mode") or row.get("mode") or "").strip()
        sid = _normalize_id(row.get("Sub Task", row.get("Subtask_ID", row.get("subtask_id"))))
        api_id = str(row.get("Selected_API") or row.get("api_id") or "").strip()
        if mode and sid and api_id:
            lookup.setdefault((mode, sid, api_id), row)
    return lookup


def load_candidate_rows(run_dir: Path, query_id: str, *, mode: str, subtask_id: str) -> list[dict[str, Any]]:
    path = _candidate_rows_path(run_dir, query_id)
    payload = _read_json(path, []) if path else []
    if not isinstance(payload, list):
        return []
    sid = _normalize_id(subtask_id)
    mode_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        row_query = row.get("Query_ID", row.get("query_id"))
        if not _is_missing(row_query) and str(row_query) != str(query_id):
            continue
        row_mode = str(row.get("Mode") or row.get("mode") or "").strip()
        row_sid = _normalize_id(row.get("Sub Task", row.get("Subtask_ID", row.get("subtask_id"))))
        if row_sid != sid:
            continue
        if row_mode == mode:
            mode_rows.append(row)
        else:
            fallback_rows.append(row)
    return mode_rows or fallback_rows


def load_ranked_candidate_lookup(run_dir: Path, mode: str, subtask_id: str) -> dict[str, dict[str, Any]]:
    sid = _normalize_id(subtask_id)
    path = run_dir / mode / f"2_ranked_s{sid}.json"
    payload = _read_json(path, [])
    rows = payload if isinstance(payload, list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        api_id = str(row.get("api_id") or row.get("Selected_API") or "").strip()
        if api_id:
            out.setdefault(api_id, row)
    return out


def load_selected_lookups(run_dir: Path, mode: str) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    by_api: dict[str, dict[str, Any]] = {}
    mode_dir = run_dir / mode
    for path in sorted(mode_dir.glob("3_selected_s*.json")):
        sid_from_path = _subtask_id_from_selected_path(path)
        payload = _read_json(path, [])
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            api_id = str(row.get("api_id") or row.get("Selected_API") or "").strip()
            sid = _normalize_id(row.get("subtask_id") or row.get("Sub Task") or sid_from_path)
            if not api_id:
                continue
            if sid:
                by_pair.setdefault((sid, api_id), row)
            by_api.setdefault(api_id, row)
    return by_pair, by_api


def load_qos_llm_lookup(run_dir: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    by_api: dict[str, dict[str, Any]] = {}
    qos_dir = run_dir / "qos_pure_llm"
    paths = sorted(qos_dir.glob("2_ranked_s*.json")) + sorted(qos_dir.glob("3_selected_s*.json"))
    for path in paths:
        sid_from_path = _subtask_id_from_stage_path(path)
        payload = _read_json(path, [])
        rows = payload if isinstance(payload, list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if _is_missing(_first_value(row, "qos_llm_score", "qos_llm_rank")):
                continue
            api_id = str(row.get("api_id") or row.get("Selected_API") or "").strip()
            sid = _normalize_id(row.get("subtask_id") or row.get("Sub Task") or sid_from_path)
            if not api_id:
                continue
            if sid:
                by_pair.setdefault((sid, api_id), row)
            by_api.setdefault(api_id, row)
    return by_pair, by_api


def _selected_for_step(
    selected_by_pair: dict[tuple[str, str], dict[str, Any]],
    selected_by_api: dict[str, dict[str, Any]],
    subtask_id: str,
    api_id: str,
) -> dict[str, Any] | None:
    return selected_by_pair.get((subtask_id, api_id)) or selected_by_api.get(api_id)


def _service_from_selected(selected: dict[str, Any] | None) -> dict[str, Any]:
    service = selected.get("service") if isinstance(selected, dict) else None
    return service if isinstance(service, dict) else {}


def _qos_from_service(selected: dict[str, Any] | None, key: str) -> float | None:
    service = _service_from_selected(selected)
    qos = service.get("qos")
    if not isinstance(qos, dict):
        qos = selected.get("qos") if isinstance(selected, dict) else None
    return _as_float(qos.get(key)) if isinstance(qos, dict) else None


def _api_name(selected: dict[str, Any] | None, api_id: str) -> str:
    service = _service_from_selected(selected)
    name = str(service.get("name") or "").strip()
    tool_name = str(service.get("tool_name") or "").strip()
    if tool_name and name and name.lower() not in tool_name.lower():
        return f"{tool_name}: {name}"
    return tool_name or name or api_id


def _filter_workflow(
    workflow_df: pd.DataFrame,
    *,
    query_id: str,
    run_dir: Path,
    run_name: str,
    mode: str,
) -> pd.DataFrame:
    if workflow_df.empty:
        return workflow_df.copy()
    filtered = workflow_df.copy()
    if "Query_ID" in filtered:
        filtered = filtered[filtered["Query_ID"].astype(str) == str(query_id)]
    if "Mode" in filtered:
        filtered = filtered[filtered["Mode"].astype(str) == str(mode)]
    if "run_dir" in filtered:
        run_dir_text = str(run_dir)
        filtered = filtered[filtered["run_dir"].astype(str) == run_dir_text]
    elif "run_name" in filtered:
        filtered = filtered[filtered["run_name"].astype(str) == str(run_name)]
    if "Step" in filtered:
        filtered = filtered.assign(_step_sort=pd.to_numeric(filtered["Step"], errors="coerce")).sort_values(
            ["_step_sort", "Subtask_ID"] if "Subtask_ID" in filtered else ["_step_sort"],
            kind="stable",
        )
        filtered = filtered.drop(columns=["_step_sort"])
    return filtered


def eval_row_for_mode(
    eval_df: pd.DataFrame,
    *,
    query_id: str,
    run_dir: Path,
    run_name: str,
    mode: str,
) -> dict[str, Any]:
    if eval_df.empty:
        return {}
    filtered = eval_df.copy()
    if "Query_ID" in filtered:
        filtered = filtered[filtered["Query_ID"].astype(str) == str(query_id)]
    if "Mode" in filtered:
        filtered = filtered[filtered["Mode"].astype(str) == str(mode)]
    if "run_dir" in filtered:
        filtered = filtered[filtered["run_dir"].astype(str) == str(run_dir)]
    elif "run_name" in filtered:
        filtered = filtered[filtered["run_name"].astype(str) == str(run_name)]
    if filtered.empty:
        return {}
    return filtered.iloc[0].to_dict()


def get_recommended_mode(
    eval_df: pd.DataFrame,
    query_id: str,
    *,
    run_dir: Path | None = None,
    run_name: str | None = None,
    modes: list[str] | None = None,
) -> dict[str, Any]:
    if eval_df.empty:
        return {"status": "unavailable", "mode": NA, "row": {}, "reason": "Composition evaluation data is missing."}

    filtered = eval_df.copy()
    if "Query_ID" in filtered:
        filtered = filtered[filtered["Query_ID"].astype(str) == str(query_id)]
    if run_dir is not None and "run_dir" in filtered:
        filtered = filtered[filtered["run_dir"].astype(str) == str(run_dir)]
    elif run_name is not None and "run_name" in filtered:
        filtered = filtered[filtered["run_name"].astype(str) == str(run_name)]
    if modes and "Mode" in filtered:
        filtered = filtered[filtered["Mode"].astype(str).isin(modes)]
    if filtered.empty or "Mode" not in filtered:
        return {"status": "unavailable", "mode": NA, "row": {}, "reason": "Recommendation unavailable."}

    filtered = filtered.copy()
    filtered["_mode_order"] = filtered["Mode"].astype(str).map({mode: idx for idx, mode in enumerate(MODE_ORDER)}).fillna(len(MODE_ORDER))
    validity_col = "Composition_Validity" if "Composition_Validity" in filtered else "Valid" if "Valid" in filtered else None
    filtered["_validity"] = filtered[validity_col].apply(_as_match_label) if validity_col else None
    filtered["_complete"] = filtered.apply(lambda row: _composition_completeness_gate(row), axis=1)
    filtered["_score"] = pd.to_numeric(filtered.get("QoS_Adjusted_Composition_Score"), errors="coerce")
    valid_rows = filtered[filtered["_validity"] == 1].copy()
    complete_valid_rows = filtered[(filtered["_validity"] == 1) & (filtered["_complete"] == 1.0)].copy()
    has_complete_valid = not complete_valid_rows.empty
    has_valid = not valid_rows.empty
    candidates = complete_valid_rows if has_complete_valid else valid_rows if has_valid else filtered.copy()

    scored = candidates[candidates["_score"].notna()].copy()
    if scored.empty:
        return {
            "status": "unavailable",
            "mode": NA,
            "row": {},
            "reason": "Recommendation unavailable.",
            "warning": "All QoS-adjusted composition scores are missing for this query.",
            "valid_candidate_count": int(len(valid_rows)),
            "complete_candidate_count": int(len(complete_valid_rows)),
        }

    tie_specs = [
        ("Functional_Coverage", False),
        ("Composition_Completeness", False),
        ("Normalized_QoS_Score", False),
        ("Total_Response_Time_s", True),
        ("Bottleneck_Throughput_kbps", False),
        ("Average_Workflow_Availability", False),
    ]
    for col, _ in tie_specs:
        if col not in scored:
            scored[col] = None
        scored[f"_{col}"] = pd.to_numeric(scored[col], errors="coerce")

    sort_cols = ["_score"] + [f"_{col}" for col, _ in tie_specs] + ["_mode_order"]
    ascending = [False] + [ascending for _, ascending in tie_specs] + [True]
    best_score = scored["_score"].max()
    tied = scored[(scored["_score"] - best_score).abs() <= BEST_MODE_TOLERANCE].copy()
    best_modes = sorted(
        tied["Mode"].dropna().astype(str).unique().tolist(),
        key=lambda mode: MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER),
    )
    best = scored.sort_values(sort_cols, ascending=ascending, na_position="last", kind="mergesort").iloc[0]
    status = "recommended" if has_complete_valid else "diagnostic"
    is_tie = len(best_modes) > 1
    result: dict[str, Any] = {
        "status": status,
        "mode": str(best.get("Mode") or NA),
        "modes": best_modes,
        "best_value": best_score,
        "is_tie": is_tie,
        "row": {key: value for key, value in best.to_dict().items() if not str(key).startswith("_")},
        "reason": (
            "Tied highest QoS-adjusted composition score among valid complete modes"
            if has_complete_valid and is_tie
            else "Highest QoS-adjusted composition score among valid complete modes"
            if has_complete_valid
            else "No valid complete workflow is available; showing tied highest-scoring diagnostic modes"
            if is_tie
            else "No valid complete workflow is available; showing the highest-scoring diagnostic mode"
        ),
        "valid_candidate_count": int(len(valid_rows)),
        "complete_candidate_count": int(len(complete_valid_rows)),
    }

    if has_complete_valid:
        recommended_fc = _as_float(best.get("Functional_Coverage"))
        complete_valid_rows = complete_valid_rows.copy()
        complete_valid_rows["_fc"] = pd.to_numeric(complete_valid_rows.get("Functional_Coverage"), errors="coerce")
        higher_fc = pd.DataFrame()
        if recommended_fc is not None:
            higher_fc = complete_valid_rows[complete_valid_rows["_fc"].notna()]
            higher_fc = higher_fc[higher_fc["_fc"] > recommended_fc]
        if not higher_fc.empty:
            best_fc = higher_fc.sort_values(["_fc", "_score", "_mode_order"], ascending=[False, False, True], na_position="last").iloc[0]
            result["tradeoff_mode"] = str(best_fc.get("Mode") or NA)
            result["tradeoff_message"] = (
                f"{result['tradeoff_mode']} has higher functional coverage, but "
                f"{result['mode']} has the stronger final QoS-adjusted score."
            )
    return result


def enrich_workflow_for_selection(
    workflow_df: pd.DataFrame,
    *,
    query_id: str,
    run_dir: Path,
    run_name: str,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    workflow = _filter_workflow(workflow_df, query_id=query_id, run_dir=run_dir, run_name=run_name, mode=mode)
    if workflow.empty:
        return pd.DataFrame(), pd.DataFrame()

    subtasks = load_subtask_descriptions(run_dir)
    candidate_lookup = load_candidate_lookup(run_dir, query_id)
    selected_by_pair, selected_by_api = load_selected_lookups(run_dir, mode)
    qos_llm_by_pair, qos_llm_by_api = load_qos_llm_lookup(run_dir)

    rows: list[dict[str, Any]] = []
    for idx, (_, step) in enumerate(workflow.iterrows(), start=1):
        sid = _normalize_id(_first_value(step, "Subtask_ID", "Sub Task", "subtask_id"))
        api_id = str(_first_value(step, "API_ID", "Selected_API", "api_id") or "").strip()
        selected = _selected_for_step(selected_by_pair, selected_by_api, sid, api_id)
        qos_llm_reference = _selected_for_step(qos_llm_by_pair, qos_llm_by_api, sid, api_id)
        candidate = candidate_lookup.get((mode, sid, api_id), {})

        functional = _as_match_label(_first_value(step, "Functional_Match", "Functional Match"))
        if functional is None:
            functional = _as_match_label(_first_value(candidate, "Functional Match (0/1)", "Functional_Match"))

        rt_s = _as_float(_first_value(step, "rt_s", "QoS_RT_s"))
        tp_kbps = _as_float(_first_value(step, "tp_kbps", "QoS_TP_kbps"))
        availability = _as_float(_first_value(step, "availability", "QoS Availability"))
        if rt_s is None:
            rt_s = _as_float(_first_value(candidate, "QoS_RT_s", "rt_s"))
            if rt_s is None:
                rt_s = _qos_from_service(selected, "rt_s")
        if tp_kbps is None:
            tp_kbps = _as_float(_first_value(candidate, "QoS_TP_kbps", "tp_kbps"))
            if tp_kbps is None:
                tp_kbps = _qos_from_service(selected, "tp_kbps")
        if availability is None:
            availability = _as_float(_first_value(candidate, "QoS Availability", "availability"))
            if availability is None:
                availability = _qos_from_service(selected, "availability")

        service = _service_from_selected(selected)
        qos_llm_score = _first_value(selected, "qos_llm_score")
        if _is_missing(qos_llm_score):
            qos_llm_score = _first_value(qos_llm_reference, "qos_llm_score")
        qos_llm_rank = _first_value(selected, "qos_llm_rank")
        if _is_missing(qos_llm_rank):
            qos_llm_rank = _first_value(qos_llm_reference, "qos_llm_rank")
        rows.append(
            {
                "Query_ID": query_id,
                "Mode": mode,
                "Step": _first_value(step, "Step") or idx,
                "Subtask_ID": sid,
                "Subtask": subtasks.get(sid, ""),
                "API_ID": api_id,
                "API_Name": _api_name(selected, api_id),
                "Functional_Match": functional,
                "rt_s": rt_s,
                "tp_kbps": tp_kbps,
                "availability": availability,
                "Mode_Rank": _first_value(selected, "mode_rank", "Mode Rank") or _first_value(candidate, "Mode Rank", "mode_rank"),
                "Selected_Rank": _first_value(selected, "selected_rank"),
                "Retrieved_Rank": _first_value(selected, "retrieved_rank") or _first_value(candidate, "Retrieved Rank"),
                "QoS_LLM_Score": qos_llm_score,
                "QoS_LLM_Rank": qos_llm_rank,
                "TOPSIS_Score": _first_value(selected, "topsis_score"),
                "TOPSIS_Rank": _first_value(selected, "topsis_rank"),
                "Selection_Score": _first_value(selected, "score"),
                "Candidate_ID": _first_value(selected, "candidate_id") or _first_value(candidate, "Candidate_ID"),
                "Category": service.get("category", ""),
                "Action": _first_value(step, "Action", "action") or "",
                "Input_From_Previous_Step": _first_value(step, "Input_From_Previous_Step", "input_from_previous_step") or "",
                "Output_To_Next_Step": _first_value(step, "Output_To_Next_Step", "output_to_next_step") or "",
                "Why": _first_value(step, "Why", "why") or "",
            }
        )

    enriched = normalize_workflow_api_qos_scores_by_subtask_candidates(
        pd.DataFrame(rows),
        run_dir=run_dir,
        query_id=query_id,
        mode=mode,
    )
    bottlenecks = identify_bottlenecks(enriched)
    bottleneck_map = _bottleneck_dimension_map(bottlenecks)
    status_rows = [api_health_status(row) for _, row in enriched.iterrows()]
    quality_rows = status_rows
    enriched["Health_Status"] = [status for status, _, _ in status_rows]
    enriched["Health_Reason"] = [reason for _, reason, _ in status_rows]
    enriched["Health_Color"] = [color for _, _, color in status_rows]
    enriched["Selection_Quality_Status"] = [status for status, _, _ in quality_rows]
    enriched["Selection_Quality_Reason"] = [reason for _, reason, _ in quality_rows]
    enriched["Selection_Quality_Color"] = [color for _, _, color in quality_rows]
    enriched["Bottleneck_Dimensions"] = [
        ", ".join(bottleneck_map.get((_normalize_id(row.get("Subtask_ID")), str(row.get("API_ID") or "")), [])) or NA
        for _, row in enriched.iterrows()
    ]
    enriched = _append_composition_risk_columns(enriched)
    return enriched, bottlenecks


def _bottleneck_dimension_map(bottlenecks: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    if bottlenecks.empty:
        return out
    for _, row in bottlenecks.iterrows():
        key = (_normalize_id(row.get("Subtask_ID")), str(row.get("API_ID") or ""))
        dimension = str(row.get("Bottleneck_Type") or row.get("Metric") or "Bottleneck")
        out[key] = _unique_text(out.get(key, []) + [dimension])
    return out


def _unique_text(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        if _is_missing(value):
            continue
        text = str(value).strip()
        if text and text != NA and text not in out:
            out.append(text)
    return out


def _workflow_row_for_bottleneck(workflow: pd.DataFrame, api_id: Any, subtask_id: Any) -> pd.Series | None:
    if workflow.empty:
        return None
    candidates = workflow
    if "API_ID" in candidates and not _is_missing(api_id):
        candidates = candidates[candidates["API_ID"].astype(str) == str(api_id)]
    if "Subtask_ID" in candidates and not _is_missing(subtask_id):
        sid = _normalize_id(subtask_id)
        candidates = candidates[candidates["Subtask_ID"].astype(str).map(_normalize_id) == sid]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _percent_text(value: float | None) -> str:
    return NA if value is None else f"{value:.1f}%"


def group_bottlenecks_by_api(workflow: pd.DataFrame, bottlenecks: pd.DataFrame) -> list[dict[str, Any]]:
    if bottlenecks.empty:
        return []

    total_rt = None
    max_tp = None
    max_av = None
    if not workflow.empty:
        if "rt_s" in workflow:
            rt_values = pd.to_numeric(workflow["rt_s"], errors="coerce").dropna()
            if not rt_values.empty:
                total_rt = float(rt_values.sum())
        if "tp_kbps" in workflow:
            tp_values = pd.to_numeric(workflow["tp_kbps"], errors="coerce").dropna()
            if not tp_values.empty:
                max_tp = float(tp_values.max())
        if "availability" in workflow:
            av_values = pd.to_numeric(workflow["availability"], errors="coerce").dropna()
            if not av_values.empty:
                max_av = float(av_values.max())

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for _, bottleneck in bottlenecks.iterrows():
        api_id = str(bottleneck.get("API_ID") or "")
        subtask_id = _normalize_id(bottleneck.get("Subtask_ID"))
        key = (api_id, subtask_id)
        workflow_row = _workflow_row_for_bottleneck(workflow, api_id, subtask_id)
        api_name = bottleneck.get("API") or api_id or NA
        if workflow_row is not None:
            api_name = workflow_row.get("API_Name") or workflow_row.get("API_ID") or api_name
        group = groups.setdefault(
            key,
            {
                "api_name": api_name,
                "api_id": api_id or NA,
                "subtask_id": subtask_id or NA,
                "subtask": workflow_row.get("Subtask") if workflow_row is not None else bottleneck.get("Subtask", NA),
                "dimensions": [],
                "reasons": [],
                "impacts": [],
                "severity": {},
                "severity_lines": [],
                "raw_metrics": {},
            },
        )
        group["dimensions"] = _unique_text(group["dimensions"] + [bottleneck.get("Bottleneck_Type") or bottleneck.get("Metric")])
        group["reasons"] = _unique_text(group["reasons"] + [bottleneck.get("Reason")])
        group["impacts"] = _unique_text(group["impacts"] + [bottleneck.get("Impact")])

    for key, group in groups.items():
        api_id, subtask_id = key
        workflow_row = _workflow_row_for_bottleneck(workflow, api_id, subtask_id)
        rt = _as_float(workflow_row.get("rt_s")) if workflow_row is not None else None
        tp = _as_float(workflow_row.get("tp_kbps")) if workflow_row is not None else None
        av = _as_float(workflow_row.get("availability")) if workflow_row is not None else None
        group["raw_metrics"] = {
            "response_time_s": rt,
            "throughput_kbps": tp,
            "availability": av,
        }

        severity: dict[str, Any] = {}
        severity_lines: list[str] = []
        dimensions = set(group.get("dimensions", []))
        if "Latency" in dimensions:
            latency_pct = (rt / total_rt * 100.0) if rt is not None and total_rt and total_rt > 0 else None
            severity["latency_contribution_pct"] = latency_pct
            severity_lines.append(f"Latency contribution: {_percent_text(latency_pct)} of total workflow response time")
        if "Throughput" in dimensions:
            throughput_gap = ((max_tp - tp) / max_tp * 100.0) if tp is not None and max_tp and max_tp > 0 and max_tp != tp else None
            severity["throughput_gap_pct"] = throughput_gap
            severity_lines.append(f"Throughput gap: {_percent_text(throughput_gap)} lower than best selected API")
        if "Availability" in dimensions:
            availability_gap = ((max_av - av) / max_av * 100.0) if av is not None and max_av and max_av > 0 and max_av != av else None
            severity["availability_gap_pct"] = availability_gap
            severity["availability_loss_note"] = "Highest contribution to workflow reliability reduction"
            severity_lines.append(f"Availability gap: {_percent_text(availability_gap)} lower than best selected API")
            severity_lines.append("Availability loss: highest contribution to workflow reliability reduction")
        group["severity"] = severity
        group["severity_lines"] = severity_lines

    return sorted(groups.values(), key=lambda item: (_sort_key(item.get("subtask_id")), str(item.get("api_name") or "")))


def bottleneck_group_summary(workflow: pd.DataFrame, bottlenecks: pd.DataFrame) -> str:
    groups = group_bottlenecks_by_api(workflow, bottlenecks)
    if not groups:
        return NA
    parts = []
    for group in groups:
        api = _shorten(group.get("api_name"), 38)
        dims = ", ".join(group.get("dimensions", [])) or NA
        parts.append(f"{api} ({dims})")
    return "; ".join(parts)


def _candidate_display_name(candidate: dict[str, Any], ranked_row: dict[str, Any] | None = None) -> str:
    ranked_row = ranked_row or {}
    service = ranked_row.get("service") if isinstance(ranked_row, dict) else None
    if isinstance(service, dict):
        name = _api_name({"service": service}, str(candidate.get("Selected_API") or candidate.get("api_id") or ""))
        if name:
            return name
    api_id = str(candidate.get("Selected_API") or candidate.get("api_id") or "").strip()
    return api_id or NA


def _candidate_to_workflow_row(
    candidate: dict[str, Any],
    ranked_row: dict[str, Any] | None,
    current_row: pd.Series,
    *,
    mode: str,
    query_id: str,
) -> dict[str, Any]:
    api_id = str(candidate.get("Selected_API") or candidate.get("api_id") or "").strip()
    ranked_row = ranked_row or {}
    service = ranked_row.get("service") if isinstance(ranked_row, dict) else {}
    service = service if isinstance(service, dict) else {}
    qos = service.get("qos") if isinstance(service.get("qos"), dict) else {}
    return {
        "Query_ID": query_id,
        "Mode": mode,
        "Step": current_row.get("Step"),
        "Subtask_ID": current_row.get("Subtask_ID"),
        "Subtask": current_row.get("Subtask"),
        "API_ID": api_id,
        "API_Name": _candidate_display_name(candidate, ranked_row),
        "Functional_Match": _as_match_label(candidate.get("Functional Match (0/1)", candidate.get("Functional_Match"))),
        "rt_s": _as_float(candidate.get("QoS_RT_s")) if _as_float(candidate.get("QoS_RT_s")) is not None else _as_float(qos.get("rt_s")),
        "tp_kbps": _as_float(candidate.get("QoS_TP_kbps")) if _as_float(candidate.get("QoS_TP_kbps")) is not None else _as_float(qos.get("tp_kbps")),
        "availability": _as_float(candidate.get("QoS Availability")) if _as_float(candidate.get("QoS Availability")) is not None else _as_float(qos.get("availability")),
        "Mode_Rank": _first_value(candidate, "Mode Rank", "mode_rank") or _first_value(ranked_row, "mode_rank"),
        "Selected_Rank": None,
        "Retrieved_Rank": _first_value(candidate, "Retrieved Rank", "retrieved_rank") or _first_value(ranked_row, "retrieved_rank"),
        "QoS_LLM_Score": _first_value(ranked_row, "qos_llm_score"),
        "QoS_LLM_Rank": _first_value(ranked_row, "qos_llm_rank"),
        "TOPSIS_Score": _first_value(ranked_row, "topsis_score"),
        "TOPSIS_Rank": _first_value(ranked_row, "topsis_rank"),
        "Selection_Score": _first_value(ranked_row, "score"),
        "Candidate_ID": _first_value(candidate, "Candidate_ID") or _first_value(ranked_row, "candidate_id"),
        "Category": service.get("category", ""),
        "Action": current_row.get("Action", ""),
        "Input_From_Previous_Step": current_row.get("Input_From_Previous_Step", ""),
        "Output_To_Next_Step": current_row.get("Output_To_Next_Step", ""),
        "Why": current_row.get("Why", ""),
    }


def _workflow_metrics(workflow: pd.DataFrame, eval_row: dict[str, Any] | None = None) -> dict[str, Any]:
    eval_row = eval_row or {}
    rt_values = pd.to_numeric(workflow.get("rt_s"), errors="coerce").dropna() if "rt_s" in workflow else pd.Series(dtype=float)
    tp_values = pd.to_numeric(workflow.get("tp_kbps"), errors="coerce").dropna() if "tp_kbps" in workflow else pd.Series(dtype=float)
    av_values = pd.to_numeric(workflow.get("availability"), errors="coerce").dropna() if "availability" in workflow else pd.Series(dtype=float)
    functional = pd.to_numeric(workflow.get("Functional_Match"), errors="coerce").dropna() if "Functional_Match" in workflow else pd.Series(dtype=float)
    return {
        "Total_Response_Time_s": float(rt_values.sum()) if not rt_values.empty else None,
        "Bottleneck_Throughput_kbps": float(tp_values.min()) if not tp_values.empty else None,
        "Average_Workflow_Availability": float(av_values.mean()) if not av_values.empty else None,
        "Workflow_Availability": float(av_values.mean()) if not av_values.empty else None,
        "Functional_Coverage": float((functional == 1).sum() / len(workflow)) if len(workflow) else None,
        "Composition_Completeness": _first_value(eval_row, "Composition_Completeness"),
        "Composition_Validity": _first_value(eval_row, "Composition_Validity"),
        "QoS_Adjusted_Composition_Score": _first_value(eval_row, "QoS_Adjusted_Composition_Score"),
    }


def _score_replacement_candidate(row: pd.Series) -> tuple:
    functional = _as_match_label(row.get("Functional_Match"))
    health = _as_float(row.get("API_QoS_Health"))
    mode_rank = _as_float(row.get("Mode_Rank"))
    api_id = str(row.get("API_ID") or "")
    return (
        0 if functional == 1 else 1,
        -(health if health is not None else -1.0),
        mode_rank if mode_rank is not None else float("inf"),
        api_id,
    )


def find_replacement_candidates(
    *,
    run_dir: Path,
    query_id: str,
    mode: str,
    current_row: pd.Series,
) -> pd.DataFrame:
    subtask_id = _normalize_id(current_row.get("Subtask_ID"))
    current_api_id = str(current_row.get("API_ID") or "")
    candidate_rows = load_candidate_rows(run_dir, query_id, mode=mode, subtask_id=subtask_id)
    ranked_lookup = load_ranked_candidate_lookup(run_dir, mode, subtask_id)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        api_id = str(candidate.get("Selected_API") or candidate.get("api_id") or "").strip()
        if not api_id or api_id == current_api_id:
            continue
        ranked_row = ranked_lookup.get(api_id)
        row = _candidate_to_workflow_row(candidate, ranked_row, current_row, mode=mode, query_id=query_id)
        if _as_float(row.get("rt_s")) is None and _as_float(row.get("tp_kbps")) is None and _as_float(row.get("availability")) is None:
            continue
        rows.append(row)
    return normalize_api_qos_scores(pd.DataFrame(rows))


def build_bottleneck_replacement_simulations(
    *,
    workflow: pd.DataFrame,
    bottlenecks: pd.DataFrame,
    eval_row: dict[str, Any],
    run_dir: Path,
    query_id: str,
    mode: str,
) -> list[dict[str, Any]]:
    groups = group_bottlenecks_by_api(workflow, bottlenecks)
    simulations: list[dict[str, Any]] = []
    if workflow.empty:
        return simulations

    current_metrics = _workflow_metrics(workflow, eval_row)
    for group in groups:
        current_row = _workflow_row_for_bottleneck(workflow, group.get("api_id"), group.get("subtask_id"))
        if current_row is None:
            simulations.append({"group": group, "status": "unavailable", "message": "Current risk-contributing API was not found in the workflow rows."})
            continue
        candidates = find_replacement_candidates(run_dir=run_dir, query_id=query_id, mode=mode, current_row=current_row)
        if candidates.empty:
            simulations.append({"group": group, "current_row": current_row.to_dict(), "status": "unavailable", "message": "No candidate replacement API available for this risk-contributing API."})
            continue
        functional_candidates = candidates[pd.to_numeric(candidates.get("Functional_Match"), errors="coerce") == 1].copy()
        candidate_pool = functional_candidates if not functional_candidates.empty else candidates.copy()
        if functional_candidates.empty:
            warning = "No functionally relevant candidate replacement API was available; showing a diagnostic candidate alternative."
        else:
            warning = ""
        candidate_pool = candidate_pool.assign(_sort_key=candidate_pool.apply(_score_replacement_candidate, axis=1))
        replacement = candidate_pool.sort_values("_sort_key", kind="stable").drop(columns=["_sort_key"]).iloc[0]

        simulated = workflow.copy()
        replace_mask = (
            simulated["API_ID"].astype(str).eq(str(current_row.get("API_ID")))
            & simulated["Subtask_ID"].astype(str).map(_normalize_id).eq(_normalize_id(current_row.get("Subtask_ID")))
        )
        for col, value in replacement.to_dict().items():
            if col in simulated.columns:
                simulated.loc[replace_mask, col] = value
        simulated = normalize_api_qos_scores(simulated)
        simulated_bottlenecks = identify_bottlenecks(simulated)
        simulated["Bottleneck_Dimensions"] = [
            ", ".join(_bottleneck_dimension_map(simulated_bottlenecks).get((_normalize_id(row.get("Subtask_ID")), str(row.get("API_ID") or "")), [])) or NA
            for _, row in simulated.iterrows()
        ]
        simulated_replacement = simulated[replace_mask].iloc[0] if not simulated[replace_mask].empty else replacement
        simulated_metrics = _workflow_metrics(simulated, {**eval_row, "QoS_Adjusted_Composition_Score": None})
        has_improvement = _simulation_has_improvement(current_metrics, simulated_metrics)
        simulations.append(
            {
                "group": group,
                "current_row": current_row.to_dict(),
                "replacement_row": simulated_replacement.to_dict(),
                "candidate_count": int(len(candidates)),
                "status": "ok",
                "warning": warning,
                "reason": _replacement_reason(current_row, replacement, has_improvement),
                "current_metrics": current_metrics,
                "simulated_metrics": simulated_metrics,
                "simulated_workflow": simulated,
                "has_improvement": has_improvement,
            }
        )
    return simulations


def _simulation_has_improvement(current: dict[str, Any], simulated: dict[str, Any]) -> bool:
    specs = [
        ("Total_Response_Time_s", False),
        ("Bottleneck_Throughput_kbps", True),
        ("Average_Workflow_Availability", True),
        ("Functional_Coverage", True),
    ]
    for metric, higher_better in specs:
        before = _as_float(current.get(metric))
        after = _as_float(simulated.get(metric))
        if before is None or after is None:
            continue
        if (higher_better and after > before) or (not higher_better and after < before):
            return True
    return False


def _replacement_reason(current_row: pd.Series, replacement: pd.Series, has_improvement: bool) -> str:
    functional = _as_match_label(replacement.get("Functional_Match"))
    if has_improvement and functional == 1:
        return "Tested because it is functionally relevant and has stronger QoS metrics than the current risk-contributing API."
    if has_improvement:
        return "Tested as a diagnostic QoS improvement, but functional relevance is not confirmed."
    return "No clear improvement detected; this candidate was tested by the visualization heuristic."


def simulation_metric_rows(current: dict[str, Any], simulated: dict[str, Any]) -> list[dict[str, str]]:
    specs = [
        ("Total_Response_Time_s", TOTAL_RESPONSE_TIME_LABEL, False, format_response_time),
        ("Bottleneck_Throughput_kbps", BOTTLENECK_THROUGHPUT_LABEL, True, format_throughput),
        ("Average_Workflow_Availability", WORKFLOW_AVAILABILITY_LABEL, True, format_value),
        ("Functional_Coverage", "Functional Coverage", True, lambda value: format_value(value, percent=True)),
        ("QoS_Adjusted_Composition_Score", "QoS-Adjusted Composition Score", True, format_value),
    ]
    rows: list[dict[str, str]] = []
    for key, label, higher_better, formatter in specs:
        before = _as_float(current.get(key))
        after = _as_float(simulated.get(key))
        change = _metric_change_text(before, after, higher_better)
        rows.append(
            {
                "Metric": label,
                "Official Planner Workflow": formatter(before),
                "What-If Replacement Workflow": formatter(after),
                "Change": change,
            }
        )
    return rows


def _metric_change_text(before: float | None, after: float | None, higher_better: bool) -> str:
    if before is None or after is None:
        return NA
    delta = after - before
    if abs(delta) < 1e-12:
        return "No change"
    improved = delta > 0 if higher_better else delta < 0
    if before != 0:
        pct = abs(delta) / abs(before) * 100.0
        return ("Improved" if improved else "Changed") + f" by {pct:.1f}%"
    return "Improved" if improved else "No clear improvement"


def build_replacement_simulation_dot(current_workflow: pd.DataFrame, simulated_workflow: pd.DataFrame, current_row: dict[str, Any], replacement_row: dict[str, Any]) -> str:
    current_api_id = str(current_row.get("API_ID") or "")
    current_sid = _normalize_id(current_row.get("Subtask_ID"))
    lines = [
        "digraph G {",
        "  rankdir=TB;",
        '  graph [bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.45"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.14,0.08"];',
        '  edge [color="#64748B", arrowsize=0.7, penwidth=1.1];',
    ]
    for prefix, label, workflow in [("cur", "Official Planner Workflow", current_workflow), ("sim", "What-If Replacement Workflow", simulated_workflow)]:
        lines.append(f'  {prefix}_title [label="{label}", fillcolor="{HEALTH_COLORS["blue"]}", color="#0284C7"];')
        previous = f"{prefix}_title"
        for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
            sid = _normalize_id(row.get("Subtask_ID")) or str(idx)
            api = _shorten(row.get("API_Name") or row.get("API_ID"), 34)
            is_replaced = sid == current_sid and (prefix == "sim" or str(row.get("API_ID") or "") == current_api_id)
            fill = "#FFF2CC" if is_replaced else _selection_color(row)
            border = "#B7791F" if is_replaced else _selection_border(row)
            node = f"{prefix}_{idx}"
            node_label = f"Subtask {sid}\\n{api}"
            if is_replaced:
                node_label += "\\nCandidate replacement" if prefix == "sim" else "\\nCurrent risk contributor"
            lines.append(f'  {node} [label="{_dot_escape(node_label)}", fillcolor="{fill}", color="{border}"];')
            lines.append(f"  {previous} -> {node};")
            previous = node
    lines.append("}")
    return "\n".join(lines)


def selected_by_other_modes(run_dir: Path, *, current_mode: str, subtask_id: Any, api_id: Any) -> list[str] | None:
    api_text = str(api_id or "").strip()
    sid = _normalize_id(subtask_id)
    if not api_text or not sid or not run_dir.exists():
        return None
    modes: list[str] = []
    mode_dirs = [path for path in run_dir.iterdir() if path.is_dir() and path.name != current_mode]
    if not mode_dirs:
        return None
    inspected_selected_outputs = False
    for mode_dir in sorted(mode_dirs, key=lambda path: MODE_ORDER.index(path.name) if path.name in MODE_ORDER else len(MODE_ORDER)):
        if not list(mode_dir.glob("3_selected_s*.json")):
            continue
        inspected_selected_outputs = True
        by_pair, _ = load_selected_lookups(run_dir, mode_dir.name)
        if (sid, api_text) in by_pair:
            modes.append(mode_dir.name)
    return modes if inspected_selected_outputs else None


def _read_text_file(path: Path, max_lines: int = 40) -> tuple[str, str]:
    if not path.exists():
        return "Not available", f"{path.name} not found"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return "Not available", f"Could not read {path.name}: {exc}"
    if not lines:
        return "Available", "File is empty"
    warning_lines = [line for line in lines if re.search(r"warning|error|invalid|failed|retry|parse", line, flags=re.IGNORECASE)]
    clipped = (warning_lines or lines)[-max_lines:]
    return "Available", "\n".join(clipped)


def _planner_steps(run_dir: Path, mode: str) -> list[dict[str, Any]]:
    payload = _read_json(run_dir / mode / "4_planner.json", {})
    primary = payload.get("primary_plan") if isinstance(payload, dict) else {}
    steps = primary.get("steps") if isinstance(primary, dict) else None
    return steps if isinstance(steps, list) else []


def build_dataflow_cards_html(
    *,
    query_context: dict[str, str],
    workflow: pd.DataFrame,
    mode: str,
    detailed: bool = False,
) -> str:
    query_text = query_context.get("goal") or query_context.get("label") or query_context.get("query_id") or NA

    def card(title: str, rows: list[tuple[str, Any]], *, css_class: str = "") -> str:
        row_html = "\n".join(
            (
                '<div class="maof-dataflow-field">'
                f'<div class="maof-dataflow-key">{escape(label)}</div>'
                f'<div class="maof-dataflow-value" title="{escape(str(value if not _is_missing(value) else NA))}">'
                f'{escape(str(value if not _is_missing(value) else NA))}'
                "</div>"
                "</div>"
            )
            for label, value in rows
        )
        return (
            f'<div class="maof-dataflow-card {css_class}">'
            f'<div class="maof-dataflow-title">{escape(title)}</div>'
            f"{row_html}"
            "</div>"
        )

    parts = [
        """
<style>
.maof-dataflow-wrap {
  max-width: 850px;
  margin: 0 auto;
}
.maof-dataflow-card {
  width: 100%;
  box-sizing: border-box;
  padding: 14px 16px;
  border-radius: 10px;
  border: 1px solid #d0d7de;
  background: #f8fafc;
  color: #1f2937;
  overflow-wrap: anywhere;
  white-space: normal;
  line-height: 1.35;
}
.maof-dataflow-step {
  background: #ffffff;
}
.maof-dataflow-final {
  background: #e8f5e9;
}
.maof-dataflow-title {
  font-size: 16px;
  font-weight: 700;
  margin-bottom: 8px;
  line-height: 1.35;
  overflow-wrap: anywhere;
}
.maof-dataflow-field {
  display: grid;
  grid-template-columns: minmax(110px, 170px) minmax(0, 1fr);
  gap: 10px;
  padding-top: 7px;
}
.maof-dataflow-key {
  font-size: 13px;
  font-weight: 650;
  color: #57606a;
}
.maof-dataflow-value {
  font-size: 14px;
  color: #24292f;
  overflow-wrap: anywhere;
  white-space: normal;
  line-height: 1.35;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.maof-dataflow-arrow {
  text-align: center;
  font-size: 28px;
  line-height: 1;
  color: #57606a;
  padding: 8px 0;
}
@media (max-width: 700px) {
  .maof-dataflow-card {
    padding: 12px 14px;
  }
  .maof-dataflow-field {
    display: block;
  }
  .maof-dataflow-key {
    margin-bottom: 2px;
  }
}
</style>
<div class="maof-dataflow-wrap">
""",
        card("User Query", [("Request", query_text)]),
    ]

    for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
        step = _first_value(row, "Step") or idx
        if detailed:
            rows = [
                ("Step number", step),
                ("Subtask", row.get("Subtask")),
                ("Selected API", row.get("API_Name") or row.get("API_ID")),
                ("Input from previous step", row.get("Input_From_Previous_Step")),
                ("Action summary", row.get("Action")),
                ("Output to next step", row.get("Output_To_Next_Step")),
            ]
        else:
            rows = [
                ("Step number", step),
                ("Selected API", row.get("API_Name") or row.get("API_ID")),
                ("Input", row.get("Input_From_Previous_Step")),
                ("Output", row.get("Output_To_Next_Step")),
            ]
        parts.append('<div class="maof-dataflow-arrow">↓</div>')
        parts.append(card(f"Step {step}", rows, css_class="maof-dataflow-step"))

    parts.append('<div class="maof-dataflow-arrow">↓</div>')
    parts.append(card("Final Planned Workflow", [("Mode", mode)], css_class="maof-dataflow-final"))
    parts.append("</div>")
    return "\n".join(parts)


def _stage_row(stage: str, status: str, evidence: str, details: str = "", warnings: str = "") -> dict[str, str]:
    return {
        "Stage": stage,
        "Status": status,
        "Evidence": evidence or NA,
        "Count/details": details or NA,
        "Warnings/errors": warnings or NA,
    }


def build_agent_observability_summary(
    *,
    run_dir: Path,
    query_id: str,
    mode: str,
    eval_row: dict[str, Any],
    workflow: pd.DataFrame,
) -> dict[str, Any]:
    meta = _read_json(run_dir / "meta.json", {})
    stages = meta.get("timing", {}).get("stages", {}) if isinstance(meta.get("timing"), dict) else {}
    rows: list[dict[str, str]] = []

    def stage_status(key: str, fallback_exists: bool = False) -> str:
        stage = stages.get(key)
        if isinstance(stage, dict) and stage.get("status"):
            raw = str(stage.get("status")).replace("_", " ").title()
            return "Completed" if raw == "Completed" else raw
        return "Completed" if fallback_exists else "Missing data"

    decomposer = _read_json(run_dir / "0_decomposer.json", {})
    subtask_count = len(decomposer.get("subtasks", [])) if isinstance(decomposer, dict) and isinstance(decomposer.get("subtasks"), list) else meta.get("num_subtasks")
    rows.append(_stage_row("Decomposition", stage_status("decomposer", (run_dir / "0_decomposer.json").exists()), "0_decomposer.json", f"{subtask_count or NA} subtasks"))

    retriever_files = sorted(run_dir.glob("1_retriever_s*.json"))
    retrieved_count = 0
    for path in retriever_files:
        payload = _read_json(path, [])
        retrieved_count += len(payload) if isinstance(payload, list) else 0
    rows.append(_stage_row("Retrieval", stage_status("retrieval", bool(retriever_files)), f"{len(retriever_files)} retriever files", f"{retrieved_count or NA} candidates"))

    fm_path = run_dir / "evaluation" / f"query_{query_id}_retrieval_functional_match_rows.json"
    fm_payload = _read_json(fm_path, [])
    fm_count = len(fm_payload) if isinstance(fm_payload, list) else 0
    rows.append(_stage_row("Functional refinement", stage_status("functional_refinement", fm_path.exists()), str(fm_path.relative_to(run_dir)) if fm_path.exists() else "Rows file missing", f"{fm_count or NA} rows"))

    ranked_files = sorted((run_dir / mode).glob("2_ranked_s*.json"))
    ranked_count = 0
    duplicate_count = 0
    for path in ranked_files:
        payload = _read_json(path, [])
        rows_list = payload if isinstance(payload, list) else []
        ids = [str(item.get("api_id") or "") for item in rows_list if isinstance(item, dict)]
        ranked_count += len(rows_list)
        duplicate_count += len(ids) - len(set(ids))
    ranking_warning = f"{duplicate_count} duplicate ranked API IDs" if duplicate_count else ""
    rows.append(_stage_row("Ranking", stage_status("ranking", bool(ranked_files)), f"{len(ranked_files)} ranked files for {mode}", f"{ranked_count or NA} ranked candidates", ranking_warning))

    selected_files = sorted((run_dir / mode).glob("3_selected_s*.json"))
    selected_count = 0
    for path in selected_files:
        payload = _read_json(path, [])
        selected_count += len(payload) if isinstance(payload, list) else 0
    missing_selected = max(len(workflow) - selected_count, 0) if not workflow.empty else 0
    selection_warning = f"{missing_selected} planned workflow rows not found in selected files" if missing_selected else ""
    rows.append(_stage_row("Selection", stage_status("selection", bool(selected_files)), f"{len(selected_files)} selected files for {mode}", f"{selected_count or NA} selected candidates", selection_warning))

    planner_path = run_dir / mode / "4_planner.json"
    validity = _as_match_label(_first_value(eval_row, "Composition_Validity"))
    invalid_reason = _first_value(eval_row, "Invalid_Reason", "invalid_reason")
    planning_status = "Completed" if planner_path.exists() and validity != 0 else "Completed with warnings" if planner_path.exists() else "Missing data"
    rows.append(_stage_row("Planning", planning_status, str(planner_path.relative_to(run_dir)) if planner_path.exists() else "Planner output missing", f"Validity: {format_flag(validity)}", str(invalid_reason or "")))

    eval_path = _first_value(eval_row, "report_path")
    rows.append(_stage_row("Composition evaluation", "Completed" if eval_row else "Missing data", str(eval_path or "evaluation row"), f"Score: {format_value(_first_value(eval_row, 'QoS_Adjusted_Composition_Score'))}"))
    rows.append(_stage_row("Visualization data loading", "Completed" if not workflow.empty else "Missing data", "Enriched workflow rows", f"{len(workflow)} rows"))

    completed = sum(1 for row in rows if str(row["Status"]).lower().startswith("completed"))
    score = completed / len(rows) if rows else None
    log_summary = []
    for filename in ["warnings.log", "errors.log", "invalid_cases.log", "ranking_anomalies.log"]:
        status, detail = _read_text_file(run_dir / filename, max_lines=10)
        log_summary.append({"Log": filename, "Status": status, "Details": detail})
    return {"rows": rows, "score": score, "logs": log_summary}


def detect_invalid_workflow_issues(
    *,
    eval_row: dict[str, Any],
    workflow: pd.DataFrame,
    run_dir: Path | None = None,
    mode: str | None = None,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    validity = _as_match_label(_first_value(eval_row, "Composition_Validity", "Valid"))
    invalid_reason = _first_value(eval_row, "Invalid_Reason", "invalid_reason", "Error", "error")
    if validity == 0:
        issues.append(
            {
                "Step": NA,
                "Subtask": NA,
                "Selected API": NA,
                "Issue": str(invalid_reason or "Workflow marked invalid, but detailed reason is unavailable."),
                "Severity": "Error",
                "Suggested interpretation": "Invalid workflow diagnostics explain why a composition received a zero or reduced validity score.",
            }
        )

    if workflow.empty:
        if validity != 1:
            return issues
        return []

    expected_subtasks = _as_float(_first_value(eval_row, "Total_Subtask_Count"))
    planned_count = _as_float(_first_value(eval_row, "Planned_API_Count"))
    if expected_subtasks is not None and planned_count is not None and planned_count < expected_subtasks:
        issues.append(
            {
                "Step": NA,
                "Subtask": NA,
                "Selected API": NA,
                "Issue": "missing_selected_api",
                "Severity": "Error",
                "Suggested interpretation": "The planned workflow contains fewer selected APIs than expected subtasks.",
            }
        )

    seen_subtasks: set[str] = set()
    selected_ids: set[str] = set()
    if run_dir is not None and mode:
        for path in sorted((run_dir / mode).glob("3_selected_s*.json")):
            payload = _read_json(path, [])
            if isinstance(payload, list):
                selected_ids.update(str(item.get("api_id") or item.get("Selected_API") or "") for item in payload if isinstance(item, dict))

    for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
        step = str(row.get("Step") or idx)
        sid = _normalize_id(row.get("Subtask_ID"))
        api_id = str(row.get("API_ID") or "").strip()
        api_name = str(row.get("API_Name") or api_id or NA)
        if not sid:
            issues.append({"Step": step, "Subtask": NA, "Selected API": api_name, "Issue": "missing_required_fields: subtask_id", "Severity": "Error", "Suggested interpretation": "A workflow step is missing its subtask identifier."})
        elif sid in seen_subtasks:
            issues.append({"Step": step, "Subtask": sid, "Selected API": api_name, "Issue": "duplicate_subtask", "Severity": "Warning", "Suggested interpretation": "More than one planned step maps to the same subtask."})
        seen_subtasks.add(sid)
        if not api_id:
            issues.append({"Step": step, "Subtask": sid or NA, "Selected API": NA, "Issue": "missing_selected_api", "Severity": "Error", "Suggested interpretation": "A planned step is missing a selected API ID."})
        elif selected_ids and api_id not in selected_ids:
            issues.append({"Step": step, "Subtask": sid or NA, "Selected API": api_name, "Issue": "unknown_api_id", "Severity": "Warning", "Suggested interpretation": "The planned API ID was not found in the selected-candidate artifacts for this mode."})
        if _as_match_label(row.get("Functional_Match")) == 0:
            issues.append({"Step": step, "Subtask": sid or NA, "Selected API": api_name, "Issue": "functional_mismatch", "Severity": "Warning", "Suggested interpretation": "The selected API is not labeled as a functional match for its subtask."})
    return issues


def build_invalid_workflow_diagnostic_dot(workflow: pd.DataFrame, issues: list[dict[str, str]]) -> str:
    issue_by_step: dict[str, str] = {}
    for issue in issues:
        step = str(issue.get("Step") or "")
        severity = str(issue.get("Severity") or "")
        if not step or step == NA:
            continue
        if severity == "Error" or issue_by_step.get(step) != "Error":
            issue_by_step[step] = severity
    lines = [
        "digraph G {",
        "  rankdir=TB;",
        '  graph [bgcolor="transparent", pad="0.25", nodesep="0.45", ranksep="0.55", splines=ortho];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.14,0.08"];',
        '  edge [color="#64748B", arrowsize=0.7, penwidth=1.1];',
        f'  start [label="Planned Workflow", fillcolor="{HEALTH_COLORS["blue"]}", color="#0284C7"];',
    ]
    previous = "start"
    for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
        step = str(row.get("Step") or idx)
        severity = issue_by_step.get(step, "OK")
        fill = HEALTH_COLORS["red"] if severity == "Error" else HEALTH_COLORS["orange"] if severity == "Warning" else _selection_fill(row)
        border = HEALTH_BORDER_COLORS["red"] if severity == "Error" else HEALTH_BORDER_COLORS["orange"] if severity == "Warning" else _selection_border(row)
        node = f"step_{idx}"
        label = f"Step {step}\\nSubtask {_normalize_id(row.get('Subtask_ID')) or NA}\\n{_shorten(row.get('API_Name') or row.get('API_ID'), 38)}"
        if severity != "OK":
            label += f"\\n{severity}"
        lines.append(f'  {node} [label="{_dot_escape(label)}", fillcolor="{fill}", color="{border}"];')
        lines.append(f"  {previous} -> {node};")
        previous = node
    lines.append(f'  final [label="Final Planned Workflow", fillcolor="{HEALTH_COLORS["final"]}", color="#548235"];')
    lines.append(f"  {previous} -> final;")
    lines.append("}")
    return "\n".join(lines)


def build_winner_heatmap(eval_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if eval_df.empty:
        return {"winners": pd.DataFrame(), "counts": pd.DataFrame()}
    metric_specs = [
        ("Best QoS-adjusted score", "QoS_Adjusted_Composition_Score", False),
        (f"Lowest {TOTAL_RESPONSE_TIME_LABEL}", "Total_Response_Time_s", True),
        (f"Highest {BOTTLENECK_THROUGHPUT_LABEL.lower()}", "Bottleneck_Throughput_kbps", False),
        (f"Highest {WORKFLOW_AVAILABILITY_LABEL}", "Average_Workflow_Availability", False),
        ("Best functional coverage", "Functional_Coverage", False),
        ("Best composition completeness", "Composition_Completeness", False),
        ("Best normalized QoS score", "Normalized_QoS_Score", False),
    ]
    rows: list[dict[str, str]] = []
    count_rows: list[dict[str, Any]] = []
    group_cols = ["Query_ID"]
    if "run_name" in eval_df and eval_df.groupby("Query_ID")["run_name"].nunique(dropna=True).max() > 1:
        group_cols.append("run_name")
    for group_key, group in eval_df.groupby(group_cols, dropna=False, sort=True):
        query_id = group_key[0] if isinstance(group_key, tuple) else group_key
        run_name = group_key[1] if isinstance(group_key, tuple) and len(group_key) > 1 else None
        label = str(query_id) if run_name is None else f"{query_id} | {run_name}"
        out_row: dict[str, str] = {"Query": label}
        validity = group["Composition_Validity"].apply(_as_match_label) if "Composition_Validity" in group else pd.Series([None] * len(group), index=group.index)
        valid_group = group[validity == 1].copy()
        candidates = valid_group if not valid_group.empty else group.iloc[0:0].copy()
        for label_name, metric, ascending in metric_specs:
            if candidates.empty:
                winner_text = "No valid mode"
            elif metric not in candidates:
                winner_text = NA
            else:
                scored = candidates[["Mode", metric]].copy()
                scored[metric] = pd.to_numeric(scored[metric], errors="coerce")
                scored = scored.dropna(subset=[metric])
                if scored.empty:
                    winner_text = NA
                else:
                    best_value = scored[metric].min() if ascending else scored[metric].max()
                    winners = sorted(
                        scored[(scored[metric] - best_value).abs() <= BEST_MODE_TOLERANCE]["Mode"].dropna().astype(str).unique().tolist(),
                        key=lambda value: MODE_ORDER.index(value) if value in MODE_ORDER else len(MODE_ORDER),
                    )
                    winner_text = ", ".join(winners) if winners else NA
            out_row[label_name] = winner_text
            for mode in [part.strip() for part in winner_text.split(",") if part.strip() and part.strip() not in {NA, "No valid mode"}]:
                count_rows.append({"Metric": label_name, "Mode": mode, "Wins": 1})
        rows.append(out_row)
    winners = pd.DataFrame(rows)
    counts = pd.DataFrame(count_rows)
    if not counts.empty:
        counts = counts.groupby(["Metric", "Mode"], as_index=False)["Wins"].sum()
    return {"winners": winners, "counts": counts}


def compute_sensitivity_scores(eval_rows: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if eval_rows.empty:
        return pd.DataFrame()
    out = eval_rows.copy()
    specs = {
        "QoS weight": "Normalized_QoS_Score",
        "Functional Coverage weight": "Functional_Coverage",
        "Composition Completeness weight": "Composition_Completeness",
        "Composition Validity weight": "Composition_Validity",
    }
    total_weight = sum(max(float(weights.get(label, 0.0)), 0.0) for label in specs) or 1.0
    score = pd.Series([0.0] * len(out), index=out.index, dtype=float)
    for label, col in specs.items():
        component = pd.to_numeric(out[col], errors="coerce").fillna(0.0) if col in out else 0.0
        score = score + (max(float(weights.get(label, 0.0)), 0.0) / total_weight) * component
    out["Sensitivity_Score"] = score
    out["Original_QoS_Adjusted_Composition_Score"] = pd.to_numeric(out.get("QoS_Adjusted_Composition_Score"), errors="coerce")
    out["_mode_order"] = out["Mode"].astype(str).map({mode: idx for idx, mode in enumerate(MODE_ORDER)}).fillna(len(MODE_ORDER)) if "Mode" in out else 0
    return out.sort_values(["Sensitivity_Score", "_mode_order"], ascending=[False, True], na_position="last").drop(columns=["_mode_order"])


def identify_bottlenecks(workflow: pd.DataFrame) -> pd.DataFrame:
    if workflow.empty:
        return pd.DataFrame()
    specs = [
        ("Latency", "rt_s", False, f"Highest {RESPONSE_TIME_LABEL.lower()}", "Increases total workflow response time"),
        ("Throughput", "tp_kbps", True, "Lowest throughput (kbps)", "Limits end-to-end workflow throughput"),
        ("Availability", "availability", True, "Lowest availability", "Lowers expected workflow reliability"),
    ]
    rows: list[dict[str, Any]] = []
    for bottleneck_type, metric, higher_is_better, reason, impact in specs:
        if metric not in workflow:
            continue
        values = pd.to_numeric(workflow[metric], errors="coerce")
        values = values.dropna()
        if values.empty:
            continue
        target = values.min() if higher_is_better else values.max()
        matches = workflow[pd.to_numeric(workflow[metric], errors="coerce") == target]
        for _, row in matches.iterrows():
            rows.append(
                {
                    "Bottleneck_Type": bottleneck_type,
                    "API": row.get("API_Name") or row.get("API_ID") or NA,
                    "API_ID": row.get("API_ID") or NA,
                    "Subtask_ID": row.get("Subtask_ID") or NA,
                    "Subtask": row.get("Subtask") or NA,
                    "Reason": reason,
                    "Metric": metric,
                    "Metric_Value": target,
                    "Impact": impact,
                }
            )
    return pd.DataFrame(rows)


def api_health_status(row: pd.Series | dict[str, Any]) -> tuple[str, str, str]:
    status, label, _, color = classify_api_health(_first_value(row, "API_QoS_Health"))
    return status, label, color


def bottleneck_summary(bottlenecks: pd.DataFrame) -> str:
    if bottlenecks.empty:
        return NA
    parts: list[str] = []
    for _, row in bottlenecks.iterrows():
        api = _shorten(row.get("API"), 34)
        btype = row.get("Bottleneck_Type") or row.get("Metric") or "Bottleneck"
        parts.append(f"{api} ({btype})")
    return "; ".join(parts)


def recommended_summary_rows(
    eval_row: dict[str, Any],
    workflow: pd.DataFrame,
    bottlenecks: pd.DataFrame,
    *,
    mode: str,
) -> list[dict[str, str]]:
    subtask_count = _first_value(eval_row, "Total_Subtask_Count")
    if _is_missing(subtask_count) and not workflow.empty:
        subtask_count = workflow["Subtask_ID"].nunique() if "Subtask_ID" in workflow else len(workflow)
    selected_count = _first_value(eval_row, "Planned_API_Count")
    if _is_missing(selected_count) and not workflow.empty:
        selected_count = len(workflow)
    validity = _first_value(eval_row, "Composition_Validity")
    validity_text = NA if _is_missing(validity) else ("Valid" if _as_float(validity) == 1 else "Invalid")
    workflow_availability = workflow_availability_value(eval_row, workflow)

    fields = [
        ("Selected Mode", mode),
        ("Number of Subtasks", str(int(float(subtask_count))) if not _is_missing(subtask_count) else NA),
        ("Number of Selected APIs", str(int(float(selected_count))) if not _is_missing(selected_count) else NA),
        ("Composition Validity", validity_text),
        ("Composition Completeness", format_value(_first_value(eval_row, "Composition_Completeness"), percent=True)),
        ("Functional Coverage", format_value(_first_value(eval_row, "Functional_Coverage"), percent=True)),
        (TOTAL_RESPONSE_TIME_LABEL, format_response_time(_first_value(eval_row, "Total_Response_Time_s"))),
        (BOTTLENECK_THROUGHPUT_LABEL, format_throughput(_first_value(eval_row, "Bottleneck_Throughput_kbps"))),
        (WORKFLOW_AVAILABILITY_LABEL, format_value(workflow_availability)),
        ("Normalized QoS Score", format_value(_first_value(eval_row, "Normalized_QoS_Score"))),
        ("QoS-Adjusted Composition Score", format_value(_first_value(eval_row, "QoS_Adjusted_Composition_Score"))),
        ("Risk-contributing API", bottleneck_group_summary(workflow, bottlenecks)),
    ]
    return [{"Metric": label, "Value": value if not _is_missing(value) else NA} for label, value in fields]


def _api_node_label(row: pd.Series, *, compact: bool) -> str:
    api_name = _shorten(row.get("API_Name") or row.get("API_ID"), 32 if compact else 44)
    api_id = _shorten(row.get("API_ID"), 30 if compact else 46)
    parts = [api_name]
    if not compact and api_id != api_name:
        parts.append(f"ID: {api_id}")
    parts.append(f"Functional fit: {format_functional_fit(row.get('Functional_Match'))} | Rank: {format_value(row.get('Mode_Rank'), 0)}")
    parts.append(
        " | ".join(
            [
                f"{RESPONSE_TIME_LABEL}: {format_response_time(row.get('rt_s'))}",
                f"{THROUGHPUT_LABEL}: {format_throughput(row.get('tp_kbps'))}",
            ]
        )
    )
    parts.append(f"Availability: {format_value(row.get('availability'))}")
    parts.append(f"API QoS Health: {format_api_health(row.get('API_QoS_Health'))}")
    if not _is_missing(row.get("API_Selection_Health")):
        parts.append(f"API Selection Health: {format_api_health(row.get('API_Selection_Health'))}")
    parts.append(f"Risk: {row.get('API_Risk_Label') or NA}")
    score_parts = []
    if not _is_missing(row.get("QoS_LLM_Score")):
        score_parts.append(f"QoS: {format_value(row.get('QoS_LLM_Score'))}")
    if not _is_missing(row.get("TOPSIS_Score")):
        score_parts.append(f"TOPSIS: {format_value(row.get('TOPSIS_Score'))}")
    if score_parts:
        parts.append(" | ".join(score_parts))
    if compact:
        return "\n".join(parts[:6])
    parts.append(f"health: {_shorten(row.get('API_Health_Label') or row.get('Selection_Quality_Reason') or row.get('Health_Reason'), 40)}")
    if not _is_missing(row.get("Bottleneck_Dimensions")):
        parts.append(f"qos signal: {_shorten(row.get('Bottleneck_Dimensions'), 40)}")
    return "\n".join(parts)


def build_workflow_graph_dot(
    *,
    query_context: dict[str, str],
    workflow: pd.DataFrame,
    mode: str,
    eval_row: dict[str, Any] | None = None,
    compact: bool = False,
) -> str:
    eval_row = eval_row or {}
    graph_label = _shorten(query_context.get("goal") or query_context.get("label") or query_context.get("query_id"), 72)
    final_bits = [f"Mode: {mode}"]
    if not _is_missing(_first_value(eval_row, "QoS_Adjusted_Composition_Score")):
        final_bits.append(f"score: {format_value(_first_value(eval_row, 'QoS_Adjusted_Composition_Score'))}")
    final_label = "Final Composed Workflow\n" + " | ".join(final_bits)
    lines = [
        "digraph G {",
        "  rankdir=TB;",
        '  graph [bgcolor="transparent", pad="0.35", margin="0.18", nodesep="0.85", ranksep="0.85", splines=ortho, concentrate=false];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.12,0.08"];',
        '  edge [color="#64748B", arrowsize=0.7, penwidth=1.2];',
        f'  query [label="User Query\\n{_dot_escape(graph_label)}", fillcolor="{HEALTH_COLORS["blue"]}", color="#0284C7"];',
        f'  final [label="{_dot_escape(final_label)}", fillcolor="{HEALTH_COLORS["final"]}", color="#548235"];',
    ]
    if workflow.empty:
        lines.append("  query -> final;")
        lines.append("}")
        return "\n".join(lines)

    subtask_nodes: list[str] = []
    for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
        sid = _normalize_id(row.get("Subtask_ID")) or str(idx)
        sub_node = f"sub_{idx}"
        api_node = f"api_{idx}"
        subtask_nodes.append(sub_node)
        subtask = _shorten(row.get("Subtask"), 34 if compact else 58)
        sub_label = f"Subtask {sid}" if compact or subtask == NA else f"Subtask {sid}\\n{subtask}"
        api_label = _api_node_label(row, compact=compact)
        fill = _selection_color(row)
        lines.extend(
            [
                f'  {sub_node} [label="{_dot_escape(sub_label)}", fillcolor="{HEALTH_COLORS["subtask"]}", color="#CBD5E1"];',
                f'  {api_node} [label="{_dot_escape(api_label)}", fillcolor="{fill}", color="#64748B"];',
                f"  {{rank=same; {sub_node}; {api_node};}}",
                f"  {sub_node} -> {api_node} [constraint=false];",
            ]
        )
    lines.append(f"  query -> {subtask_nodes[0]};")
    for previous, current in zip(subtask_nodes, subtask_nodes[1:]):
        lines.append(f"  {previous} -> {current};")
    lines.append(f"  {subtask_nodes[-1]} -> final;")
    lines.append("}")
    return "\n".join(lines)


def _finish_diagram_layout(fig: go.Figure, *, height: int, title: str | None = None, show_axes: bool = False) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=18, r=18, t=52 if title else 20, b=18),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
        xaxis=dict(visible=show_axes, fixedrange=not show_axes),
        yaxis=dict(visible=show_axes, fixedrange=not show_axes),
        font=dict(family="Arial, sans-serif", size=13, color="#1F2937"),
    )
    return fig


def _add_box(
    fig: go.Figure,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    fill: str,
    border: str,
    hover: str,
    text_size: int = 13,
    line_width: int = 2,
) -> None:
    fig.add_shape(
        type="rect",
        x0=x - width / 2,
        x1=x + width / 2,
        y0=y - height / 2,
        y1=y + height / 2,
        line=dict(color=border, width=line_width),
        fillcolor=fill,
        layer="below",
    )
    fig.add_annotation(
        x=x,
        y=y,
        text=label,
        showarrow=False,
        align="center",
        font=dict(size=text_size, color="#111827"),
    )
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(size=max(width, height) * 45, color="rgba(0,0,0,0)"),
            hovertext=[hover],
            hoverinfo="text",
            showlegend=False,
        )
    )


def _add_arrow(fig: go.Figure, start: tuple[float, float], end: tuple[float, float]) -> None:
    fig.add_annotation(
        x=end[0],
        y=end[1],
        ax=start[0],
        ay=start[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1.7,
        arrowcolor="#64748B",
    )


def build_workflow_figure(
    *,
    query_context: dict[str, str],
    workflow: pd.DataFrame,
    mode: str,
    eval_row: dict[str, Any] | None = None,
) -> go.Figure:
    eval_row = eval_row or {}
    fig = go.Figure()
    steps = len(workflow)
    row_gap = 1.32
    top_y = steps * row_gap + 1.05
    left_x = 0.0
    api_x = 2.05
    query_label = "<b>User Query</b><br>" + _wrap_text(
        query_context.get("goal") or query_context.get("label") or query_context.get("query_id"),
        width=48,
        max_lines=3,
    )
    _add_box(
        fig,
        x=left_x,
        y=top_y,
        width=1.8,
        height=0.78,
        label=query_label,
        fill=HEALTH_COLORS["blue"],
        border=HEALTH_BORDER_COLORS["blue"],
        hover=escape(str(query_context.get("goal") or query_context.get("label") or NA)),
        text_size=12,
    )
    previous_left = (left_x, top_y - 0.48)
    last_subtask_bottom = previous_left
    for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
        y = top_y - idx * row_gap
        sid = _normalize_id(row.get("Subtask_ID")) or str(idx)
        sub_label = f"<b>Subtask {escape(sid)}</b><br>{_wrap_text(row.get('Subtask'), width=36, max_lines=3)}"
        api_label = (
            f"<b>Selected API {idx}</b><br>"
            f"{_wrap_text(row.get('API_Name') or row.get('API_ID'), width=40, max_lines=2)}<br>"
            f"Fit: {format_functional_fit(row.get('Functional_Match'))} | Rank: {format_value(row.get('Mode_Rank'), 0)}<br>"
            f"{RESPONSE_TIME_LABEL}: {format_response_time(row.get('rt_s'))} | TP (kbps): {format_throughput(row.get('tp_kbps'))} | Avail: {format_value(row.get('availability'))}<br>"
            f"QoS Health: {format_api_health(row.get('API_QoS_Health'))} | Selection: {format_api_health(row.get('API_Selection_Health'))}<br>"
            f"Risk: {row.get('API_Risk_Label') or NA}"
        )
        if not _is_missing(row.get("Bottleneck_Dimensions")):
            api_label += f"<br>QoS Signal: {_wrap_text(row.get('Bottleneck_Dimensions'), width=34, max_lines=1)}"
        sub_pos = (left_x, y)
        api_pos = (api_x, y)
        _add_box(
            fig,
            x=sub_pos[0],
            y=sub_pos[1],
            width=1.45,
            height=0.9,
            label=sub_label,
            fill=HEALTH_COLORS["subtask"],
            border=HEALTH_BORDER_COLORS["subtask"],
            hover=f"<b>Subtask {escape(sid)}</b><br>{escape(str(row.get('Subtask') or NA))}",
            text_size=12,
        )
        _add_box(
            fig,
            x=api_pos[0],
            y=api_pos[1],
            width=1.92,
            height=1.28 if not _is_missing(row.get("Bottleneck_Dimensions")) else 1.18,
            label=api_label,
            fill=_selection_fill(row),
            border=_selection_border(row),
            hover=_hover_for_step(row),
            text_size=12,
            line_width=2,
        )
        _add_arrow(fig, previous_left, (sub_pos[0], sub_pos[1] + 0.48))
        _add_arrow(fig, (sub_pos[0] + 0.75, sub_pos[1]), (api_pos[0] - 1.0, api_pos[1]))
        last_subtask_bottom = (sub_pos[0], sub_pos[1] - 0.48)
        previous_left = last_subtask_bottom

    final_y = top_y - (steps + 1) * row_gap
    score = format_value(_first_value(eval_row, "QoS_Adjusted_Composition_Score"))
    validity = _first_value(eval_row, "Composition_Validity")
    validity_text = "Valid" if _as_float(validity) == 1 else ("Invalid" if not _is_missing(validity) else NA)
    final_label = f"<b>Final Composed Workflow</b><br>Mode: {escape(mode)} | Score: {score}<br>{validity_text}"
    _add_box(
        fig,
        x=left_x,
        y=final_y,
        width=1.85,
        height=0.82,
        label=final_label,
        fill=HEALTH_COLORS["final"],
        border=HEALTH_BORDER_COLORS["final"],
        hover="<br>".join(
            [
                f"Mode: {escape(mode)}",
                f"Composition validity: {validity_text}",
                f"Completeness: {format_value(_first_value(eval_row, 'Composition_Completeness'), percent=True)}",
                f"Functional coverage: {format_value(_first_value(eval_row, 'Functional_Coverage'), percent=True)}",
                f"QoS-adjusted score: {score}",
            ]
        ),
        text_size=12,
    )
    _add_arrow(fig, last_subtask_bottom, (left_x, final_y + 0.45))
    fig.update_xaxes(range=[-1.1, 3.25])
    fig.update_yaxes(range=[final_y - 0.75, top_y + 0.75])
    return _finish_diagram_layout(fig, height=max(560, 135 * (steps + 2)), title="Composed Workflow")


def build_sequence_figure(workflow: pd.DataFrame, *, kind: str) -> go.Figure:
    fig = go.Figure()
    if kind == "agent":
        items: list[dict[str, Any]] = [
            {"label": "User", "fill": HEALTH_COLORS["blue"], "border": HEALTH_BORDER_COLORS["blue"], "hover": "User submits the request."},
            {"label": "Decomposer Agent", "fill": HEALTH_COLORS["blue"], "border": HEALTH_BORDER_COLORS["blue"], "hover": "Creates ordered subtasks."},
            {"label": "Retriever Agent", "fill": HEALTH_COLORS["blue"], "border": HEALTH_BORDER_COLORS["blue"], "hover": "Retrieves candidate APIs for each subtask."},
            {"label": "Ranker Agent", "fill": HEALTH_COLORS["blue"], "border": HEALTH_BORDER_COLORS["blue"], "hover": "Ranks candidates for the selected mode."},
            {"label": "QoS Selector", "fill": HEALTH_COLORS["blue"], "border": HEALTH_BORDER_COLORS["blue"], "hover": "Selects APIs passed to the planner."},
            {"label": "Planner Agent", "fill": HEALTH_COLORS["blue"], "border": HEALTH_BORDER_COLORS["blue"], "hover": "Builds the final composition plan."},
        ]
        for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
            items.append(
                {
                    "label": (
                        f"Selected API {idx}<br>{_wrap_text(row.get('API_Name') or row.get('API_ID'), width=38, max_lines=2)}"
                        f"<br>Health {format_api_health(row.get('API_QoS_Health'))} | Risk {row.get('API_Risk_Label') or NA}"
                    ),
                    "fill": _selection_fill(row),
                    "border": _selection_border(row),
                    "hover": _hover_for_step(row),
                }
            )
        items.append({"label": "Final Workflow Output", "fill": HEALTH_COLORS["final"], "border": HEALTH_BORDER_COLORS["final"], "hover": "Final planned workflow output returned to the user."})
        title = "Planned Agent and API Flow"
    else:
        items = [{"label": "User Request", "fill": HEALTH_COLORS["blue"], "border": HEALTH_BORDER_COLORS["blue"], "hover": "Incoming user request."}]
        for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
            sid = _normalize_id(row.get("Subtask_ID")) or str(idx)
            items.append(
                {
                    "label": (
                        f"Subtask {escape(sid)} API<br>{_wrap_text(row.get('API_Name') or row.get('API_ID'), width=40, max_lines=2)}"
                        f"<br>Health {format_api_health(row.get('API_QoS_Health'))} | Risk {row.get('API_Risk_Label') or NA}"
                    ),
                    "fill": _selection_fill(row),
                    "border": _selection_border(row),
                    "hover": _hover_for_step(row),
                }
            )
        items.append({"label": "Final Workflow Output", "fill": HEALTH_COLORS["final"], "border": HEALTH_BORDER_COLORS["final"], "hover": "Final planned workflow output."})
        title = "Planned API Composition Sequence"

    top_y = len(items) * 1.15
    previous: tuple[float, float] | None = None
    for idx, item in enumerate(items):
        y = top_y - idx * 1.15
        x = 0.0 if idx % 2 == 0 else 1.18
        _add_box(
            fig,
            x=x,
            y=y,
            width=1.62,
            height=0.82,
            label=f"<b>{item['label']}</b>",
            fill=item["fill"],
            border=item["border"],
            hover=item["hover"],
            text_size=12,
        )
        if previous is not None:
            _add_arrow(fig, previous, (x, y + 0.38))
        previous = (x, y - 0.38)
    fig.update_xaxes(range=[-1.0, 2.15])
    fig.update_yaxes(range=[top_y - len(items) * 1.15 - 0.2, top_y + 0.65])
    return _finish_diagram_layout(fig, height=max(620, 95 * len(items)), title=title)


def best_composition_modes(eval_rows_by_mode: dict[str, dict[str, Any]], modes: list[str]) -> set[str]:
    candidates: list[tuple[str, float]] = []
    for mode in modes:
        eval_row = eval_rows_by_mode.get(mode) or {}
        if _composition_validity(eval_row) != 1:
            continue
        if _composition_completeness_gate(eval_row) != 1.0:
            continue
        score = _as_float(_first_value(eval_row, "QoS_Adjusted_Composition_Score"))
        if score is not None:
            candidates.append((mode, score))
    if not candidates:
        return set()
    best_score = max(score for _, score in candidates)
    return {mode for mode, score in candidates if abs(score - best_score) <= BEST_MODE_TOLERANCE}


def mode_comparison_has_eval_rows(eval_rows_by_mode: dict[str, dict[str, Any]] | None) -> bool:
    return bool(eval_rows_by_mode) and any(bool(row) for row in eval_rows_by_mode.values())


def _mode_comparison_status(
    row: pd.Series,
    *,
    mode: str,
    eval_row: dict[str, Any] | None,
    recommended_modes: set[str],
    use_composition_status: bool,
) -> dict[str, str]:
    functional_risk = functional_risk_label(row.get("Functional_Match"))
    qos_risk = qos_risk_label(row)
    if not use_composition_status:
        status = str(row.get("API_Health_Status") or row.get("Health_Status") or "gray").lower()
        return {
            "fill": _selection_fill(row),
            "border": _selection_border(row),
            "overall_status": row.get("API_Health_Label") or "Missing Data",
            "functional_risk": functional_risk,
            "qos_risk": qos_risk,
            "composition_risk": "Missing composition evaluation data",
            "reason": "Composition evaluation rows were unavailable, so this graph is temporarily using API QoS health colors.",
            "legend_key": status if status in {"green", "orange", "red", "gray"} else "gray",
        }

    validity = _composition_validity(eval_row)
    complete = _composition_completeness_gate(eval_row)
    functional = _as_match_label(row.get("Functional_Match"))
    if not eval_row:
        key = "missing"
        overall = "Missing Data"
        reason = "Composition evaluation row is missing for this mode."
    elif validity == 0:
        key = "risk"
        overall = "Invalid"
        reason = "Composition_Validity is 0 for this mode."
    elif complete == 0.0:
        key = "risk"
        overall = "Incomplete"
        reason = "Composition_Completeness_Gate is 0 for this mode."
    elif functional == 0:
        key = "risk"
        overall = "Functional Risk"
        reason = "Functional Match is No for this planned API."
    elif mode in recommended_modes:
        key = "recommended"
        overall = "Recommended"
        reason = "This mode has the best QoS-adjusted composition score among valid complete modes."
    elif functional == 1 and qos_risk == "High":
        key = "qos_risk"
        overall = "Valid Alternative with QoS Risk"
        reason = "The mode is valid and complete, but this API has high QoS risk."
    elif functional is None:
        key = "missing"
        overall = "Missing Data"
        reason = "Functional Match is missing for this planned API."
    else:
        key = "valid_alternative"
        overall = "Valid Alternative"
        reason = "The mode is valid and complete but is not the best final composition score for this query."

    return {
        "fill": COMPOSITION_STATUS_COLORS[key],
        "border": COMPOSITION_STATUS_BORDERS[key],
        "overall_status": overall,
        "functional_risk": functional_risk,
        "qos_risk": qos_risk,
        "composition_risk": overall,
        "reason": reason,
        "legend_key": key,
    }


def _hover_for_mode_comparison(row: pd.Series, *, mode: str, idx: int, status: dict[str, str]) -> str:
    api_name = row.get("API_Name") or row.get("API_ID") or NA
    return "<br>".join(
        [
            f"<b>{escape(str(api_name))}</b>",
            f"Mode: {escape(str(mode))}",
            f"Step / Subtask: {escape(str(row.get('Step') or idx))} / {escape(str(row.get('Subtask_ID') or NA))}",
            f"API name: {escape(str(api_name))}",
            f"API ID: {escape(str(row.get('API_ID') or NA))}",
            f"Functional Match: {format_functional_fit(row.get('Functional_Match'))}",
            f"Composition Risk: {escape(status['composition_risk'])}",
            f"Functional Risk: {escape(status['functional_risk'])}",
            f"QoS Risk: {escape(status['qos_risk'])}",
            f"QoS Health: {format_api_health(row.get('API_QoS_Health'))}",
            f"{RESPONSE_TIME_LABEL}: {format_response_time(row.get('rt_s'))}",
            f"{THROUGHPUT_LABEL}: {format_throughput(row.get('tp_kbps'))}",
            f"Availability: {format_value(row.get('availability'))}",
            f"Overall Status: {escape(status['overall_status'])}",
            f"Reason: {escape(status['reason'])}",
        ]
    )


def build_mode_comparison_figure(
    workflows_by_mode: dict[str, pd.DataFrame],
    modes: list[str],
    *,
    eval_rows_by_mode: dict[str, dict[str, Any]] | None = None,
) -> go.Figure:
    fig = go.Figure()
    max_steps = max((len(df) for df in workflows_by_mode.values() if not df.empty), default=0)
    if max_steps == 0:
        return _finish_diagram_layout(fig, height=420, title="Mode Comparison")
    eval_rows_by_mode = eval_rows_by_mode or {}
    use_composition_status = mode_comparison_has_eval_rows(eval_rows_by_mode)
    recommended_modes = best_composition_modes(eval_rows_by_mode, modes) if use_composition_status else set()
    for mode_idx, mode in enumerate(modes):
        workflow = workflows_by_mode.get(mode, pd.DataFrame())
        if workflow.empty:
            continue
        y = len(modes) - mode_idx
        xs = list(range(1, len(workflow) + 1))
        ys = [y] * len(workflow)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="#94A3B8", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
            label = f"S{idx}<br>{_wrap_text(row.get('API_Name') or row.get('API_ID'), width=18, max_lines=1)}"
            status = _mode_comparison_status(
                row,
                mode=mode,
                eval_row=eval_rows_by_mode.get(mode),
                recommended_modes=recommended_modes,
                use_composition_status=use_composition_status,
            )
            fig.add_trace(
                go.Scatter(
                    x=[idx],
                    y=[y],
                    mode="markers+text",
                    marker=dict(
                        symbol="square",
                        size=72,
                        color=status["fill"],
                        line=dict(color=status["border"], width=2),
                    ),
                    text=[label],
                    textposition="middle center",
                    textfont=dict(size=11, color="#111827"),
                    hovertext=[_hover_for_mode_comparison(row, mode=mode, idx=idx, status=status)],
                    hoverinfo="text",
                    showlegend=False,
                )
            )
    for x in range(1, max_steps + 1):
        fig.add_vline(x=x, line_width=1, line_color="#E5E7EB")
    fig.update_xaxes(
        range=[0.45, max_steps + 0.55],
        visible=True,
        title="Workflow Step / Subtask",
        tickmode="array",
        tickvals=list(range(1, max_steps + 1)),
    )
    fig.update_yaxes(
        range=[0.25, len(modes) + 0.75],
        visible=True,
        title="Mode",
        tickmode="array",
        tickvals=[len(modes) - idx for idx, _ in enumerate(modes)],
        ticktext=modes,
    )
    fig.update_layout(showlegend=False)
    return _finish_diagram_layout(fig, height=max(560, 150 * len(modes)), title="Selected APIs Across Ranking Modes", show_axes=True)


def build_bottleneck_figure(workflow: pd.DataFrame, bottlenecks: pd.DataFrame) -> go.Figure:
    labels = [
        _shorten(row.get("API_Name") or row.get("API_ID"), 28)
        for _, row in workflow.iterrows()
    ]
    specs = [
        ("rt_s", RESPONSE_TIME_LABEL, "Higher is worse"),
        ("tp_kbps", THROUGHPUT_LABEL, "Lower is worse"),
        ("availability", "Availability", "Lower is worse"),
    ]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"{title}<br><sup>{direction}</sup>" for _, title, direction in specs],
        horizontal_spacing=0.08,
    )
    bottleneck_pairs = {
        (str(row.get("API_ID") or ""), str(row.get("Metric") or ""))
        for _, row in bottlenecks.iterrows()
    } if not bottlenecks.empty else set()
    for col_idx, (metric, title, _) in enumerate(specs, start=1):
        values = pd.to_numeric(workflow[metric], errors="coerce") if metric in workflow else pd.Series(dtype=float)
        colors = []
        for _, row in workflow.iterrows():
            pair = (str(row.get("API_ID") or ""), metric)
            colors.append("#C53030" if pair in bottleneck_pairs else _selection_fill(row))
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                hovertext=[
                    _hover_for_step(row, title=f"{title}: {format_response_time(row.get(metric))}" if metric == "rt_s" else f"{title}: {format_throughput(row.get(metric))}" if metric == "tp_kbps" else f"{title}: {format_value(row.get(metric))}")
                    for _, row in workflow.iterrows()
                ],
                hoverinfo="text",
                text=[format_response_time(value) if metric == "rt_s" else format_throughput(value) if metric == "tp_kbps" else format_value(value) for value in values],
                textposition="outside",
                cliponaxis=False,
            ),
            row=1,
            col=col_idx,
        )
    fig.update_xaxes(tickangle=25)
    fig.update_layout(
        height=430,
        margin=dict(l=12, r=12, t=70, b=85),
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial, sans-serif", size=12),
    )
    return fig


def build_quality_score_figure(eval_row: dict[str, Any], workflow: pd.DataFrame) -> go.Figure:
    availability = workflow_availability_value(eval_row, workflow)
    metrics = [
        ("QoS-Adjusted Score", _first_value(eval_row, "QoS_Adjusted_Composition_Score")),
        ("Normalized QoS", _first_value(eval_row, "Normalized_QoS_Score")),
        (WORKFLOW_AVAILABILITY_LABEL, availability),
        ("Functional Coverage", _first_value(eval_row, "Functional_Coverage")),
        ("Completeness", _first_value(eval_row, "Composition_Completeness")),
    ]
    names = [label for label, _ in metrics]
    values = [_as_float(value) for _, value in metrics]
    colors = ["#2F855A" if value is not None and value >= 0.8 else "#B7791F" if value is not None and value >= 0.5 else "#C53030" for value in values]
    fig = go.Figure(
        go.Bar(
            x=[value if value is not None else 0 for value in values],
            y=names,
            orientation="h",
            marker_color=colors,
            text=[format_value(value) for value in values],
            textposition="auto",
            cliponaxis=False,
            hovertext=[f"{name}: {format_value(value)}" for name, value in metrics],
            hoverinfo="text",
        )
    )
    fig.update_xaxes(range=[0, 1.05], title="Score")
    fig.update_yaxes(title=None, automargin=True)
    fig.update_layout(
        height=360,
        margin=dict(l=135, r=20, t=28, b=45),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
        font=dict(family="Arial, sans-serif", size=12),
    )
    return fig


def build_agent_sequence_dot(workflow: pd.DataFrame, *, rankdir: str = "LR") -> str:
    rankdir = rankdir if rankdir in {"LR", "TB"} else "LR"
    graph_spacing = 'nodesep="0.55", ranksep="0.65"' if rankdir == "LR" else 'nodesep="0.35", ranksep="0.45"'
    lines = [
        "digraph G {",
        f"  rankdir={rankdir};",
        f'  graph [bgcolor="transparent", pad="0.25", {graph_spacing}];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.16,0.10"];',
        '  edge [color="#64748B", arrowsize=0.7, penwidth=1.2];',
    ]
    agents = [
        ("user", "User"),
        ("decomposer", "Decomposer Agent"),
        ("retriever", "Retriever Agent"),
        ("ranker", "Ranker Agent"),
        ("selector", "QoS Selector"),
        ("planner", "Planner Agent"),
    ]
    for node_id, label in agents:
        lines.append(f'  {node_id} [label="{label}", fillcolor="{HEALTH_COLORS["blue"]}", color="#0284C7"];')
    lines.extend(
        [
            "  user -> decomposer;",
            "  decomposer -> retriever;",
            "  retriever -> ranker;",
            "  ranker -> selector;",
            "  selector -> planner;",
        ]
    )
    previous = "planner"
    for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
        node = f"api_{idx}"
        label = (
            f"Selected API {idx}\\n{_shorten(row.get('API_Name') or row.get('API_ID'), 38)}"
            f"\\nHealth {format_api_health(row.get('API_QoS_Health'))} | Risk {row.get('API_Risk_Label') or NA}"
        )
        fill = _selection_color(row)
        lines.append(f'  {node} [label="{_dot_escape(label)}", fillcolor="{fill}", color="#64748B"];')
        lines.append(f"  {previous} -> {node};")
        previous = node
    lines.append(f'  final [label="Final Workflow Output", fillcolor="{HEALTH_COLORS["final"]}", color="#548235"];')
    lines.append(f"  {previous} -> final;")
    lines.append("}")
    return "\n".join(lines)


def build_planned_api_flow_dot(workflow: pd.DataFrame, *, rankdir: str = "LR") -> str:
    rankdir = rankdir if rankdir in {"LR", "TB"} else "LR"
    graph_spacing = 'nodesep="0.55", ranksep="0.65"' if rankdir == "LR" else 'nodesep="0.35", ranksep="0.45"'
    lines = [
        "digraph G {",
        f"  rankdir={rankdir};",
        f'  graph [bgcolor="transparent", pad="0.25", {graph_spacing}];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.16,0.10"];',
        '  edge [color="#64748B", arrowsize=0.7, penwidth=1.2];',
        f'  request [label="User Request", fillcolor="{HEALTH_COLORS["blue"]}", color="#0284C7"];',
    ]
    previous = "request"
    for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
        node = f"planned_api_{idx}"
        sid = _normalize_id(row.get("Subtask_ID")) or str(idx)
        label = (
            f"Subtask {sid} API\\n{_shorten(row.get('API_Name') or row.get('API_ID'), 38)}"
            f"\\nHealth {format_api_health(row.get('API_QoS_Health'))} | Risk {row.get('API_Risk_Label') or NA}"
        )
        fill = _selection_color(row)
        lines.append(f'  {node} [label="{_dot_escape(label)}", fillcolor="{fill}", color="#64748B"];')
        lines.append(f"  {previous} -> {node};")
        previous = node
    lines.append(f'  result [label="Final Workflow Output", fillcolor="{HEALTH_COLORS["final"]}", color="#548235"];')
    lines.append(f"  {previous} -> result;")
    lines.append("}")
    return "\n".join(lines)


def build_agent_mermaid(workflow: pd.DataFrame) -> str:
    lines = [
        "sequenceDiagram",
        "    autonumber",
        "    participant U as User",
        "    participant D as Decomposer Agent",
        "    participant R as Retriever Agent",
        "    participant K as Ranker Agent",
        "    participant Q as QoS Selector",
        "    participant P as Planner Agent",
        "    U->>D: Submit query",
        "    D->>R: Send subtasks",
        "    R->>K: Return candidate APIs",
        "    K->>Q: Ranked candidates by mode",
        "    Q->>P: Selected APIs",
    ]
    for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
        api_id = f"A{idx}"
        api_label = _shorten(row.get("API_Name") or row.get("API_ID"), 40).replace(":", "-")
        lines.append(f"    participant {api_id} as {api_label}")
        lines.append(f"    P->>{api_id}: Plan step {idx}")
    lines.append("    P-->>U: Final workflow output")
    return "\n".join(lines)


def build_api_mermaid(workflow: pd.DataFrame) -> str:
    lines = [
        "sequenceDiagram",
        "    autonumber",
        "    participant U as User Request",
    ]
    previous = "U"
    for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
        api_id = f"A{idx}"
        sid = _normalize_id(row.get("Subtask_ID")) or str(idx)
        api_label = _shorten(row.get("API_Name") or row.get("API_ID"), 40).replace(":", "-")
        lines.append(f"    participant {api_id} as Subtask {sid} API - {api_label}")
        lines.append(f"    {previous}->>{api_id}: Input from previous step")
        previous = api_id
    lines.append(f"    {previous}-->>U: Final workflow output")
    return "\n".join(lines)


def workflow_difference_table(workflows_by_mode: dict[str, pd.DataFrame], modes: list[str]) -> pd.DataFrame:
    subtask_ids = sorted(
        {
            _normalize_id(sid)
            for workflow in workflows_by_mode.values()
            if not workflow.empty and "Subtask_ID" in workflow
            for sid in workflow["Subtask_ID"].tolist()
        },
        key=_sort_key,
    )
    rows: list[dict[str, Any]] = []
    for sid in subtask_ids:
        row: dict[str, Any] = {"Subtask_ID": sid}
        subtask = next(
            (
                str(match["Subtask"].dropna().iloc[0])
                for match in (workflow[workflow["Subtask_ID"].astype(str) == sid] for workflow in workflows_by_mode.values() if not workflow.empty)
                if "Subtask" in match and not match["Subtask"].dropna().empty
            ),
            "",
        )
        row["Subtask"] = _shorten(subtask, 64) if subtask else NA
        selected_values: list[str] = []
        for mode in modes:
            workflow = workflows_by_mode.get(mode, pd.DataFrame())
            label = NA
            if not workflow.empty and "Subtask_ID" in workflow:
                match = workflow[workflow["Subtask_ID"].astype(str) == sid]
                if not match.empty:
                    labels = [
                        _shorten(value, 42)
                        for value in (match["API_Name"] if "API_Name" in match else match["API_ID"]).dropna().astype(str).tolist()
                    ]
                    label = " -> ".join(dict.fromkeys(labels)) if labels else NA
            row[mode] = label
            if label != NA:
                selected_values.append(label)
        distinct = len(set(selected_values))
        row["Distinct_API_Count"] = distinct
        row["Selection_Difference"] = "Different APIs" if distinct > 1 else "Same API"
        rows.append(row)
    return pd.DataFrame(rows)


def mode_summary_table(
    eval_df: pd.DataFrame,
    workflows_by_mode: dict[str, pd.DataFrame],
    bottlenecks_by_mode: dict[str, pd.DataFrame],
    *,
    query_id: str,
    run_dir: Path,
    run_name: str,
    modes: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        eval_row = eval_row_for_mode(eval_df, query_id=query_id, run_dir=run_dir, run_name=run_name, mode=mode)
        workflow = workflows_by_mode.get(mode, pd.DataFrame())
        rows.append(
            {
                "Mode": mode,
                "Selected_APIs": len(workflow) if not workflow.empty else 0,
                "Composition_Validity": _first_value(eval_row, "Composition_Validity") if eval_row else NA,
                "Composition_Completeness": _first_value(eval_row, "Composition_Completeness") if eval_row else NA,
                "Functional_Coverage": _first_value(eval_row, "Functional_Coverage") if eval_row else NA,
                "Total_Response_Time_s": _first_value(eval_row, "Total_Response_Time_s") if eval_row else NA,
                "Bottleneck_Throughput_kbps": _first_value(eval_row, "Bottleneck_Throughput_kbps") if eval_row else NA,
                "Average_Workflow_Availability": workflow_availability_value(eval_row, workflow) if eval_row else NA,
                "Normalized_QoS_Score": _first_value(eval_row, "Normalized_QoS_Score") if eval_row else NA,
                "QoS_Adjusted_Composition_Score": _first_value(eval_row, "QoS_Adjusted_Composition_Score") if eval_row else NA,
                "Composition_Risk_API": bottleneck_group_summary(workflow, bottlenecks_by_mode.get(mode, pd.DataFrame())),
                "Risk_Summary": bottleneck_group_summary(workflow, bottlenecks_by_mode.get(mode, pd.DataFrame())),
            }
        )
    return pd.DataFrame(rows)


def comparison_highlights(summary: pd.DataFrame, differences: pd.DataFrame) -> list[str]:
    highlights: list[str] = []
    if not differences.empty and "Selection_Difference" in differences:
        changed = int((differences["Selection_Difference"] == "Different APIs").sum())
        total = len(differences)
        highlights.append(f"API selection differs across modes for {changed}/{total} subtasks.")

    metric_specs = [
        ("QoS_Adjusted_Composition_Score", "Highest QoS-adjusted composition score", True),
        ("Total_Response_Time_s", "Lowest total response time (s)", False),
        ("Functional_Coverage", "Highest functional coverage", True),
        ("Normalized_QoS_Score", "Highest normalized QoS score", True),
    ]
    for metric, label, higher_better in metric_specs:
        if summary.empty or metric not in summary:
            continue
        values = pd.to_numeric(summary[metric], errors="coerce")
        if values.dropna().empty:
            continue
        target = values.max() if higher_better else values.min()
        modes = summary.loc[(values - target).abs() <= BEST_MODE_TOLERANCE, "Mode"].astype(str).tolist()
        highlights.append(f"{label}: {', '.join(modes)} ({format_value(target)}).")

    risk_col = "Composition_Risk_API" if "Composition_Risk_API" in summary else "Bottleneck_API" if "Bottleneck_API" in summary else None
    if risk_col:
        risk_apis = [value for value in summary[risk_col].astype(str).tolist() if value and value != NA]
        if len(set(risk_apis)) > 1:
            highlights.append("Risk-contributing APIs change across ranking modes.")
    return highlights
