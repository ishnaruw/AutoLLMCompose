from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
NA = "N/A"


@dataclass(frozen=True)
class HealthThresholds:
    """Display-only thresholds for coloring API nodes when no project thresholds exist."""

    good_rt_ms: float = 500.0
    moderate_rt_ms: float = 1000.0
    good_tp_rps: float = 10.0
    moderate_tp_rps: float = 1.0
    good_availability: float = 0.99
    moderate_availability: float = 0.95


DEFAULT_HEALTH_THRESHOLDS = HealthThresholds()

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


def _line_fill(status: Any) -> str:
    status_key = str(status or "gray").lower()
    return HEALTH_COLORS.get(status_key, HEALTH_COLORS["gray"])


def _line_border(status: Any) -> str:
    status_key = str(status or "gray").lower()
    return HEALTH_BORDER_COLORS.get(status_key, HEALTH_BORDER_COLORS["gray"])


def _hover_for_step(row: pd.Series, *, title: str | None = None) -> str:
    title_text = title or row.get("API_Name") or row.get("API_ID") or "Workflow step"
    parts = [
        f"<b>{escape(str(title_text))}</b>",
        f"Subtask: {escape(str(row.get('Subtask_ID') or NA))}",
        f"API ID: {escape(str(row.get('API_ID') or NA))}",
        f"Functional match: {format_flag(row.get('Functional_Match'))}",
        f"Mode rank: {format_value(row.get('Mode_Rank'), 0)}",
        f"rt_ms: {format_value(row.get('rt_ms'))}",
        f"tp_rps: {format_value(row.get('tp_rps'))}",
        f"availability: {format_value(row.get('availability'))}",
        f"QoS LLM score: {format_value(row.get('QoS_LLM_Score'))}",
        f"TOPSIS score: {format_value(row.get('TOPSIS_Score'))}",
        f"Health: {escape(str(row.get('Health_Reason') or NA))}",
    ]
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


def format_flag(value: Any) -> str:
    label = _as_match_label(value)
    return NA if label is None else str(label)


def _subtask_id_from_selected_path(path: Path) -> str:
    match = re.search(r"3_selected_s([^./]+)", path.name)
    return _normalize_id(match.group(1)) if match else ""


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
        sid = _normalize_id(row.get("Sub Task", row.get("subtask_id")))
        api_id = str(row.get("Selected_API") or row.get("api_id") or "").strip()
        if mode and sid and api_id:
            lookup.setdefault((mode, sid, api_id), row)
    return lookup


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


def enrich_workflow_for_selection(
    workflow_df: pd.DataFrame,
    *,
    query_id: str,
    run_dir: Path,
    run_name: str,
    mode: str,
    thresholds: HealthThresholds = DEFAULT_HEALTH_THRESHOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    workflow = _filter_workflow(workflow_df, query_id=query_id, run_dir=run_dir, run_name=run_name, mode=mode)
    if workflow.empty:
        return pd.DataFrame(), pd.DataFrame()

    subtasks = load_subtask_descriptions(run_dir)
    candidate_lookup = load_candidate_lookup(run_dir, query_id)
    selected_by_pair, selected_by_api = load_selected_lookups(run_dir, mode)

    rows: list[dict[str, Any]] = []
    for idx, (_, step) in enumerate(workflow.iterrows(), start=1):
        sid = _normalize_id(_first_value(step, "Subtask_ID", "Sub Task", "subtask_id"))
        api_id = str(_first_value(step, "API_ID", "Selected_API", "api_id") or "").strip()
        selected = _selected_for_step(selected_by_pair, selected_by_api, sid, api_id)
        candidate = candidate_lookup.get((mode, sid, api_id), {})

        functional = _as_match_label(_first_value(step, "Functional_Match", "Functional Match"))
        if functional is None:
            functional = _as_match_label(_first_value(candidate, "Functional Match (0/1)", "Functional_Match"))

        rt_ms = _as_float(_first_value(step, "rt_ms", "QoS_RT"))
        tp_rps = _as_float(_first_value(step, "tp_rps", "QoS_TP"))
        availability = _as_float(_first_value(step, "availability", "QoS Availability"))
        if rt_ms is None:
            rt_ms = _as_float(_first_value(candidate, "QoS_RT", "rt_ms"))
            if rt_ms is None:
                rt_ms = _qos_from_service(selected, "rt_ms")
        if tp_rps is None:
            tp_rps = _as_float(_first_value(candidate, "QoS_TP", "tp_rps"))
            if tp_rps is None:
                tp_rps = _qos_from_service(selected, "tp_rps")
        if availability is None:
            availability = _as_float(_first_value(candidate, "QoS Availability", "availability"))
            if availability is None:
                availability = _qos_from_service(selected, "availability")

        service = _service_from_selected(selected)
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
                "rt_ms": rt_ms,
                "tp_rps": tp_rps,
                "availability": availability,
                "Mode_Rank": _first_value(selected, "mode_rank", "Mode Rank") or _first_value(candidate, "Mode Rank", "mode_rank"),
                "Selected_Rank": _first_value(selected, "selected_rank"),
                "Retrieved_Rank": _first_value(selected, "retrieved_rank") or _first_value(candidate, "Retrieved Rank"),
                "QoS_LLM_Score": _first_value(selected, "qos_llm_score"),
                "QoS_LLM_Rank": _first_value(selected, "qos_llm_rank"),
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

    enriched = pd.DataFrame(rows)
    bottlenecks = identify_bottlenecks(enriched)
    bottleneck_map = _bottleneck_dimension_map(bottlenecks)
    status_rows = [
        api_health_status(row, bottleneck_map=bottleneck_map, thresholds=thresholds)
        for _, row in enriched.iterrows()
    ]
    enriched["Health_Status"] = [status for status, _, _ in status_rows]
    enriched["Health_Reason"] = [reason for _, reason, _ in status_rows]
    enriched["Health_Color"] = [color for _, _, color in status_rows]
    enriched["Bottleneck_Dimensions"] = [
        ", ".join(bottleneck_map.get((_normalize_id(row.get("Subtask_ID")), str(row.get("API_ID") or "")), [])) or NA
        for _, row in enriched.iterrows()
    ]
    return enriched, bottlenecks


def _bottleneck_dimension_map(bottlenecks: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    if bottlenecks.empty:
        return out
    for _, row in bottlenecks.iterrows():
        key = (_normalize_id(row.get("Subtask_ID")), str(row.get("API_ID") or ""))
        out.setdefault(key, []).append(str(row.get("Bottleneck_Type") or row.get("Metric") or "Bottleneck"))
    return out


def identify_bottlenecks(workflow: pd.DataFrame) -> pd.DataFrame:
    if workflow.empty:
        return pd.DataFrame()
    specs = [
        ("Latency", "rt_ms", False, "Highest response time", "Increases total workflow response time"),
        ("Throughput", "tp_rps", True, "Lowest throughput", "Limits end-to-end workflow throughput"),
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


def api_health_status(
    row: pd.Series,
    *,
    bottleneck_map: dict[tuple[str, str], list[str]],
    thresholds: HealthThresholds,
) -> tuple[str, str, str]:
    key = (_normalize_id(row.get("Subtask_ID")), str(row.get("API_ID") or ""))
    if key in bottleneck_map:
        return "red", "Bottleneck API", HEALTH_COLORS["red"]

    functional = _as_match_label(row.get("Functional_Match"))
    rt_ms = _as_float(row.get("rt_ms"))
    tp_rps = _as_float(row.get("tp_rps"))
    availability = _as_float(row.get("availability"))

    if functional == 0:
        return "red", "Functional mismatch", HEALTH_COLORS["red"]
    if functional is None and rt_ms is None and tp_rps is None and availability is None:
        return "gray", "Missing functional and QoS data", HEALTH_COLORS["gray"]

    poor_qos = (
        (rt_ms is not None and rt_ms > thresholds.moderate_rt_ms)
        or (tp_rps is not None and tp_rps < thresholds.moderate_tp_rps)
        or (availability is not None and availability < thresholds.moderate_availability)
    )
    if poor_qos:
        return "red", "Poor QoS", HEALTH_COLORS["red"]

    good_qos = (
        rt_ms is not None
        and tp_rps is not None
        and availability is not None
        and rt_ms <= thresholds.good_rt_ms
        and tp_rps >= thresholds.good_tp_rps
        and availability >= thresholds.good_availability
    )
    if functional == 1 and good_qos:
        return "green", "Functional match with strong QoS", HEALTH_COLORS["green"]
    if functional == 1:
        return "orange", "Functional match with moderate or incomplete QoS", HEALTH_COLORS["orange"]
    return "gray", "Unknown selection quality", HEALTH_COLORS["gray"]


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
    average_availability = _first_value(eval_row, "Average_Workflow_Availability")
    if _is_missing(average_availability) and not workflow.empty and "availability" in workflow:
        average_availability = pd.to_numeric(workflow["availability"], errors="coerce").mean()

    fields = [
        ("Selected Mode", mode),
        ("Number of Subtasks", str(int(float(subtask_count))) if not _is_missing(subtask_count) else NA),
        ("Number of Selected APIs", str(int(float(selected_count))) if not _is_missing(selected_count) else NA),
        ("Composition Validity", validity_text),
        ("Composition Completeness", format_value(_first_value(eval_row, "Composition_Completeness"), percent=True)),
        ("Functional Coverage", format_value(_first_value(eval_row, "Functional_Coverage"), percent=True)),
        ("Total Response Time", format_value(_first_value(eval_row, "Total_Response_Time"))),
        ("Bottleneck Throughput", format_value(_first_value(eval_row, "Bottleneck_Throughput"))),
        ("Average Workflow Availability", format_value(average_availability)),
        ("Normalized QoS Score", format_value(_first_value(eval_row, "Normalized_QoS_Score"))),
        ("QoS-Adjusted Composition Score", format_value(_first_value(eval_row, "QoS_Adjusted_Composition_Score"))),
        ("Bottleneck API", bottleneck_summary(bottlenecks)),
    ]
    return [{"Metric": label, "Value": value if not _is_missing(value) else NA} for label, value in fields]


def _api_node_label(row: pd.Series, *, compact: bool) -> str:
    api_name = _shorten(row.get("API_Name") or row.get("API_ID"), 32 if compact else 44)
    api_id = _shorten(row.get("API_ID"), 30 if compact else 46)
    parts = [api_name]
    if not compact and api_id != api_name:
        parts.append(f"ID: {api_id}")
    parts.append(f"FM: {format_flag(row.get('Functional_Match'))} | rank: {format_value(row.get('Mode_Rank'), 0)}")
    parts.append(
        " | ".join(
            [
                f"rt_ms: {format_value(row.get('rt_ms'))}",
                f"tp_rps: {format_value(row.get('tp_rps'))}",
            ]
        )
    )
    parts.append(f"availability: {format_value(row.get('availability'))}")
    score_parts = []
    if not _is_missing(row.get("QoS_LLM_Score")):
        score_parts.append(f"QoS: {format_value(row.get('QoS_LLM_Score'))}")
    if not _is_missing(row.get("TOPSIS_Score")):
        score_parts.append(f"TOPSIS: {format_value(row.get('TOPSIS_Score'))}")
    if score_parts:
        parts.append(" | ".join(score_parts))
    if compact:
        return "\n".join(parts[:4])
    parts.append(f"status: {_shorten(row.get('Health_Reason'), 40)}")
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
        "  rankdir=LR;",
        '  graph [bgcolor="transparent", pad="0.2", nodesep="0.55", ranksep="0.75", splines=ortho];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.12,0.08"];',
        '  edge [color="#64748B", arrowsize=0.7, penwidth=1.2];',
        f'  query [label="User Query\\n{_dot_escape(graph_label)}", fillcolor="{HEALTH_COLORS["blue"]}", color="#0284C7"];',
        f'  final [label="{_dot_escape(final_label)}", fillcolor="{HEALTH_COLORS["final"]}", color="#548235"];',
    ]
    if workflow.empty:
        lines.append("  query -> final;")
        lines.append("}")
        return "\n".join(lines)

    previous = "query"
    for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
        sid = _normalize_id(row.get("Subtask_ID")) or str(idx)
        sub_node = f"sub_{idx}"
        api_node = f"api_{idx}"
        subtask = _shorten(row.get("Subtask"), 34 if compact else 58)
        sub_label = f"Subtask {sid}" if compact or subtask == NA else f"Subtask {sid}\\n{subtask}"
        api_label = _api_node_label(row, compact=compact)
        fill = row.get("Health_Color") or HEALTH_COLORS["gray"]
        lines.extend(
            [
                f'  {sub_node} [label="{_dot_escape(sub_label)}", fillcolor="{HEALTH_COLORS["subtask"]}", color="#CBD5E1"];',
                f'  {api_node} [label="{_dot_escape(api_label)}", fillcolor="{fill}", color="#64748B"];',
                f"  {previous} -> {sub_node};",
                f"  {sub_node} -> {api_node};",
            ]
        )
        previous = api_node
    lines.append(f"  {previous} -> final;")
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
    top_y = steps * 1.55 + 1.2
    query_label = "<b>User Query</b><br>" + _wrap_text(
        query_context.get("goal") or query_context.get("label") or query_context.get("query_id"),
        width=54,
        max_lines=3,
    )
    _add_box(
        fig,
        x=0.5,
        y=top_y,
        width=1.7,
        height=0.78,
        label=query_label,
        fill=HEALTH_COLORS["blue"],
        border=HEALTH_BORDER_COLORS["blue"],
        hover=escape(str(query_context.get("goal") or query_context.get("label") or NA)),
        text_size=12,
    )
    previous = (0.5, top_y - 0.48)
    for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
        y = top_y - idx * 1.55
        sid = _normalize_id(row.get("Subtask_ID")) or str(idx)
        sub_label = f"<b>Subtask {escape(sid)}</b><br>{_wrap_text(row.get('Subtask'), width=36, max_lines=3)}"
        api_label = (
            f"<b>Selected API {idx}</b><br>"
            f"{_wrap_text(row.get('API_Name') or row.get('API_ID'), width=42, max_lines=2)}<br>"
            f"FM {format_flag(row.get('Functional_Match'))} | rank {format_value(row.get('Mode_Rank'), 0)} | "
            f"rt {format_value(row.get('rt_ms'))}"
        )
        sub_pos = (0.0, y)
        api_pos = (1.62, y)
        _add_box(
            fig,
            x=sub_pos[0],
            y=sub_pos[1],
            width=1.35,
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
            width=1.65,
            height=0.98,
            label=api_label,
            fill=_line_fill(row.get("Health_Status")),
            border=_line_border(row.get("Health_Status")),
            hover=_hover_for_step(row),
            text_size=12,
            line_width=2,
        )
        _add_arrow(fig, previous, (sub_pos[0], sub_pos[1] + 0.48))
        _add_arrow(fig, (sub_pos[0] + 0.68, sub_pos[1]), (api_pos[0] - 0.84, api_pos[1]))
        previous = (api_pos[0], api_pos[1] - 0.52)

    final_y = top_y - (steps + 1) * 1.55
    score = format_value(_first_value(eval_row, "QoS_Adjusted_Composition_Score"))
    validity = _first_value(eval_row, "Composition_Validity")
    validity_text = "Valid" if _as_float(validity) == 1 else ("Invalid" if not _is_missing(validity) else NA)
    final_label = f"<b>Final Composed Workflow</b><br>Mode: {escape(mode)} | Score: {score}<br>{validity_text}"
    _add_box(
        fig,
        x=0.5,
        y=final_y,
        width=1.75,
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
    _add_arrow(fig, previous, (0.5, final_y + 0.45))
    fig.update_xaxes(range=[-0.9, 2.6])
    fig.update_yaxes(range=[final_y - 0.75, top_y + 0.75])
    return _finish_diagram_layout(fig, height=max(620, 170 * (steps + 2)), title="Composed Workflow")


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
                    "label": f"Selected API {idx}<br>{_wrap_text(row.get('API_Name') or row.get('API_ID'), width=38, max_lines=2)}",
                    "fill": _line_fill(row.get("Health_Status")),
                    "border": _line_border(row.get("Health_Status")),
                    "hover": _hover_for_step(row),
                }
            )
        items.append({"label": "Final Response", "fill": HEALTH_COLORS["final"], "border": HEALTH_BORDER_COLORS["final"], "hover": "Composed response returned to the user."})
        title = "Agent and API Execution Flow"
    else:
        items = [{"label": "User Request", "fill": HEALTH_COLORS["blue"], "border": HEALTH_BORDER_COLORS["blue"], "hover": "Incoming user request."}]
        for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
            sid = _normalize_id(row.get("Subtask_ID")) or str(idx)
            items.append(
                {
                    "label": f"Subtask {escape(sid)} API<br>{_wrap_text(row.get('API_Name') or row.get('API_ID'), width=40, max_lines=2)}",
                    "fill": _line_fill(row.get("Health_Status")),
                    "border": _line_border(row.get("Health_Status")),
                    "hover": _hover_for_step(row),
                }
            )
        items.append({"label": "Final Composed Result", "fill": HEALTH_COLORS["final"], "border": HEALTH_BORDER_COLORS["final"], "hover": "Final composed result."})
        title = "API Execution Sequence"

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
            height=0.7,
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


def build_mode_comparison_figure(workflows_by_mode: dict[str, pd.DataFrame], modes: list[str]) -> go.Figure:
    fig = go.Figure()
    max_steps = max((len(df) for df in workflows_by_mode.values() if not df.empty), default=0)
    if max_steps == 0:
        return _finish_diagram_layout(fig, height=420, title="Mode Comparison")
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
            fig.add_trace(
                go.Scatter(
                    x=[idx],
                    y=[y],
                    mode="markers+text",
                    marker=dict(
                        symbol="square",
                        size=72,
                        color=_line_fill(row.get("Health_Status")),
                        line=dict(color=_line_border(row.get("Health_Status")), width=2),
                    ),
                    text=[label],
                    textposition="middle center",
                    textfont=dict(size=11, color="#111827"),
                    hovertext=[_hover_for_step(row, title=f"{mode} step {idx}")],
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
        ("rt_ms", "Response Time", "Higher is worse"),
        ("tp_rps", "Throughput", "Lower is worse"),
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
    for col_idx, (metric, _, _) in enumerate(specs, start=1):
        values = pd.to_numeric(workflow[metric], errors="coerce") if metric in workflow else pd.Series(dtype=float)
        colors = []
        for _, row in workflow.iterrows():
            pair = (str(row.get("API_ID") or ""), metric)
            colors.append("#C53030" if pair in bottleneck_pairs else _line_fill(row.get("Health_Status")))
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                hovertext=[
                    _hover_for_step(row, title=f"{metric}: {format_value(row.get(metric))}")
                    for _, row in workflow.iterrows()
                ],
                hoverinfo="text",
                text=[format_value(value) for value in values],
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
    availability = _first_value(eval_row, "Average_Workflow_Availability")
    if _is_missing(availability) and not workflow.empty and "availability" in workflow:
        availability = pd.to_numeric(workflow["availability"], errors="coerce").mean()
    metrics = [
        ("QoS-Adjusted Score", _first_value(eval_row, "QoS_Adjusted_Composition_Score")),
        ("Normalized QoS", _first_value(eval_row, "Normalized_QoS_Score")),
        ("Avg. Availability", availability),
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


def build_agent_sequence_dot(workflow: pd.DataFrame) -> str:
    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  graph [bgcolor="transparent", pad="0.2", nodesep="0.55", ranksep="0.65"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.12,0.08"];',
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
        label = f"Selected API {idx}\\n{_shorten(row.get('API_Name') or row.get('API_ID'), 38)}"
        fill = row.get("Health_Color") or HEALTH_COLORS["gray"]
        lines.append(f'  {node} [label="{_dot_escape(label)}", fillcolor="{fill}", color="#64748B"];')
        lines.append(f"  {previous} -> {node};")
        previous = node
    lines.append(f'  final [label="Final Response", fillcolor="{HEALTH_COLORS["final"]}", color="#548235"];')
    lines.append(f"  {previous} -> final;")
    lines.append("}")
    return "\n".join(lines)


def build_api_execution_dot(workflow: pd.DataFrame) -> str:
    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  graph [bgcolor="transparent", pad="0.2", nodesep="0.55", ranksep="0.65"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.12,0.08"];',
        '  edge [color="#64748B", arrowsize=0.7, penwidth=1.2];',
        f'  request [label="User Request", fillcolor="{HEALTH_COLORS["blue"]}", color="#0284C7"];',
    ]
    previous = "request"
    for idx, (_, row) in enumerate(workflow.iterrows(), start=1):
        node = f"exec_api_{idx}"
        sid = _normalize_id(row.get("Subtask_ID")) or str(idx)
        label = f"Subtask {sid} API\\n{_shorten(row.get('API_Name') or row.get('API_ID'), 38)}"
        fill = row.get("Health_Color") or HEALTH_COLORS["gray"]
        lines.append(f'  {node} [label="{_dot_escape(label)}", fillcolor="{fill}", color="#64748B"];')
        lines.append(f"  {previous} -> {node};")
        previous = node
    lines.append(f'  result [label="Final Composed Result", fillcolor="{HEALTH_COLORS["final"]}", color="#548235"];')
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
        lines.append(f"    P->>{api_id}: Execute step {idx}")
    lines.append("    P-->>U: Final response")
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
    lines.append(f"    {previous}-->>U: Final composed result")
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
                "Total_Response_Time": _first_value(eval_row, "Total_Response_Time") if eval_row else NA,
                "Bottleneck_Throughput": _first_value(eval_row, "Bottleneck_Throughput") if eval_row else NA,
                "Workflow_Availability": _first_value(eval_row, "Workflow_Availability", "Average_Workflow_Availability") if eval_row else NA,
                "Normalized_QoS_Score": _first_value(eval_row, "Normalized_QoS_Score") if eval_row else NA,
                "QoS_Adjusted_Composition_Score": _first_value(eval_row, "QoS_Adjusted_Composition_Score") if eval_row else NA,
                "Bottleneck_API": bottleneck_summary(bottlenecks_by_mode.get(mode, pd.DataFrame())),
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
        ("Total_Response_Time", "Lowest total response time", False),
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
        modes = summary.loc[values == target, "Mode"].astype(str).tolist()
        highlights.append(f"{label}: {', '.join(modes)} ({format_value(target)}).")

    if not summary.empty and "Bottleneck_API" in summary:
        bottlenecks = [value for value in summary["Bottleneck_API"].astype(str).tolist() if value and value != NA]
        if len(set(bottlenecks)) > 1:
            highlights.append("Bottleneck APIs change across ranking modes.")
    return highlights
