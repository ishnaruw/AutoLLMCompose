from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable

try:
    import streamlit as st
except Exception:  # pragma: no cover - keeps loader importable outside Streamlit.
    st = None

from src.ui import live_demo_catalog as catalog


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results/logs"
QUERY_CATALOG_PATH = PROJECT_ROOT / "data/queries/all_user_query.jsonl"

MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
MODE_LABELS = {
    "no_qos": "No-QoS",
    "qos_pure_llm": "QoS-Pure-LLM",
    "qos_topsis": "QoS-TOPSIS",
    "qos_hybrid": "QoS-Hybrid",
}
OFFICIAL_QUERY_IDS = tuple(f"q{idx:02d}" for idx in range(1, 16))
COMPOSITION_SCORE_COL = "QoS_Adjusted_Composition_Score"
AGGREGATE_COMPONENT_COLUMNS = (
    "Functional_Coverage",
    "Normalized_QoS_Score",
    "Total_Response_Time_s",
    "Bottleneck_Throughput_kbps",
    "Average_Workflow_Availability",
)


def _cache_data(func: Callable) -> Callable:
    if st is None:
        return func
    return st.cache_data(show_spinner=False)(func)


def resolve_path(path_text: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(str(path_text or "")).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def query_id_from_name(name: str) -> str | None:
    match = re.match(r"^(q\d{1,3})(?:[_-]|$)", str(name).strip(), flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def timestamp_from_name(name: str) -> str | None:
    match = re.search(r"(20\d{6}T\d{6})", str(name))
    return match.group(1) if match else None


def query_sort_key(query_id: str) -> tuple[int, str]:
    text = str(query_id or "").lower()
    if text.startswith("q") and text[1:].isdigit():
        return int(text[1:]), text
    return 9999, text


def mode_label(mode: str) -> str:
    return MODE_LABELS.get(str(mode), str(mode))


def mode_sort_key(mode: str) -> tuple[int, str]:
    mode = str(mode)
    return (MODE_ORDER.index(mode), mode) if mode in MODE_ORDER else (999, mode)


def short_api_name(api_id: Any, *, max_parts: int = 4) -> str:
    text = str(api_id or "").strip()
    if not text:
        return ""
    parts = [part for part in re.split(r"[_\s]+", text) if part]
    if len(parts) <= max_parts:
        return text
    return "_".join(parts[:max_parts]) + "..."


def _json_path_from_reference(reference: Any, *, query_dir: Path | None = None) -> Path | None:
    if not reference:
        return None
    raw = Path(str(reference)).expanduser()
    if raw.is_absolute():
        return raw
    project_candidate = (PROJECT_ROOT / raw).resolve()
    if project_candidate.exists():
        return project_candidate
    if query_dir is not None:
        query_candidate = (query_dir / raw).resolve()
        if query_candidate.exists():
            return query_candidate
    return project_candidate


@_cache_data
def read_json_file(path_text: str) -> tuple[Any | None, str | None]:
    path = resolve_path(path_text)
    if not path.exists():
        return None, f"Missing file: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"Could not read {path}: {exc}"


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    return read_json_file(str(path))


@_cache_data
def load_query_catalog(path_text: str = str(QUERY_CATALOG_PATH)) -> dict[str, dict[str, str]]:
    path = resolve_path(path_text)
    catalog: dict[str, dict[str, str]] = {}
    if not path.exists():
        return catalog
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return catalog
    for idx, line in enumerate(lines, start=1):
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        query_id = str(payload.get("id") or f"q{idx:02d}").lower()
        catalog[query_id] = {
            "title": str(payload.get("title") or ""),
            "goal": str(payload.get("goal") or ""),
            "category": str(payload.get("category") or ""),
            "domain": str(payload.get("domain") or ""),
        }
    return catalog


def _looks_like_query_dir(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith(".") or query_id_from_name(path.name) is None:
        return False
    markers = [
        "meta.json",
        "0_decomposer.json",
        "evaluation_result.json",
        "evaluation",
        "no_qos",
        "qos_pure_llm",
        "qos_topsis",
        "qos_hybrid",
    ]
    return any((path / marker).exists() for marker in markers)


def _query_dirs_in_run(run_dir: Path) -> list[Path]:
    if not run_dir.exists() or not run_dir.is_dir():
        return []
    return [child for child in run_dir.iterdir() if _looks_like_query_dir(child)]


def _available_modes_for_query(query_dir: Path, meta: dict[str, Any] | None = None) -> list[str]:
    modes: set[str] = set()
    if isinstance(meta, dict) and isinstance(meta.get("modes"), list):
        modes.update(str(mode) for mode in meta["modes"] if mode)
    for mode in MODE_ORDER:
        if (query_dir / mode / "4_planner.json").exists() or (query_dir / mode).is_dir():
            modes.add(mode)
    return sorted(modes, key=mode_sort_key)


@_cache_data
def discover_run_folders(base_dir_text: str = str(DEFAULT_RESULTS_ROOT)) -> list[dict[str, Any]]:
    base_dir = resolve_path(base_dir_text)
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    rows: list[dict[str, Any]] = []
    candidates = [base_dir] + [path for path in base_dir.rglob("*") if path.is_dir() and not path.name.startswith(".")]
    seen: set[str] = set()
    for path in candidates:
        query_dirs = _query_dirs_in_run(path)
        if not query_dirs:
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        query_ids = sorted({query_id_from_name(qdir.name) or "" for qdir in query_dirs}, key=query_sort_key)
        timestamps = [timestamp_from_name(qdir.name) for qdir in query_dirs if timestamp_from_name(qdir.name)]
        try:
            label_root = path.relative_to(PROJECT_ROOT)
        except ValueError:
            label_root = path
        rows.append(
            {
                "run_dir": str(path),
                "label": f"{label_root} ({len(query_dirs)} queries)",
                "query_count": len(query_dirs),
                "query_ids": query_ids,
                "latest_timestamp": max(timestamps) if timestamps else "",
            }
        )
    return sorted(rows, key=lambda row: (str(row.get("latest_timestamp") or ""), str(row.get("label") or "")), reverse=True)


@_cache_data
def scan_run_folder(run_dir_text: str) -> dict[str, Any]:
    run_dir = resolve_path(run_dir_text)
    warnings: list[str] = []
    catalog = load_query_catalog()
    if not run_dir.exists():
        return {"run_dir": str(run_dir), "queries": [], "warnings": [f"Run folder not found: {run_dir}"]}
    if not run_dir.is_dir():
        return {"run_dir": str(run_dir), "queries": [], "warnings": [f"Run path is not a folder: {run_dir}"]}

    by_query: dict[str, dict[str, Any]] = {}
    for query_dir in _query_dirs_in_run(run_dir):
        query_id = query_id_from_name(query_dir.name)
        if not query_id:
            continue
        meta, meta_error = _read_json(query_dir / "meta.json")
        if meta_error and (query_dir / "meta.json").exists():
            warnings.append(meta_error)
        meta = meta if isinstance(meta, dict) else {}
        num_subtasks = meta.get("num_subtasks")
        if num_subtasks in (None, "") and (query_dir / "0_decomposer.json").exists():
            decomposer_payload, decomposer_error = _read_json(query_dir / "0_decomposer.json")
            if decomposer_error is None:
                num_subtasks = len(_subtasks_from_payload(decomposer_payload))
        catalog_row = catalog.get(query_id, {})
        row = {
            "query_id": query_id,
            "query_dir": str(query_dir),
            "folder_name": query_dir.name,
            "timestamp": timestamp_from_name(query_dir.name) or "",
            "title": str(meta.get("query_title") or catalog_row.get("title") or ""),
            "user_goal": str(meta.get("user_goal") or catalog_row.get("goal") or ""),
            "category": str(meta.get("category") or meta.get("query_category") or catalog_row.get("category") or ""),
            "domain": str(meta.get("domain") or meta.get("query_domain") or catalog_row.get("domain") or ""),
            "num_subtasks": num_subtasks,
            "available_modes": _available_modes_for_query(query_dir, meta),
        }
        existing = by_query.get(query_id)
        if existing is None or str(row["timestamp"]) > str(existing.get("timestamp") or ""):
            if existing is not None:
                warnings.append(
                    f"Multiple folders found for {query_id}; using the latest timestamped folder `{row['folder_name']}`."
                )
            by_query[query_id] = row

    queries = sorted(by_query.values(), key=lambda row: query_sort_key(str(row["query_id"])))
    if not queries:
        warnings.append(f"No qXX timestamped query folders were found directly inside: {run_dir}")
    return {"run_dir": str(run_dir), "queries": queries, "warnings": warnings}


def _load_artifact(
    query_dir: Path,
    relative_path: str,
    *,
    eval_result: dict[str, Any] | None = None,
    eval_result_key: str | None = None,
) -> tuple[Any | None, str | None, str | None]:
    candidates: list[Path] = [query_dir / relative_path]
    if eval_result_key and isinstance(eval_result, dict):
        referenced = _json_path_from_reference(eval_result.get(eval_result_key), query_dir=query_dir)
        if referenced is not None and referenced not in candidates:
            candidates.append(referenced)
    for path in candidates:
        payload, error = _read_json(path)
        if error is None:
            return payload, None, str(path)
    return None, f"Missing artifact: {relative_path}", None


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "summary_rows", "data"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _subtask_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    if text.lower().startswith("s") and text[1:].isdigit():
        return text[1:]
    return text


def _first_present(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _as_intish(value: Any) -> Any:
    try:
        parsed = float(value)
    except Exception:
        return value
    if parsed.is_integer():
        return int(parsed)
    return parsed


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return None if math.isnan(parsed) else parsed


def _load_composition_metric_rows(
    query_dir: Path,
    query_id: str,
    *,
    eval_result: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    composition_payload, composition_error, composition_path = _load_artifact(
        query_dir,
        f"evaluation/query_{query_id}_composition_qos_eval_rows.json",
        eval_result=eval_result,
        eval_result_key="composition_qos_eval_rows_json",
    )
    composition_rows = _rows_from_payload(composition_payload)
    if composition_rows:
        return composition_rows, composition_path, None

    summary_payload, summary_error, summary_path = _load_artifact(
        query_dir,
        f"evaluation/query_{query_id}_composition_qos_eval_summary.json",
        eval_result=eval_result,
        eval_result_key="composition_qos_eval_summary_json",
    )
    summary_rows = _rows_from_payload(summary_payload)
    if summary_rows:
        return summary_rows, summary_path, None

    return [], None, composition_error or summary_error or f"No composition metric rows found for {query_id}."


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _first_mode_row(rows: list[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("Mode") or "") == mode:
            return row
    return None


@_cache_data
def load_official_query_aggregate(run_dir_text: str) -> dict[str, Any]:
    scan = scan_run_folder(run_dir_text)
    queries = {
        str(row.get("query_id") or ""): row
        for row in scan.get("queries", [])
        if str(row.get("query_id") or "") in OFFICIAL_QUERY_IDS
    }
    found_query_ids = [query_id for query_id in OFFICIAL_QUERY_IDS if query_id in queries]
    missing_query_ids = [query_id for query_id in OFFICIAL_QUERY_IDS if query_id not in queries]
    warnings: list[str] = []
    artifact_sources: dict[str, str] = {}
    loaded_query_ids: set[str] = set()
    mode_values: dict[str, dict[str, list[float]]] = {
        mode: {COMPOSITION_SCORE_COL: [], **{metric: [] for metric in AGGREGATE_COMPONENT_COLUMNS}}
        for mode in MODE_ORDER
    }
    mode_query_ids: dict[str, set[str]] = {mode: set() for mode in MODE_ORDER}

    for query_id in OFFICIAL_QUERY_IDS:
        query_row = queries.get(query_id)
        if query_row is None:
            continue
        query_dir = Path(str(query_row.get("query_dir") or ""))
        eval_result, eval_error = _read_json(query_dir / "evaluation_result.json")
        if eval_error and (query_dir / "evaluation_result.json").exists():
            warnings.append(f"{query_id}: {eval_error}")
        eval_result = eval_result if isinstance(eval_result, dict) else {}
        rows, source, load_warning = _load_composition_metric_rows(query_dir, query_id, eval_result=eval_result)
        if load_warning:
            warnings.append(f"{query_id}: {load_warning}")
            continue
        if source:
            artifact_sources[query_id] = source
        if rows:
            loaded_query_ids.add(query_id)

        for mode in MODE_ORDER:
            row = _first_mode_row(rows, mode)
            if row is None:
                continue
            score = _as_float(row.get(COMPOSITION_SCORE_COL))
            if score is None:
                continue
            mode_values[mode][COMPOSITION_SCORE_COL].append(score)
            mode_query_ids[mode].add(query_id)
            for metric in AGGREGATE_COMPONENT_COLUMNS:
                value = _as_float(row.get(metric))
                if value is not None:
                    mode_values[mode][metric].append(value)

    score_rows = []
    component_rows = []
    official_count = len(OFFICIAL_QUERY_IDS)
    for mode in MODE_ORDER:
        included_count = len(mode_query_ids[mode])
        score_rows.append(
            {
                "Mode": mode,
                "Mode_Label": mode_label(mode),
                "Average_QoS_Adjusted_Composition_Score": _mean(mode_values[mode][COMPOSITION_SCORE_COL]),
                "Number_of_queries_included": included_count,
                "Missing_query_count": official_count - included_count,
            }
        )
        component_rows.append(
            {
                "Mode": mode,
                "Mode_Label": mode_label(mode),
                "Average_Functional_Coverage": _mean(mode_values[mode]["Functional_Coverage"]),
                "Average_Normalized_QoS_Score": _mean(mode_values[mode]["Normalized_QoS_Score"]),
                "Average_Total_Response_Time": _mean(mode_values[mode]["Total_Response_Time_s"]),
                "Average_Bottleneck_Throughput": _mean(mode_values[mode]["Bottleneck_Throughput_kbps"]),
                "Average_Workflow_Availability": _mean(mode_values[mode]["Average_Workflow_Availability"]),
            }
        )

    return {
        "run_dir": scan.get("run_dir") or str(resolve_path(run_dir_text)),
        "official_query_count": official_count,
        "found_query_ids": found_query_ids,
        "missing_query_ids": missing_query_ids,
        "loaded_query_ids": sorted(loaded_query_ids, key=query_sort_key),
        "score_rows": score_rows,
        "component_rows": component_rows,
        "warnings": warnings,
        "artifact_sources": artifact_sources,
    }


def _functional_match(value: Any) -> int | None:
    parsed = _as_float(value)
    if parsed is None:
        return None
    return 1 if parsed >= 0.5 else 0


def _query_category_from_payloads(*payloads: Any) -> str:
    keys = ["category", "domain", "query_category", "query_domain"]
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in keys:
            value = payload.get(key)
            if value not in (None, ""):
                return str(value)
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in keys:
                value = metadata.get(key)
                if value not in (None, ""):
                    return str(value)
    return ""


def _subtasks_from_payload(payload: Any) -> list[dict[str, Any]]:
    rows = _rows_from_payload(payload)
    subtasks: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        subtask_id = _subtask_id(row.get("id") or row.get("subtask_id") or row.get("Subtask_ID") or idx)
        subtasks.append(
            {
                "subtask_id": subtask_id,
                "description": str(row.get("description") or row.get("goal") or row.get("Subtask") or ""),
            }
        )
    return subtasks


def _api_lookup_from_rows(
    candidate_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    *,
    retriever_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
    catalog_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    retriever_lookup = retriever_lookup or {}
    catalog_by_id = catalog_by_id or {}

    def add(row: dict[str, Any], *, prefer_existing: bool) -> None:
        subtask_id = _subtask_id(_first_present(row, ["Sub Task", "Subtask_ID", "subtask_id", "Subtask ID"]))
        api_id = str(_first_present(row, ["Selected_API", "API_ID", "api_id", "API ID"]) or "").strip()
        if not subtask_id or not api_id:
            return
        key = (subtask_id, api_id)
        if prefer_existing and key in lookup:
            return
        existing = lookup.get(key, {})
        artifact_info = retriever_lookup.get(key, {})
        fields = _source_api_fields(api_id, catalog_by_id=catalog_by_id, artifact_info=artifact_info)
        lookup[key] = {
            **existing,
            **{k: v for k, v in fields.items() if v not in (None, "") or k not in existing},
            "api_id": api_id,
            "display_name": existing.get("display_name") or fields.get("display_name") or short_api_name(api_id),
            "category": existing.get("category") or fields.get("category"),
            "tool_name": existing.get("tool_name") or fields.get("tool_name"),
            "api_name": existing.get("api_name") or fields.get("api_name"),
            "http_method": existing.get("http_method") or fields.get("http_method"),
            "endpoint_path": existing.get("endpoint_path") or fields.get("endpoint_path"),
            "endpoint_url": existing.get("endpoint_url") or fields.get("endpoint_url"),
            "rag_score": existing.get("rag_score") or artifact_info.get("rag_score"),
            "functional_match": _functional_match(
                _first_present(row, ["Functional Match (0/1)", "Functional_Match", "functional_match"])
            )
            if _first_present(row, ["Functional Match (0/1)", "Functional_Match", "functional_match"]) is not None
            else existing.get("functional_match"),
            "retrieved_rank": _as_intish(_first_present(row, ["Retrieved Rank", "retrieved_rank"])),
            "mode_rank": _as_intish(_first_present(row, ["Mode Rank", "mode_rank"])),
            "qos_rt_s": _first_present(row, ["QoS_RT_s", "rt_s", "response_time_s"]),
            "qos_tp_kbps": _first_present(row, ["QoS_TP_kbps", "tp_kbps", "throughput_kbps"]),
            "qos_availability": _first_present(row, ["QoS Availability", "availability"]),
            "ranker_reason": _first_present(row, ["Ranker Reason", "reason", "ranker_reason"]),
            "comments": _first_present(row, ["Comments", "comments"]),
        }

    for row in retrieval_rows:
        add(row, prefer_existing=True)
    for row in candidate_rows:
        add(row, prefer_existing=False)
    return lookup


def _service_name(service: Any) -> str:
    if not isinstance(service, dict):
        return ""
    return str(
        service.get("name")
        or service.get("toolbench_endpoint_description")
        or service.get("tool_name")
        or service.get("toolbench_tool_name")
        or ""
    )


def _qos_from_service(service: Any) -> dict[str, Any]:
    if not isinstance(service, dict):
        return {}
    qos = service.get("qos")
    return qos if isinstance(qos, dict) else {}


def _source_api_fields(
    api_id: str,
    *,
    catalog_by_id: dict[str, dict[str, Any]] | None = None,
    artifact_info: dict[str, Any] | None = None,
    service: dict[str, Any] | None = None,
) -> dict[str, Any]:
    catalog_row = (catalog_by_id or {}).get(api_id) or {}
    enrichment = service.get("toolbench_enrichment", {}) if isinstance(service, dict) else {}
    if not isinstance(enrichment, dict):
        enrichment = {}
    artifact_info = artifact_info or {}
    display_name = (
        catalog_row.get("display_name")
        or artifact_info.get("display_name")
        or _service_name(service)
        or short_api_name(api_id)
    )
    return {
        "display_name": display_name,
        "category": catalog_row.get("category") or artifact_info.get("category") or (service or {}).get("category") or enrichment.get("category"),
        "tool_name": catalog_row.get("tool_name") or artifact_info.get("tool_name") or (service or {}).get("tool_name") or enrichment.get("tool_name"),
        "api_name": catalog_row.get("api_name") or artifact_info.get("api_name") or (service or {}).get("name") or enrichment.get("endpoint_name"),
        "http_method": catalog_row.get("http_method") or artifact_info.get("http_method") or (service or {}).get("method") or enrichment.get("endpoint_method"),
        "endpoint_path": catalog_row.get("endpoint_path") or artifact_info.get("endpoint_path"),
        "endpoint_url": catalog_row.get("endpoint_url") or artifact_info.get("endpoint_url") or (service or {}).get("url") or enrichment.get("endpoint_url"),
        "catalog_found": bool(catalog_row),
        "catalog_source_file": catalog_row.get("source_file"),
        "metadata_source_file": catalog_row.get("source_file") or artifact_info.get("metadata_source_file"),
    }


def _retriever_file_lookup(query_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(query_dir.glob("1_retriever_s*.json")):
        match = re.search(r"_s(\d+)", path.stem)
        subtask_id = match.group(1) if match else ""
        if not subtask_id:
            continue
        payload, error = _read_json(path)
        if error is not None:
            continue
        for row in _rows_from_payload(payload):
            compressed = row.get("compressed") if isinstance(row.get("compressed"), dict) else {}
            api_id = str(row.get("api_id") or compressed.get("api_id") or "").strip()
            if not api_id:
                continue
            lookup[(subtask_id, api_id)] = {
                "api_id": api_id,
                "display_name": compressed.get("name") or compressed.get("tool_name") or short_api_name(api_id),
                "category": compressed.get("category"),
                "tool_name": compressed.get("tool_name"),
                "api_name": compressed.get("name"),
                "http_method": compressed.get("method"),
                "endpoint_url": compressed.get("url"),
                "endpoint_path": compressed.get("url"),
                "rag_score": row.get("rag_score"),
                "metadata_source_file": str(path),
            }
    return lookup


def _selected_file_lookup(
    query_dir: Path,
    mode: str,
    *,
    catalog_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    mode_dir = query_dir / mode
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not mode_dir.exists():
        return lookup
    for path in sorted(mode_dir.glob("3_selected_s*.json")):
        payload, error = _read_json(path)
        if error is not None:
            continue
        for row in _rows_from_payload(payload):
            subtask_id = _subtask_id(row.get("subtask_id") or row.get("Subtask_ID") or query_id_from_name(path.stem) or "")
            if not subtask_id:
                match = re.search(r"_s(\d+)", path.stem)
                subtask_id = match.group(1) if match else ""
            api_id = str(row.get("api_id") or row.get("API_ID") or "").strip()
            if not subtask_id or not api_id:
                continue
            service = row.get("service")
            qos = _qos_from_service(service)
            fields = _source_api_fields(
                api_id,
                catalog_by_id=catalog_by_id,
                service=service if isinstance(service, dict) else None,
                artifact_info={"metadata_source_file": str(path)},
            )
            lookup[(subtask_id, api_id)] = {
                "api_id": api_id,
                **fields,
                "candidate_id": row.get("candidate_id"),
                "retrieved_rank": row.get("retrieved_rank"),
                "mode_rank": row.get("mode_rank"),
                "ranker_reason": row.get("reason"),
                "qos_rt_s": qos.get("rt_s"),
                "qos_tp_kbps": qos.get("tp_kbps"),
                "qos_availability": qos.get("availability"),
                "selected_by": row.get("selected_by"),
            }
    return lookup


def _extract_planner_steps(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    steps = None
    workflow = payload.get("execution_workflow")
    if isinstance(workflow, dict) and isinstance(workflow.get("steps"), list):
        steps = workflow["steps"]
    if steps is None:
        primary = payload.get("primary_plan")
        if isinstance(primary, dict) and isinstance(primary.get("steps"), list):
            steps = primary["steps"]
    if not isinstance(steps, list):
        return []

    normalized: list[dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        normalized.append(
            {
                "step": step.get("step") or idx,
                "subtask_id": _subtask_id(step.get("subtask_id") or step.get("subtask") or idx),
                "api_id": str(step.get("api_id") or step.get("API_ID") or "").strip(),
                "action": str(step.get("action") or step.get("Action") or ""),
                "why": str(step.get("why") or step.get("rationale") or ""),
                "input_from_previous_step": step.get("input_from_previous_step") or step.get("input_mapping"),
                "output_to_next_step": step.get("output_to_next_step") or step.get("output_mapping"),
            }
        )
    return [step for step in normalized if step.get("api_id")]


def _fallback_steps_from_selected_files(selected_lookup: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    steps = []
    for idx, ((subtask_id, api_id), _) in enumerate(sorted(selected_lookup.items(), key=lambda item: query_sort_key("q" + item[0][0])), start=1):
        steps.append({"step": idx, "subtask_id": subtask_id, "api_id": api_id, "action": "", "why": ""})
    return steps


def _enrich_selected_step(
    step: dict[str, Any],
    *,
    subtasks_by_id: dict[str, str],
    row_lookup: dict[tuple[str, str], dict[str, Any]],
    selected_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    key = (str(step.get("subtask_id") or ""), str(step.get("api_id") or ""))
    row_info = row_lookup.get(key, {})
    selected_info = selected_lookup.get(key, {})
    return {
        **step,
        "subtask": subtasks_by_id.get(str(step.get("subtask_id") or ""), ""),
        "display_name": selected_info.get("display_name") or row_info.get("display_name") or short_api_name(step.get("api_id")),
        "category": selected_info.get("category") or row_info.get("category"),
        "tool_name": selected_info.get("tool_name") or row_info.get("tool_name"),
        "api_name": selected_info.get("api_name") or row_info.get("api_name"),
        "http_method": selected_info.get("http_method") or row_info.get("http_method"),
        "endpoint_path": selected_info.get("endpoint_path") or row_info.get("endpoint_path"),
        "endpoint_url": selected_info.get("endpoint_url") or row_info.get("endpoint_url"),
        "catalog_found": selected_info.get("catalog_found", row_info.get("catalog_found")),
        "metadata_source_file": selected_info.get("metadata_source_file") or row_info.get("metadata_source_file"),
        "functional_match": row_info.get("functional_match"),
        "retrieved_rank": row_info.get("retrieved_rank", selected_info.get("retrieved_rank")),
        "mode_rank": row_info.get("mode_rank", selected_info.get("mode_rank")),
        "qos_rt_s": row_info.get("qos_rt_s", selected_info.get("qos_rt_s")),
        "qos_tp_kbps": row_info.get("qos_tp_kbps", selected_info.get("qos_tp_kbps")),
        "qos_availability": row_info.get("qos_availability", selected_info.get("qos_availability")),
        "ranker_reason": row_info.get("ranker_reason", selected_info.get("ranker_reason")),
        "comments": row_info.get("comments"),
    }


def _normalize_ranking_rows(
    candidate_rows: list[dict[str, Any]],
    selected_by_mode_subtask: dict[tuple[str, str], set[str]],
    row_lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        mode = str(_first_present(row, ["Mode", "mode"]) or "")
        subtask_id = _subtask_id(_first_present(row, ["Sub Task", "Subtask_ID", "subtask_id", "Subtask ID"]))
        api_id = str(_first_present(row, ["Selected_API", "API_ID", "api_id", "API ID"]) or "").strip()
        if not mode or not subtask_id or not api_id:
            continue
        info = row_lookup.get((subtask_id, api_id), {})
        rows.append(
            {
                "query_id": _first_present(row, ["Query_ID", "query_id"]),
                "mode": mode,
                "subtask_id": subtask_id,
                "mode_rank": _as_intish(_first_present(row, ["Mode Rank", "mode_rank"])),
                "retrieved_rank": _as_intish(_first_present(row, ["Retrieved Rank", "retrieved_rank"])),
                "api_id": api_id,
                "display_name": info.get("display_name") or short_api_name(api_id),
                "category": info.get("category"),
                "functional_match": _functional_match(_first_present(row, ["Functional Match (0/1)", "Functional_Match"]))
                if _first_present(row, ["Functional Match (0/1)", "Functional_Match"]) is not None
                else info.get("functional_match"),
                "selected_for_planner": "Yes" if api_id in selected_by_mode_subtask.get((mode, subtask_id), set()) else "No",
                "qos_rt_s": _first_present(row, ["QoS_RT_s", "rt_s"]) or info.get("qos_rt_s"),
                "qos_tp_kbps": _first_present(row, ["QoS_TP_kbps", "tp_kbps"]) or info.get("qos_tp_kbps"),
                "qos_availability": _first_present(row, ["QoS Availability", "availability"]) or info.get("qos_availability"),
                "ranker_reason": _first_present(row, ["Ranker Reason", "reason"]),
                "comments": _first_present(row, ["Comments", "comments"]),
                "catalog_found": info.get("catalog_found"),
                "metadata_source_file": info.get("metadata_source_file"),
            }
        )
    return sorted(rows, key=lambda row: (mode_sort_key(str(row["mode"])), int(str(row["subtask_id"]).isdigit() and row["subtask_id"] or 999), float(row["mode_rank"] or 9999)))


def _normalize_retrieval_rows(
    retrieval_rows: list[dict[str, Any]],
    row_lookup: dict[tuple[str, str], dict[str, Any]],
    selected_by_mode_subtask: dict[tuple[str, str], set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in retrieval_rows:
        subtask_id = _subtask_id(_first_present(row, ["Sub Task", "Subtask_ID", "subtask_id", "Subtask ID"]))
        api_id = str(_first_present(row, ["Selected_API", "API_ID", "api_id", "API ID"]) or "").strip()
        if not subtask_id or not api_id:
            continue
        info = row_lookup.get((subtask_id, api_id), {})
        selected_modes = [
            mode_label(mode)
            for mode in MODE_ORDER
            if api_id in selected_by_mode_subtask.get((mode, subtask_id), set())
        ]
        rows.append(
            {
                "subtask_id": subtask_id,
                "retrieved_rank": _as_intish(_first_present(row, ["Retrieved Rank", "retrieved_rank"])),
                "api_id": api_id,
                "display_name": info.get("display_name") or short_api_name(api_id),
                "category": info.get("category"),
                "functional_match": _functional_match(_first_present(row, ["Functional Match (0/1)", "Functional_Match"])),
                "qos_rt_s": info.get("qos_rt_s"),
                "qos_tp_kbps": info.get("qos_tp_kbps"),
                "qos_availability": info.get("qos_availability"),
                "selected_by_modes": ", ".join(selected_modes) if selected_modes else "",
                "catalog_found": info.get("catalog_found"),
                "metadata_source_file": info.get("metadata_source_file"),
                "comments": _first_present(row, ["Comments", "comments"]) or info.get("comments"),
            }
        )
    return sorted(rows, key=lambda row: (int(str(row["subtask_id"]).isdigit() and row["subtask_id"] or 999), float(row["retrieved_rank"] or 9999)))


@_cache_data
def load_live_demo_query(run_dir_text: str, query_id: str) -> dict[str, Any]:
    scan = scan_run_folder(run_dir_text)
    query_id = str(query_id or "").lower()
    warnings: list[str] = list(scan.get("warnings") or [])
    query_row = next((row for row in scan.get("queries", []) if str(row.get("query_id")) == query_id), None)
    if query_row is None:
        return {
            "run_dir": scan.get("run_dir") or str(resolve_path(run_dir_text)),
            "query_id": query_id,
            "query_dir": "",
            "warnings": warnings + [f"Query {query_id} was not found in the selected run folder."],
            "available": False,
        }

    query_dir = Path(str(query_row["query_dir"]))
    meta, meta_error = _read_json(query_dir / "meta.json")
    if meta_error:
        warnings.append(meta_error)
    meta = meta if isinstance(meta, dict) else {}

    run_config, run_config_error = _read_json(query_dir / "run_config.json")
    if run_config_error and (query_dir / "run_config.json").exists():
        warnings.append(run_config_error)
    run_config = run_config if isinstance(run_config, dict) else {}

    eval_result, eval_error = _read_json(query_dir / "evaluation_result.json")
    if eval_error and (query_dir / "evaluation_result.json").exists():
        warnings.append(eval_error)
    eval_result = eval_result if isinstance(eval_result, dict) else {}

    catalog_bundle = catalog.load_api_catalog()
    for catalog_warning in catalog_bundle.get("warnings") or []:
        warnings.append(catalog_warning)
    catalog_by_id = catalog_bundle.get("by_id") if isinstance(catalog_bundle.get("by_id"), dict) else {}

    decomposer, decomposer_error, _ = _load_artifact(query_dir, "0_decomposer.json")
    if decomposer_error:
        warnings.append(decomposer_error)
    subtasks = _subtasks_from_payload(decomposer)

    composition_rows, composition_source, composition_warning = _load_composition_metric_rows(
        query_dir,
        query_id,
        eval_result=eval_result,
    )
    if composition_warning:
        warnings.append(composition_warning)

    candidate_payload, candidate_error, candidate_path = _load_artifact(
        query_dir,
        f"evaluation/query_{query_id}_candidate_api_rankings_rows.json",
        eval_result=eval_result,
        eval_result_key="candidate_api_rankings_rows_json",
    )
    candidate_rows = _rows_from_payload(candidate_payload)
    if candidate_error:
        warnings.append(candidate_error)

    retrieval_payload, retrieval_error, retrieval_path = _load_artifact(
        query_dir,
        f"evaluation/query_{query_id}_retrieval_functional_match_rows.json",
        eval_result=eval_result,
        eval_result_key="retrieval_functional_match_rows_json",
    )
    retrieval_rows = _rows_from_payload(retrieval_payload)
    if retrieval_error:
        warnings.append(retrieval_error)

    retriever_lookup = _retriever_file_lookup(query_dir)
    row_lookup = _api_lookup_from_rows(
        candidate_rows,
        retrieval_rows,
        retriever_lookup=retriever_lookup,
        catalog_by_id=catalog_by_id,
    )
    subtasks_by_id = {str(row.get("subtask_id")): str(row.get("description") or "") for row in subtasks}

    modes = sorted(
        set(query_row.get("available_modes") or []) | {str(row.get("Mode")) for row in composition_rows if row.get("Mode")},
        key=mode_sort_key,
    )
    selected_paths: dict[str, list[dict[str, Any]]] = {}
    raw_planners: dict[str, Any] = {}
    planner_sources: dict[str, str] = {}
    selected_by_mode_subtask: dict[tuple[str, str], set[str]] = {}
    for mode in modes:
        selected_lookup = _selected_file_lookup(query_dir, mode, catalog_by_id=catalog_by_id)
        planner_path = query_dir / mode / "4_planner.json"
        planner_payload, planner_error = _read_json(planner_path)
        if planner_error:
            warnings.append(planner_error)
            planner_payload = {}
        else:
            raw_planners[mode] = planner_payload
            planner_sources[mode] = str(planner_path)
        steps = _extract_planner_steps(planner_payload)
        if not steps and selected_lookup:
            steps = _fallback_steps_from_selected_files(selected_lookup)
            warnings.append(f"{mode}: planner steps unavailable; selected path was inferred from 3_selected_s*.json artifacts.")
        enriched_steps = [
            _enrich_selected_step(
                step,
                subtasks_by_id=subtasks_by_id,
                row_lookup=row_lookup,
                selected_lookup=selected_lookup,
            )
            for step in steps
        ]
        selected_paths[mode] = sorted(enriched_steps, key=lambda row: float(row.get("step") or 999))
        for step in enriched_steps:
            selected_by_mode_subtask.setdefault((mode, str(step.get("subtask_id") or "")), set()).add(str(step.get("api_id") or ""))

    ranking_rows = _normalize_ranking_rows(candidate_rows, selected_by_mode_subtask, row_lookup)
    retrieval_snapshot_rows = _normalize_retrieval_rows(retrieval_rows, row_lookup, selected_by_mode_subtask)

    return {
        "available": True,
        "run_dir": str(scan["run_dir"]),
        "query_id": query_id,
        "query_dir": str(query_dir),
        "folder_name": query_row.get("folder_name") or query_dir.name,
        "timestamp": query_row.get("timestamp") or timestamp_from_name(query_dir.name) or "",
        "query_title": str(meta.get("query_title") or query_row.get("title") or query_id),
        "user_goal": str(meta.get("user_goal") or query_row.get("user_goal") or ""),
        "query_category": _query_category_from_payloads(meta, run_config, eval_result, query_row),
        "query_domain": str(query_row.get("domain") or ""),
        "num_subtasks": meta.get("num_subtasks") or len(subtasks),
        "available_modes": modes,
        "subtasks": subtasks,
        "composition_rows": composition_rows,
        "ranking_rows": ranking_rows,
        "retrieval_rows": retrieval_snapshot_rows,
        "selected_paths": selected_paths,
        "warnings": warnings,
        "artifact_sources": {
            "composition_rows": composition_source,
            "candidate_api_rankings_rows": candidate_path,
            "retrieval_functional_match_rows": retrieval_path,
            "retriever_files": {key[0]: value.get("metadata_source_file") for key, value in retriever_lookup.items()},
            "api_catalog": catalog_bundle.get("sources") or [],
            "planner_files": planner_sources,
        },
        "raw_artifacts": {
            "meta.json": meta,
            "0_decomposer.json": decomposer,
            "evaluation_result.json": eval_result,
            "composition_qos_eval_rows": composition_rows,
            "candidate_api_rankings_rows": candidate_rows,
            "retrieval_functional_match_rows": retrieval_rows,
            "planner files by mode": raw_planners,
        },
    }
