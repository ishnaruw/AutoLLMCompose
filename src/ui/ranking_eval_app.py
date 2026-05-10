from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.ranking_metrics import (  # noqa: E402
    DEFAULT_RBO_P,
    METRIC_NAMES,
    MODE_ORDER,
    aggregate_matrices_with_counts,
    cases_to_frame,
    compute_case_matrices,
    evaluate_parent_runs,
    matrices_to_pairwise_table,
    overlap_by_depth,
    top_lists_to_wide_frame,
)
from src.llm.backends import fireworks_model_options  # noqa: E402

DEFAULT_RUN_EXPLORER_PARENT = PROJECT_ROOT / "results/logs"
DEFAULT_PARENT = DEFAULT_RUN_EXPLORER_PARENT
DEFAULT_EXPERIMENT_RUN_TAG = "STREAMLIT_RUNS"
QUERIES_PATH = PROJECT_ROOT / "data/queries/all_user_query.jsonl"

PROVIDER_MODELS = {
    "mistral": ["mistral-small-latest", "mistral-large-latest"],
    "fireworks": fireworks_model_options(),
    "groq": ["multi", "llama-3.3-70b-versatile"],
    "lmstudio": ["meta-llama-3.1-8b-instruct"],
    "lmstudio_qwen": ["qwen2.5-3b-instruct.gguf"],
    "azure": ["gpt-4o-dspy"],
    "azure_foundry": ["DeepSeek-R1-0528"],
    "gemini": ["gemini-2.5-flash"],
    "together": ["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"],
}

METRIC_LABELS = {
    "spearman": "Spearman (All Candidates)",
    "average_overlap": "Top-K Average Overlap",
    "rbo": "Top-K RBO",
    "jaccard": "Top-K Jaccard",
}

METRIC_HELP = {
    "spearman": "Standard Spearman correlation over the complete shared ranked candidate list for each mode.",
    "average_overlap": "Top-K mean overlap across depths 1..K.",
    "rbo": "Top-K RBO with p=0.9 by default; emphasizes highly ranked overlap and is less sensitive to exact position changes when lists contain similar APIs.",
    "jaccard": "Top-K set overlap only; it ignores within-set order.",
}

K_HELP = (
    "Top-K metrics use a shared K per query/subtask. K is derived from the qos_hybrid functional-match "
    "count because qos_hybrid is the functional-refinement mode used to define the meaningful valid "
    "selection depth. This keeps AO, RBO, and Jaccard comparisons at the same cutoff across all modes."
)

STAGE_LABELS = [
    ("Decomposition", ("decomposer",)),
    ("Retrieval", ("retrieval", "retrieval_functional_match_evaluation")),
    ("Ranking", ("ranking", "ranker_no_qos", "ranker_qos_pure_llm", "qos_scorer", "qos_topsis", "qos_hybrid")),
    ("Selection", ("selection", "selector")),
    ("Planning", ("planner",)),
    ("Reports", ("evaluation_outputs",)),
]

LOG_FILENAMES = [
    ("Run Log", "run.log"),
    ("Errors Log", "errors.log"),
    ("Invalid Cases Log", "invalid_cases.log"),
    ("Warnings Log", "warnings.log"),
]


@dataclass(frozen=True)
class QueryRun:
    run_dir: Path
    query_id: str
    run_name: str
    model_label: str
    provider: str | None
    timestamp: str | None


def _read_json_file(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@st.cache_data(show_spinner=False)
def _load_query_options(path_text: str) -> list[dict]:
    path = Path(path_text)
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                query_id = str(payload.get("id") or f"q{idx:02d}")
                rows.append(
                    {
                        "id": query_id,
                        "title": str(payload.get("title") or ""),
                        "goal": str(payload.get("goal") or ""),
                    }
                )
    except Exception:
        return []
    return rows


def _query_option_label(query: dict) -> str:
    title = query.get("title") or ""
    return f"{query['id']} | {title}" if title else str(query["id"])


def _is_process_running(process: subprocess.Popen | None, pid: int | None) -> bool:
    if process is not None:
        return process.poll() is None
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _launch_experiment(query_ids: list[str], provider: str, model: str, run_tag: str) -> dict:
    launch_dir = PROJECT_ROOT / "results/logs/streamlit_launches"
    launch_dir.mkdir(parents=True, exist_ok=True)
    run_parent = PROJECT_ROOT / "results/logs" / run_tag
    run_parent.mkdir(parents=True, exist_ok=True)
    launch_log = launch_dir / f"launch_{time.strftime('%Y%m%dT%H%M%S')}.log"
    cmd = [
        sys.executable,
        "-m",
        "src.driver.run_autogen_pipeline",
        "--query-ids",
        ",".join(query_ids),
        "--provider",
        provider,
        "--run-tag",
        run_tag,
    ]
    if model.strip():
        cmd.extend(["--model", model.strip()])
    handle = launch_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    handle.close()
    return {
        "process": process,
        "pid": process.pid,
        "cmd": cmd,
        "query_ids": query_ids,
        "provider": provider,
        "model": model,
        "run_tag": run_tag,
        "run_parent": str(run_parent),
        "launch_log": str(launch_log),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _format_command(cmd: list[str]) -> str:
    return " ".join(f"'{part}'" if " " in part else part for part in cmd)


def _path_for_input(text: str) -> Path:
    raw = Path((text or "").strip()).expanduser()
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def _safe_relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())) or "."
    except Exception:
        return str(path)


def _choose_directory_with_native_dialog(initial_dir: Path, prompt: str) -> Path | None:
    initial_dir = initial_dir if initial_dir.exists() and initial_dir.is_dir() else DEFAULT_RUN_EXPLORER_PARENT
    script = "\n".join(
        [
            f"set defaultFolder to POSIX file {json.dumps(str(initial_dir))} as alias",
            f"set selectedFolder to choose folder with prompt {json.dumps(prompt)} default location defaultFolder",
            "POSIX path of selectedFolder",
        ]
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except Exception as exc:
        st.error(f"Could not open macOS folder picker: {exc}")
        return None
    if result.returncode != 0:
        if result.stderr and "User canceled" not in result.stderr:
            st.warning(f"Folder picker did not return a directory: {result.stderr.strip()}")
        return None
    selected = Path(result.stdout.strip()).expanduser()
    return selected if selected.exists() and selected.is_dir() else None


def _render_directory_selector(label: str, default_path: Path, key: str, root: Path | None = None) -> str:
    root = root or DEFAULT_RUN_EXPLORER_PARENT
    state_key = f"{key}_path"
    if state_key not in st.session_state:
        st.session_state[state_key] = str(default_path)

    current = _path_for_input(st.session_state[state_key])
    if not current.exists() or not current.is_dir():
        current = default_path
        st.session_state[state_key] = str(current)

    st.caption(f"{label}: `{current}`")
    if st.button("Choose folder...", key=f"{key}_choose"):
        selected = _choose_directory_with_native_dialog(root.resolve(), f"Select {label.lower()}")
        if selected is not None:
            st.session_state[state_key] = str(selected)
            _discover_query_runs.clear()
            _load_excel_sheet.clear()
        st.rerun()
    return st.session_state[state_key]


def _query_id_from_name(name: str) -> str | None:
    match = re.match(r"^(q\d{1,3})(?:[_-]|$)", name.strip(), flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _timestamp_from_name(name: str) -> str | None:
    match = re.search(r"(20\d{6}T\d{6})", name)
    return match.group(1) if match else None


def _looks_like_run_dir(path: Path) -> bool:
    if not path.is_dir() or path.name.startswith("."):
        return False
    if _query_id_from_name(path.name) is None:
        return False
    markers = [
        "meta.json",
        "run_config.json",
        "run.log",
        "errors.log",
        "warnings.log",
        "invalid_cases.log",
        "0_decomposer.json",
        "evaluation_result.json",
    ]
    return any((path / marker).exists() for marker in markers) or any(path.glob("*.xlsx")) or any((path / "evaluation").glob("*.xlsx"))


def _model_label_for_run(run_dir: Path, parent_dir: Path, meta: dict) -> str:
    model_tag = str(meta.get("model_tag") or "").strip()
    active_model = str(meta.get("active_model") or "").strip()
    provider = str(meta.get("provider") or "").strip()
    if model_tag:
        return model_tag.split(":", 1)[1] if ":" in model_tag else model_tag
    if active_model:
        return active_model
    if provider:
        return provider
    try:
        relative_parts = run_dir.relative_to(parent_dir).parts
    except ValueError:
        relative_parts = run_dir.parts
    if len(relative_parts) >= 2:
        return relative_parts[-2]
    return run_dir.parent.name if run_dir.parent != run_dir else "unknown"


@st.cache_data(show_spinner=False)
def _discover_query_runs(parent_dir_text: str) -> tuple[list[dict], list[str]]:
    parent_dir = _path_for_input(parent_dir_text)
    warnings: list[str] = []
    if not parent_dir.exists():
        return [], [f"Parent runs directory not found: {parent_dir}"]
    if not parent_dir.is_dir():
        return [], [f"Parent runs path is not a directory: {parent_dir}"]

    discovered: list[QueryRun] = []
    candidates = [parent_dir] + [path for path in parent_dir.rglob("*") if path.is_dir() and not path.name.startswith(".")]
    for path in candidates:
        if not _looks_like_run_dir(path):
            continue
        meta = _read_json_file(path / "meta.json")
        query_id = str(meta.get("query_id") or _query_id_from_name(path.name) or "").lower()
        if not query_id:
            continue
        discovered.append(
            QueryRun(
                run_dir=path,
                query_id=query_id,
                run_name=path.name,
                model_label=_model_label_for_run(path, parent_dir, meta),
                provider=str(meta.get("provider") or "").strip() or None,
                timestamp=_timestamp_from_name(path.name),
            )
        )

    deduped = {str(run.run_dir.resolve()): run for run in discovered}
    runs = sorted(
        deduped.values(),
        key=lambda run: (run.query_id, run.model_label.lower(), run.timestamp or "", run.run_name),
    )
    rows = [
        {
            "run_dir": str(run.run_dir),
            "query_id": run.query_id,
            "run_name": run.run_name,
            "model_label": run.model_label,
            "provider": run.provider,
            "timestamp": run.timestamp,
        }
        for run in runs
    ]
    return rows, warnings


def _selected_run_from_row(row: dict) -> QueryRun:
    return QueryRun(
        run_dir=Path(row["run_dir"]),
        query_id=str(row["query_id"]),
        run_name=str(row["run_name"]),
        model_label=str(row["model_label"]),
        provider=row.get("provider"),
        timestamp=row.get("timestamp"),
    )


def _run_label(row: dict) -> str:
    timestamp = row.get("timestamp") or "no timestamp"
    return f"{row['run_name']} | {timestamp}"


def _list_excel_files(run_dir: Path) -> list[Path]:
    files = [path for path in run_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".xlsx", ".xls"}]
    return sorted(files, key=lambda path: (0 if "evaluation" in path.parts else 1, path.name.lower()))


def _find_excel_target(run_dir: Path, token: str) -> tuple[Path | None, str | None, list[str], str | None]:
    token_lower = token.lower()
    sheet_aliases = {
        "candidate_api_rankings": ("candidate_api_rankings", "ranked apis", "rankings", "query"),
        "mode_anomalies": ("mode_anomalies", "mode anomalies", "anomalies"),
    }
    preferred_sheet_tokens = sheet_aliases.get(token_lower, (token_lower,))
    excel_files = _list_excel_files(run_dir)
    preferred_files = sorted(
        excel_files,
        key=lambda path: (0 if token_lower in path.name.lower() else 1, path.name.lower()),
    )
    errors: list[str] = []

    fallback: tuple[Path | None, str | None, list[str], str | None] = (None, None, [], None)
    for path in preferred_files:
        try:
            workbook = pd.ExcelFile(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        sheets = workbook.sheet_names
        matched_sheet = next(
            (sheet for token_part in preferred_sheet_tokens for sheet in sheets if token_part in sheet.lower()),
            None,
        )
        if matched_sheet:
            return path, matched_sheet, sheets, None
        if token_lower in path.name.lower() and sheets:
            return path, sheets[0], sheets, None
        if fallback[0] is None and sheets:
            fallback = (path, sheets[0], sheets, None)

    if errors:
        return None, None, [], "; ".join(errors)
    if token_lower == "candidate_api_rankings":
        return fallback
    return None, None, [], None


@st.cache_data(show_spinner=False)
def _load_excel_sheet(path_text: str, sheet_name: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_excel(path_text, sheet_name=sheet_name)
    except Exception as exc:
        return None, str(exc)
    return df, None


def _matching_column(df: pd.DataFrame, desired: str) -> str | None:
    aliases = {
        "Subtask ID": ("Subtask ID", "Sub Task", "subtask_id", "Subtask"),
        "Functional Match Label": ("Functional Match Label", "Functional Match (0/1)", "functional_match"),
    }
    desired_options = aliases.get(desired, (desired,))
    normalized_columns = [(str(col), str(col).strip().lower()) for col in df.columns]
    for option in desired_options:
        desired_lower = option.lower()
        for col, col_lower in normalized_columns:
            if col_lower == desired_lower:
                return col
    desired_lower = desired.lower()
    for col in df.columns:
        if str(col).strip().lower() == desired_lower:
            return str(col)
    compact_options = [re.sub(r"[^a-z0-9]+", "", option.lower()) for option in desired_options]
    for col in df.columns:
        col_compact = re.sub(r"[^a-z0-9]+", "", str(col).lower())
        if any(compact and compact in col_compact for compact in compact_options):
            return str(col)
    return None


def _render_dataframe_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    filter_specs = [
        "Mode",
        "Subtask ID",
        "Selected for Planner",
        "Functional Match Label",
    ]
    selected_filters: dict[str, list] = {}
    cols = st.columns(4)
    for idx, label in enumerate(filter_specs):
        col_name = _matching_column(df, label)
        if not col_name:
            continue
        values = sorted([value for value in df[col_name].dropna().unique().tolist()], key=lambda value: str(value))
        if not values or len(values) > 200:
            continue
        selected_filters[col_name] = cols[idx].multiselect(label, values, default=values, key=f"{key_prefix}_{idx}_{col_name}")

    filtered = df
    for col_name, values in selected_filters.items():
        filtered = filtered[filtered[col_name].isin(values)]
    return filtered


def _render_excel_viewer(run_dir: Path, token: str, title: str, missing_message: str, key_prefix: str) -> None:
    st.subheader(title)
    path, sheet_name, sheets, error = _find_excel_target(run_dir, token)
    if error:
        st.warning(f"Could not read Excel report: {error}")
        return
    if path is None or sheet_name is None:
        st.info(missing_message)
        return
    st.caption(f"{path.relative_to(run_dir)} | sheet: {sheet_name}")
    if len(sheets) > 1:
        sheet_name = st.selectbox("Sheet", sheets, index=sheets.index(sheet_name), key=f"{key_prefix}_sheet")
    df, load_error = _load_excel_sheet(str(path), sheet_name)
    if load_error:
        st.warning(f"Could not load sheet `{sheet_name}`: {load_error}")
        return
    if df is None or df.empty:
        st.info("The selected sheet is empty.")
        return
    view = _render_dataframe_filters(df, key_prefix) if token == "candidate_api_rankings" else df
    st.dataframe(view, width="stretch", hide_index=True)


def _tail_text(path: Path, line_count: int) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, "Log file not found."
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return None, f"Could not read log file: {exc}"
    clipped = lines[-line_count:] if line_count > 0 else lines
    prefix = f"Showing last {len(clipped)} of {len(lines)} lines.\n\n" if len(lines) > len(clipped) else ""
    return prefix + "\n".join(clipped), None


def _stage_status_from_meta(run_dir: Path, stage_keys: tuple[str, ...], label: str) -> tuple[str, str]:
    meta = _read_json_file(run_dir / "meta.json")
    stages = meta.get("timing", {}).get("stages", {}) if isinstance(meta.get("timing"), dict) else {}
    statuses = []
    details = []
    for key in stage_keys:
        stage = stages.get(key)
        if isinstance(stage, dict) and stage.get("status"):
            statuses.append(str(stage["status"]))
            if stage.get("duration_seconds") is not None:
                details.append(f"{key}: {stage['duration_seconds']}s")

    if label == "Reports":
        return ("available", "Excel report files found") if _list_excel_files(run_dir) else ("missing", "No Excel files found")
    if label == "Selection" and not statuses:
        planner_stage = stages.get("planner")
        if isinstance(planner_stage, dict) and planner_stage.get("status") == "skipped":
            return "skipped", str(planner_stage.get("reason") or "")
        return ("completed", "Selection files found") if list(run_dir.glob("*/3_selected_s*.json")) else ("unknown", "")
    if label == "Planning" and not statuses:
        if list(run_dir.glob("*/4_planner.json")):
            return "completed", "Planner output files found"
        return "unknown", ""
    if not statuses:
        return "unknown", ""
    if "failed" in statuses:
        if label == "Ranking" and "completed" in statuses and (run_dir / "invalid_cases.log").exists():
            return "completed with invalid cases", "; ".join(details)
        return "invalid", "; ".join(details)
    if any(status == "running" for status in statuses):
        return "running", "; ".join(details)
    if label == "Ranking" and ((run_dir / "invalid_cases.log").exists() or (run_dir / "ranking_anomalies.log").exists()):
        return "completed with invalid cases", "; ".join(details)
    if all(status in {"completed", "skipped"} for status in statuses):
        return statuses[0] if len(set(statuses)) == 1 else "completed", "; ".join(details)
    return statuses[0], "; ".join(details)


def _stage_rows_for_run(run_dir: Path) -> list[dict]:
    rows = []
    for label, keys in STAGE_LABELS:
        status, detail = _stage_status_from_meta(run_dir, keys, label)
        rows.append({"Stage": label, "Status": status, "Detail": detail})
    return rows


def _display_status(raw_status: str, *, is_run_status: bool = False) -> str:
    normalized = str(raw_status or "unknown").strip().lower().replace("_", " ")
    if is_run_status:
        if normalized in {"completed", "completed with warnings"}:
            return "Success"
        if normalized in {"failed", "invalid"}:
            return "Failed"
        if normalized == "running":
            return "Running"
        return "Unknown"
    if normalized in {"completed", "completed with invalid cases", "available"}:
        return "Done"
    if normalized == "running":
        return "Running"
    if normalized == "skipped":
        return "Skipped"
    if normalized in {"failed", "invalid"}:
        return "Failed"
    if normalized == "missing":
        return "Missing"
    return "Pending"


def _run_progress_info(run_dir: Path) -> dict:
    rows = _stage_rows_for_run(run_dir)
    status_scores = {
        "completed": 1.0,
        "completed with invalid cases": 1.0,
        "available": 1.0,
        "skipped": 1.0,
        "running": 0.5,
        "invalid": 1.0,
        "failed": 1.0,
    }
    total = len(rows) or 1
    score = sum(status_scores.get(str(row["Status"]), 0.0) for row in rows)
    percent = min(100, int(round((score / total) * 100)))
    active = next((row["Stage"] for row in rows if row["Status"] == "running"), None)
    if active is None:
        active = next((row["Stage"] for row in rows if row["Status"] in {"unknown", "missing"}), rows[-1]["Stage"] if rows else "Unknown")
    meta = _read_json_file(run_dir / "meta.json")
    raw_run_status = str(meta.get("status") or "unknown")
    return {
        "percent": percent,
        "active_stage": active,
        "run_status": _display_status(raw_run_status, is_run_status=True),
        "raw_run_status": raw_run_status,
        "rows": rows,
    }


def _render_process_tracker(run_dir: Path) -> None:
    st.subheader("Process Tracker")
    rows = _stage_rows_for_run(run_dir)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_run_progress_panel(run_dir: Path) -> None:
    info = _run_progress_info(run_dir)
    st.subheader("Live Run Progress")
    cols = st.columns(3)
    cols[0].metric("Run status", info["run_status"])
    cols[1].metric("Current stage", info["active_stage"])
    cols[2].metric("Progress", f"{info['percent']}%")
    st.progress(info["percent"] / 100.0, text=f"{info['active_stage']} - {info['run_status']}")
    stage_cols = st.columns(len(info["rows"]))
    for idx, row in enumerate(info["rows"]):
        stage_cols[idx].markdown(
            f"**{row['Stage']}**  \n"
            f"`{_display_status(str(row['Status']))}`"
        )
    with st.expander("Stage details", expanded=True):
        st.dataframe(pd.DataFrame(info["rows"]), width="stretch", hide_index=True)


def _is_run_live(run_dir: Path) -> bool:
    info = _run_progress_info(run_dir)
    if info["run_status"] == "Running":
        return True
    return any(str(row["Status"]) == "running" for row in info["rows"])


def _clear_run_view_caches() -> None:
    _discover_query_runs.clear()
    _load_excel_sheet.clear()


def _render_run_output_tabs(run_dir: Path, key_prefix: str) -> None:
    report_tab, logs_tab = st.tabs(["Excel Reports", "Logs"])
    with report_tab:
        _render_excel_viewer(
            run_dir,
            "candidate_api_rankings",
            "Candidate API Rankings",
            "No candidate API rankings sheet found for this run yet.",
            f"{key_prefix}_candidate_api_rankings",
        )
        _render_excel_viewer(
            run_dir,
            "mode_anomalies",
            "Mode Anomalies",
            "No mode anomalies sheet found for this run yet.",
            f"{key_prefix}_mode_anomalies",
        )
    with logs_tab:
        _render_logs(run_dir, key_prefix=f"{key_prefix}_logs")


def _render_completed_run_selector(parent_dir: str, key_prefix: str) -> QueryRun | None:
    rows, warnings = _discover_query_runs(parent_dir)
    for warning in warnings:
        st.warning(warning)
    if not rows:
        st.info("No query run folders were discovered under the selected directory.")
        return None

    runs_df = pd.DataFrame(rows)
    with st.sidebar:
        model_options = ["All"] + sorted(runs_df["model_label"].dropna().unique().tolist(), key=str.lower)
        selected_model = st.selectbox("Model/provider filter", model_options, key=f"{key_prefix}_model")
        model_filtered = runs_df if selected_model == "All" else runs_df[runs_df["model_label"] == selected_model]

        query_options = sorted(model_filtered["query_id"].dropna().unique().tolist())
        selected_query = st.selectbox("Query ID", query_options, key=f"{key_prefix}_query")
        query_filtered = model_filtered[model_filtered["query_id"] == selected_query].copy()

        run_options = query_filtered.sort_values(["timestamp", "run_name"]).to_dict(orient="records")
        labels = [_run_label(row) for row in run_options]
        selected_label = st.selectbox("Run", labels, index=max(0, len(labels) - 1), key=f"{key_prefix}_run")

    selected_row = next(row for row in run_options if _run_label(row) == selected_label)
    return _selected_run_from_row(selected_row)


def _render_run_inspection(run: QueryRun, key_prefix: str, *, force_live: bool = False) -> None:
    st.write(f"**Run folder:** `{run.run_dir}`")
    is_live = force_live or _is_run_live(run.run_dir)
    refresh_cols = st.columns([1.1, 1.3, 4.6])
    with refresh_cols[0]:
        if st.button("Refresh", key=f"{key_prefix}_refresh", use_container_width=True):
            _clear_run_view_caches()
            st.rerun()
    with refresh_cols[1]:
        auto_refresh = st.toggle(
            "Auto-refresh",
            value=True,
            key=f"{key_prefix}_auto_refresh",
            help="Refreshes every 2 seconds while this run is still active.",
        )
    with refresh_cols[2]:
        if is_live:
            st.caption("Live monitor: updates every 2 seconds when auto-refresh is on.")
        else:
            st.caption("Snapshot view: use Refresh to check for newer files.")
    _render_run_progress_panel(run.run_dir)
    with st.expander("Live Run Log", expanded=is_live):
        text, error = _tail_text(run.run_dir / "run.log", 80)
        if error:
            st.info(error)
        else:
            st.code(text or "", language="text")
    _render_run_output_tabs(run.run_dir, key_prefix=key_prefix)
    if is_live and auto_refresh:
        time.sleep(2)
        st.rerun()


def _render_logs(run_dir: Path, key_prefix: str = "logs") -> None:
    st.subheader("Logs")
    tail_lines = st.slider("Tail lines", min_value=50, max_value=1000, value=50, step=50, key=f"{key_prefix}_tail")
    for label, filename in LOG_FILENAMES:
        with st.expander(label, expanded=filename == "run.log"):
            text, error = _tail_text(run_dir / filename, tail_lines)
            if error:
                st.info(error)
            else:
                st.code(text or "", language="text")


def render_query_run_explorer() -> None:
    st.title("Run Experiments")
    st.caption("Launch MAOF query-level runs, watch stage progress, then inspect the generated reports and logs.")

    queries = _load_query_options(str(QUERIES_PATH))
    if not queries:
        st.error(f"No queries could be loaded from `{QUERIES_PATH}`.")
        return

    with st.sidebar:
        st.header("Experiment")
        query_labels = [_query_option_label(query) for query in queries]
        selected_labels = st.multiselect("Queries", query_labels, default=query_labels[:1])
        selected_queries = [query for query in queries if _query_option_label(query) in selected_labels]
        selected_query_ids = [str(query["id"]) for query in selected_queries]

        provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()), index=0)
        model_options = PROVIDER_MODELS[provider] + ["Custom..."]
        model_choice = st.selectbox("Model", model_options)
        model = st.text_input("Custom model", value="") if model_choice == "Custom..." else model_choice
        run_tag = st.text_input("Run tag", value=DEFAULT_EXPERIMENT_RUN_TAG)

        start_disabled = not selected_query_ids or not provider or _is_process_running(
            st.session_state.get("experiment_process"),
            st.session_state.get("experiment_pid"),
        )
        if st.button("Start Experiment", disabled=start_disabled):
            try:
                st.session_state["experiment"] = _launch_experiment(selected_query_ids, provider, model, run_tag)
                st.session_state["experiment_process"] = st.session_state["experiment"]["process"]
                st.session_state["experiment_pid"] = st.session_state["experiment"]["pid"]
                _clear_run_view_caches()
                st.rerun()
            except Exception as exc:
                st.error(f"Could not start experiment: {exc}")

        active_process = st.session_state.get("experiment_process")
        if _is_process_running(active_process, st.session_state.get("experiment_pid")):
            if st.button("Stop Running Experiment"):
                active_process.terminate()
                st.warning("Stop signal sent to the running experiment.")

    experiment = st.session_state.get("experiment")
    if not experiment:
        st.info("Choose one or more queries, select a provider/model, then start an experiment.")
        st.code(
            "python -m src.driver.run_autogen_pipeline --query-ids q01 --provider mistral --model mistral-small-latest --run-tag STREAMLIT_RUNS",
            language="bash",
        )
        return

    process = st.session_state.get("experiment_process")
    is_running = _is_process_running(process, experiment.get("pid"))
    if process is not None and process.poll() is not None:
        experiment["returncode"] = process.returncode

    status_cols = st.columns(4)
    status_cols[0].metric("Status", "running" if is_running else "finished")
    status_cols[1].metric("PID", experiment.get("pid", ""))
    status_cols[2].metric("Queries", len(experiment.get("query_ids", [])))
    status_cols[3].metric("Provider", experiment.get("provider", ""))
    st.code(_format_command(experiment.get("cmd", [])), language="bash")

    rows, warnings = _discover_query_runs(experiment["run_parent"])
    for warning in warnings:
        st.warning(warning)
    query_ids = set(experiment.get("query_ids", []))
    rows = [row for row in rows if row["query_id"] in query_ids]
    runs_df = pd.DataFrame(rows)

    if rows:
        progress_rows = []
        for row in rows:
            run = _selected_run_from_row(row)
            meta = _read_json_file(run.run_dir / "meta.json")
            progress = _run_progress_info(run.run_dir)
            stage_statuses = {
                label: _stage_status_from_meta(run.run_dir, keys, label)[0]
                for label, keys in STAGE_LABELS
            }
            progress_rows.append(
                {
                    "Query": run.query_id,
                    "Run": run.run_name,
                    "Model": run.model_label,
                    "Run Status": _display_status(str(meta.get("status") or "unknown"), is_run_status=True),
                    "Current Stage": progress["active_stage"],
                    "Progress": f"{progress['percent']}%",
                    **{stage: _display_status(status) for stage, status in stage_statuses.items()},
                }
            )
        st.subheader("Current Batch Progress")
        st.dataframe(pd.DataFrame(progress_rows), width="stretch", hide_index=True)
    else:
        st.info("Waiting for the first query run folder to appear.")

    launch_log = Path(experiment["launch_log"])
    with st.expander("Process Launcher Log (stdout/stderr)", expanded=not rows):
        text, error = _tail_text(launch_log, 200)
        if error:
            st.info(error)
        else:
            st.code(text or "", language="text")

    if not rows and is_running:
        time.sleep(2)
        st.rerun()

    if rows:
        st.subheader("Inspect Run Output")
        run_options = runs_df.sort_values(["query_id", "timestamp", "run_name"]).to_dict(orient="records")
        default_index = max(0, len(run_options) - 1)
        selected_label = st.selectbox(
            "Generated run",
            [_run_label(row) for row in run_options],
            index=default_index,
        )
        selected_row = next(row for row in run_options if _run_label(row) == selected_label)
        selected_run = _selected_run_from_row(selected_row)
        _render_run_inspection(selected_run, key_prefix="current_run", force_live=is_running)


def render_completed_runs() -> None:
    st.title("Completed Runs")
    st.caption("Browse MAOF query runs, including still-running experiments, with the same live progress and report view used by Run Experiments.")

    with st.sidebar:
        st.header("Completed Run Directory")
        parent_dir = _render_directory_selector(
            "Runs directory",
            default_path=DEFAULT_RUN_EXPLORER_PARENT,
            key="completed_runs_dir",
        )
        if st.button("Reload completed runs"):
            _clear_run_view_caches()
            st.rerun()

    selected_run = _render_completed_run_selector(parent_dir, key_prefix="completed_runs")
    if selected_run is None:
        return
    _render_run_inspection(selected_run, key_prefix="completed_runs_output")


@st.cache_data(show_spinner=False)
def _load_bundle(parent_dir: str, rbo_p: float, selected_modes: tuple[str, ...]):
    return evaluate_parent_runs(parent_dir, p=rbo_p, selected_modes=list(selected_modes))


def _heatmap(matrix: pd.DataFrame, title: str, value_range: tuple[float, float]):
    fig = px.imshow(
        matrix.astype(float),
        x=matrix.columns,
        y=matrix.index,
        text_auto=".3f",
        color_continuous_scale="RdYlGn",
        zmin=value_range[0],
        zmax=value_range[1],
        aspect="auto",
        title=title,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=410)
    fig.update_xaxes(side="top")
    return fig


def _case_label(row: pd.Series) -> str:
    fallback = " fallback K" if bool(row.get("k_fallback_used")) else ""
    run_name = Path(str(row["run_dir"])).name
    return f"{row['query_id']} / subtask {row['subtask_id']} / K={row['k']}{fallback} / {run_name}"


def _filter_pairwise(pairwise: pd.DataFrame, metrics: list[str], modes: list[str]) -> pd.DataFrame:
    if pairwise.empty:
        return pairwise
    return pairwise[
        pairwise["metric"].isin(metrics)
        & pairwise["mode_a"].isin(modes)
        & pairwise["mode_b"].isin(modes)
    ].copy()


def render_ranking_evaluation() -> None:
    st.title("MAOF Ranking Evaluation")

    with st.sidebar:
        st.header("Input")
        parent_dir = _render_directory_selector(
            "Parent runs directory",
            default_path=DEFAULT_PARENT,
            key="ranking_eval_dir",
        )
        rbo_p = st.slider(
            "RBO p",
            min_value=0.5,
            max_value=0.99,
            value=DEFAULT_RBO_P,
            step=0.01,
            help=METRIC_HELP["rbo"],
        )
        selected_modes = st.multiselect(
            "Modes",
            MODE_ORDER,
            default=MODE_ORDER,
            help="Evaluation includes only query/subtask cases where all selected modes have usable outputs for the metric being computed.",
        )
        st.caption(K_HELP)
        if st.button("Reload reports"):
            _load_bundle.clear()

    if len(selected_modes) < 2:
        st.warning("Select at least two modes.")
        return

    bundle = _load_bundle(parent_dir, rbo_p, tuple(selected_modes))
    cases_df = cases_to_frame(bundle.cases)

    if bundle.warnings:
        with st.expander(f"Warnings ({len(bundle.warnings)})", expanded=False):
            for warning in bundle.warnings:
                st.warning(warning)

    if not bundle.cases:
        st.error("No query/subtask cases were available for the selected modes.")
        st.caption("Check that the selected parent directory contains q* run folders with Excel reports.")
        if not bundle.invalid_cases.empty:
            with st.expander(f"Invalid mode/subtask cases ({len(bundle.invalid_cases)})", expanded=True):
                st.dataframe(bundle.invalid_cases, width="stretch", hide_index=True)
        return

    with st.sidebar:
        st.header("Filters")
        query_options = sorted(cases_df["query_id"].unique().tolist())
        selected_queries = st.multiselect("Query ID", query_options, default=query_options)

        visible_cases_df = cases_df[cases_df["query_id"].isin(selected_queries)]
        subtask_options = sorted(visible_cases_df["subtask_id"].unique().tolist(), key=lambda value: str(value))
        selected_subtasks = st.multiselect("Subtask ID", subtask_options, default=subtask_options)

        selected_metrics = st.multiselect(
            "Metric",
            METRIC_NAMES,
            default=METRIC_NAMES,
            format_func=lambda value: METRIC_LABELS.get(value, value),
        )

    if not selected_metrics:
        st.warning("Select at least one metric.")
        return

    filtered_ids = set(
        cases_df[
            cases_df["query_id"].isin(selected_queries)
            & cases_df["subtask_id"].isin(selected_subtasks)
        ]["case_id"]
    )
    filtered_cases = [case for case in bundle.cases if case.case_id in filtered_ids]
    if not filtered_cases:
        st.error("No cases match the selected filters.")
        return

    filtered_matrices, filtered_counts = aggregate_matrices_with_counts(filtered_cases, p=rbo_p)
    filtered_pairwise = matrices_to_pairwise_table(filtered_matrices, filtered_counts)

    fallback_count = sum(case.k_fallback_used for case in filtered_cases)
    included_pairwise = (
        int(filtered_pairwise["included_cases"].sum())
        if "included_cases" in filtered_pairwise.columns and not filtered_pairwise.empty
        else 0
    )
    stat_cols = st.columns(7)
    stat_cols[0].metric("Included cases", len(filtered_cases))
    stat_cols[1].metric("Discovered runs", len(bundle.discovered_run_dirs))
    stat_cols[2].metric("Loaded reports", len(bundle.loaded_report_paths))
    stat_cols[3].metric("Fallback K cases", fallback_count)
    stat_cols[4].metric("Metric comparisons", included_pairwise)
    stat_cols[5].metric("Invalid cases", len(bundle.invalid_cases))
    stat_cols[6].metric("Rows loaded", len(bundle.raw_rows))
    st.caption("Evaluation rule: strict selected-mode evaluation. Final reporting should use all four modes selected.")

    if not bundle.invalid_cases.empty:
        st.subheader("Invalid Evaluation Cases")
        invalid_cols = st.columns(3)
        invalid_cols[0].dataframe(
            bundle.invalid_cases.groupby("mode", dropna=False).size().reset_index(name="count"),
            width="stretch",
            hide_index=True,
        )
        invalid_cols[1].dataframe(
            bundle.invalid_cases.groupby("failure_reason", dropna=False).size().reset_index(name="count"),
            width="stretch",
            hide_index=True,
        )
        invalid_cols[2].dataframe(
            bundle.invalid_cases.groupby(["query_id", "subtask_id"], dropna=False).size().reset_index(name="count"),
            width="stretch",
            hide_index=True,
        )
        with st.expander("Excluded invalid mode/subtask rows", expanded=False):
            st.dataframe(bundle.invalid_cases, width="stretch", hide_index=True)

    st.subheader("Overall Mode Similarity")
    with st.expander("Metric notes", expanded=False):
        for metric in METRIC_NAMES:
            st.markdown(f"**{METRIC_LABELS[metric]}**: {METRIC_HELP[metric]}")
        st.markdown(f"**K**: {K_HELP}")
    for metric in selected_metrics:
        matrix = filtered_matrices[metric].loc[selected_modes, selected_modes]
        value_range = (-1.0, 1.0) if metric == "spearman" else (0.0, 1.0)
        st.plotly_chart(
            _heatmap(matrix, METRIC_LABELS.get(metric, metric), value_range),
            width="stretch",
        )

    st.subheader("Pairwise Scores")
    pairwise_view = _filter_pairwise(filtered_pairwise, selected_metrics, selected_modes)
    pairwise_table = pairwise_view.copy()
    if not pairwise_table.empty:
        pairwise_table["metric"] = pairwise_table["metric"].map(METRIC_LABELS)
    st.dataframe(pairwise_table.round({"score": 4}), width="stretch", hide_index=True)

    if not pairwise_view.empty:
        pairwise_plot = pairwise_view.copy()
        pairwise_plot["metric_label"] = pairwise_plot["metric"].map(METRIC_LABELS)
        pairwise_plot["mode_pair"] = pairwise_plot["mode_a"] + " vs " + pairwise_plot["mode_b"]
        fig = px.bar(
            pairwise_plot,
            x="mode_pair",
            y="score",
            color="metric_label",
            barmode="group",
            range_y=[-1, 1] if "spearman" in selected_metrics else [0, 1],
            labels={"mode_pair": "Mode pair", "score": "Score", "metric_label": "Metric"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=380)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Selected Query/Subtask Details")
    detail_df = cases_to_frame(filtered_cases)
    detail_df["label"] = detail_df.apply(_case_label, axis=1)
    selected_label = st.selectbox("Query/subtask", detail_df["label"].tolist())
    selected_case_id = detail_df.loc[detail_df["label"] == selected_label, "case_id"].iloc[0]
    selected_case = next(case for case in filtered_cases if case.case_id == selected_case_id)

    k_note = (
        "fallback K used because qos_hybrid had 0 functional matches"
        if selected_case.k_fallback_used
        else "K from qos_hybrid functional-match count"
    )
    ranked_count = min((len(selected_case.ranked_lists.get(mode, [])) for mode in selected_case.valid_modes), default=0)
    st.caption(f"K = {selected_case.k} ({k_note}); Spearman uses all {ranked_count} ranked candidates.")
    st.dataframe(top_lists_to_wide_frame(selected_case), width="stretch", hide_index=True)

    st.subheader("Case-Level Metrics")
    case_matrices = compute_case_matrices(selected_case, p=rbo_p)
    selected_case_metric = st.selectbox(
        "Case metric",
        METRIC_NAMES,
        format_func=lambda value: METRIC_LABELS.get(value, value),
    )
    value_range = (-1.0, 1.0) if selected_case_metric == "spearman" else (0.0, 1.0)
    st.plotly_chart(
        _heatmap(
            case_matrices[selected_case_metric].loc[selected_modes, selected_modes],
            f"{METRIC_LABELS.get(selected_case_metric, selected_case_metric)} for selected case",
            value_range,
        ),
        width="stretch",
    )

    st.subheader("Average Overlap by Depth")
    pair_cols = st.columns(2)
    left_mode = pair_cols[0].selectbox("Mode A", selected_modes, index=0)
    default_right = selected_modes.index("qos_hybrid") if "qos_hybrid" in selected_modes else len(selected_modes) - 1
    right_mode = pair_cols[1].selectbox("Mode B", selected_modes, index=default_right)
    if left_mode not in selected_case.top_lists or right_mode not in selected_case.top_lists:
        st.warning("The selected pair is not available for this query/subtask under the current inclusion policy.")
        return
    ao_depth = overlap_by_depth(
        selected_case.top_lists[left_mode],
        selected_case.top_lists[right_mode],
        selected_case.k,
    )
    fig = px.line(
        ao_depth,
        x="depth",
        y="overlap_ratio",
        markers=True,
        range_y=[0, 1],
        labels={"depth": "Depth", "overlap_ratio": "Overlap ratio"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=320)
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(page_title="MAOF Dashboard", layout="wide")
    with st.sidebar:
        page = st.radio(
            "Page",
            ["Ranking Evaluation", "Run Experiments", "Completed Runs"],
            index=0,
        )

    if page == "Ranking Evaluation":
        render_ranking_evaluation()
    elif page == "Run Experiments":
        render_query_run_explorer()
    else:
        render_completed_runs()


if __name__ == "__main__":
    main()
