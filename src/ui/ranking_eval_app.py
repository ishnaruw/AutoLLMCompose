from __future__ import annotations

import io
import json
import importlib
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Callable
import time
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runtime_bootstrap import harden_scientific_runtime

harden_scientific_runtime()

import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore", message="coroutine 'expire_cache' was never awaited", category=RuntimeWarning)

from src.eval import ranking_metrics as _ranking_metrics  # noqa: E402
from src.llm.backends import fireworks_model_options  # noqa: E402
from src.ui import composition_visualization_helpers as viz  # noqa: E402
from src.ui.thesis_results_page import render_thesis_results_figure_generator  # noqa: E402

if not hasattr(_ranking_metrics, "build_ranking_eval_report_files"):
    _ranking_metrics = importlib.reload(_ranking_metrics)

DEFAULT_RBO_P = _ranking_metrics.DEFAULT_RBO_P
METRIC_NAMES = _ranking_metrics.METRIC_NAMES
MODE_ORDER = _ranking_metrics.MODE_ORDER
aggregate_matrices_with_counts = _ranking_metrics.aggregate_matrices_with_counts
build_ranking_eval_report_files = _ranking_metrics.build_ranking_eval_report_files
cases_to_frame = _ranking_metrics.cases_to_frame
compute_case_matrices = _ranking_metrics.compute_case_matrices
evaluate_parent_runs = _ranking_metrics.evaluate_parent_runs
matrices_to_pairwise_table = _ranking_metrics.matrices_to_pairwise_table
overlap_by_depth = _ranking_metrics.overlap_by_depth
top_lists_to_wide_frame = _ranking_metrics.top_lists_to_wide_frame

if not hasattr(viz, "THROUGHPUT_LABEL") or not hasattr(viz, "format_throughput"):
    viz = importlib.reload(viz)
if not hasattr(viz, "THROUGHPUT_LABEL"):
    viz.THROUGHPUT_LABEL = "Throughput (kbps)"
if not hasattr(viz, "BOTTLENECK_THROUGHPUT_LABEL"):
    viz.BOTTLENECK_THROUGHPUT_LABEL = "Bottleneck Throughput (kbps)"
if not hasattr(viz, "format_throughput"):
    viz.format_throughput = lambda value: (
        viz.NA if viz.format_value(value) == viz.NA else f"{viz.format_value(value)} kbps"
    )

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
    ("Retrieval", ("retrieval",)),
    ("Functional Refinement", ("functional_refinement", "retrieval_functional_match_evaluation")),
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


@st.cache_data(show_spinner=False)
def _load_composition_reports(parent_dir_text: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    parent_dir = _path_for_input(parent_dir_text)
    warnings: list[str] = []
    eval_frames: list[pd.DataFrame] = []
    workflow_frames: list[pd.DataFrame] = []
    if not parent_dir.exists() or not parent_dir.is_dir():
        return pd.DataFrame(), pd.DataFrame(), [f"Composition reports directory not found: {parent_dir}"]

    rows_paths = sorted(parent_dir.rglob("evaluation/query_*_composition_qos_eval_rows.json"))
    for rows_path in rows_paths:
        run_dir = rows_path.parent.parent
        try:
            rows_payload = json.loads(rows_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"{rows_path}: {exc}")
            continue
        if not isinstance(rows_payload, list):
            warnings.append(f"{rows_path}: expected a list of rows")
            continue
        rows_df = pd.DataFrame(rows_payload)
        if rows_df.empty:
            continue
        rows_df["run_dir"] = str(run_dir)
        rows_df["run_name"] = run_dir.name
        rows_df["report_path"] = str(rows_path)
        eval_frames.append(rows_df)

        query_id = str(rows_df["Query_ID"].dropna().iloc[0]) if "Query_ID" in rows_df and not rows_df["Query_ID"].dropna().empty else "*"
        xlsx_candidates = sorted(rows_path.parent.glob(f"query_{query_id}_composition_qos_eval.xlsx"))
        if not xlsx_candidates:
            xlsx_candidates = sorted(rows_path.parent.glob("query_*_composition_qos_eval.xlsx"))
        if not xlsx_candidates:
            continue
        workflow_df, load_error = _load_excel_sheet(str(xlsx_candidates[0]), "Planned_Workflow")
        if load_error or workflow_df is None:
            warnings.append(f"{xlsx_candidates[0]} Planned_Workflow: {load_error}")
            continue
        if not workflow_df.empty:
            workflow_df["run_dir"] = str(run_dir)
            workflow_df["run_name"] = run_dir.name
            workflow_df["report_path"] = str(xlsx_candidates[0])
            workflow_frames.append(workflow_df)

    eval_frames = [frame.dropna(axis=1, how="all") for frame in eval_frames if not frame.empty and not frame.dropna(axis=1, how="all").empty]
    workflow_frames = [frame.dropna(axis=1, how="all") for frame in workflow_frames if not frame.empty and not frame.dropna(axis=1, how="all").empty]
    eval_df = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    workflow_df = pd.concat(workflow_frames, ignore_index=True) if workflow_frames else pd.DataFrame()
    return eval_df, workflow_df, warnings


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
    render_sticky_table(
        view,
        sticky_columns=["Query_ID", "query_id", "Mode", "mode", "Subtask_ID", "subtask_id", "API_Name", "API_ID", "api_id"],
        height=520,
        key=f"{key_prefix}_{sheet_name}",
    )


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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
        st.dataframe(pd.DataFrame(info["rows"]), use_container_width=True, hide_index=True)


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
    st.caption("Launch AutoLLMCompose query-level runs, watch stage progress, then inspect the generated reports and logs.")

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
        st.dataframe(pd.DataFrame(progress_rows), use_container_width=True, hide_index=True)
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
    st.caption("Browse AutoLLMCompose query runs, including still-running experiments, with the same live progress and report view used by Run Experiments.")

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


def _build_deterministic_zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for filename in sorted(files):
            info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, files[filename].encode("utf-8"))
    return buffer.getvalue()


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


def _render_composition_evaluation(parent_dir: str, selected_modes: list[str], selected_queries: list[str]) -> None:
    st.subheader("Composition Workflow Evaluation")
    eval_df, workflow_df, warnings = _load_composition_reports(parent_dir)
    for warning in warnings:
        st.warning(warning)
    if eval_df.empty:
        st.info("No composition QoS evaluation reports found under the selected parent directory.")
        return

    query_col = "Query_ID"
    mode_col = "Mode"
    selected_queries = [str(query) for query in selected_queries]
    selected_composition_modes = [mode for mode in selected_modes if mode in eval_df[mode_col].dropna().astype(str).unique().tolist()]
    st.caption(
        "Using shared sidebar filters: "
        f"queries={', '.join(selected_queries) if selected_queries else 'all'}; "
        f"modes={', '.join(selected_composition_modes) if selected_composition_modes else 'all'}"
    )

    filtered = eval_df.copy()
    if selected_queries:
        filtered = filtered[filtered[query_col].astype(str).isin(selected_queries)]
    if selected_composition_modes:
        filtered = filtered[filtered[mode_col].astype(str).isin(selected_composition_modes)]
    if filtered.empty:
        st.info("No composition rows match the selected filters.")
        return

    availability_col = "Average_Workflow_Availability" if "Average_Workflow_Availability" in filtered else "Workflow_Availability"
    numeric_cols = [
        "Composition_Validity",
        "Composition_Completeness",
        "Functional_Coverage",
        "Total_Response_Time_s",
        "Bottleneck_Throughput_kbps",
        availability_col,
        "Normalized_Response_Time_Score",
        "Normalized_Throughput_Score",
        "Normalized_Availability_Score",
        "Normalized_QoS_Score",
        "QoS_Adjusted_Composition_Score",
    ]
    for col in numeric_cols:
        if col in filtered:
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce")

    render_sticky_table(
        _composition_display_frame(filtered[
            [
                col
                for col in [
                    "Query_ID",
                    "run_name",
                    "Mode",
                    "Composition_Validity",
                    "Composition_Completeness",
                    "Functional_Coverage",
                    "Total_Response_Time_s",
                    "Bottleneck_Throughput_kbps",
                    availability_col,
                    "Normalized_QoS_Score",
                    "QoS_Adjusted_Composition_Score",
                ]
                if col in filtered.columns
            ]
        ].round(4)),
        sticky_columns=["Query_ID", "Mode"],
        height=360,
        key="composition_eval_overview",
    )
    if availability_col in filtered:
        st.caption(viz.WORKFLOW_AVAILABILITY_HELP)

    qos_hybrid_best = viz.build_qos_hybrid_best_table(filtered)
    if not qos_hybrid_best.empty:
        st.markdown("**QoS Hybrid Best Mode Summary**")
        render_sticky_table(
            qos_hybrid_best,
            sticky_columns=["query_id"],
            height=260,
            key="qos_hybrid_best_summary",
            cell_style=_qos_summary_cell_style,
        )

    chart_tab, raw_tab, workflow_tab = st.tabs(["Score & QoS Charts", "Ranks & Validity", "Planned Workflow"])
    with chart_tab:
        score_overview = filtered.copy()
        score_overview["Query_ID"] = score_overview["Query_ID"].astype(str)
        score_overview["QoS_Adjusted_Composition_Score"] = pd.to_numeric(
            score_overview["QoS_Adjusted_Composition_Score"],
            errors="coerce",
        )
        overall_score = (
            score_overview.dropna(subset=["QoS_Adjusted_Composition_Score"])
            .groupby("Mode", as_index=False)
            .agg(
                Mean_QoS_Adjusted_Composition_Score=("QoS_Adjusted_Composition_Score", "mean"),
                Std_QoS_Adjusted_Composition_Score=("QoS_Adjusted_Composition_Score", "std"),
                Query_Count=("Query_ID", "nunique"),
                Valid_Composition_Rate=("Composition_Validity", "mean"),
                Mean_Functional_Coverage=("Functional_Coverage", "mean"),
                Mean_Normalized_QoS_Score=("Normalized_QoS_Score", "mean"),
            )
        )
        if not overall_score.empty:
            overall_score["Mode"] = pd.Categorical(overall_score["Mode"], categories=MODE_ORDER, ordered=True)
            overall_score = overall_score.sort_values("Mode")
            overall_score["Std_QoS_Adjusted_Composition_Score"] = overall_score["Std_QoS_Adjusted_Composition_Score"].fillna(0.0)
            overall_fig = px.bar(
                overall_score,
                x="Mode",
                y="Mean_QoS_Adjusted_Composition_Score",
                color="Mode",
                error_y="Std_QoS_Adjusted_Composition_Score",
                text_auto=".3f",
                title="Overall Mean QoS-Adjusted Composition Score by Mode",
                range_y=[0, 1],
                category_orders={"Mode": MODE_ORDER},
                hover_data=[
                    "Query_Count",
                    "Valid_Composition_Rate",
                    "Mean_Functional_Coverage",
                    "Mean_Normalized_QoS_Score",
                ],
            )
            overall_fig.update_traces(textposition="outside", cliponaxis=False)
            overall_fig.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=60, b=60),
                xaxis_title="Mode",
                yaxis_title="Mean QoS-Adjusted Composition Score",
                showlegend=False,
            )
            st.plotly_chart(overall_fig, use_container_width=True)
            render_sticky_table(
                overall_score.round(
                    {
                        "Mean_QoS_Adjusted_Composition_Score": 4,
                        "Std_QoS_Adjusted_Composition_Score": 4,
                        "Valid_Composition_Rate": 4,
                        "Mean_Functional_Coverage": 4,
                        "Mean_Normalized_QoS_Score": 4,
                    }
                ),
                sticky_columns=["Mode"],
                height=320,
                key="composition_overall_score",
            )

        query_order = sorted(
            score_overview["Query_ID"].dropna().unique().tolist(),
            key=lambda value: int(value[1:]) if value.lower().startswith("q") and value[1:].isdigit() else value,
        )
        all_query_fig = px.bar(
            score_overview,
            x="Query_ID",
            y="QoS_Adjusted_Composition_Score",
            color="Mode",
            barmode="group",
            text_auto=".3f",
            title="QoS-Adjusted Composition Score Across Queries",
            range_y=[0, 1],
            category_orders={"Query_ID": query_order, "Mode": MODE_ORDER},
            hover_data=["run_name", "Composition_Validity", "Functional_Coverage", "Normalized_QoS_Score"],
        )
        all_query_fig.update_traces(textposition="outside", cliponaxis=False)
        all_query_fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=60, b=80),
            xaxis_title="Query",
            yaxis_title="QoS-Adjusted Composition Score",
            legend_title_text="Mode",
            bargap=0.2,
            bargroupgap=0.05,
        )
        all_query_fig.update_xaxes(tickangle=0)
        st.plotly_chart(all_query_fig, use_container_width=True)

        normalized_cols = [
            "Normalized_Response_Time_Score",
            "Normalized_Throughput_Score",
            "Normalized_Availability_Score",
        ]
        norm_long = filtered.melt(
            id_vars=["Query_ID", "run_name", "Mode"],
            value_vars=[col for col in normalized_cols if col in filtered],
            var_name="Metric",
            value_name="Score",
        )
        if not norm_long.empty:
            norm_fig = px.bar(
                norm_long,
                x="Mode",
                y="Score",
                color="Metric",
                barmode="group",
                facet_col="Query_ID" if norm_long["Query_ID"].nunique() > 1 else None,
                title="Normalized QoS Dimension Scores",
                range_y=[0, 1],
            )
            norm_fig.update_layout(height=400, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(norm_fig, use_container_width=True)

        scatter_fig = px.scatter(
            filtered,
            x="Functional_Coverage",
            y="Normalized_QoS_Score",
            color="Mode",
            symbol="Composition_Validity",
            hover_data=["Query_ID", "run_name", "QoS_Adjusted_Composition_Score"],
            text="Mode",
            title="Functional Coverage vs Normalized QoS",
            range_x=[-0.05, 1.05],
            range_y=[-0.05, 1.05],
        )
        scatter_fig.update_traces(textposition="top center")
        scatter_fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(scatter_fig, use_container_width=True)

    with raw_tab:
        raw_cols = [
            "Total_Response_Time_s",
            "Bottleneck_Throughput_kbps",
            availability_col,
        ]
        raw_long = filtered.melt(
            id_vars=["Query_ID", "run_name", "Mode"],
            value_vars=[col for col in raw_cols if col in filtered],
            var_name="Metric",
            value_name="Value",
        ).dropna(subset=["Value"])
        if not raw_long.empty:
            raw_long["Metric"] = raw_long["Metric"].map(_composition_metric_label)
            raw_fig = px.bar(
                raw_long,
                x="Mode",
                y="Value",
                color="Mode",
                facet_col="Metric",
                facet_col_wrap=3,
                text_auto=".3f",
                title="Raw Composition-Level QoS Metrics",
                labels={"Value": "Value", "Metric": "Metric"},
            )
            raw_fig.update_yaxes(matches=None)
            raw_fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10), showlegend=False)
            st.plotly_chart(raw_fig, use_container_width=True)

        validity_long = filtered.melt(
            id_vars=["Query_ID", "run_name", "Mode"],
            value_vars=[col for col in ["Composition_Completeness", "Composition_Validity"] if col in filtered],
            var_name="Metric",
            value_name="Score",
        )
        validity_fig = px.bar(
            validity_long,
            x="Mode",
            y="Score",
            color="Metric",
            barmode="group",
            title="Composition Completeness and Validity",
            range_y=[0, 1],
        )
        validity_fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(validity_fig, use_container_width=True)

        rank_cols = {
            "QoS_Adjusted_Composition_Score": False,
            "Total_Response_Time_s": True,
            "Bottleneck_Throughput_kbps": False,
            availability_col: False,
        }
        rank_rows: list[dict] = []
        for (query_id, run_name), group in filtered.groupby(["Query_ID", "run_name"], dropna=False):
            for metric, ascending in rank_cols.items():
                if metric not in group:
                    continue
                ranked = group[["Mode", metric]].dropna().copy()
                if ranked.empty:
                    continue
                ranked["Rank"] = ranked[metric].rank(method="min", ascending=ascending)
                for _, row in ranked.iterrows():
                    rank_rows.append(
                        {
                            "Query_ID": query_id,
                            "run_name": run_name,
                            "Mode": row["Mode"],
                            "Metric": _composition_metric_label(metric),
                            "Rank": row["Rank"],
                        }
                    )
        rank_df = pd.DataFrame(rank_rows)
        if not rank_df.empty:
            selected_rank_query = str(filtered["Query_ID"].iloc[0])
            selected_rank_run = str(filtered["run_name"].iloc[0])
            heat_df = rank_df[
                (rank_df["Query_ID"].astype(str) == selected_rank_query)
                & (rank_df["run_name"].astype(str) == selected_rank_run)
            ].pivot(index="Mode", columns="Metric", values="Rank")
            if not heat_df.empty:
                fig = px.imshow(
                    heat_df.astype(float),
                    text_auto=".0f",
                    color_continuous_scale="RdYlGn_r",
                    title=f"Mode Rank Heatmap: {selected_rank_query} / {selected_rank_run}",
                    aspect="auto",
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)

    with workflow_tab:
        workflow_view = workflow_df.copy()
        if not workflow_view.empty:
            if selected_queries and "Query_ID" in workflow_view:
                workflow_view = workflow_view[workflow_view["Query_ID"].astype(str).isin(selected_queries)]
            if selected_composition_modes and "Mode" in workflow_view:
                workflow_view = workflow_view[workflow_view["Mode"].astype(str).isin(selected_composition_modes)]
        if workflow_view.empty:
            st.info("No planned workflow rows found for the selected composition reports.")
            return
        for col in ["Step", "rt_s", "tp_kbps", "availability", "Functional_Match"]:
            if col in workflow_view:
                workflow_view[col] = pd.to_numeric(workflow_view[col], errors="coerce")

        render_sticky_table(
            _composition_display_frame(workflow_view[
                [
                    col
                    for col in [
                        "Query_ID",
                        "run_name",
                        "Mode",
                        "Step",
                        "Subtask_ID",
                        "API_ID",
                        "Functional_Match",
                        "rt_s",
                        "tp_kbps",
                        "availability",
                        "Action",
                    ]
                    if col in workflow_view.columns
                ]
            ]),
            sticky_columns=["Query_ID", "Mode", "Step", "API_ID"],
            height=480,
            key="composition_planned_workflow",
        )

        timeline_source = workflow_view.dropna(subset=["Step"]).copy()
        if not timeline_source.empty:
            timeline_source["y_label"] = timeline_source["Mode"].astype(str) + " | " + timeline_source["run_name"].astype(str)
            timeline_fig = px.scatter(
                timeline_source,
                x="Step",
                y="y_label",
                color="Mode",
                text="API_ID",
                hover_data=["Subtask_ID", "Action", "Input_From_Previous_Step", "Output_To_Next_Step"],
                title="Planned Workflow Timeline",
            )
            timeline_fig.update_traces(mode="markers+text", textposition="top center")
            timeline_fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10), yaxis_title="Mode / Run")
            st.plotly_chart(timeline_fig, use_container_width=True)

        step_metric_options = [col for col in ["rt_s", "tp_kbps", "availability"] if col in workflow_view.columns]
        if not step_metric_options:
            st.info("No step-level QoS columns found for this workflow report.")
            return
        step_metric = st.selectbox(
            "Step-level QoS metric",
            step_metric_options,
            format_func=_composition_metric_label,
            key="composition_step_metric",
        )
        step_df = workflow_view.dropna(subset=["Step", step_metric]).copy()
        if not step_df.empty:
            line_fig = px.line(
                step_df,
                x="Step",
                y=step_metric,
                color="Mode",
                markers=True,
                line_group="run_name",
                hover_data=["Query_ID", "run_name", "API_ID", "Subtask_ID"],
                title=f"Step-Level {_composition_metric_label(step_metric)}",
                labels={step_metric: _composition_metric_label(step_metric)},
            )
            line_fig.update_layout(height=380, margin=dict(l=10, r=10, t=55, b=10), yaxis_title=_composition_metric_label(step_metric))
            st.plotly_chart(line_fig, use_container_width=True)


def _display_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.astype(object).where(pd.notna(df), viz.NA).astype(str)


DEFAULT_STICKY_IDENTIFIER_COLUMNS = [
    "Query_ID",
    "Query ID",
    "query_id",
    "Mode",
    "mode",
    "Step",
    "Subtask_ID",
    "Subtask ID",
    "subtask_id",
    "Subtask",
    "API_Name",
    "API Name",
    "API",
    "API_ID",
    "API ID",
    "api_id",
    "Metric",
]


def _sticky_columns_for(df: pd.DataFrame, preferred: list[str] | tuple[str, ...] | None = None, *, max_columns: int = 2) -> list[str]:
    if df.empty:
        return []
    candidates = list(preferred or []) + [col for col in DEFAULT_STICKY_IDENTIFIER_COLUMNS if col not in set(preferred or [])]
    sticky: list[str] = []
    for col in candidates:
        if col in df.columns and col not in sticky:
            sticky.append(col)
        if len(sticky) >= max_columns:
            break
    return sticky


def _html_value(value: Any) -> str:
    if value is None:
        return viz.NA
    if isinstance(value, float) and pd.isna(value):
        return viz.NA
    text = str(value)
    return text if text.strip() else viz.NA


def render_sticky_table(
    df: pd.DataFrame,
    sticky_columns: list[str] | tuple[str, ...] | None = None,
    *,
    height: int = 420,
    key: str | None = None,
    note: bool = True,
    cell_style: Callable[[str, str, pd.Series], str] | None = None,
) -> None:
    if df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    display = _display_frame(df)
    sticky = _sticky_columns_for(display, sticky_columns)
    if not sticky:
        st.dataframe(display, use_container_width=True, hide_index=True, height=height)
        return

    ordered_cols = sticky + [col for col in display.columns if col not in sticky]
    display = display[ordered_cols]
    safe_key = re.sub(r"[^a-zA-Z0-9_-]+", "-", key or "default").strip("-") or "default"
    sticky_widths = [180, 220]
    sticky_lefts = [0]
    for idx in range(1, len(sticky)):
        sticky_lefts.append(sum(sticky_widths[:idx]))

    sticky_css = []
    for idx, col in enumerate(sticky, start=1):
        width = sticky_widths[min(idx - 1, len(sticky_widths) - 1)]
        left = sticky_lefts[idx - 1]
        sticky_css.append(
            f"""
.maof-sticky-{safe_key} th:nth-child({idx}),
.maof-sticky-{safe_key} td:nth-child({idx}) {{
  position: sticky;
  left: {left}px;
  min-width: {width}px;
  max-width: {width}px;
  background: #ffffff;
  z-index: 4;
  box-shadow: 1px 0 0 #d0d7de;
}}
.maof-sticky-{safe_key} th:nth-child({idx}) {{
  z-index: 6;
  background: #f6f8fa;
}}
"""
        )

    header_html = "".join(f"<th>{escape(str(col))}</th>" for col in display.columns)
    body_rows = []
    for _, row in display.iterrows():
        cells = []
        for col in display.columns:
            value = _html_value(row.get(col))
            style = cell_style(col, value, row) if cell_style else ""
            style_attr = f' style="{escape(style)}"' if style else ""
            cells.append(f'<td title="{escape(value)}"{style_attr}><div class="maof-sticky-cell-text">{escape(value)}</div></td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    if note:
        st.caption("Primary identifier columns remain fixed while scrolling horizontally.")
    st.markdown(
        f"""
<style>
.maof-sticky-{safe_key}-wrap {{
  max-height: {int(height)}px;
  overflow: auto;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  background: #ffffff;
}}
.maof-sticky-{safe_key} {{
  width: max-content;
  min-width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 13px;
  line-height: 1.35;
  color: #24292f;
}}
.maof-sticky-{safe_key} th,
.maof-sticky-{safe_key} td {{
  box-sizing: border-box;
  min-width: 150px;
  max-width: 280px;
  padding: 8px 10px;
  border-right: 1px solid #d8dee4;
  border-bottom: 1px solid #d8dee4;
  vertical-align: top;
  overflow-wrap: anywhere;
  white-space: normal;
}}
.maof-sticky-{safe_key} th {{
  position: sticky;
  top: 0;
  background: #f6f8fa;
  color: #57606a;
  font-weight: 650;
  z-index: 3;
}}
.maof-sticky-{safe_key} td {{
  background: #ffffff;
  display: table-cell;
}}
.maof-sticky-{safe_key} .maof-sticky-cell-text {{
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  white-space: normal;
  overflow-wrap: anywhere;
}}
.maof-sticky-{safe_key} tbody tr:nth-child(even) td {{
  background: #fbfcfe;
}}
.maof-sticky-{safe_key} tbody tr:nth-child(even) td:nth-child(-n+{len(sticky)}) {{
  background: #fbfcfe;
}}
{''.join(sticky_css)}
</style>
<div class="maof-sticky-{safe_key}-wrap">
  <table class="maof-sticky-{safe_key}">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</div>
""",
        unsafe_allow_html=True,
    )


COMPOSITION_METRIC_LABELS = {
    "rt_s": viz.RESPONSE_TIME_LABEL,
    "Total_Response_Time_s": viz.TOTAL_RESPONSE_TIME_LABEL,
    "Average_Workflow_Availability": viz.WORKFLOW_AVAILABILITY_LABEL,
    "Workflow_Availability": viz.WORKFLOW_AVAILABILITY_LABEL,
    "Composition_Risk_API": "Composition Risk API",
    "Risk_Summary": "Risk Summary",
    "tp_kbps": viz.THROUGHPUT_LABEL,
    "availability": "Availability",
}


def _composition_metric_label(metric: str) -> str:
    return COMPOSITION_METRIC_LABELS.get(metric, metric)


def _composition_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    return _display_frame(df).rename(columns=COMPOSITION_METRIC_LABELS)


def _bottleneck_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    display = _display_frame(df).copy()
    if "Metric" in display:
        display["Metric"] = display["Metric"].astype(str).map(_composition_metric_label)
    return display.rename(
        columns={
            "Bottleneck_Type": "QoS Risk Type",
            "Metric_Value": "Metric Value",
            "Impact": "Risk Summary",
        }
    )


def _render_grouped_bottlenecks(workflow: pd.DataFrame, bottlenecks: pd.DataFrame) -> None:
    groups = viz.group_bottlenecks_by_api(workflow, bottlenecks)
    if not groups:
        st.info("No grouped composition-risk APIs were identified.")
        return

    st.caption(
        "Composition risk combines functional suitability and QoS signals. "
        "Functional risk is based on Functional Match, while QoS risk summarizes latency, throughput, and availability signals."
    )
    for group in groups:
        api_name = group.get("api_name") or group.get("api_id") or viz.NA
        dimensions = ", ".join(group.get("dimensions", [])) or viz.NA
        title = f"{api_name} | {dimensions}"
        with st.expander(title, expanded=True):
            st.markdown(f"**Risk-contributing API:** {api_name}")
            st.markdown(f"**QoS Signal:** {dimensions}")
            subtask = group.get("subtask")
            subtask_id = group.get("subtask_id")
            if subtask_id and subtask_id != viz.NA:
                subtask_text = f"Subtask {subtask_id}"
                if subtask and subtask != viz.NA:
                    subtask_text += f": {subtask}"
                st.markdown(f"**Subtask:** {subtask_text}")

            reasons = group.get("reasons", [])
            if reasons:
                st.markdown("**Risk Summary:**")
                for reason in reasons:
                    st.markdown(f"- {reason}")

            severity_lines = group.get("severity_lines", [])
            if severity_lines:
                st.markdown("**QoS Signal Details:**")
                for line in severity_lines:
                    st.markdown(f"- {line}")

            raw = group.get("raw_metrics", {})
            raw_rows = [
                {"Metric": viz.RESPONSE_TIME_LABEL, "Value": viz.format_response_time(raw.get("response_time_s"))},
                {"Metric": viz.THROUGHPUT_LABEL, "Value": viz.format_throughput(raw.get("throughput_kbps"))},
                {"Metric": "Availability", "Value": viz.format_value(raw.get("availability"))},
            ]
            st.dataframe(pd.DataFrame(raw_rows), use_container_width=True, hide_index=True)


def _api_detail_rows(row: dict[str, Any] | pd.Series, *, dimensions: str | None = None, reason: str | None = None) -> pd.DataFrame:
    rows = [
        {"Field": "Subtask", "Value": f"{row.get('Subtask_ID', viz.NA)} | {row.get('Subtask', viz.NA)}"},
        {"Field": "API", "Value": row.get("API_Name") or row.get("API_ID") or viz.NA},
        {"Field": "Functional Fit", "Value": viz.format_functional_fit(row.get("Functional_Match"))},
        {"Field": "Rank", "Value": viz.format_value(row.get("Mode_Rank"), 0)},
        {"Field": viz.RESPONSE_TIME_LABEL, "Value": viz.format_response_time(row.get("rt_s"))},
        {"Field": viz.THROUGHPUT_LABEL, "Value": viz.format_throughput(row.get("tp_kbps"))},
        {"Field": "Availability", "Value": viz.format_value(row.get("availability"))},
        {"Field": "API QoS Health", "Value": viz.format_api_health(row.get("API_QoS_Health"))},
        {"Field": "API QoS Health Source", "Value": row.get("API_QoS_Health_Source", viz.NA)},
    ]
    if dimensions is not None:
        rows.append({"Field": "QoS Risk Dimensions", "Value": dimensions})
    if reason is not None:
        rows.append({"Field": "Why This Candidate Was Tested", "Value": reason})
    return pd.DataFrame(rows)


def _optional_float(value: Any) -> float | None:
    if value is None or value == viz.NA:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return None if pd.isna(parsed) else parsed


def _what_if_worsens_major_metric(current: dict[str, Any], tested: dict[str, Any]) -> bool:
    specs = [
        ("Total_Response_Time_s", False),
        ("Bottleneck_Throughput_kbps", True),
        ("Average_Workflow_Availability", True),
        ("Functional_Coverage", True),
    ]
    for metric, higher_better in specs:
        before = _optional_float(current.get(metric))
        after = _optional_float(tested.get(metric))
        if before is None or after is None:
            continue
        if (higher_better and after < before) or (not higher_better and after > before):
            return True
    return False


def _render_replacement_simulation(
    *,
    workflow: pd.DataFrame,
    bottlenecks: pd.DataFrame,
    eval_row: dict[str, Any],
    run_dir: Path,
    query_id: str,
    mode: str,
    key_prefix: str,
) -> None:
    def available(value: Any) -> bool:
        return value is not None and not (isinstance(value, float) and pd.isna(value)) and value != viz.NA

    st.subheader("What-If Composition Risk Replacement Analysis")
    st.caption(
        "This what-if analysis tests how workflow-level QoS might change if a risk-contributing API were replaced with another candidate "
        "from the same subtask pool. The replacement is selected by visualization logic for diagnostic comparison. It is not "
        "generated by the LLM planner, not an official pipeline output, and does not modify stored experiment results."
    )
    simulation_bottlenecks = bottlenecks if not bottlenecks.empty else viz.identify_bottlenecks(workflow)
    simulations = viz.build_bottleneck_replacement_simulations(
        workflow=workflow,
        bottlenecks=simulation_bottlenecks,
        eval_row=eval_row,
        run_dir=run_dir,
        query_id=query_id,
        mode=mode,
    )
    if not simulations:
        st.info("No risk-contributing APIs were available for what-if replacement analysis.")
        return

    labels = []
    for idx, simulation in enumerate(simulations, start=1):
        group = simulation.get("group", {})
        dims = ", ".join(group.get("dimensions", [])) or viz.NA
        labels.append(f"{idx}. {group.get('api_name', viz.NA)} | {dims}")
    selected_label = st.selectbox("Current Risk-Contributing API", labels, key=f"{key_prefix}_replacement_pick")
    simulation = simulations[labels.index(selected_label)]
    group = simulation.get("group", {})
    if simulation.get("status") != "ok":
        st.warning(simulation.get("message") or "No candidate replacement API available for this risk-contributing API.")
        return
    if simulation.get("warning"):
        st.warning(simulation["warning"])

    current_row = simulation["current_row"]
    replacement_row = simulation["replacement_row"]
    other_modes = viz.selected_by_other_modes(
        run_dir,
        current_mode=mode,
        subtask_id=replacement_row.get("Subtask_ID"),
        api_id=replacement_row.get("API_ID"),
    )
    dimensions = ", ".join(group.get("dimensions", [])) or viz.NA
    left, right = st.columns(2)
    with left:
        st.markdown("**Current Risk-Contributing API**")
        st.dataframe(_display_frame(_api_detail_rows(current_row, dimensions=dimensions)), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Candidate Replacement API**")
        st.dataframe(_display_frame(_api_detail_rows(replacement_row, reason=simulation.get("reason"))), use_container_width=True, hide_index=True)
        if other_modes is None:
            st.caption("This candidate may or may not appear in another mode's selected workflow.")
        else:
            label = "Yes" if other_modes else "No"
            detail = f" ({', '.join(other_modes)})" if other_modes else ""
            st.caption(f"Selected by another mode: {label}{detail}")

    st.markdown("**What-If Metric Change**")
    comparison = pd.DataFrame(viz.simulation_metric_rows(simulation["current_metrics"], simulation["simulated_metrics"]))
    st.dataframe(_display_frame(comparison), use_container_width=True, hide_index=True)
    worsens_major_metric = _what_if_worsens_major_metric(simulation["current_metrics"], simulation["simulated_metrics"])
    if worsens_major_metric:
        st.warning("The candidate replacement does not improve the workflow overall and may weaken one or more QoS metrics.")
    elif simulation.get("has_improvement"):
        st.info("The candidate replacement improves the what-if workflow metrics, but it remains a diagnostic alternative and is not part of the official planner output.")
    else:
        st.info("No clear improvement detected. The official planner workflow remains the better supported result for this comparison.")

    severity_lines = group.get("severity_lines", [])
    explanation_lines = list(severity_lines)
    current_metrics = simulation.get("current_metrics", {})
    simulated_metrics = simulation.get("simulated_metrics", {})
    before_tp = current_metrics.get("Bottleneck_Throughput_kbps")
    after_tp = simulated_metrics.get("Bottleneck_Throughput_kbps")
    if available(before_tp) and available(after_tp) and before_tp != after_tp:
        explanation_lines.append(f"The candidate replacement changes bottleneck throughput from {viz.format_value(before_tp)} to {viz.format_value(after_tp)}.")
    before_av = current_metrics.get("Average_Workflow_Availability", current_metrics.get("Workflow_Availability"))
    after_av = simulated_metrics.get("Average_Workflow_Availability", simulated_metrics.get("Workflow_Availability"))
    if available(before_av) and available(after_av) and before_av != after_av:
        explanation_lines.append(f"The candidate replacement changes workflow availability from {viz.format_value(before_av)} to {viz.format_value(after_av)}.")
    if explanation_lines:
        st.markdown("**Simulation Notes**")
        for line in explanation_lines[:5]:
            st.markdown(f"- {line}")

    with st.expander("What-If Replacement Workflow Mini Graph", expanded=False):
        st.caption(
            "The left side shows the official workflow generated by the selected mode. The right side shows a diagnostic what-if workflow where only the risk-contributing API is replaced."
        )
        st.graphviz_chart(
            viz.build_replacement_simulation_dot(workflow, simulation["simulated_workflow"], current_row, replacement_row),
            use_container_width=True,
        )


def _render_dataflow_graph(workflow: pd.DataFrame, query_context: dict[str, str], selected_mode: str) -> None:
    if workflow.empty:
        st.info("No planned workflow rows are available for the dataflow graph.")
        return

    view_mode = st.selectbox(
        "View mode",
        ["Compact graph", "Detailed graph"],
        index=0,
        key=f"dataflow_view_mode_{selected_mode}",
    )
    st.markdown(
        viz.build_dataflow_cards_html(
            query_context=query_context,
            workflow=workflow,
            mode=selected_mode,
            detailed=view_mode == "Detailed graph",
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "This view shows the planned dataflow between selected APIs. Long action and rationale details are shown below to keep the diagram readable."
    )

    st.markdown("**Step Details**")
    for _, row in workflow.iterrows():
        title = f"Step {row.get('Step', viz.NA)} | {row.get('API_Name') or row.get('API_ID') or viz.NA}"
        with st.expander(title, expanded=False):
            detail = pd.DataFrame(
                [
                    {"Field": "Subtask", "Value": row.get("Subtask", viz.NA)},
                    {"Field": "Selected API", "Value": row.get("API_Name") or row.get("API_ID") or viz.NA},
                    {"Field": "Input from previous step", "Value": row.get("Input_From_Previous_Step", viz.NA)},
                    {"Field": "Action", "Value": row.get("Action", viz.NA)},
                    {"Field": "Output to next step", "Value": row.get("Output_To_Next_Step", viz.NA)},
                    {"Field": "Why", "Value": row.get("Why", viz.NA)},
                ]
            )
            st.dataframe(_display_frame(detail), use_container_width=True, hide_index=True)


def _render_agent_observability(
    *,
    run_dir: Path,
    query_id: str,
    mode: str,
    eval_row: dict[str, Any],
    workflow: pd.DataFrame,
) -> None:
    st.caption(
        "This view summarizes the health of the agentic pipeline using existing run artifacts and logs. "
        "It is for debugging and interpretation only."
    )
    summary = viz.build_agent_observability_summary(
        run_dir=run_dir,
        query_id=query_id,
        mode=mode,
        eval_row=eval_row,
        workflow=workflow,
    )
    score = summary.get("score")
    st.metric("Pipeline Health Score", viz.format_value(score, percent=True))
    st.dataframe(_display_frame(pd.DataFrame(summary.get("rows", []))), use_container_width=True, hide_index=True)
    logs = pd.DataFrame(summary.get("logs", []))
    if logs.empty:
        st.info("No warning/error log available for this run.")
        return
    st.markdown("**Warning and Error Log Signals**")
    st.dataframe(_display_frame(logs), use_container_width=True, hide_index=True)
    available_logs = logs[logs["Status"].astype(str) == "Available"] if "Status" in logs else pd.DataFrame()
    if available_logs.empty:
        st.info("No warning/error log available for this run.")
    else:
        for _, row in available_logs.iterrows():
            with st.expander(str(row.get("Log") or "Log"), expanded=False):
                st.code(str(row.get("Details") or ""), language="text")


def _render_invalid_workflow_diagnostics(
    *,
    workflow: pd.DataFrame,
    eval_row: dict[str, Any],
    run_dir: Path,
    mode: str,
) -> None:
    st.caption("Invalid workflow diagnostics explain why invalid plans receive zero or reduced validity.")
    issues = viz.detect_invalid_workflow_issues(eval_row=eval_row, workflow=workflow, run_dir=run_dir, mode=mode)
    if not issues:
        st.info("No invalid workflow issues detected.")
        return
    st.graphviz_chart(viz.build_invalid_workflow_diagnostic_dot(workflow, issues), use_container_width=True)
    issues_df = pd.DataFrame(issues)
    render_sticky_table(_display_frame(issues_df), sticky_columns=["Step", "API", "Issue"], height=360, key="invalid_workflow_issues")


def _render_winner_heatmap(eval_df: pd.DataFrame, run_dir: Path) -> None:
    st.caption("This heatmap summarizes which ranking/selection mode performs best for each query and metric.")
    collection_eval = _eval_rows_for_run_collection(eval_df, run_dir)
    result = viz.build_winner_heatmap(collection_eval)
    winners = result.get("winners", pd.DataFrame())
    counts = result.get("counts", pd.DataFrame())
    if winners.empty:
        st.info("Winner heatmap unavailable because composition evaluation rows are missing.")
        return
    render_sticky_table(winners, sticky_columns=["Query"], height=420, key="winner_heatmap")
    if not counts.empty:
        st.markdown("**Win Counts by Metric**")
        render_sticky_table(_display_frame(counts), sticky_columns=["Metric", "Mode"], height=320, key="winner_counts")
        overall = counts.groupby("Mode", as_index=False)["Wins"].sum().sort_values(["Wins", "Mode"], ascending=[False, True])
        if not overall.empty:
            st.caption(f"Overall most frequent winner: `{overall.iloc[0]['Mode']}` with {int(overall.iloc[0]['Wins'])} metric wins.")


def _render_sensitivity_analysis(eval_df: pd.DataFrame, *, query_id: str, run_dir: Path, run_name: str, recommended_mode: str | None) -> None:
    st.caption("Sensitivity analysis is exploratory. It does not change the official evaluation results.")
    filtered = eval_df.copy()
    if "Query_ID" in filtered:
        filtered = filtered[filtered["Query_ID"].astype(str) == str(query_id)]
    if "run_dir" in filtered:
        filtered = filtered[filtered["run_dir"].astype(str) == str(run_dir)]
    elif "run_name" in filtered:
        filtered = filtered[filtered["run_name"].astype(str) == str(run_name)]
    if filtered.empty:
        st.info("Sensitivity analysis unavailable because composition evaluation rows are missing.")
        return

    st.markdown("**Visualization-only score weights**")
    cols = st.columns(3)
    raw_weights = {
        "QoS weight": cols[0].slider("QoS weight", 0.0, 1.0, 0.4, 0.05, key="sensitivity_qos"),
        "Functional Coverage weight": cols[1].slider("Functional coverage weight", 0.0, 1.0, 0.3, 0.05, key="sensitivity_functional"),
        "Composition Completeness weight": cols[2].slider("Completeness weight", 0.0, 1.0, 0.2, 0.05, key="sensitivity_completeness"),
    }
    total = sum(raw_weights.values()) or 1.0
    st.caption("Weights are normalized internally so the sensitivity score always sums to 1.")
    weights = {key: value / total for key, value in raw_weights.items()}
    scores = viz.compute_sensitivity_scores(filtered, weights)
    if scores.empty:
        st.info("Sensitivity analysis could not compute scores from the available metrics.")
        return
    display_cols = [
        "Mode",
        "Original_QoS_Adjusted_Composition_Score",
        "Sensitivity_Score",
        "Normalized_QoS_Score",
        "Functional_Coverage",
        "Composition_Completeness",
        "Composition_Validity",
    ]
    display_scores = scores[[col for col in display_cols if col in scores]].copy()
    render_sticky_table(
        _composition_display_frame(display_scores.round(4)),
        sticky_columns=["Mode"],
        height=320,
        key="sensitivity_scores",
    )

    chart_source = scores.melt(
        id_vars=["Mode"],
        value_vars=[col for col in ["Original_QoS_Adjusted_Composition_Score", "Sensitivity_Score"] if col in scores],
        var_name="Score Type",
        value_name="Score",
    )
    if not chart_source.empty:
        chart_source["Score Type"] = chart_source["Score Type"].map(
            {
                "Original_QoS_Adjusted_Composition_Score": "Original QoS-adjusted composition score",
                "Sensitivity_Score": "Sensitivity score",
            }
        ).fillna(chart_source["Score Type"])
        fig = px.bar(
            chart_source,
            x="Mode",
            y="Score",
            color="Score Type",
            barmode="group",
            text_auto=".3f",
            range_y=[0, 1],
            title="Original Score vs Sensitivity Score",
            category_orders={"Mode": MODE_ORDER},
        )
        fig.update_layout(height=410, margin=dict(l=10, r=10, t=55, b=60), yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True, config=_plotly_config())

    sensitivity_winner = str(scores.iloc[0].get("Mode") or viz.NA)
    if recommended_mode and sensitivity_winner != recommended_mode:
        st.info(f"With these weights, the recommended mode changes from `{recommended_mode}` to `{sensitivity_winner}`.")
    else:
        st.success(f"With these weights, the top sensitivity mode is `{sensitivity_winner}`.")
    norm_components = ["Normalized_Response_Time_Score", "Normalized_Throughput_Score", "Normalized_Availability_Score"]
    if not all(col in filtered for col in norm_components):
        st.caption("QoS subweight sensitivity is unavailable because normalized QoS components are missing.")


def _render_observability_diagnostics(
    *,
    eval_df: pd.DataFrame,
    workflow: pd.DataFrame,
    eval_row: dict[str, Any],
    query_context: dict[str, str],
    query_id: str,
    run_dir: Path,
    run_name: str,
    mode: str,
    recommended_mode: str | None,
) -> None:
    st.subheader("Observability and Diagnostics")
    st.caption("These views are visualization and diagnostic aids only. They do not change the stored experiment results.")
    dataflow_tab, agent_tab, invalid_tab, heatmap_tab, sensitivity_tab = st.tabs(
        ["Dataflow Graph", "Agent Observability", "Invalid Workflow Diagnostics", "Winner Heatmap", "Sensitivity Analysis"]
    )
    with dataflow_tab:
        _render_dataflow_graph(workflow, query_context, mode)
    with agent_tab:
        _render_agent_observability(run_dir=run_dir, query_id=query_id, mode=mode, eval_row=eval_row, workflow=workflow)
    with invalid_tab:
        _render_invalid_workflow_diagnostics(workflow=workflow, eval_row=eval_row, run_dir=run_dir, mode=mode)
    with heatmap_tab:
        _render_winner_heatmap(eval_df, run_dir)
    with sensitivity_tab:
        _render_sensitivity_analysis(eval_df, query_id=query_id, run_dir=run_dir, run_name=run_name, recommended_mode=recommended_mode)


def _composition_visual_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"] div[data-testid="stPlotlyChart"] {
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 0.25rem;
            background: #FFFFFF;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            background: #FFFFFF;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.75rem;
        }
        .AutoLLMCompose-color-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem 1rem;
            align-items: center;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 0.65rem 0.8rem;
            margin: 0.2rem 0 0.7rem 0;
            background: #FFFFFF;
            font-size: 0.92rem;
            color: #374151;
        }
        .AutoLLMCompose-color-legend .legend-title {
            font-weight: 700;
            color: #111827;
        }
        .AutoLLMCompose-color-legend .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            white-space: nowrap;
        }
        .AutoLLMCompose-color-legend .legend-chip {
            display: inline-block;
            width: 0.95rem;
            height: 0.95rem;
            border-radius: 3px;
            border: 2px solid #64748B;
        }
        .AutoLLMCompose-color-legend .legend-green {
            background: #C6EFCE;
            border-color: #2F855A;
        }
        .AutoLLMCompose-color-legend .legend-orange {
            background: #FCE4D6;
            border-color: #B7791F;
        }
        .AutoLLMCompose-color-legend .legend-red {
            background: #F4CCCC;
            border-color: #C53030;
        }
        .AutoLLMCompose-color-legend .legend-gray {
            background: #E5E7EB;
            border-color: #64748B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plotly_config() -> dict:
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "responsive": True,
    }


def _viz_key(*parts: object) -> str:
    raw = "_".join(str(part) for part in parts if part is not None)
    return re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()


def _ordered_mode_subset(values: list[str]) -> list[str]:
    value_set = {str(value) for value in values}
    ordered = [mode for mode in MODE_ORDER if mode in value_set]
    ordered.extend(sorted(value_set - set(ordered)))
    return ordered


def _visual_run_label(row: pd.Series) -> str:
    run_dir = Path(str(row.get("run_dir") or ""))
    parent_name = run_dir.parent.name if str(run_dir) else ""
    suffix = f" | {parent_name}" if parent_name else ""
    return f"{row.get('run_name')}{suffix}"


def _highlight_selection_difference(row: pd.Series) -> list[str]:
    color = "background-color: #FFF2CC" if row.get("Selection_Difference") == "Different APIs" else ""
    return [color for _ in row]


def _mode_cell_style(value: Any) -> str:
    text = str(value or "")
    colors = {
        "no_qos": "#E5E7EB",
        "qos_pure_llm": "#D9EAF7",
        "qos_topsis": "#E2F0D9",
        "qos_hybrid": "#FFF2CC",
        "No composition-complete mode": "#F4CCCC",
    }
    if "," in text:
        return "background-color: #F8FAFC; color: #111827;"
    for mode, color in colors.items():
        if text == mode:
            return f"background-color: {color}; color: #111827;"
    return ""


def _qos_summary_cell_style(column: str, value: str, row: pd.Series) -> str:
    pure_vs_no_qos_col = "Is Qos_pure_llm better than no_qos"
    colors = {
        "Yes": "#C6EFCE",
        "No": "#F4CCCC",
        "Tie": "#FFF2CC",
    }
    if column == "query_id":
        hybrid_status = str(row.get("is_QoS_Hybrid_best", "")).strip()
        pure_status = str(row.get(pure_vs_no_qos_col, "")).strip()
        if hybrid_status == "Yes" and pure_status == "Yes":
            color = colors["Yes"]
        elif pure_status == "Tie" or "No" in {hybrid_status, pure_status}:
            color = colors["No"]
        elif hybrid_status == "Tie" and pure_status == "Yes":
            color = colors["Tie"]
        else:
            color = ""
        return f"background-color: {color}; color: #111827;" if color else ""
    if column not in {"is_QoS_Hybrid_best", pure_vs_no_qos_col}:
        return ""
    color = colors.get(str(value).strip())
    return f"background-color: {color}; color: #111827;" if color else ""


def _eval_rows_for_run_collection(eval_df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    if eval_df.empty or "run_dir" not in eval_df:
        return eval_df.copy()
    collection_root = run_dir.parent.resolve()

    def same_collection(value: Any) -> bool:
        try:
            return Path(str(value)).parent.resolve() == collection_root
        except Exception:
            return False

    filtered = eval_df[eval_df["run_dir"].apply(same_collection)].copy()
    return filtered if not filtered.empty else eval_df.copy()


def _render_summary_metrics(summary_rows: list[dict[str, str]]) -> None:
    summary = {row["Metric"]: row["Value"] for row in summary_rows}
    metric_order = [
        "Selected Mode",
        "Number of Subtasks",
        "Number of Selected APIs",
        "Composition Validity",
        "Composition Completeness",
        "Functional Coverage",
        viz.TOTAL_RESPONSE_TIME_LABEL,
        viz.BOTTLENECK_THROUGHPUT_LABEL,
        viz.WORKFLOW_AVAILABILITY_LABEL,
        "Normalized QoS Score",
        "QoS-Adjusted Composition Score",
    ]
    cols = st.columns(4)
    for idx, label in enumerate(metric_order):
        cols[idx % 4].metric(label, summary.get(label, viz.NA))
    st.markdown(f"**Risk-contributing API:** {summary.get('Risk-contributing API', summary.get('Bottleneck API', viz.NA))}")


def _render_recommendation_summary(recommendation: dict[str, Any]) -> None:
    row = recommendation.get("row") or {}
    modes = [str(mode) for mode in recommendation.get("modes") or [] if str(mode)]
    mode_label = ", ".join(modes) if recommendation.get("is_tie") and modes else recommendation.get("mode") or viz.NA
    if recommendation.get("is_tie") and mode_label != viz.NA:
        mode_label = f"{mode_label} (tie)"
    reason = recommendation.get("reason") or "Highest QoS-adjusted composition score among composition-complete modes"
    mode_field = "Diagnostic Mode" if recommendation.get("status") == "diagnostic" else "Recommended Mode"
    workflow_availability = viz.workflow_availability_value(row)
    fields = [
        (mode_field, mode_label),
        ("Reason", reason),
        ("QoS-Adjusted Composition Score", viz.format_value(row.get("QoS_Adjusted_Composition_Score"))),
        ("Functional Coverage", viz.format_value(row.get("Functional_Coverage"), percent=True)),
        ("Composition Completeness", viz.format_value(row.get("Composition_Completeness"), percent=True)),
        ("Normalized QoS Score", viz.format_value(row.get("Normalized_QoS_Score"))),
        (viz.TOTAL_RESPONSE_TIME_LABEL, viz.format_response_time(row.get("Total_Response_Time_s"))),
        (viz.BOTTLENECK_THROUGHPUT_LABEL, viz.format_throughput(row.get("Bottleneck_Throughput_kbps"))),
        (viz.WORKFLOW_AVAILABILITY_LABEL, viz.format_value(workflow_availability)),
    ]
    st.dataframe(_display_frame(pd.DataFrame(fields, columns=["Metric", "Value"])), use_container_width=True, hide_index=True)
    st.caption(viz.WORKFLOW_AVAILABILITY_HELP)


def _render_quality_legend() -> None:
    legend = pd.DataFrame(
        [
            {"Color": "Green", "Meaning": "Strong QoS / Low Risk"},
            {"Color": "Orange", "Meaning": "Moderate QoS / Medium Risk"},
            {"Color": "Red", "Meaning": "Weak QoS / High Risk"},
            {"Color": "Gray", "Meaning": "Missing or unknown QoS data"},
        ]
    )
    st.dataframe(legend, use_container_width=True, hide_index=True, height=180)
    st.caption(viz.API_HEALTH_HELP)


def _render_color_chip_legend(*, composition_status: bool = True) -> None:
    if composition_status:
        legend_items = """
            <span class="legend-item"><span class="legend-chip legend-green"></span>Recommended composition path</span>
            <span class="legend-item"><span class="legend-chip legend-orange"></span>Valid alternative path</span>
            <span class="legend-item"><span class="legend-chip legend-red"></span>Functional or composition risk</span>
            <span class="legend-item"><span class="legend-chip legend-gray"></span>Missing/unknown data</span>
        """
        caption = (
            "Composition Recommendation Status. Functional suitability is prioritized first, "
            "while QoS health is used as a secondary signal."
        )
    else:
        legend_items = """
            <span class="legend-item"><span class="legend-chip legend-green"></span>Strong QoS / Low Risk</span>
            <span class="legend-item"><span class="legend-chip legend-orange"></span>Moderate QoS / Medium Risk</span>
            <span class="legend-item"><span class="legend-chip legend-red"></span>Weak QoS / High Risk</span>
            <span class="legend-item"><span class="legend-chip legend-gray"></span>Missing/unknown QoS data</span>
        """
        caption = "Composition evaluation rows are missing, so node colors fall back to API QoS Health."
    st.markdown(
        f"""
        <div class="AutoLLMCompose-color-legend">
            <span class="legend-title">Node color guide:</span>
            {legend_items}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(caption)


def _render_qos_health_warnings(workflow: pd.DataFrame) -> None:
    if workflow.empty or "API_QoS_Health_Warning" not in workflow:
        return
    warnings = [
        str(value).strip()
        for value in workflow["API_QoS_Health_Warning"].dropna().tolist()
        if str(value).strip()
    ]
    if warnings:
        st.warning(warnings[0])


def _render_workflow_interpretation(workflow: pd.DataFrame, bottlenecks: pd.DataFrame) -> None:
    if workflow.empty:
        return
    status_col = "API_Health_Status" if "API_Health_Status" in workflow else "Selection_Quality_Status" if "Selection_Quality_Status" in workflow else "Health_Status"
    status_counts = workflow[status_col].value_counts().to_dict() if status_col in workflow else {}
    mismatch_count = int((pd.to_numeric(workflow.get("Functional_Match"), errors="coerce") == 0).sum()) if "Functional_Match" in workflow else 0
    functional_risk_counts = workflow["Functional_Risk"].value_counts().to_dict() if "Functional_Risk" in workflow else {}
    qos_risk_counts = workflow["QoS_Risk"].value_counts().to_dict() if "QoS_Risk" in workflow else {}
    bottleneck_text = viz.bottleneck_group_summary(workflow, bottlenecks)
    rows = [
        {"Signal": "Selected APIs", "Interpretation": str(len(workflow))},
        {"Signal": "Functional Risk Low APIs", "Interpretation": str(functional_risk_counts.get("Low", max(len(workflow) - mismatch_count, 0)))},
        {"Signal": "Functional Risk High APIs", "Interpretation": str(functional_risk_counts.get("High", mismatch_count))},
        {"Signal": "QoS Risk Low APIs", "Interpretation": str(qos_risk_counts.get("Low", status_counts.get("green", 0)))},
        {"Signal": "QoS Risk Medium APIs", "Interpretation": str(qos_risk_counts.get("Medium", status_counts.get("orange", 0)))},
        {"Signal": "QoS Risk High APIs", "Interpretation": str(qos_risk_counts.get("High", status_counts.get("red", 0)))},
        {"Signal": "Unknown QoS APIs", "Interpretation": str(status_counts.get("gray", 0))},
        {"Signal": "Risk-contributing API", "Interpretation": bottleneck_text},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=250)


def _render_step_cards(workflow: pd.DataFrame) -> None:
    if workflow.empty:
        return
    for _, row in workflow.iterrows():
        title = f"Step {row.get('Step', viz.NA)} | Subtask {row.get('Subtask_ID', viz.NA)} | {row.get('API_Name', row.get('API_ID', viz.NA))}"
        with st.expander(title, expanded=False):
            cols = st.columns(4)
            cols[0].metric("Functional Fit", viz.format_functional_fit(row.get("Functional_Match")))
            cols[1].metric(viz.RESPONSE_TIME_LABEL, viz.format_response_time(row.get("rt_s")))
            cols[2].metric(viz.THROUGHPUT_LABEL, viz.format_throughput(row.get("tp_kbps")))
            cols[3].metric("Availability", viz.format_value(row.get("availability")))
            detail = pd.DataFrame(
                [
                    {"Field": "Subtask", "Value": row.get("Subtask", viz.NA)},
                    {"Field": "API ID", "Value": row.get("API_ID", viz.NA)},
                    {"Field": "Mode Rank", "Value": viz.format_value(row.get("Mode_Rank"), 0)},
                    {"Field": "API QoS Health", "Value": viz.format_api_health(row.get("API_QoS_Health"))},
                    {"Field": "API QoS Health Source", "Value": row.get("API_QoS_Health_Source", viz.NA)},
                    {"Field": "API Selection Health", "Value": viz.format_api_health(row.get("API_Selection_Health"))},
                    {"Field": "API Health Label", "Value": row.get("API_Health_Label", viz.NA)},
                    {"Field": "Functional Risk", "Value": row.get("Functional_Risk", viz.NA)},
                    {"Field": "QoS Risk", "Value": row.get("QoS_Risk", row.get("API_Risk_Label", viz.NA))},
                    {"Field": "Composition Risk", "Value": row.get("Composition_Risk", viz.NA)},
                    {"Field": "QoS LLM Score", "Value": viz.format_value(row.get("QoS_LLM_Score"))},
                    {"Field": "TOPSIS Score", "Value": viz.format_value(row.get("TOPSIS_Score"))},
                    {"Field": "QoS Risk Dimensions", "Value": row.get("Bottleneck_Dimensions", viz.NA)},
                    {"Field": "Action", "Value": row.get("Action", viz.NA)},
                    {"Field": "Why", "Value": row.get("Why", viz.NA)},
                ]
            )
            st.dataframe(_display_frame(detail), use_container_width=True, hide_index=True)


def _workflow_detail_view(workflow: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Step",
        "Subtask_ID",
        "Subtask",
        "API_Name",
        "API_ID",
        "Functional_Match",
        "rt_s",
        "tp_kbps",
        "availability",
        "Mode_Rank",
        "API_QoS_Health",
        "API_QoS_Health_Source",
        "API_Selection_Health",
        "API_Health_Label",
        "API_Risk_Label",
        "Functional_Risk",
        "QoS_Risk",
        "Composition_Risk",
        "QoS_LLM_Score",
        "TOPSIS_Score",
        "Bottleneck_Dimensions",
    ]
    display_source = workflow[[col for col in cols if col in workflow.columns]].copy()
    if "Functional_Match" in display_source:
        display_source["Functional_Match"] = display_source["Functional_Match"].apply(viz.format_functional_fit)
    for health_col in ["API_QoS_Health", "API_Selection_Health"]:
        if health_col in display_source:
            display_source[health_col] = display_source[health_col].apply(viz.format_api_health)
    display = _display_frame(display_source)
    return display.rename(
        columns={
            "Subtask_ID": "Subtask ID",
            "Functional_Match": "Functional Fit",
            "rt_s": viz.RESPONSE_TIME_LABEL,
            "tp_kbps": viz.THROUGHPUT_LABEL,
            "availability": "Availability",
            "Mode_Rank": "Mode Rank",
            "API_QoS_Health": "API QoS Health",
            "API_QoS_Health_Source": "API QoS Health Source",
            "API_Selection_Health": "API Selection Health",
            "API_Health_Label": "API Health",
            "API_Risk_Label": "QoS Risk",
            "Functional_Risk": "Functional Risk",
            "QoS_Risk": "QoS Risk",
            "Composition_Risk": "Composition Risk",
            "QoS_LLM_Score": "QoS LLM Score",
            "TOPSIS_Score": "TOPSIS Score",
            "Bottleneck_Dimensions": "QoS Risk Dimensions",
        }
    )


def _render_mode_comparison(
    *,
    eval_df: pd.DataFrame,
    workflow_df: pd.DataFrame,
    query_context: dict[str, str],
    query_id: str,
    run_dir: Path,
    run_name: str,
    modes: list[str],
) -> None:
    key_prefix = _viz_key("mode_compare", query_id, run_name)
    workflows_by_mode: dict[str, pd.DataFrame] = {}
    bottlenecks_by_mode: dict[str, pd.DataFrame] = {}
    eval_rows_by_mode: dict[str, dict] = {}
    for mode in modes:
        mode_workflow, mode_bottlenecks = viz.enrich_workflow_for_selection(
            workflow_df,
            query_id=query_id,
            run_dir=run_dir,
            run_name=run_name,
            mode=mode,
        )
        workflows_by_mode[mode] = mode_workflow
        bottlenecks_by_mode[mode] = mode_bottlenecks
        eval_rows_by_mode[mode] = viz.eval_row_for_mode(
            eval_df,
            query_id=query_id,
            run_dir=run_dir,
            run_name=run_name,
            mode=mode,
        )

    has_eval_rows = viz.mode_comparison_has_eval_rows(eval_rows_by_mode)
    _render_color_chip_legend(composition_status=has_eval_rows)
    if not has_eval_rows:
        st.warning("Composition evaluation rows are missing for this run, so the Mode Comparison graph is falling back to API QoS health colors.")
    for mode_workflow in workflows_by_mode.values():
        _render_qos_health_warnings(mode_workflow)
    st.plotly_chart(
        viz.build_mode_comparison_figure(workflows_by_mode, modes, eval_rows_by_mode=eval_rows_by_mode),
        use_container_width=True,
        config=_plotly_config(),
        key=f"{key_prefix}_matrix",
    )
    st.caption(
        "Node colors show composition recommendation status. Functional suitability is prioritized first, "
        "while QoS health is used as a secondary signal. Green indicates the recommended composition path "
        "for the query, red indicates invalid, incomplete, or functionally weak steps, and orange indicates "
        "valid alternatives with lower final score or QoS risk."
    )

    st.subheader("Compact Comparison Table")
    difference_df = viz.workflow_difference_table(workflows_by_mode, modes)
    if not difference_df.empty:
        render_sticky_table(
            difference_df,
            sticky_columns=["Subtask_ID", "Subtask"],
            height=420,
            key=f"{key_prefix}_difference_table",
        )
    else:
        st.info("No selected API rows were available for comparison.")

    summary_df = viz.mode_summary_table(
        eval_df,
        workflows_by_mode,
        bottlenecks_by_mode,
        query_id=query_id,
        run_dir=run_dir,
        run_name=run_name,
        modes=modes,
    )
    if not summary_df.empty:
        render_sticky_table(
            _composition_display_frame(summary_df),
            sticky_columns=["Mode"],
            height=360,
            key=f"{key_prefix}_mode_summary",
        )
        highlights = viz.comparison_highlights(summary_df, difference_df)
        if highlights:
            st.markdown("**Mode Differences**")
            for item in highlights:
                st.markdown(f"- {item}")

    mode_tabs = st.tabs(modes)
    for mode_tab, mode in zip(mode_tabs, modes):
        workflow = workflows_by_mode.get(mode, pd.DataFrame())
        with mode_tab:
            if workflow.empty:
                st.info("No planned workflow rows.")
                continue
            left, right = st.columns([1.45, 1])
            with left:
                st.plotly_chart(
                    viz.build_workflow_figure(
                        query_context=query_context,
                        workflow=workflow,
                        mode=mode,
                        eval_row=eval_rows_by_mode.get(mode),
                    ),
                    use_container_width=True,
                    config=_plotly_config(),
                    key=f"{key_prefix}_{_viz_key(mode)}_workflow",
                )
                st.caption(
                    "This graph shows the planned composition structure. Subtasks are shown vertically, "
                    "and the selected API for each subtask is shown to the right."
                )
            with right:
                summary_rows = viz.recommended_summary_rows(
                    eval_rows_by_mode.get(mode, {}),
                    workflow,
                    bottlenecks_by_mode.get(mode, pd.DataFrame()),
                    mode=mode,
                )
                st.dataframe(_display_frame(pd.DataFrame(summary_rows)), use_container_width=True, hide_index=True, height=430)
                _render_workflow_interpretation(workflow, bottlenecks_by_mode.get(mode, pd.DataFrame()))


def render_composition_visualizations() -> None:
    _composition_visual_css()
    st.title("Composition Visualizations")
    st.caption("Visual inspection of composed API workflows using existing AutoLLMCompose run artifacts.")

    with st.sidebar:
        st.header("Composition Visualizations")
        parent_dir = _render_directory_selector(
            "Runs directory",
            default_path=DEFAULT_PARENT,
            key="composition_visuals_dir",
        )
        if st.button("Reload visualization data"):
            _load_composition_reports.clear()
            st.rerun()

    eval_df, workflow_df, warnings = _load_composition_reports(parent_dir)
    for warning in warnings:
        st.warning(warning)
    if eval_df.empty:
        st.info("No composition QoS evaluation reports found under the selected parent directory.")
        return

    query_lookup = {row["id"]: {"title": row.get("title", ""), "goal": row.get("goal", "")} for row in _load_query_options(str(QUERIES_PATH))}
    query_options = sorted(
        eval_df["Query_ID"].dropna().astype(str).unique().tolist(),
        key=lambda value: int(value[1:]) if value.lower().startswith("q") and value[1:].isdigit() else value,
    )
    if not query_options:
        st.info("No query IDs were found in composition reports.")
        return

    with st.sidebar:
        selected_query = st.selectbox("Query ID", query_options, key="composition_visual_query")
        run_rows = (
            eval_df[eval_df["Query_ID"].astype(str) == selected_query][["run_dir", "run_name"]]
            .drop_duplicates()
            .sort_values(["run_name", "run_dir"])
            .reset_index(drop=True)
        )
        if run_rows.empty:
            st.info("No composition runs are available for the selected query.")
            return
        run_labels = [_visual_run_label(row) for _, row in run_rows.iterrows()]
        selected_run_label = st.selectbox("Run", run_labels, key="composition_visual_run")
        selected_run = run_rows.iloc[run_labels.index(selected_run_label)]
        run_dir = Path(str(selected_run["run_dir"]))
        run_name = str(selected_run["run_name"])
        mode_values = eval_df[
            (eval_df["Query_ID"].astype(str) == selected_query)
            & (eval_df["run_dir"].astype(str) == str(run_dir))
        ]["Mode"].dropna().astype(str).unique().tolist()
        available_modes = _ordered_mode_subset(mode_values)
        if not available_modes:
            st.info("No ranking modes are available for the selected run.")
            return
        recommendation = viz.get_recommended_mode(
            eval_df,
            selected_query,
            run_dir=run_dir,
            run_name=run_name,
            modes=available_modes,
        )
        recommended_mode = recommendation.get("mode")
        default_mode_idx = available_modes.index(recommended_mode) if recommended_mode in available_modes else 0
        selected_mode = st.selectbox(
            "Selected Mode Workflow",
            available_modes,
            index=default_mode_idx,
            key="composition_visual_mode",
        )

    query_context = viz.load_query_context(run_dir, selected_query, query_lookup)
    eval_row = viz.eval_row_for_mode(
        eval_df,
        query_id=selected_query,
        run_dir=run_dir,
        run_name=run_name,
        mode=selected_mode,
    )
    workflow, bottlenecks = viz.enrich_workflow_for_selection(
        workflow_df,
        query_id=selected_query,
        run_dir=run_dir,
        run_name=run_name,
        mode=selected_mode,
    )
    recommended_workflow = pd.DataFrame()
    recommended_bottlenecks = pd.DataFrame()
    recommended_eval_row = recommendation.get("row") or {}
    if recommended_mode in available_modes:
        recommended_workflow, recommended_bottlenecks = viz.enrich_workflow_for_selection(
            workflow_df,
            query_id=selected_query,
            run_dir=run_dir,
            run_name=run_name,
            mode=recommended_mode,
        )
    chart_key_prefix = _viz_key("composition_visuals", selected_query, run_name, selected_mode)
    recommendation_key_prefix = _viz_key("recommended", selected_query, run_name, recommended_mode or "none")

    st.write(f"**Query:** {query_context['label']}")
    if query_context.get("goal"):
        st.caption(query_context["goal"])
    st.caption(f"Run folder: `{run_dir}`")

    tabs = st.tabs(
        [
            "Selected Mode Workflow",
            "Composition Risk Analysis",
            "Recommended Composition Path",
            "Mode Comparison",
            "Observability and Diagnostics",
            "Sequence Diagram",
            "Raw Workflow Data",
        ]
    )

    with tabs[0]:
        if workflow.empty:
            st.info("No planned workflow rows found for this query and mode.")
        else:
            st.plotly_chart(
                viz.build_workflow_figure(
                    query_context=query_context,
                    workflow=workflow,
                    mode=selected_mode,
                    eval_row=eval_row,
                ),
                use_container_width=True,
                config=_plotly_config(),
                key=f"{chart_key_prefix}_workflow_graph",
            )
            st.caption(
                "This graph shows the planned composition structure. Subtasks are shown vertically, "
                "and the selected API for each subtask is shown to the right."
            )
            left, right = st.columns([1.2, 1])
            with left:
                st.dataframe(
                    _display_frame(pd.DataFrame(viz.recommended_summary_rows(eval_row, workflow, bottlenecks, mode=selected_mode))),
                    use_container_width=True,
                    hide_index=True,
                    height=430,
                )
            with right:
                _render_quality_legend()
                _render_qos_health_warnings(workflow)
                _render_workflow_interpretation(workflow, bottlenecks)
            if not bottlenecks.empty:
                st.subheader("Grouped Composition Risks")
                _render_grouped_bottlenecks(workflow, bottlenecks)

    with tabs[1]:
        if bottlenecks.empty:
            st.info("No response time, throughput, or availability composition risks could be inferred from available workflow metrics.")
        else:
            st.plotly_chart(
                viz.build_bottleneck_figure(workflow, bottlenecks),
                use_container_width=True,
                config=_plotly_config(),
                key=f"{chart_key_prefix}_bottleneck",
            )
            _render_workflow_interpretation(workflow, bottlenecks)
            st.subheader("Grouped Composition Risk Details")
            _render_grouped_bottlenecks(workflow, bottlenecks)
            st.subheader("Raw Composition Risk Rows")
            display_cols = ["Bottleneck_Type", "API", "Subtask_ID", "Reason", "Metric", "Metric_Value", "Impact"]
            render_sticky_table(
                _bottleneck_display_frame(bottlenecks[[col for col in display_cols if col in bottlenecks.columns]]),
                height=240,
                sticky_columns=["QoS Risk Type", "API"],
                key=f"{chart_key_prefix}_raw_bottleneck_rows",
            )
        if not workflow.empty:
            _render_replacement_simulation(
                workflow=workflow,
                bottlenecks=bottlenecks,
                eval_row=eval_row,
                run_dir=run_dir,
                query_id=selected_query,
                mode=selected_mode,
                key_prefix=chart_key_prefix,
            )
            st.subheader("Selected API Metrics")
            render_sticky_table(
                _workflow_detail_view(workflow),
                sticky_columns=["Step", "API_Name", "API Name"],
                height=420,
                key=f"{chart_key_prefix}_selected_api_metrics",
            )

    with tabs[2]:
        status = recommendation.get("status")
        if status == "unavailable":
            st.warning(recommendation.get("warning") or "Recommendation unavailable.")
            _render_recommendation_summary(recommendation)
        elif status == "diagnostic":
            st.warning("No composition-complete workflow is available. Showing diagnostics only, not a recommended path.")
            _render_recommendation_summary(recommendation)
        else:
            _render_recommendation_summary(recommendation)
            if recommendation.get("tradeoff_message"):
                st.info(f"Tradeoff: {recommendation['tradeoff_message']}")

        if status == "recommended" and recommended_workflow.empty:
            st.warning(f"The recommended mode `{recommended_mode}` has no workflow rows.")
        if status == "recommended" and not recommended_workflow.empty:
            chart_cols = st.columns([1.05, 1])
            with chart_cols[0]:
                st.plotly_chart(
                    viz.build_quality_score_figure(recommended_eval_row, recommended_workflow),
                    use_container_width=True,
                    config=_plotly_config(),
                    key=f"{recommendation_key_prefix}_quality_scores",
                )
            with chart_cols[1]:
                _render_workflow_interpretation(recommended_workflow, recommended_bottlenecks)
                _render_quality_legend()
                _render_qos_health_warnings(recommended_workflow)
            st.plotly_chart(
                viz.build_workflow_figure(
                    query_context=query_context,
                    workflow=recommended_workflow,
                    mode=recommended_mode,
                    eval_row=recommended_eval_row,
                ),
                use_container_width=True,
                config=_plotly_config(),
                key=f"{recommendation_key_prefix}_workflow",
            )
            st.caption(
                "This graph shows the planned composition structure. Subtasks are shown vertically, "
                "and the selected API for each subtask is shown to the right."
            )
            if not recommended_bottlenecks.empty:
                st.subheader("Grouped Composition Risks")
                _render_grouped_bottlenecks(recommended_workflow, recommended_bottlenecks)
            st.subheader("Recommended Composition Path Steps")
            step_cols = [
                "Step",
                "Subtask_ID",
                "Subtask",
                "API_Name",
                "Functional_Match",
                "rt_s",
                "tp_kbps",
                "availability",
                "Mode_Rank",
                "API_QoS_Health",
                "API_QoS_Health_Source",
                "API_Selection_Health",
                "API_Health_Label",
                "API_Risk_Label",
                "Functional_Risk",
                "QoS_Risk",
                "Composition_Risk",
                "QoS_LLM_Score",
                "TOPSIS_Score",
                "Bottleneck_Dimensions",
            ]
            path_steps = recommended_workflow[[col for col in step_cols if col in recommended_workflow.columns]].copy()
            if "Functional_Match" in path_steps:
                path_steps["Functional_Match"] = path_steps["Functional_Match"].apply(viz.format_functional_fit)
            for health_col in ["API_QoS_Health", "API_Selection_Health"]:
                if health_col in path_steps:
                    path_steps[health_col] = path_steps[health_col].apply(viz.format_api_health)
            render_sticky_table(
                _display_frame(path_steps).rename(
                    columns={
                        "Subtask_ID": "Subtask ID",
                        "Functional_Match": "Functional Fit",
                        "rt_s": viz.RESPONSE_TIME_LABEL,
                        "tp_kbps": viz.THROUGHPUT_LABEL,
                        "availability": "Availability",
                        "Mode_Rank": "Mode Rank",
                        "API_QoS_Health": "API QoS Health",
                        "API_QoS_Health_Source": "API QoS Health Source",
                        "API_Selection_Health": "API Selection Health",
                        "API_Health_Label": "API Health",
                        "API_Risk_Label": "QoS Risk",
                        "Functional_Risk": "Functional Risk",
                        "QoS_Risk": "QoS Risk",
                        "Composition_Risk": "Composition Risk",
                        "QoS_LLM_Score": "QoS LLM Score",
                        "TOPSIS_Score": "TOPSIS Score",
                        "Bottleneck_Dimensions": "QoS Risk Dimensions",
                    }
                ),
                sticky_columns=["Step", "API_Name", "API Name"],
                height=480,
                key=f"{recommendation_key_prefix}_path_steps",
            )
            _render_step_cards(recommended_workflow)

    with tabs[3]:
        _render_mode_comparison(
            eval_df=eval_df,
            workflow_df=workflow_df,
            query_context=query_context,
            query_id=selected_query,
            run_dir=run_dir,
            run_name=run_name,
            modes=available_modes,
        )

    with tabs[4]:
        _render_observability_diagnostics(
            eval_df=eval_df,
            workflow=workflow,
            eval_row=eval_row,
            query_context=query_context,
            query_id=selected_query,
            run_dir=run_dir,
            run_name=run_name,
            mode=selected_mode,
            recommended_mode=recommended_mode,
        )

    with tabs[5]:
        if workflow.empty:
            st.info("No workflow steps are available for sequence visualization.")
        else:
            st.subheader("Planned API Composition Sequence")
            st.caption(
                "This view represents the planned API composition sequence generated by the planner. "
                "It does not imply that external APIs were executed during visualization."
            )
            agent_tab, api_tab, graphviz_tab, dot_tab, mermaid_tab = st.tabs(
                ["Planned Agent Flow", "Planned API Flow", "Graphviz Diagram", "DOT Source", "Mermaid Source"]
            )
            with agent_tab:
                st.plotly_chart(
                    viz.build_sequence_figure(workflow, kind="agent"),
                    use_container_width=True,
                    config=_plotly_config(),
                    key=f"{chart_key_prefix}_sequence_agent",
                )
            with api_tab:
                st.plotly_chart(
                    viz.build_sequence_figure(workflow, kind="api"),
                    use_container_width=True,
                    config=_plotly_config(),
                    key=f"{chart_key_prefix}_sequence_api",
                )
            with graphviz_tab:
                st.caption("Rendered Graphviz view using a top-down layout so the planned sequence remains readable.")
                graphviz_agent_tab, graphviz_api_tab = st.tabs(["Planned Agent Flow", "Planned API Flow"])
                with graphviz_agent_tab:
                    st.graphviz_chart(viz.build_agent_sequence_dot(workflow, rankdir="TB"), use_container_width=True)
                with graphviz_api_tab:
                    st.graphviz_chart(viz.build_planned_api_flow_dot(workflow, rankdir="TB"), use_container_width=True)
            with dot_tab:
                st.caption("Copyable Graphviz DOT source for the rendered diagrams.")
                agent_dot_tab, api_dot_tab = st.tabs(["Planned Agent Flow DOT", "Planned API Flow DOT"])
                with agent_dot_tab:
                    st.code(viz.build_agent_sequence_dot(workflow, rankdir="TB"), language="dot")
                with api_dot_tab:
                    st.code(viz.build_planned_api_flow_dot(workflow, rankdir="TB"), language="dot")
            with mermaid_tab:
                st.code(viz.build_agent_mermaid(workflow), language="mermaid")
                st.code(viz.build_api_mermaid(workflow), language="mermaid")

    with tabs[6]:
        st.subheader("Enriched Workflow Rows")
        render_sticky_table(
            _workflow_detail_view(workflow),
            sticky_columns=["Step", "API_Name", "API Name"],
            height=520,
            key=f"{chart_key_prefix}_raw_enriched_workflow",
        )
        st.subheader("Composition Evaluation Row")
        render_sticky_table(
            _composition_display_frame(pd.DataFrame([eval_row])),
            sticky_columns=["Query_ID", "Mode"],
            height=260,
            key=f"{chart_key_prefix}_raw_eval_row",
        )
        st.subheader("Composition Risk Rows")
        render_sticky_table(
            _bottleneck_display_frame(bottlenecks),
            sticky_columns=["QoS Risk Type", "API"],
            height=420,
            key=f"{chart_key_prefix}_raw_bottlenecks",
        )


def render_ranking_evaluation() -> None:
    st.title("AutoLLMCompose Ranking Evaluation")

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
        discovered_rows, discovery_warnings = _discover_query_runs(parent_dir)
        for warning in discovery_warnings:
            st.warning(warning)
        query_options = sorted({str(row["query_id"]) for row in discovered_rows if row.get("query_id")})
        query_filter_key = "ranking_eval_shared_query_filter"
        query_options_signature_key = f"{query_filter_key}_options"
        if st.session_state.get(query_options_signature_key) != query_options:
            st.session_state[query_filter_key] = query_options
            st.session_state[query_options_signature_key] = query_options
        else:
            current_queries = st.session_state.get(query_filter_key, query_options)
            sanitized_queries = [query for query in current_queries if query in query_options]
            if sanitized_queries != current_queries or not sanitized_queries:
                st.session_state[query_filter_key] = query_options
        selected_queries_common = st.multiselect(
            "Query ID",
            query_options,
            default=query_options,
            help="Shared query filter for ranking similarity and composition workflow reports.",
            key=query_filter_key,
        )
        st.caption(K_HELP)
        if st.button("Reload reports"):
            _load_bundle.clear()
            _load_composition_reports.clear()

    if len(selected_modes) < 2:
        st.warning("Select at least two modes.")
        return

    composition_tab, ranking_tab = st.tabs(["Composition Workflow Evaluation", "Ranking Similarity Evaluation"])
    with composition_tab:
        _render_composition_evaluation(parent_dir, list(selected_modes), list(selected_queries_common))

    with ranking_tab:
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
                    render_sticky_table(
                        bundle.invalid_cases,
                        sticky_columns=["query_id", "subtask_id"],
                        height=420,
                        key="invalid_cases_empty_bundle",
                    )
            return

        with st.sidebar:
            st.header("Filters")
            query_options = sorted(cases_df["query_id"].unique().tolist())
            selected_queries = [query for query in selected_queries_common if query in query_options] or query_options
            st.caption("Using shared Query ID filter: " + ", ".join(selected_queries))

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
        pairwise_view = _filter_pairwise(filtered_pairwise, selected_metrics, selected_modes)

        fallback_count = sum(case.k_fallback_used for case in filtered_cases)
        included_pairwise = (
            int(pairwise_view["included_cases"].sum())
            if "included_cases" in pairwise_view.columns and not pairwise_view.empty
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

        report_files = build_ranking_eval_report_files(
            cases=filtered_cases,
            matrices=filtered_matrices,
            pairwise_counts=filtered_counts,
            raw_rows=bundle.raw_rows,
            invalid_cases=bundle.invalid_cases,
            warnings=bundle.warnings,
            discovered_run_dirs=bundle.discovered_run_dirs,
            loaded_report_paths=bundle.loaded_report_paths,
            inclusion_policy=bundle.inclusion_policy,
            selected_modes=selected_modes,
            selected_metrics=selected_metrics,
            selected_queries=selected_queries,
            selected_subtasks=selected_subtasks,
            parent_runs_dir=parent_dir,
            p=rbo_p,
        )
        st.download_button(
            "Download Filtered Ranking Reports",
            data=_build_deterministic_zip(report_files),
            file_name="ranking_eval_filtered_reports.zip",
            mime="application/zip",
            key="download_filtered_ranking_reports",
            help=(
                "Exports the current filtered ranking-similarity report set as CSV/JSON files: "
                "metric matrices, included counts, pairwise scores, included cases, invalid cases, loaded rows, warnings, and summary."
            ),
        )

    if not bundle.invalid_cases.empty:
        st.subheader("Invalid Evaluation Cases")
        invalid_cols = st.columns(3)
        invalid_cols[0].dataframe(
            bundle.invalid_cases.groupby("mode", dropna=False).size().reset_index(name="count"),
            use_container_width=True,
            hide_index=True,
        )
        invalid_cols[1].dataframe(
            bundle.invalid_cases.groupby("failure_reason", dropna=False).size().reset_index(name="count"),
            use_container_width=True,
            hide_index=True,
        )
        invalid_cols[2].dataframe(
            bundle.invalid_cases.groupby(["query_id", "subtask_id"], dropna=False).size().reset_index(name="count"),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Excluded invalid mode/subtask rows", expanded=False):
            render_sticky_table(
                bundle.invalid_cases,
                sticky_columns=["query_id", "subtask_id"],
                height=420,
                key="excluded_invalid_cases",
            )

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
            use_container_width=True,
        )

    st.subheader("Pairwise Scores")
    pairwise_table = pairwise_view.copy()
    if not pairwise_table.empty:
        pairwise_table["metric"] = pairwise_table["metric"].map(METRIC_LABELS)
    render_sticky_table(
        pairwise_table.round({"score": 4}),
        sticky_columns=["mode_a", "mode_b"],
        height=420,
        key="pairwise_scores",
    )

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
        st.plotly_chart(fig, use_container_width=True)

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
    render_sticky_table(
        top_lists_to_wide_frame(selected_case),
        sticky_columns=["rank"],
        height=420,
        key=f"top_lists_{selected_case_id}",
    )

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
        use_container_width=True,
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
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="AutoLLMCompose Dashboard", layout="wide")
    with st.sidebar:
        page = st.radio(
            "Page",
            [
                "Ranking Evaluation",
                "Composition Visualizations",
                "Thesis Results Figure Generator",
                "Run Experiments",
                "Completed Runs",
            ],
            index=0,
        )

    if page == "Ranking Evaluation":
        render_ranking_evaluation()
    elif page == "Composition Visualizations":
        render_composition_visualizations()
    elif page == "Thesis Results Figure Generator":
        render_thesis_results_figure_generator(
            directory_selector=_render_directory_selector,
            default_run_dir=DEFAULT_RUN_EXPLORER_PARENT / "RUNS_MAY_31_NEW_5/fireworks_gpt-oss-120b",
        )
    elif page == "Run Experiments":
        render_query_run_explorer()
    else:
        render_completed_runs()


if __name__ == "__main__":
    main()
