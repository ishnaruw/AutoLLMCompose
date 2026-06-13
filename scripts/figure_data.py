#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


MODE_ORDER = ("no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid")
MODE_LABELS = {
    "no_qos": "No-QoS",
    "qos_pure_llm": "QoS-Pure-LLM",
    "qos_topsis": "QoS-TOPSIS",
    "qos_hybrid": "QoS-Hybrid",
}
MODE_COLORS = {
    "no_qos": "#0B7DB4",
    "qos_pure_llm": "#E9A300",
    "qos_topsis": "#0AA177",
    "qos_hybrid": "#C875A7",
}
QUERY_DIR_RE = re.compile(r"^(q\d{2})_")
QUERY_ID_RE = re.compile(r"^q?(\d+)$", re.IGNORECASE)
SUMMARY_SCORE_FIELD = "QoS_Adjusted_Composition_Score"
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class QueryRun:
    query_id: str
    path: Path


@dataclass(frozen=True)
class ModeResult:
    query_id: str
    query_dir: Path
    mode: str
    score: float
    functional_coverage: float
    normalized_qos: float
    response_time_s: float | None
    throughput_kbps: float | None
    availability: float | None
    source_path: Path


@dataclass(frozen=True)
class AggregateResult:
    mode: str
    query_count: int
    mean_score: float
    std_score: float
    mean_functional_coverage: float
    mean_normalized_qos: float
    mean_response_time_s: float
    mean_throughput_kbps: float
    mean_availability: float
    component_query_count: int


def configure_matplotlib() -> None:
    cache_root = Path(tempfile.gettempdir()) / "autollmcompose_figures"
    mpl_cache_dir = cache_root / "matplotlib"
    xdg_cache_dir = cache_root / "xdg"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir.as_posix())
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache_dir.as_posix())


def load_pyplot():
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.max_open_warning": 0,
        }
    )
    return plt


def resolve_run_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Run folder does not exist or is not a directory: {path}")
    return path.resolve()


def figures_dir(run_root: Path, output_dir: str | Path | None = None) -> Path:
    if output_dir is None:
        path = run_root / "figures"
    else:
        path = Path(output_dir).expanduser()
        if not path.is_absolute():
            path = run_root / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def parse_query_id(value: str | int) -> str:
    if isinstance(value, int):
        number = value
    else:
        text = str(value).strip()
        match = QUERY_ID_RE.fullmatch(text)
        if not match:
            raise ValueError(f"Invalid query id {value!r}; use q01, 01, or 1.")
        number = int(match.group(1))
    if number <= 0:
        raise ValueError(f"Query id must be positive: {value!r}")
    return f"q{number:02d}"


def parse_query_ids(value: str) -> tuple[str, ...]:
    query_ids: list[str] = []
    seen: set[str] = set()
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        query_id = parse_query_id(raw)
        if query_id not in seen:
            query_ids.append(query_id)
            seen.add(query_id)
    if not query_ids:
        raise ValueError("At least one query id is required.")
    return tuple(query_ids)


def query_sort_key(query_id: str) -> tuple[int, str]:
    try:
        return (int(query_id[1:]), query_id)
    except ValueError:
        return (10_000, query_id)


def discover_query_runs(run_root: Path) -> list[QueryRun]:
    runs: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in run_root.iterdir():
        if not path.is_dir():
            continue
        match = QUERY_DIR_RE.match(path.name)
        if not match:
            continue
        query_id = match.group(1)
        if query_id in runs:
            duplicates.setdefault(query_id, [runs[query_id]]).append(path)
        else:
            runs[query_id] = path
    if duplicates:
        details = "; ".join(
            f"{query_id}: {', '.join(str(path) for path in paths)}"
            for query_id, paths in sorted(duplicates.items(), key=lambda item: query_sort_key(item[0]))
        )
        raise ValueError(f"Ambiguous duplicate query folders under {run_root}: {details}")
    if not runs:
        raise ValueError(f"No qXX_* query folders found under {run_root}")
    return [QueryRun(query_id, runs[query_id]) for query_id in sorted(runs, key=query_sort_key)]


def query_run_by_id(run_root: Path, query_id: str) -> QueryRun:
    runs = {run.query_id: run for run in discover_query_runs(run_root)}
    if query_id not in runs:
        available = ", ".join(sorted(runs, key=query_sort_key))
        raise ValueError(f"Query {query_id} was not found under {run_root}. Available: {available}")
    return runs[query_id]


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def finite_float(value: Any, field: str, source_path: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{source_path} has boolean {field}: {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_path} has non-numeric {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{source_path} has non-finite {field}: {value!r}")
    return number


def optional_finite_float(value: Any, field: str, source_path: Path) -> float | None:
    if value is None or value == "":
        return None
    return finite_float(value, field, source_path)


def summary_path(query_run: QueryRun) -> Path:
    return query_run.path / "evaluation" / f"query_{query_run.query_id}_composition_qos_eval_summary.json"


def load_query_results(query_run: QueryRun) -> list[ModeResult]:
    path = summary_path(query_run)
    if not path.exists():
        raise FileNotFoundError(f"Missing composition summary for {query_run.query_id}: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Composition summary must be a JSON object: {path}")
    if payload.get("query_id") not in (None, query_run.query_id):
        raise ValueError(f"{path} query_id is {payload.get('query_id')!r}; expected {query_run.query_id!r}")
    raw_rows = payload.get("summary_rows")
    if not isinstance(raw_rows, list):
        raise ValueError(f"{path} does not contain summary_rows.")

    rows_by_mode: dict[str, ModeResult] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ValueError(f"{path} contains a non-object summary row.")
        mode = raw_row.get("Mode")
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError(f"{path} contains a summary row without a valid Mode.")
        mode = mode.strip()
        if mode in rows_by_mode:
            raise ValueError(f"{path} contains duplicate mode row: {mode}")
        required = (
            SUMMARY_SCORE_FIELD,
            "Functional_Coverage",
            "Normalized_QoS_Score",
        )
        missing = [field for field in required if field not in raw_row]
        if missing:
            raise ValueError(f"{path} mode {mode} is missing fields: {', '.join(missing)}")
        rows_by_mode[mode] = ModeResult(
            query_id=query_run.query_id,
            query_dir=query_run.path,
            mode=mode,
            score=finite_float(raw_row[SUMMARY_SCORE_FIELD], SUMMARY_SCORE_FIELD, path),
            functional_coverage=finite_float(raw_row["Functional_Coverage"], "Functional_Coverage", path),
            normalized_qos=finite_float(raw_row["Normalized_QoS_Score"], "Normalized_QoS_Score", path),
            response_time_s=optional_finite_float(raw_row.get("Total_Response_Time_s"), "Total_Response_Time_s", path),
            throughput_kbps=optional_finite_float(
                raw_row.get("Bottleneck_Throughput_kbps"),
                "Bottleneck_Throughput_kbps",
                path,
            ),
            availability=optional_finite_float(
                raw_row.get("Average_Workflow_Availability"),
                "Average_Workflow_Availability",
                path,
            ),
            source_path=path,
        )

    missing_modes = [mode for mode in MODE_ORDER if mode not in rows_by_mode]
    if missing_modes:
        raise ValueError(f"{path} is missing required modes: {', '.join(missing_modes)}")
    return [rows_by_mode[mode] for mode in MODE_ORDER]


def load_all_results(run_root: Path) -> list[ModeResult]:
    rows: list[ModeResult] = []
    for query_run in discover_query_runs(run_root):
        rows.extend(load_query_results(query_run))
    return rows


def aggregate_results(rows: Sequence[ModeResult]) -> list[AggregateResult]:
    grouped = group_results_by_query(rows)
    component_query_ids = []
    for query_id, query_rows in grouped.items():
        if all(
            row.response_time_s is not None
            and row.throughput_kbps is not None
            and row.availability is not None
            for row in query_rows
        ):
            component_query_ids.append(query_id)
    if not component_query_ids:
        raise ValueError("No queries have complete QoS component metrics across all modes.")

    aggregates: list[AggregateResult] = []
    for mode in MODE_ORDER:
        mode_rows = [row for row in rows if row.mode == mode]
        if not mode_rows:
            raise ValueError(f"No rows available for mode {mode}")
        scores = [row.score for row in mode_rows]
        component_rows = [row for row in mode_rows if row.query_id in component_query_ids]
        aggregates.append(
            AggregateResult(
                mode=mode,
                query_count=len(mode_rows),
                mean_score=statistics.fmean(scores),
                std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
                mean_functional_coverage=statistics.fmean(row.functional_coverage for row in mode_rows),
                mean_normalized_qos=statistics.fmean(row.normalized_qos for row in mode_rows),
                mean_response_time_s=statistics.fmean(
                    row.response_time_s for row in component_rows if row.response_time_s is not None
                ),
                mean_throughput_kbps=statistics.fmean(
                    row.throughput_kbps for row in component_rows if row.throughput_kbps is not None
                ),
                mean_availability=statistics.fmean(
                    row.availability for row in component_rows if row.availability is not None
                ),
                component_query_count=len(component_rows),
            )
        )
    counts = {item.query_count for item in aggregates}
    if len(counts) != 1:
        raise ValueError(f"Modes have unequal query counts: {counts}")
    return aggregates


def best_modes_for_query(results: Sequence[ModeResult]) -> tuple[str, ...]:
    if not results:
        raise ValueError("Cannot compute best modes for an empty query.")
    best = max(row.score for row in results)
    return tuple(
        mode
        for mode in MODE_ORDER
        for row in results
        if row.mode == mode and math.isclose(row.score, best, rel_tol=TIE_TOLERANCE, abs_tol=TIE_TOLERANCE)
    )


def group_results_by_query(rows: Sequence[ModeResult]) -> dict[str, list[ModeResult]]:
    grouped: dict[str, list[ModeResult]] = {}
    for row in rows:
        grouped.setdefault(row.query_id, []).append(row)
    return {
        query_id: sorted(grouped[query_id], key=lambda row: MODE_ORDER.index(row.mode))
        for query_id in sorted(grouped, key=query_sort_key)
    }


def selected_functional_match(query_run: QueryRun, mode: str, subtask_id: str, api_id: str) -> int | None:
    rows_path = query_run.path / "evaluation" / f"query_{query_run.query_id}_candidate_api_rankings_rows.json"
    if not rows_path.exists():
        return None
    payload = read_json(rows_path)
    if not isinstance(payload, list):
        raise ValueError(f"Candidate ranking rows must be a JSON list: {rows_path}")
    matches = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        if row.get("Mode") != mode:
            continue
        if str(row.get("Sub Task")) != str(subtask_id):
            continue
        if row.get("Selected_API") != api_id:
            continue
        matches.append(row)
    if not matches:
        return None
    matches.sort(key=lambda row: finite_float(row.get("Mode Rank", 10_000), "Mode Rank", rows_path))
    value = matches[0].get("Functional Match (0/1)")
    if value in (0, 1):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{rows_path} has invalid Functional Match value for {mode}/{subtask_id}/{api_id}") from exc


def load_csv_matrix(path: Path) -> tuple[list[str], list[list[float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ranking matrix CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"Ranking matrix CSV has no header: {path}")
    modes = [cell.strip() for cell in rows[0][1:]]
    matrix: list[list[float]] = []
    row_labels: list[str] = []
    for row in rows[1:]:
        if len(row) != len(modes) + 1:
            raise ValueError(f"Malformed row in {path}: {row}")
        row_labels.append(row[0].strip())
        matrix.append([finite_float(cell, "matrix value", path) for cell in row[1:]])
    if row_labels != modes:
        raise ValueError(f"{path} row labels {row_labels} do not match column labels {modes}")
    missing_modes = [mode for mode in MODE_ORDER if mode not in modes]
    if missing_modes:
        raise ValueError(f"{path} is missing modes: {', '.join(missing_modes)}")
    order = [modes.index(mode) for mode in MODE_ORDER]
    reordered = [[matrix[row_index][col_index] for col_index in order] for row_index in order]
    return list(MODE_ORDER), reordered


def safe_filename_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return token or "figure"


def query_span_token(query_ids: Sequence[str]) -> str:
    ordered = sorted(dict.fromkeys(query_ids), key=query_sort_key)
    if len(ordered) == 1:
        return ordered[0]
    numbers = [int(query_id[1:]) for query_id in ordered]
    if numbers == list(range(numbers[0], numbers[-1] + 1)):
        return f"{ordered[0]}_{ordered[-1]}"
    return "_".join(ordered)
