from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.core.runtime_bootstrap import harden_scientific_runtime

harden_scientific_runtime()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

PREFERRED_MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
SECTION_511_QUERY_IDS = [f"q{idx:02d}" for idx in range(1, 16)]
SECTION_511_MODE_ORDER = PREFERRED_MODE_ORDER.copy()
WEIGHT_SETTINGS: tuple[dict[str, Any], ...] = (
    {"alpha": 1.00, "beta": 0.00, "label": "100 functional / 0 QoS"},
    {"alpha": 0.75, "beta": 0.25, "label": "75 functional / 25 QoS"},
    {"alpha": 0.70, "beta": 0.30, "label": "70 functional / 30 QoS"},
    {"alpha": 0.50, "beta": 0.50, "label": "50 functional / 50 QoS"},
    {"alpha": 0.30, "beta": 0.70, "label": "30 functional / 70 QoS"},
    {"alpha": 0.00, "beta": 1.00, "label": "0 functional / 100 QoS"},
)
RANKING_MATRIX_FILES = {
    "spearman": "spearman_matrix.csv",
    "average_overlap": "average_overlap_matrix.csv",
    "rbo": "rbo_matrix.csv",
    "jaccard": "jaccard_matrix.csv",
}
RANKING_METRIC_LABELS = {
    "spearman": "Spearman",
    "average_overlap": "Average Overlap@K",
    "rbo": "RBO",
    "jaccard": "Jaccard@K",
}
DEFAULT_SCORE_COL = "QoS_Adjusted_Composition_Score"
TOLERANCE = 1e-9
FIGURE_5_2_MODE_ORDER = ["qos_hybrid", "qos_pure_llm", "no_qos", "qos_topsis"]
FIGURE_5_2_REQUIRED_COLUMNS = [
    "Mode",
    "Query_Count",
    "Average_QoS_Adjusted_Composition_Score",
    "Std_QoS_Adjusted_Composition_Score",
]
FIGURE_5_7_MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
FIGURE_5_7_REQUIRED_COLUMNS = [
    "Mode",
    "Query_Count",
    "Average_Functional_Coverage",
    "Average_Normalized_QoS_Score",
]
FIGURE_5_7_OFFICIAL_POINTS = {
    "no_qos": (0.955556, 0.525717),
    "qos_pure_llm": (1.000000, 0.822242),
    "qos_topsis": (0.238889, 0.790922),
    "qos_hybrid": (1.000000, 0.967079),
}
FIGURE_5_8_MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
FIGURE_5_8_METRIC_ORDER = ["spearman", "average_overlap", "rbo", "jaccard"]
FIGURE_5_10_TITLE = "Hybrid Mode Rationale and Evidence Flow"
SECTION_55_QUERY_ID = "q02"
SECTION_55_EXPECTED_STEPS = 3
SECTION_55_SCORE_TRACE_COLUMNS = [
    "Mode",
    "Composition_Completeness",
    "Functional_Coverage",
    "Total_Response_Time_s",
    "Bottleneck_Throughput_kbps",
    "Average_Workflow_Availability",
    "Normalized_QoS_Score",
    "QoS_Adjusted_Composition_Score",
]
SECTION_55_FIGURE_COLUMNS = [
    "Mode",
    "Subtask 1",
    "Subtask 2",
    "Subtask 3",
    "Functional Coverage",
    "Normalized QoS",
    "Final Score",
]


@dataclass
class LoadedThesisArtifacts:
    run_dir: Path
    summary_dir: Path
    ranking_dir: Path
    aggregate_scores: pd.DataFrame
    composition_results: pd.DataFrame
    consolidation_report: dict[str, Any]
    summary_readme: str | None
    ranking_summary: dict[str, Any]
    pairwise_scores: pd.DataFrame
    matrices: dict[str, pd.DataFrame]
    included_cases: pd.DataFrame
    loaded_rows: pd.DataFrame
    invalid_cases: pd.DataFrame
    ranking_warnings: list[str]
    file_status: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.max_open_warning": 0,
        }
    )


def resolve_path(text: str | Path, project_root: Path) -> Path:
    raw = Path(str(text or "").strip()).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (project_root / raw).resolve()


def _relative_to(path: Path, base: Path) -> Path | None:
    try:
        return path.resolve().relative_to(base.resolve())
    except Exception:
        return None


def resolve_summary_dir(project_root: Path, run_dir: Path) -> Path:
    logs_root = project_root / "results" / "logs"
    summaries_root = project_root / "results" / "summaries"
    rel = _relative_to(run_dir, logs_root)
    candidates: list[Path] = []
    if rel is not None:
        candidates.extend(
            [
                summaries_root / rel / "q01_q15_official",
                summaries_root / rel,
            ]
        )
    candidates.extend([run_dir / "q01_q15_official", run_dir])
    for candidate in candidates:
        if (candidate / "all_15_query_composition_results.csv").exists():
            return candidate
    return candidates[0] if candidates else summaries_root


def resolve_ranking_eval_dir(project_root: Path, run_dir: Path, consolidation_report: dict[str, Any] | None = None) -> Path:
    candidates = [run_dir / "ranking_eval"]
    input_path = (consolidation_report or {}).get("input_path")
    if input_path:
        candidates.append(resolve_path(str(input_path), project_root) / "ranking_eval")
    logs_root = project_root / "results" / "logs"
    rel = _relative_to(run_dir, logs_root)
    if rel is not None:
        candidates.append(logs_root / rel / "ranking_eval")
    for candidate in candidates:
        if (candidate / "summary.json").exists() or (candidate / "loaded_rows.csv").exists():
            return candidate
    return candidates[0]


def load_csv_if_exists(path: Path, *, index_col: int | None = None) -> tuple[pd.DataFrame, str | None]:
    if not path.exists():
        return pd.DataFrame(), f"Missing file: {path}"
    try:
        return pd.read_csv(path, index_col=index_col), None
    except Exception as exc:
        return pd.DataFrame(), f"Could not read CSV {path}: {exc}"


def load_json_if_exists(path: Path) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    if not path.exists():
        return None, f"Missing file: {path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"Could not read JSON {path}: {exc}"


def require_columns(df: pd.DataFrame, required_columns: Iterable[str], label: str) -> list[str]:
    missing = [col for col in required_columns if col not in df.columns]
    return [f"{label}: missing required column `{col}`" for col in missing]


def normalize_mode_order(modes: Iterable[Any]) -> list[str]:
    unique = {str(mode) for mode in modes if str(mode).strip()}
    preferred = [mode for mode in PREFERRED_MODE_ORDER if mode in unique]
    remaining = sorted(unique.difference(preferred))
    return preferred + remaining


def sorted_query_ids(values: Iterable[Any]) -> list[str]:
    def key(value: Any) -> tuple[int, str]:
        text = str(value)
        if len(text) >= 2 and text[0].lower() == "q" and text[1:].isdigit():
            return int(text[1:]), text
        return 10_000, text

    return sorted({str(value) for value in values if str(value).strip()}, key=key)


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _truthy_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(series, errors="coerce")
    return text.isin({"1", "1.0", "true", "yes", "y"}) | (numeric == 1)


def _format_list(values: Iterable[Any]) -> str:
    cleaned = [str(value) for value in values if str(value).strip()]
    return ", ".join(cleaned)


def _file_status_row(label: str, path: Path, df: pd.DataFrame | None = None, warning: str | None = None) -> dict[str, Any]:
    return {
        "artifact": label,
        "path": str(path),
        "exists": path.exists(),
        "rows": None if df is None or df.empty else int(len(df)),
        "columns": None if df is None or df.empty else int(len(df.columns)),
        "warning": warning or "",
    }


def load_thesis_artifacts(
    project_root: Path,
    run_dir: Path,
    summary_dir: Path | None = None,
    ranking_dir: Path | None = None,
) -> LoadedThesisArtifacts:
    run_dir = run_dir.resolve()
    guessed_summary_dir = summary_dir or resolve_summary_dir(project_root, run_dir)
    aggregate_path = guessed_summary_dir / "aggregate_mode_scores.csv"
    composition_path = guessed_summary_dir / "all_15_query_composition_results.csv"
    report_path = guessed_summary_dir / "consolidation_report.json"
    readme_path = guessed_summary_dir / "README.md"

    warnings: list[str] = []
    status_rows: list[dict[str, Any]] = []

    aggregate_df, warning = load_csv_if_exists(aggregate_path)
    if warning:
        warnings.append(warning)
    status_rows.append(_file_status_row("aggregate_mode_scores.csv", aggregate_path, aggregate_df, warning))

    composition_df, warning = load_csv_if_exists(composition_path)
    if warning:
        warnings.append(warning)
    status_rows.append(_file_status_row("all_15_query_composition_results.csv", composition_path, composition_df, warning))

    report_payload, warning = load_json_if_exists(report_path)
    consolidation_report = report_payload if isinstance(report_payload, dict) else {}
    if warning:
        warnings.append(warning)
    status_rows.append(_file_status_row("consolidation_report.json", report_path, None, warning))

    summary_readme = None
    readme_warning = None
    if readme_path.exists():
        try:
            summary_readme = readme_path.read_text(encoding="utf-8")
        except Exception as exc:
            readme_warning = f"Could not read README {readme_path}: {exc}"
            warnings.append(readme_warning)
    else:
        readme_warning = f"Missing file: {readme_path}"
    status_rows.append(_file_status_row("README.md", readme_path, None, readme_warning))

    guessed_ranking_dir = ranking_dir or resolve_ranking_eval_dir(project_root, run_dir, consolidation_report)
    ranking_summary_payload, warning = load_json_if_exists(guessed_ranking_dir / "summary.json")
    ranking_summary = ranking_summary_payload if isinstance(ranking_summary_payload, dict) else {}
    if warning:
        warnings.append(warning)
    status_rows.append(_file_status_row("ranking_eval/summary.json", guessed_ranking_dir / "summary.json", None, warning))

    pairwise_scores, warning = load_csv_if_exists(guessed_ranking_dir / "pairwise_scores.csv")
    if warning:
        warnings.append(warning)
    status_rows.append(_file_status_row("ranking_eval/pairwise_scores.csv", guessed_ranking_dir / "pairwise_scores.csv", pairwise_scores, warning))

    included_cases, warning = load_csv_if_exists(guessed_ranking_dir / "included_cases.csv")
    if warning:
        warnings.append(warning)
    status_rows.append(_file_status_row("ranking_eval/included_cases.csv", guessed_ranking_dir / "included_cases.csv", included_cases, warning))

    loaded_rows, warning = load_csv_if_exists(guessed_ranking_dir / "loaded_rows.csv")
    if warning:
        warnings.append(warning)
    status_rows.append(_file_status_row("ranking_eval/loaded_rows.csv", guessed_ranking_dir / "loaded_rows.csv", loaded_rows, warning))

    invalid_cases, warning = load_csv_if_exists(guessed_ranking_dir / "invalid_cases.csv")
    if warning:
        warnings.append(warning)
    status_rows.append(_file_status_row("ranking_eval/invalid_cases.csv", guessed_ranking_dir / "invalid_cases.csv", invalid_cases, warning))

    ranking_warnings_payload, warning = load_json_if_exists(guessed_ranking_dir / "warnings.json")
    ranking_warnings = ranking_warnings_payload if isinstance(ranking_warnings_payload, list) else []
    if warning:
        warnings.append(warning)
    status_rows.append(_file_status_row("ranking_eval/warnings.json", guessed_ranking_dir / "warnings.json", None, warning))

    matrices: dict[str, pd.DataFrame] = {}
    for metric, filename in RANKING_MATRIX_FILES.items():
        matrix, warning = load_csv_if_exists(guessed_ranking_dir / filename, index_col=0)
        if warning:
            warnings.append(warning)
        else:
            matrices[metric] = matrix
        status_rows.append(_file_status_row(f"ranking_eval/{filename}", guessed_ranking_dir / filename, matrix, warning))

    return LoadedThesisArtifacts(
        run_dir=run_dir,
        summary_dir=guessed_summary_dir.resolve(),
        ranking_dir=guessed_ranking_dir.resolve(),
        aggregate_scores=aggregate_df,
        composition_results=composition_df,
        consolidation_report=consolidation_report,
        summary_readme=summary_readme,
        ranking_summary=ranking_summary,
        pairwise_scores=pairwise_scores,
        matrices=matrices,
        included_cases=included_cases,
        loaded_rows=loaded_rows,
        invalid_cases=invalid_cases,
        ranking_warnings=[str(item) for item in ranking_warnings],
        file_status=pd.DataFrame(status_rows),
        warnings=warnings,
    )


def official_query_ids(report: dict[str, Any], composition_df: pd.DataFrame) -> list[str]:
    for key in ("include_query_ids", "query_ids_found"):
        values = report.get(key)
        if isinstance(values, list) and values:
            return sorted_query_ids(values)
    if "Query_ID" in composition_df:
        return sorted_query_ids(composition_df["Query_ID"].dropna().astype(str))
    return []


def filter_composition(df: pd.DataFrame, query_ids: list[str], modes: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    filtered = df.copy()
    if "Query_ID" in filtered and query_ids:
        filtered = filtered[filtered["Query_ID"].astype(str).isin(query_ids)]
    if "Mode" in filtered and modes:
        filtered = filtered[filtered["Mode"].astype(str).isin(modes)]
    return filtered.copy()


def validation_summary(
    artifacts: LoadedThesisArtifacts,
    selected_query_ids: list[str],
    selected_modes: list[str],
) -> dict[str, pd.DataFrame]:
    composition = artifacts.composition_results
    composition_selected = filter_composition(composition, selected_query_ids, selected_modes)
    rows: list[dict[str, Any]] = []
    report = artifacts.consolidation_report

    queries_found = sorted_query_ids(composition["Query_ID"].dropna().astype(str)) if "Query_ID" in composition else []
    modes_found = normalize_mode_order(composition["Mode"].dropna().astype(str)) if "Mode" in composition else []
    missing_queries = [query_id for query_id in selected_query_ids if query_id not in queries_found]

    rows.extend(
        [
            {"check": "Summary directory", "value": str(artifacts.summary_dir)},
            {"check": "Ranking directory", "value": str(artifacts.ranking_dir)},
            {"check": "Query IDs found", "value": _format_list(queries_found)},
            {"check": "Modes found", "value": _format_list(modes_found)},
            {"check": "Selected query IDs missing from composition rows", "value": _format_list(missing_queries) or "None"},
            {
                "check": "q16-q18 official status",
                "value": q16_q18_status(report, composition),
            },
            {"check": "Ranking included cases", "value": len(artifacts.included_cases)},
            {"check": "Ranking invalid cases", "value": len(artifacts.invalid_cases)},
            {"check": "Ranking warnings", "value": len(artifacts.ranking_warnings)},
        ]
    )

    missing_mode_rows: list[dict[str, str]] = []
    if not composition_selected.empty and {"Query_ID", "Mode"}.issubset(composition_selected.columns):
        observed = set(zip(composition_selected["Query_ID"].astype(str), composition_selected["Mode"].astype(str)))
        for query_id in selected_query_ids:
            for mode in selected_modes:
                if (query_id, mode) not in observed:
                    missing_mode_rows.append({"Query_ID": query_id, "Mode": mode})

    duplicate_rows = pd.DataFrame()
    if not composition_selected.empty and {"Query_ID", "Mode"}.issubset(composition_selected.columns):
        duplicate_rows = composition_selected[
            composition_selected.duplicated(["Query_ID", "Mode"], keep=False)
        ].copy()

    malformed_rows = []
    for key in ("malformed_files", "missing_eval_files", "missing_requested_query_ids", "expected_modes_missing_by_query"):
        value = report.get(key)
        if value:
            malformed_rows.append({"artifact": key, "value": json.dumps(value, ensure_ascii=False)})
    for warning in artifacts.warnings:
        malformed_rows.append({"artifact": "loader_warning", "value": warning})

    return {
        "summary": pd.DataFrame(rows),
        "file_status": artifacts.file_status,
        "missing_mode_entries": pd.DataFrame(missing_mode_rows),
        "duplicate_query_mode_rows": duplicate_rows,
        "malformed_or_missing": pd.DataFrame(malformed_rows),
        "included_cases": artifacts.included_cases,
        "invalid_cases": artifacts.invalid_cases,
        "ranking_warnings": pd.DataFrame({"warning": artifacts.ranking_warnings}),
    }


def q16_q18_status(report: dict[str, Any], composition_df: pd.DataFrame) -> str:
    extended = {"q16", "q17", "q18"}
    found = set()
    if "Query_ID" in composition_df:
        found.update(composition_df["Query_ID"].dropna().astype(str))
    for key in ("query_ids_excluded_by_filter", "query_ids_found"):
        values = report.get(key)
        if isinstance(values, list):
            found.update(str(value) for value in values)
    excluded = set(str(value) for value in report.get("query_ids_excluded_by_filter", []) if str(value) in extended)
    included = sorted(extended.intersection(set(official_query_ids(report, composition_df))))
    visible = sorted(extended.intersection(found))
    if excluded:
        return f"Visible in run but excluded by official scope: {_format_list(sorted(excluded))}"
    if included:
        return f"Included in selected scope: {_format_list(included)}"
    if visible:
        return f"Visible in loaded rows: {_format_list(visible)}"
    return "Not present in loaded artifacts"


def build_official_scope_table(artifacts: LoadedThesisArtifacts) -> pd.DataFrame:
    report = artifacts.consolidation_report
    composition = artifacts.composition_results
    queries = official_query_ids(report, composition)
    modes = normalize_mode_order(report.get("modes_found", [])) or (
        normalize_mode_order(composition["Mode"].dropna().astype(str)) if "Mode" in composition else []
    )
    rows = [
        ("Summary directory", artifacts.summary_dir),
        ("Run/log directory", artifacts.run_dir),
        ("Ranking directory", artifacts.ranking_dir),
        ("Official query IDs", _format_list(queries)),
        ("Queries excluded by scope filter", _format_list(report.get("query_ids_excluded_by_filter", [])) or "None"),
        ("Modes found", _format_list(modes)),
        ("Composition rows loaded", len(composition)),
        ("Aggregate rows loaded", len(artifacts.aggregate_scores)),
        ("Duplicate query IDs", _format_list(report.get("duplicate_query_ids", [])) or "None"),
        ("Missing requested query IDs", _format_list(report.get("missing_requested_query_ids", [])) or "None"),
        ("Malformed files", len(report.get("malformed_files", []) or [])),
        ("Missing evaluation files", len(report.get("missing_eval_files", []) or [])),
        ("Ranking included cases", len(artifacts.included_cases)),
        ("Ranking invalid cases", len(artifacts.invalid_cases)),
        ("Ranking warnings", len(artifacts.ranking_warnings)),
        ("q16-q18 status", q16_q18_status(report, composition)),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def compute_unique_and_tied_best(
    query_df: pd.DataFrame,
    score_col: str,
    tolerance: float = TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"Query_ID", "Mode", score_col}
    if query_df.empty or not required.issubset(query_df.columns):
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for query_id, group in query_df.groupby("Query_ID", sort=True):
        scores = _as_numeric(group[score_col])
        valid = group.loc[scores.notna()].copy()
        valid[score_col] = scores[scores.notna()]
        if valid.empty:
            rows.append({"Query_ID": query_id, "Best_Score": np.nan, "Best_Modes": "", "Best_Status": "no numeric score"})
            continue
        max_score = float(valid[score_col].max())
        best_modes = normalize_mode_order(valid.loc[(max_score - valid[score_col]).abs() <= tolerance, "Mode"])
        rows.append(
            {
                "Query_ID": query_id,
                "Best_Score": max_score,
                "Best_Modes": ", ".join(best_modes),
                "Best_Status": "tied best" if len(best_modes) > 1 else "unique best",
            }
        )

    best_df = pd.DataFrame(rows)
    count_rows = []
    all_modes = normalize_mode_order(query_df["Mode"].dropna().astype(str))
    for mode in all_modes:
        unique_count = 0
        tied_count = 0
        for _, row in best_df.iterrows():
            best_modes = [part.strip() for part in str(row.get("Best_Modes") or "").split(",") if part.strip()]
            if mode not in best_modes:
                continue
            if len(best_modes) == 1:
                unique_count += 1
            else:
                tied_count += 1
        count_rows.append({"Mode": mode, "unique_best_count": unique_count, "tied_best_count": tied_count})
    return best_df, pd.DataFrame(count_rows)


def recompute_score_sensitivity(composition_df: pd.DataFrame, alpha: float, beta: float) -> pd.DataFrame:
    result = composition_df.copy()
    score_col = f"score_alpha_{alpha:.2f}_beta_{beta:.2f}"
    result[score_col] = (
        _as_numeric(result["Composition_Completeness"])
        * (alpha * _as_numeric(result["Functional_Coverage"]) + beta * _as_numeric(result["Normalized_QoS_Score"]))
    )
    return result


def build_weight_sensitivity_tables(
    composition_df: pd.DataFrame,
    selected_query_ids: list[str],
    selected_modes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    required = ["Query_ID", "Mode", "Composition_Completeness", "Functional_Coverage", "Normalized_QoS_Score"]
    warnings = require_columns(composition_df, required, "Weight sensitivity")
    if warnings:
        return pd.DataFrame(), pd.DataFrame(), warnings

    filtered = filter_composition(composition_df, selected_query_ids, selected_modes)
    rows: list[pd.DataFrame] = []
    for idx, setting in enumerate(WEIGHT_SETTINGS, start=1):
        scored = recompute_score_sensitivity(filtered, setting["alpha"], setting["beta"])
        score_col = f"score_alpha_{setting['alpha']:.2f}_beta_{setting['beta']:.2f}"
        view = scored[["Query_ID", "Mode", score_col]].copy()
        view = view.rename(columns={score_col: "Sensitivity_Score"})
        view["alpha"] = setting["alpha"]
        view["beta"] = setting["beta"]
        view["Weight_Setting"] = setting["label"]
        view["Weight_Order"] = idx
        rows.append(view)
    long_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary = (
        long_df.groupby(["Weight_Order", "Weight_Setting", "alpha", "beta", "Mode"], dropna=False)
        .agg(
            Query_Count=("Query_ID", "nunique"),
            Average_Sensitivity_Score=("Sensitivity_Score", "mean"),
            Std_Sensitivity_Score=("Sensitivity_Score", "std"),
        )
        .reset_index()
    )
    summary["Mode"] = pd.Categorical(summary["Mode"], categories=normalize_mode_order(summary["Mode"]), ordered=True)
    summary = summary.sort_values(["Weight_Order", "Mode"]).reset_index(drop=True)
    return long_df, summary, warnings


def compute_candidate_qos_score(candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = ["query_id", "subtask_id", "api_id", "functional_match_label"]
    warnings = require_columns(candidate_df, required, "Candidate QoS sensitivity")
    if warnings:
        return pd.DataFrame(), warnings

    qos_cols = {
        "QoS_RT_s": "response_time_score",
        "QoS_TP_kbps": "throughput_score",
        "QoS Availability": "availability_score",
    }
    missing_qos = [col for col in qos_cols if col not in candidate_df.columns]
    if missing_qos:
        warnings.append(
            "Candidate QoS sensitivity: missing QoS dimensions ignored: " + ", ".join(missing_qos)
        )

    working = candidate_df.copy()
    if "mode_rank" in working:
        working["_sort_rank"] = _as_numeric(working["mode_rank"])
    else:
        working["_sort_rank"] = np.nan
    sort_cols = [col for col in ["query_id", "subtask_id", "api_id", "mode", "_sort_rank"] if col in working.columns]
    working = working.sort_values(sort_cols, kind="mergesort")
    for col in ["functional_match_label", *qos_cols.keys()]:
        if col in working:
            working[col] = _as_numeric(working[col])

    conflict_cols = [col for col in ["functional_match_label", *qos_cols.keys()] if col in working]
    conflict_counts = (
        working.groupby(["query_id", "subtask_id", "api_id"], dropna=False)[conflict_cols]
        .nunique(dropna=True)
        .reset_index()
    )
    for col in conflict_cols:
        conflicts = conflict_counts[conflict_counts[col] > 1]
        if not conflicts.empty:
            warnings.append(
                f"Candidate QoS sensitivity: {len(conflicts)} deduplicated candidates have conflicting `{col}` values; using the maximum numeric value."
            )

    aggregations: dict[str, str] = {"functional_match_label": "max"}
    for col in qos_cols:
        if col in working:
            aggregations[col] = "max"
    deduped = (
        working.groupby(["query_id", "subtask_id", "api_id"], dropna=False)
        .agg(aggregations)
        .reset_index()
    )

    score_columns: list[str] = []
    for source_col, score_col in qos_cols.items():
        if source_col not in deduped:
            continue
        values = deduped[source_col]
        group_keys = [deduped["query_id"], deduped["subtask_id"]]
        group_min = values.groupby(group_keys).transform("min")
        group_max = values.groupby(group_keys).transform("max")
        spread = group_max - group_min
        if source_col == "QoS_RT_s":
            normalized = (group_max - values) / spread
        else:
            normalized = (values - group_min) / spread
        normalized = normalized.where(spread != 0, 1.0)
        normalized = normalized.where(values.notna(), np.nan)
        deduped[score_col] = normalized
        score_columns.append(score_col)

    if not score_columns:
        warnings.append("Candidate QoS sensitivity: no QoS dimensions were available.")
        deduped["candidate_qos_score"] = np.nan
    else:
        missing_any = deduped[score_columns].isna().any(axis=1).sum()
        if missing_any:
            warnings.append(
                f"Candidate QoS sensitivity: {int(missing_any)} candidates have missing normalized QoS dimensions; available dimensions are averaged."
            )
        deduped["candidate_qos_score"] = deduped[score_columns].mean(axis=1, skipna=True)
    return deduped, warnings


def build_candidate_top10_sensitivity(
    loaded_rows: pd.DataFrame,
    selected_query_ids: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if loaded_rows.empty:
        return pd.DataFrame(), ["Candidate QoS sensitivity: ranking_eval/loaded_rows.csv is unavailable."]
    filtered = loaded_rows.copy()
    if "query_id" in filtered and selected_query_ids:
        filtered = filtered[filtered["query_id"].astype(str).isin(selected_query_ids)]
    candidates, warnings = compute_candidate_qos_score(filtered)
    if candidates.empty:
        return pd.DataFrame(), warnings

    rows: list[dict[str, Any]] = []
    for setting in WEIGHT_SETTINGS:
        scored = candidates.copy()
        scored["candidate_score_alpha_beta"] = (
            setting["alpha"] * _as_numeric(scored["functional_match_label"])
            + setting["beta"] * _as_numeric(scored["candidate_qos_score"])
        )
        scored = scored.sort_values(
            ["query_id", "subtask_id", "candidate_score_alpha_beta", "api_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        scored["candidate_rank"] = scored.groupby(["query_id", "subtask_id"], dropna=False).cumcount() + 1
        top10 = scored[scored["candidate_rank"] <= 10]
        for (query_id, subtask_id), group in top10.groupby(["query_id", "subtask_id"], dropna=False):
            invalid = group[_as_numeric(group["functional_match_label"]) != 1]
            rows.append(
                {
                    "Weight_Setting": setting["label"],
                    "alpha": setting["alpha"],
                    "beta": setting["beta"],
                    "query_id": query_id,
                    "subtask_id": subtask_id,
                    "top10_candidate_count": len(group),
                    "invalid_top10_candidate_count": len(invalid),
                    "invalid_top10_api_ids": "; ".join(invalid["api_id"].astype(str).tolist()),
                }
            )
    return pd.DataFrame(rows), warnings


def build_aggregate_performance_table(
    composition_df: pd.DataFrame,
    selected_query_ids: list[str],
    selected_modes: list[str],
    tolerance: float = TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    required = ["Query_ID", "Mode", DEFAULT_SCORE_COL]
    warnings = require_columns(composition_df, required, "Aggregate performance")
    if warnings:
        return pd.DataFrame(), pd.DataFrame(), warnings
    filtered = filter_composition(composition_df, selected_query_ids, selected_modes)
    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame(), ["Aggregate performance: no rows match the selected scope."]

    numeric_cols = [
        DEFAULT_SCORE_COL,
        "Functional_Coverage",
        "Normalized_QoS_Score",
        "Total_Response_Time_s",
        "Bottleneck_Throughput_kbps",
        "Average_Workflow_Availability",
        "Composition_Completeness",
        "Composition_Validity",
    ]
    for col in numeric_cols:
        if col in filtered:
            filtered[col] = _as_numeric(filtered[col])
        else:
            warnings.append(f"Aggregate performance: optional column `{col}` is unavailable.")

    best_df, counts_df = compute_unique_and_tied_best(filtered, DEFAULT_SCORE_COL, tolerance)
    rows: list[dict[str, Any]] = []
    for mode in normalize_mode_order(filtered["Mode"].dropna().astype(str)):
        group = filtered[filtered["Mode"].astype(str) == mode]
        row = {
            "Mode": mode,
            "Query_Count": int(group["Query_ID"].nunique()),
            "Average_QoS_Adjusted_Composition_Score": group[DEFAULT_SCORE_COL].mean() if DEFAULT_SCORE_COL in group else np.nan,
            "Std_QoS_Adjusted_Composition_Score": group[DEFAULT_SCORE_COL].std(ddof=1) if DEFAULT_SCORE_COL in group else np.nan,
            "Average_Functional_Coverage": group["Functional_Coverage"].mean() if "Functional_Coverage" in group else np.nan,
            "Average_Normalized_QoS_Score": group["Normalized_QoS_Score"].mean() if "Normalized_QoS_Score" in group else np.nan,
            "Average_Total_Response_Time_s": group["Total_Response_Time_s"].mean() if "Total_Response_Time_s" in group else np.nan,
            "Average_Bottleneck_Throughput_kbps": group["Bottleneck_Throughput_kbps"].mean()
            if "Bottleneck_Throughput_kbps" in group
            else np.nan,
            "Average_Average_Workflow_Availability": group["Average_Workflow_Availability"].mean()
            if "Average_Workflow_Availability" in group
            else np.nan,
            "Complete_Composition_Count": int((group["Composition_Completeness"] >= 1).sum())
            if "Composition_Completeness" in group
            else np.nan,
            "Invalid_Composition_Count": int((group["Composition_Validity"] != 1).sum())
            if "Composition_Validity" in group
            else np.nan,
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    if not counts_df.empty:
        table = table.merge(counts_df, on="Mode", how="left")
    else:
        table["unique_best_count"] = 0
        table["tied_best_count"] = 0
    return table, best_df, warnings


def build_figure_5_2_average_score_table(
    aggregate_df: pd.DataFrame,
    *,
    expected_query_count: int = 15,
) -> tuple[pd.DataFrame, list[str]]:
    warnings = require_columns(aggregate_df, FIGURE_5_2_REQUIRED_COLUMNS, "Figure 5.2 aggregate source")
    if warnings:
        return pd.DataFrame(), warnings
    working = aggregate_df.copy()
    for column in [
        "Query_Count",
        "Average_QoS_Adjusted_Composition_Score",
        "Std_QoS_Adjusted_Composition_Score",
    ]:
        working[column] = _as_numeric(working[column])

    if working["Mode"].astype(str).duplicated().any():
        warnings.append("Figure 5.2 aggregate source: duplicate mode rows found; using first row in CSV order.")

    rows: list[dict[str, Any]] = []
    observed_modes = set(working["Mode"].astype(str))
    missing_modes = [mode for mode in FIGURE_5_2_MODE_ORDER if mode not in observed_modes]
    warnings.extend(f"Figure 5.2 aggregate source: missing mode `{mode}`." for mode in missing_modes)
    for mode in FIGURE_5_2_MODE_ORDER:
        group = working[working["Mode"].astype(str) == mode]
        if group.empty:
            continue
        row = group.iloc[0]
        query_count = row["Query_Count"]
        if pd.isna(query_count) or int(query_count) != expected_query_count:
            warnings.append(
                f"Figure 5.2 aggregate source: mode `{mode}` has Query_Count={query_count}; "
                f"expected {expected_query_count} for official q01-q15."
            )
        rows.append(
            {
                "Mode": mode,
                "Query_Count": int(query_count) if not pd.isna(query_count) else np.nan,
                "Average_QoS_Adjusted_Composition_Score": row["Average_QoS_Adjusted_Composition_Score"],
                "Std_QoS_Adjusted_Composition_Score": row["Std_QoS_Adjusted_Composition_Score"],
            }
        )
    return pd.DataFrame(rows), warnings


def build_figure_5_7_functional_vs_qos_table(
    aggregate_df: pd.DataFrame,
    *,
    expected_query_count: int = 15,
) -> tuple[pd.DataFrame, list[str]]:
    warnings = require_columns(aggregate_df, FIGURE_5_7_REQUIRED_COLUMNS, "Figure 5.7 aggregate source")
    if warnings:
        return pd.DataFrame(), warnings
    working = aggregate_df.copy()
    for column in ["Query_Count", "Average_Functional_Coverage", "Average_Normalized_QoS_Score"]:
        working[column] = _as_numeric(working[column])

    if working["Mode"].astype(str).duplicated().any():
        warnings.append("Figure 5.7 aggregate source: duplicate mode rows found; using first row in CSV order.")

    rows: list[dict[str, Any]] = []
    observed_modes = set(working["Mode"].astype(str))
    missing_modes = [mode for mode in FIGURE_5_7_MODE_ORDER if mode not in observed_modes]
    warnings.extend(f"Figure 5.7 aggregate source: missing mode `{mode}`." for mode in missing_modes)
    for mode in FIGURE_5_7_MODE_ORDER:
        group = working[working["Mode"].astype(str) == mode]
        if group.empty:
            continue
        row = group.iloc[0]
        query_count = row["Query_Count"]
        if pd.isna(query_count) or int(query_count) != expected_query_count:
            warnings.append(
                f"Figure 5.7 aggregate source: mode `{mode}` has Query_Count={query_count}; "
                f"expected {expected_query_count} for official q01-q15."
            )
        avg_functional = row["Average_Functional_Coverage"]
        avg_qos = row["Average_Normalized_QoS_Score"]
        expected_point = FIGURE_5_7_OFFICIAL_POINTS.get(mode)
        if expected_point is not None and not (
            np.isclose(avg_functional, expected_point[0], atol=5e-7, rtol=0.0)
            and np.isclose(avg_qos, expected_point[1], atol=5e-7, rtol=0.0)
        ):
            warnings.append(
                f"Figure 5.7 aggregate source: mode `{mode}` point is "
                f"({avg_functional:.6f}, {avg_qos:.6f}); expected "
                f"({expected_point[0]:.6f}, {expected_point[1]:.6f}) for official q01-q15."
            )
        rows.append(
            {
                "Mode": mode,
                "Query_Count": int(query_count) if not pd.isna(query_count) else np.nan,
                "Average_Functional_Coverage": avg_functional,
                "Average_Normalized_QoS_Score": avg_qos,
            }
        )
    return pd.DataFrame(rows), warnings


def build_pairwise_similarity_table(pairwise_scores: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = ["metric", "mode_a", "mode_b", "score", "included_cases"]
    warnings = require_columns(pairwise_scores, required, "Pairwise ranking similarity")
    if warnings:
        return pd.DataFrame(), warnings
    working = pairwise_scores.copy()
    working["score"] = _as_numeric(working["score"])
    working["included_cases"] = _as_numeric(working["included_cases"])
    pivot = working.pivot_table(index=["mode_a", "mode_b"], columns="metric", values="score", aggfunc="mean").reset_index()
    counts = working.groupby(["mode_a", "mode_b"], dropna=False)["included_cases"].max().reset_index()
    table = pivot.merge(counts, on=["mode_a", "mode_b"], how="left")
    table = table.rename(
        columns={
            "spearman": "Spearman",
            "average_overlap": "Average Overlap@K",
            "rbo": "RBO",
            "jaccard": "Jaccard@K",
        }
    )
    return table, []


def build_figure_5_8_ranking_similarity_matrices(
    matrices: dict[str, pd.DataFrame],
    *,
    mode_order: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[str]]:
    mode_order = mode_order or FIGURE_5_8_MODE_ORDER
    ordered_matrices: dict[str, pd.DataFrame] = {}
    source_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for metric in FIGURE_5_8_METRIC_ORDER:
        filename = RANKING_MATRIX_FILES[metric]
        label = RANKING_METRIC_LABELS[metric]
        matrix = matrices.get(metric, pd.DataFrame())
        if matrix.empty:
            warnings.append(f"Figure 5.8 ranking source: missing or empty `{filename}`.")
            continue

        working = matrix.copy()
        working.index = working.index.astype(str)
        working.columns = working.columns.astype(str)
        missing_rows = [mode for mode in mode_order if mode not in working.index]
        missing_columns = [mode for mode in mode_order if mode not in working.columns]
        if missing_rows or missing_columns:
            if missing_rows:
                warnings.append(
                    f"Figure 5.8 ranking source: `{filename}` missing row modes: {', '.join(missing_rows)}."
                )
            if missing_columns:
                warnings.append(
                    f"Figure 5.8 ranking source: `{filename}` missing column modes: {', '.join(missing_columns)}."
                )
            continue

        view = working.loc[mode_order, mode_order].apply(pd.to_numeric, errors="coerce")
        if view.isna().any().any():
            warnings.append(f"Figure 5.8 ranking source: `{filename}` contains nonnumeric or missing matrix cells.")

        finite_values = view.to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size and (finite_values.min() < -TOLERANCE or finite_values.max() > 1.0 + TOLERANCE):
            warnings.append(
                f"Figure 5.8 ranking source: `{filename}` has values outside the required 0.0-1.0 color scale."
            )

        ordered_matrices[metric] = view
        for row_mode in mode_order:
            for column_mode in mode_order:
                source_rows.append(
                    {
                        "Metric": label,
                        "Source_File": filename,
                        "Row_Mode": row_mode,
                        "Column_Mode": column_mode,
                        "Similarity": view.loc[row_mode, column_mode],
                    }
                )

    return ordered_matrices, pd.DataFrame(source_rows), warnings


def build_query_score_matrix(
    composition_df: pd.DataFrame,
    selected_query_ids: list[str],
    selected_modes: list[str],
    tolerance: float = TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    required = ["Query_ID", "Mode", DEFAULT_SCORE_COL]
    warnings = require_columns(composition_df, required, "Query score matrix")
    if warnings:
        return pd.DataFrame(), pd.DataFrame(), warnings
    filtered = filter_composition(composition_df, selected_query_ids, selected_modes)
    if filtered.empty:
        return pd.DataFrame(), pd.DataFrame(), ["Query score matrix: no rows match the selected scope."]
    working = filtered.copy()
    working["Query_ID"] = working["Query_ID"].astype(str)
    working["Mode"] = working["Mode"].astype(str)
    working[DEFAULT_SCORE_COL] = _as_numeric(working[DEFAULT_SCORE_COL])
    observed_queries = set(working["Query_ID"])
    observed_modes = set(working["Mode"])
    missing_queries = [
        query_id
        for query_id in sorted_query_ids(selected_query_ids)
        if query_id not in observed_queries
    ]
    missing_modes = [mode for mode in selected_modes if mode not in observed_modes]
    if missing_queries:
        warnings.append(f"Query score matrix: missing selected query IDs: {', '.join(missing_queries)}.")
    if missing_modes:
        warnings.append(f"Query score matrix: missing selected modes: {', '.join(missing_modes)}.")
    duplicate_mask = working.duplicated(["Query_ID", "Mode"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = (
            working.loc[duplicate_mask, ["Query_ID", "Mode"]]
            .drop_duplicates()
            .sort_values(["Query_ID", "Mode"])
            .astype(str)
        )
        duplicate_labels = [f"{row.Query_ID}/{row.Mode}" for row in duplicate_keys.itertuples(index=False)]
        warnings.append(
            "Query score matrix: duplicate Query_ID/Mode score rows found; using the first source row for "
            f"{', '.join(duplicate_labels)}."
        )
    unique_scores = working.drop_duplicates(["Query_ID", "Mode"], keep="first")
    pivot = unique_scores.pivot(index="Query_ID", columns="Mode", values=DEFAULT_SCORE_COL)
    for mode in selected_modes:
        if mode not in pivot.columns:
            pivot[mode] = np.nan
    pivot = pivot[[mode for mode in selected_modes if mode in pivot.columns]]
    ordered_queries = sorted_query_ids(selected_query_ids) if selected_query_ids else sorted_query_ids(pivot.index)
    pivot = pivot.reindex(ordered_queries)
    best_df, _counts = compute_unique_and_tied_best(working, DEFAULT_SCORE_COL, tolerance)
    matrix = pivot.reset_index()
    if not best_df.empty:
        matrix = matrix.merge(best_df[["Query_ID", "Best_Modes", "Best_Status"]], on="Query_ID", how="left")
        matrix = matrix.rename(columns={"Best_Modes": "Best_Mode_or_Tied_Modes"})
    return matrix, best_df, warnings


def build_per_query_metrics_table(
    composition_df: pd.DataFrame,
    query_id: str,
    selected_modes: list[str],
    metrics: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    required = ["Query_ID", "Mode"]
    warnings = require_columns(composition_df, required, "Per-query metrics")
    if warnings:
        return pd.DataFrame(), warnings
    available = [metric for metric in metrics if metric in composition_df.columns]
    missing = [metric for metric in metrics if metric not in composition_df.columns]
    warnings.extend([f"Per-query metrics: missing optional column `{metric}`." for metric in missing])
    columns = ["Mode", *available]
    filtered = filter_composition(composition_df, [query_id], selected_modes)
    if filtered.empty:
        warnings.append(f"Per-query metrics: no rows found for {query_id}.")
        return pd.DataFrame(), warnings
    table = filtered[columns].copy()
    table["Mode"] = pd.Categorical(table["Mode"], categories=selected_modes, ordered=True)
    return table.sort_values("Mode").reset_index(drop=True), warnings


def _first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _format_exact_decimal(value: Any, *, digits: int = 6) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return f"{float(numeric):.{digits}f}".rstrip("0").rstrip(".")


def _format_functional_match_cell(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "FM=n/a"
    if abs(float(numeric) - 1.0) <= TOLERANCE:
        return "FM=1"
    if abs(float(numeric)) <= TOLERANCE:
        return "FM=0"
    return f"FM={_format_exact_decimal(numeric)}"


def _parse_step_index(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, float) and not math.isnan(value) and value.is_integer():
        return int(value)
    text = str(value or "").strip()
    if text.isdigit():
        return int(text)
    return fallback


def _extract_planner_steps_for_dashboard(payload: dict[str, Any], planner_path: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    workflow = payload.get("execution_workflow") if isinstance(payload.get("execution_workflow"), dict) else {}
    steps = workflow.get("steps") if isinstance(workflow.get("steps"), list) else None
    if steps is None:
        primary_plan = payload.get("primary_plan") if isinstance(payload.get("primary_plan"), dict) else {}
        steps = primary_plan.get("steps") if isinstance(primary_plan.get("steps"), list) else None
    if not isinstance(steps, list):
        return pd.DataFrame(), [f"Section 5.5 q02 path: planner has no step list: {planner_path}"]

    rows: list[dict[str, Any]] = []
    for fallback_step, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, dict):
            warnings.append(f"Section 5.5 q02 path: ignored non-object planner step in {planner_path}.")
            continue
        api_id = str(raw_step.get("api_id") or "").strip()
        if not api_id:
            warnings.append(f"Section 5.5 q02 path: planner step {fallback_step} has no api_id in {planner_path}.")
            continue
        step_number = _parse_step_index(raw_step.get("step"), fallback_step)
        rows.append(
            {
                "Step": step_number,
                "Subtask_ID": str(raw_step.get("subtask_id") or step_number).strip(),
                "API_ID": api_id,
                "Method": str(raw_step.get("method") or "").strip().upper(),
                "URL": str(raw_step.get("url") or "").strip(),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table, warnings
    table = table.sort_values(["Step", "Subtask_ID", "API_ID"], kind="mergesort").reset_index(drop=True)

    selected_api_ids = payload.get("selected_api_ids")
    if not isinstance(selected_api_ids, list):
        selected_api_ids = payload.get("planner_selected_api_ids")
    if isinstance(selected_api_ids, list):
        selected_api_ids = [str(item) for item in selected_api_ids]
        workflow_api_ids = table["API_ID"].astype(str).tolist()
        if selected_api_ids and workflow_api_ids != selected_api_ids:
            warnings.append(
                "Section 5.5 q02 path: selected_api_ids does not match planner step order "
                f"for {planner_path}."
            )
    return table, warnings


def _functional_match_lookup(loaded_rows: pd.DataFrame, query_id: str) -> tuple[dict[tuple[str, str, str], Any], list[str]]:
    if loaded_rows.empty:
        return {}, ["Section 5.5 q02 path: ranking_eval/loaded_rows.csv is unavailable; functional-match labels are omitted."]
    query_col = _first_column(loaded_rows, ["query_id", "Query_ID"])
    mode_col = _first_column(loaded_rows, ["mode", "Mode"])
    subtask_col = _first_column(loaded_rows, ["subtask_id", "Sub Task", "Subtask_ID"])
    api_col = _first_column(loaded_rows, ["api_id", "Selected_API", "API_ID"])
    match_col = _first_column(loaded_rows, ["functional_match_label", "Functional Match (0/1)", "Functional_Match"])
    missing = [
        label
        for label, col in {
            "query": query_col,
            "mode": mode_col,
            "subtask": subtask_col,
            "api": api_col,
            "functional match": match_col,
        }.items()
        if col is None
    ]
    if missing:
        return {}, [f"Section 5.5 q02 path: loaded ranking rows are missing {', '.join(missing)} columns."]

    working = loaded_rows[loaded_rows[query_col].astype(str) == str(query_id)].copy()
    if working.empty:
        return {}, [f"Section 5.5 q02 path: no loaded ranking rows found for {query_id}."]
    planner_col = _first_column(working, ["selected_for_planner", "Selected for Planner"])
    rank_col = _first_column(working, ["mode_rank", "Mode Rank"])
    working["_planner_selected_sort"] = _truthy_series(working[planner_col]).astype(int) if planner_col else 0
    working["_mode_rank_sort"] = _as_numeric(working[rank_col]) if rank_col else np.nan
    working = working.sort_values(
        ["_planner_selected_sort", "_mode_rank_sort", mode_col, subtask_col, api_col],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )

    lookup: dict[tuple[str, str, str], Any] = {}
    for _, row in working.iterrows():
        key = (str(row[mode_col]), str(row[subtask_col]), str(row[api_col]))
        lookup.setdefault(key, row[match_col])
    return lookup, []


def build_selected_api_path_comparison(
    run_dir: Path,
    composition_df: pd.DataFrame,
    loaded_rows: pd.DataFrame,
    *,
    query_id: str = SECTION_55_QUERY_ID,
    selected_modes: list[str] | None = None,
    expected_steps: int = SECTION_55_EXPECTED_STEPS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Build Section 5.5 q02 dashboard figure rows from official CSV/JSON artifacts."""
    modes = selected_modes or PREFERRED_MODE_ORDER
    required = [
        "Query_ID",
        "Mode",
        "Run_Folder",
        "Planner_Output_File",
        "Functional_Coverage",
        "Normalized_QoS_Score",
        "QoS_Adjusted_Composition_Score",
    ]
    warnings = require_columns(composition_df, required, "Section 5.5 q02 path")
    if warnings:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), warnings

    filtered = filter_composition(composition_df, [query_id], modes)
    if filtered.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            [f"Section 5.5 q02 path: no composition rows found for {query_id}."],
        )

    observed_modes = set(filtered["Mode"].astype(str))
    missing_modes = [mode for mode in modes if mode not in observed_modes]
    warnings.extend(f"Section 5.5 q02 path: missing q02 row for mode `{mode}`." for mode in missing_modes)
    if filtered.duplicated(["Query_ID", "Mode"]).any():
        warnings.append("Section 5.5 q02 path: duplicate q02/mode rows found; using first row in CSV order.")

    match_lookup, match_warnings = _functional_match_lookup(loaded_rows, query_id)
    warnings.extend(match_warnings)

    figure_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for mode in modes:
        mode_rows = filtered[filtered["Mode"].astype(str) == mode]
        if mode_rows.empty:
            continue
        comp_row = mode_rows.iloc[0]
        planner_path = Path(str(comp_row["Planner_Output_File"]))
        if not planner_path.is_absolute():
            planner_path = run_dir / str(comp_row["Run_Folder"]) / planner_path
        payload, warning = load_json_if_exists(planner_path)
        if warning or not isinstance(payload, dict):
            warnings.append(warning or f"Section 5.5 q02 path: planner JSON is not an object: {planner_path}")
            continue
        steps_df, step_warnings = _extract_planner_steps_for_dashboard(payload, planner_path)
        warnings.extend(step_warnings)
        if steps_df.empty:
            continue
        if expected_steps and len(steps_df) != expected_steps:
            warnings.append(
                f"Section 5.5 q02 path: mode `{mode}` has {len(steps_df)} planner steps; expected {expected_steps}."
            )

        row: dict[str, Any] = {
            "Mode": mode,
            "Functional Coverage": _format_exact_decimal(comp_row["Functional_Coverage"]),
            "Normalized QoS": _format_exact_decimal(comp_row["Normalized_QoS_Score"]),
            "Final Score": _format_exact_decimal(comp_row["QoS_Adjusted_Composition_Score"]),
        }
        for step_index in range(1, expected_steps + 1):
            step_match = steps_df[steps_df["Step"] == step_index]
            if step_match.empty:
                row[f"Subtask {step_index}"] = ""
                continue
            step_row = step_match.iloc[0]
            subtask_id = str(step_row["Subtask_ID"])
            api_id = str(step_row["API_ID"])
            fm_value = match_lookup.get((mode, subtask_id, api_id))
            fm_label = _format_functional_match_cell(fm_value)
            row[f"Subtask {step_index}"] = f"{api_id}\n{fm_label}"
            detail_rows.append(
                {
                    "Query_ID": query_id,
                    "Mode": mode,
                    "Step": int(step_row["Step"]),
                    "Subtask_ID": subtask_id,
                    "API_ID": api_id,
                    "Functional_Match": "" if fm_value is None else fm_value,
                    "Functional_Match_Label": fm_label,
                    "Method": step_row.get("Method", ""),
                    "URL": step_row.get("URL", ""),
                    "Planner_Output_File": str(comp_row["Planner_Output_File"]),
                    "Source_Planner_Path": str(planner_path),
                }
            )
        figure_rows.append(row)
        trace_row = {"Query_ID": query_id}
        for col in SECTION_55_SCORE_TRACE_COLUMNS:
            trace_row[col] = comp_row[col] if col in comp_row.index else np.nan
        trace_rows.append(trace_row)

    figure_df = pd.DataFrame(figure_rows)
    if not figure_df.empty:
        for column in SECTION_55_FIGURE_COLUMNS:
            if column not in figure_df:
                figure_df[column] = ""
        figure_df = figure_df[SECTION_55_FIGURE_COLUMNS]
    score_trace_df = pd.DataFrame(trace_rows)
    detail_df = pd.DataFrame(detail_rows)
    return figure_df, score_trace_df, detail_df, warnings


def build_diagnostic_exceptions_table(
    composition_df: pd.DataFrame,
    loaded_rows: pd.DataFrame,
    best_df: pd.DataFrame,
    selected_query_ids: list[str],
    selected_modes: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    filtered = filter_composition(composition_df, selected_query_ids, selected_modes)
    if filtered.empty:
        return pd.DataFrame(), ["Diagnostics: no composition rows match the selected scope."]
    for col in ["Composition_Validity", "Composition_Completeness"]:
        if col in filtered:
            filtered[col] = _as_numeric(filtered[col])
        else:
            warnings.append(f"Diagnostics: optional column `{col}` is unavailable.")

    rows: list[dict[str, Any]] = []
    tied_by_mode: dict[str, list[str]] = {mode: [] for mode in selected_modes}
    if not best_df.empty:
        for _, row in best_df.iterrows():
            best_modes = [part.strip() for part in str(row.get("Best_Modes") or "").split(",") if part.strip()]
            if len(best_modes) <= 1:
                continue
            for mode in best_modes:
                tied_by_mode.setdefault(mode, []).append(str(row["Query_ID"]))

    selected_loaded = loaded_rows.copy()
    if not selected_loaded.empty and "query_id" in selected_loaded:
        selected_loaded = selected_loaded[selected_loaded["query_id"].astype(str).isin(selected_query_ids)]
    if not selected_loaded.empty and "mode" in selected_loaded:
        selected_loaded = selected_loaded[selected_loaded["mode"].astype(str).isin(selected_modes)]

    for mode in selected_modes:
        group = filtered[filtered["Mode"].astype(str) == mode] if "Mode" in filtered else pd.DataFrame()
        row: dict[str, Any] = {
            "Mode": mode,
            "Average_Composition_Validity": group["Composition_Validity"].mean()
            if "Composition_Validity" in group
            else np.nan,
            "Average_Composition_Completeness": group["Composition_Completeness"].mean()
            if "Composition_Completeness" in group
            else np.nan,
            "Complete_Composition_Count": int((group["Composition_Completeness"] >= 1).sum())
            if "Composition_Completeness" in group
            else np.nan,
            "Invalid_Composition_Count": int((group["Composition_Validity"] != 1).sum())
            if "Composition_Validity" in group
            else np.nan,
            "Tied_Best_Queries": ", ".join(tied_by_mode.get(mode, [])),
        }
        mode_rows = selected_loaded[selected_loaded["mode"].astype(str) == mode] if "mode" in selected_loaded else pd.DataFrame()
        if not mode_rows.empty:
            if "selected_for_planner" in mode_rows:
                planner_rows = mode_rows[_truthy_series(mode_rows["selected_for_planner"])]
            else:
                planner_rows = mode_rows
            qos_cols = [col for col in ["QoS_RT_s", "QoS_TP_kbps", "QoS Availability"] if col in planner_rows]
            if qos_cols:
                row["Missing_QoS_Selected_API_Count"] = int(
                    planner_rows[qos_cols].replace("", np.nan).isna().any(axis=1).sum()
                )
            if "is_hallucinated" in planner_rows:
                row["Selected_API_Hallucination_Count"] = int(_truthy_series(planner_rows["is_hallucinated"]).sum())
            if "is_duplicated" in planner_rows:
                row["Selected_API_Duplicate_Count"] = int(_truthy_series(planner_rows["is_duplicated"]).sum())
            if "ranking_anomaly" in planner_rows:
                row["Selected_API_Ranking_Anomaly_Count"] = int(_truthy_series(planner_rows["ranking_anomaly"]).sum())
        rows.append(row)
    return pd.DataFrame(rows), warnings


def parse_planner_trace(run_dir: Path, composition_df: pd.DataFrame, query_id: str, selected_modes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    required = ["Query_ID", "Mode", "Run_Folder", "Planner_Output_File"]
    missing = require_columns(composition_df, required, "Workflow trace")
    if missing:
        return pd.DataFrame(), missing
    filtered = filter_composition(composition_df, [query_id], selected_modes)
    rows: list[dict[str, Any]] = []
    for _, comp_row in filtered.iterrows():
        mode = str(comp_row["Mode"])
        run_folder = str(comp_row["Run_Folder"])
        planner_file = str(comp_row["Planner_Output_File"])
        planner_path = Path(planner_file)
        if not planner_path.is_absolute():
            planner_path = run_dir / run_folder / planner_path
        payload, warning = load_json_if_exists(planner_path)
        if warning or not isinstance(payload, dict):
            warnings.append(warning or f"Workflow trace: planner JSON is not an object: {planner_path}")
            continue
        primary_plan = payload.get("primary_plan") if isinstance(payload.get("primary_plan"), dict) else {}
        steps = primary_plan.get("steps")
        if not isinstance(steps, list):
            workflow = payload.get("execution_workflow") if isinstance(payload.get("execution_workflow"), dict) else {}
            steps = workflow.get("steps")
        if not isinstance(steps, list):
            warnings.append(f"Workflow trace: no step list found in {planner_path}")
            continue
        for idx, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            qos = step.get("qos") if isinstance(step.get("qos"), dict) else {}
            rows.append(
                {
                    "Query_ID": query_id,
                    "Mode": mode,
                    "Step": step.get("step", idx),
                    "Subtask_ID": step.get("subtask_id"),
                    "API_ID": step.get("api_id"),
                    "Functional_Match_Label": step.get("functional_match_label"),
                    "QoS_RT_s": qos.get("rt_s"),
                    "QoS_TP_kbps": qos.get("tp_kbps"),
                    "QoS_Availability": qos.get("availability"),
                    "Selected_By": step.get("selected_by"),
                    "Planner_Override_Attempted": step.get("planner_override_attempted"),
                    "Action": step.get("action"),
                    "Why": step.get("why"),
                    "Planner_Output_File": str(planner_path),
                }
            )
    return pd.DataFrame(rows), warnings


def _ranked_lists(case_df: pd.DataFrame, modes: list[str], top_n: int | None = None) -> dict[str, list[str]]:
    lists: dict[str, list[str]] = {}
    if case_df.empty or "api_id" not in case_df or "mode" not in case_df:
        return lists
    rank_col = "mode_rank" if "mode_rank" in case_df else "Retrieved Rank"
    for mode in modes:
        group = case_df[case_df["mode"].astype(str) == mode].copy()
        if group.empty:
            continue
        if rank_col in group:
            group["_rank"] = _as_numeric(group[rank_col])
            group = group.sort_values(["_rank", "api_id"], kind="mergesort")
        else:
            group = group.sort_values("api_id", kind="mergesort")
        values = group["api_id"].astype(str).tolist()
        lists[mode] = values[:top_n] if top_n else values
    return lists


def ranked_api_lists(case_df: pd.DataFrame, modes: list[str], top_n: int | None = None) -> dict[str, list[str]]:
    return _ranked_lists(case_df, modes, top_n=top_n)


def case_mode_ranking_table(
    loaded_rows: pd.DataFrame,
    query_id: str,
    subtask_id: str,
    selected_modes: list[str],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if loaded_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    case_df = loaded_rows[
        (loaded_rows["query_id"].astype(str) == str(query_id))
        & (loaded_rows["subtask_id"].astype(str) == str(subtask_id))
        & (loaded_rows["mode"].astype(str).isin(selected_modes))
    ].copy()
    if case_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    case_df["mode"] = pd.Categorical(case_df["mode"], categories=selected_modes, ordered=True)
    if "mode_rank" in case_df:
        case_df["_rank"] = _as_numeric(case_df["mode_rank"])
    elif "Retrieved Rank" in case_df:
        case_df["_rank"] = _as_numeric(case_df["Retrieved Rank"])
    else:
        case_df["_rank"] = np.nan
    case_df = case_df.sort_values(["mode", "_rank", "api_id"], kind="mergesort")

    top_lists = _ranked_lists(case_df, selected_modes, top_n=top_n)
    wide_rows = []
    for rank in range(1, top_n + 1):
        row = {"rank": rank}
        for mode in selected_modes:
            values = top_lists.get(mode, [])
            row[mode] = values[rank - 1] if rank <= len(values) else ""
        wide_rows.append(row)
    topn_wide = pd.DataFrame(wide_rows)

    selected_rows = pd.DataFrame()
    if "selected_for_planner" in case_df:
        selected_rows = case_df[_truthy_series(case_df["selected_for_planner"])].drop(columns=["_rank"], errors="ignore")
    return case_df.drop(columns=["_rank"], errors="ignore"), topn_wide, selected_rows


def _jaccard(left: list[str], right: list[str]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return np.nan
    union = a | b
    return len(a & b) / len(union) if union else np.nan


def average_overlap_by_depth(left: list[str], right: list[str], max_depth: int) -> pd.DataFrame:
    rows = []
    if max_depth <= 0:
        return pd.DataFrame(columns=["depth", "overlap_ratio"])
    for depth in range(1, max_depth + 1):
        left_set = set(left[:depth])
        right_set = set(right[:depth])
        denom = max(len(left_set), len(right_set), 1)
        rows.append({"depth": depth, "overlap_ratio": len(left_set & right_set) / denom})
    return pd.DataFrame(rows)


def _rbo(left: list[str], right: list[str], p: float = 0.9, max_depth: int | None = None) -> float:
    depth_limit = max_depth or max(len(left), len(right))
    if depth_limit <= 0:
        return np.nan
    score = 0.0
    overlap = 0
    left_seen: set[str] = set()
    right_seen: set[str] = set()
    for depth in range(1, depth_limit + 1):
        if depth <= len(left):
            left_seen.add(left[depth - 1])
        if depth <= len(right):
            right_seen.add(right[depth - 1])
        overlap = len(left_seen & right_seen)
        score += (p ** (depth - 1)) * (overlap / depth)
    extrapolated = (overlap / depth_limit) * (p**depth_limit)
    return float((1 - p) * score + extrapolated)


def _spearman_from_lists(left: list[str], right: list[str]) -> float:
    shared = [api_id for api_id in left if api_id in set(right)]
    if len(shared) < 2:
        return np.nan
    left_rank = {api_id: idx + 1 for idx, api_id in enumerate(left)}
    right_rank = {api_id: idx + 1 for idx, api_id in enumerate(right)}
    x = np.array([left_rank[api_id] for api_id in shared], dtype=float)
    y = np.array([right_rank[api_id] for api_id in shared], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compute_case_similarity_matrix(case_df: pd.DataFrame, selected_modes: list[str], metric: str, top_n: int) -> pd.DataFrame:
    lists = _ranked_lists(case_df, selected_modes, top_n=None)
    matrix = pd.DataFrame(index=selected_modes, columns=selected_modes, dtype=float)
    for mode_a in selected_modes:
        for mode_b in selected_modes:
            left = lists.get(mode_a, [])[:top_n]
            right = lists.get(mode_b, [])[:top_n]
            if mode_a == mode_b:
                matrix.loc[mode_a, mode_b] = 1.0
            elif metric == "jaccard":
                matrix.loc[mode_a, mode_b] = _jaccard(left, right)
            elif metric == "average_overlap":
                ao = average_overlap_by_depth(left, right, min(top_n, max(len(left), len(right))))
                matrix.loc[mode_a, mode_b] = ao["overlap_ratio"].mean() if not ao.empty else np.nan
            elif metric == "rbo":
                matrix.loc[mode_a, mode_b] = _rbo(left, right, max_depth=top_n)
            elif metric == "spearman":
                matrix.loc[mode_a, mode_b] = _spearman_from_lists(lists.get(mode_a, []), lists.get(mode_b, []))
            else:
                matrix.loc[mode_a, mode_b] = np.nan
    return matrix


def compute_llm_topsis_agreement(loaded_rows: pd.DataFrame, selected_query_ids: list[str], top_k: int) -> tuple[pd.DataFrame, list[str]]:
    required = ["query_id", "subtask_id", "mode", "api_id"]
    warnings = require_columns(loaded_rows, required, "LLM-TOPSIS agreement")
    if warnings:
        return pd.DataFrame(), warnings
    modes = ["qos_pure_llm", "qos_topsis"]
    filtered = loaded_rows[
        loaded_rows["query_id"].astype(str).isin(selected_query_ids)
        & loaded_rows["mode"].astype(str).isin(modes)
    ].copy()
    if filtered.empty:
        return pd.DataFrame(), ["LLM-TOPSIS agreement: no loaded rows for qos_pure_llm and qos_topsis in selected scope."]
    rows: list[dict[str, Any]] = []
    for (query_id, subtask_id), case_df in filtered.groupby(["query_id", "subtask_id"], dropna=False):
        lists = _ranked_lists(case_df, modes, top_n=None)
        if any(mode not in lists or not lists[mode] for mode in modes):
            continue
        pure_top1 = lists["qos_pure_llm"][0]
        topsis_top1 = lists["qos_topsis"][0]
        row: dict[str, Any] = {
            "query_id": query_id,
            "subtask_id": subtask_id,
            "qos_pure_llm_top1_api": pure_top1,
            "qos_topsis_top1_api": topsis_top1,
            "top1_agreement_status": "agreement" if pure_top1 == topsis_top1 else "disagreement",
            f"top{top_k}_jaccard": _jaccard(lists["qos_pure_llm"][:top_k], lists["qos_topsis"][:top_k]),
        }
        if "selected_for_planner" in case_df:
            selected_sets = {}
            for mode in modes:
                selected = case_df[
                    (case_df["mode"].astype(str) == mode) & _truthy_series(case_df["selected_for_planner"])
                ]["api_id"].astype(str)
                selected_sets[mode] = set(selected.tolist())
            row["selected_for_planner_agreement_status"] = (
                "agreement" if selected_sets["qos_pure_llm"] == selected_sets["qos_topsis"] else "disagreement"
            )
            row["qos_pure_llm_selected_api_ids"] = "; ".join(sorted(selected_sets["qos_pure_llm"]))
            row["qos_topsis_selected_api_ids"] = "; ".join(sorted(selected_sets["qos_topsis"]))

        for mode, api_id in [("qos_pure_llm", pure_top1), ("qos_topsis", topsis_top1)]:
            top_row = case_df[(case_df["mode"].astype(str) == mode) & (case_df["api_id"].astype(str) == api_id)]
            if top_row.empty:
                continue
            top_row = top_row.iloc[0]
            for source, target in [
                ("functional_match_label", "functional_match_label"),
                ("QoS_RT_s", "rt_s"),
                ("QoS_TP_kbps", "tp_kbps"),
                ("QoS Availability", "availability"),
            ]:
                if source in top_row:
                    row[f"{mode}_top1_{target}"] = top_row[source]
        rows.append(row)
    return pd.DataFrame(rows), warnings


def figure_to_bytes(fig: plt.Figure, fmt: str, dpi: int = 300) -> bytes:
    buffer = io.BytesIO()
    metadata = {"Creator": "AutoLLMCompose thesis figure generator"}
    if fmt.lower() == "pdf":
        metadata.update({"CreationDate": None, "ModDate": None})
    fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white", metadata=metadata)
    return buffer.getvalue()


def table_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def plot_table_pdf(df: pd.DataFrame, title: str, max_rows: int = 30) -> plt.Figure:
    configure_matplotlib()
    view = df.head(max_rows).copy()
    fig_height = max(2.4, 0.35 * (len(view) + 2))
    fig_width = min(18, max(8, 1.8 * max(1, len(view.columns))))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, loc="left", pad=10)
    table = ax.table(cellText=view.astype(str).values, colLabels=view.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
    fig.tight_layout()
    return fig


def _mode_colors(modes: list[str]) -> dict[str, str]:
    palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
    return {mode: palette[idx % len(palette)] for idx, mode in enumerate(modes)}


def plot_figure_5_2_average_score_by_mode(table: pd.DataFrame) -> plt.Figure | None:
    if table.empty:
        return None
    configure_matplotlib()
    modes = table["Mode"].astype(str).tolist()
    means = _as_numeric(table["Average_QoS_Adjusted_Composition_Score"]).tolist()
    stds = _as_numeric(table["Std_QoS_Adjusted_Composition_Score"]).fillna(0.0).tolist()
    colors = {
        "qos_hybrid": "#0072B2",
        "qos_pure_llm": "#009E73",
        "no_qos": "#999999",
        "qos_topsis": "#D55E00",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    bars = ax.bar(
        modes,
        means,
        yerr=stds,
        capsize=5,
        color=[colors.get(mode, "#4C78A8") for mode in modes],
        edgecolor="#222222",
        linewidth=0.8,
        error_kw={"elinewidth": 1.15, "ecolor": "#222222", "capthick": 1.15, "clip_on": False},
    )

    ax.set_xlabel("Evaluation Mode")
    ax.set_ylabel("Average QoS-Adjusted Composition Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.tick_params(axis="x", rotation=22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, mean in zip(bars, means):
        mean_value = float(mean)
        near_ceiling = mean_value >= 0.955
        label_y = mean_value - 0.035 if near_ceiling else min(mean_value + 0.022, 0.985)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{mean_value:.3f}",
            ha="center",
            va="top" if near_ceiling else "bottom",
            fontsize=9,
            fontweight="bold" if near_ceiling else "normal",
            color="white" if near_ceiling else "#111111",
            clip_on=True,
        )
    return fig


def plot_figure_5_7_functional_vs_qos(table: pd.DataFrame) -> plt.Figure | None:
    if table.empty:
        return None
    configure_matplotlib()
    colors = {
        "no_qos": "#999999",
        "qos_pure_llm": "#009E73",
        "qos_topsis": "#D55E00",
        "qos_hybrid": "#0072B2",
    }
    label_offsets = {
        "no_qos": (-8, -14, "right", "top"),
        "qos_pure_llm": (-9, 10, "right", "bottom"),
        "qos_topsis": (8, 8, "left", "bottom"),
        "qos_hybrid": (-9, -14, "right", "top"),
    }

    fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
    for _, row in table.iterrows():
        mode = str(row["Mode"])
        x = float(row["Average_Functional_Coverage"])
        y = float(row["Average_Normalized_QoS_Score"])
        ax.scatter(
            [x],
            [y],
            s=86,
            color=colors.get(mode, "#4C78A8"),
            edgecolor="#222222",
            linewidth=0.8,
            clip_on=False,
            zorder=3,
        )
        dx, dy, ha, va = label_offsets.get(mode, (7, 7, "left", "bottom"))
        ax.annotate(
            mode,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=9,
            fontweight="bold",
            color="#111111",
            clip_on=True,
        )

    ax.set_xlabel("Average Functional Coverage")
    ax.set_ylabel("Average Normalized QoS Score")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axisbelow(True)
    ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def plot_weight_sensitivity_by_mode(summary: pd.DataFrame, selected_modes: list[str]) -> plt.Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = _mode_colors(selected_modes)
    for mode in selected_modes:
        group = summary[summary["Mode"].astype(str) == mode].sort_values("Weight_Order")
        if group.empty:
            continue
        ax.plot(group["Weight_Setting"], group["Average_Sensitivity_Score"], marker="o", label=mode, color=colors[mode])
    ax.set_title("Average Score by Mode Across Weight Settings")
    ax.set_ylabel("Average sensitivity score")
    ax.set_xlabel("Weight setting")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_mode_ranking_stability(summary: pd.DataFrame, selected_modes: list[str]) -> plt.Figure:
    configure_matplotlib()
    ranked = summary.copy()
    ranked["Mode_Rank"] = ranked.groupby("Weight_Order")["Average_Sensitivity_Score"].rank(
        method="min", ascending=False
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = _mode_colors(selected_modes)
    for mode in selected_modes:
        group = ranked[ranked["Mode"].astype(str) == mode].sort_values("Weight_Order")
        if group.empty:
            continue
        ax.plot(group["Weight_Setting"], group["Mode_Rank"], marker="o", label=mode, color=colors[mode])
    ax.set_title("Mode Ranking Stability Across Weight Settings")
    ax.set_ylabel("Rank (1 = highest average score)")
    ax.set_xlabel("Weight setting")
    ax.set_yticks(range(1, max(len(selected_modes), 1) + 1))
    ax.invert_yaxis()
    ax.tick_params(axis="x", rotation=30)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_selected_query_sensitivity(long_df: pd.DataFrame, query_id: str, selected_modes: list[str]) -> plt.Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = _mode_colors(selected_modes)
    view = long_df[long_df["Query_ID"].astype(str) == str(query_id)]
    for mode in selected_modes:
        group = view[view["Mode"].astype(str) == mode].sort_values("Weight_Order")
        if group.empty:
            continue
        ax.plot(group["Weight_Setting"], group["Sensitivity_Score"], marker="o", label=mode, color=colors[mode])
    ax.set_title(f"Selected Query Score Sensitivity by Mode ({query_id})")
    ax.set_ylabel("Sensitivity score")
    ax.set_xlabel("Weight setting")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_candidate_invalid_top10(candidate_table: pd.DataFrame) -> plt.Figure:
    configure_matplotlib()
    summary = (
        candidate_table.groupby(["Weight_Setting", "alpha", "beta"], dropna=False)["invalid_top10_candidate_count"]
        .sum()
        .reset_index()
        .sort_values(["alpha", "beta"], ascending=[False, True])
    )
    order = [setting["label"] for setting in WEIGHT_SETTINGS]
    summary["Weight_Setting"] = pd.Categorical(summary["Weight_Setting"], categories=order, ordered=True)
    summary = summary.sort_values("Weight_Setting")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(summary["Weight_Setting"].astype(str), summary["invalid_top10_candidate_count"], color="#E45756")
    ax.set_title("Invalid Top-10 Candidate Count by Weight Setting")
    ax.set_ylabel("Invalid Top-10 candidate count")
    ax.set_xlabel("Weight setting")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_aggregate_score_by_mode(table: pd.DataFrame) -> plt.Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    modes = table["Mode"].astype(str).tolist()
    means = _as_numeric(table["Average_QoS_Adjusted_Composition_Score"]).fillna(0.0).astype(float).tolist()
    stds = _as_numeric(table["Std_QoS_Adjusted_Composition_Score"]).fillna(0.0).astype(float).tolist()
    colors = [_mode_colors(modes)[mode] for mode in modes]
    bars = ax.bar(
        modes,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        error_kw={"clip_on": False},
    )
    label_anchors = [mean + max(std, 0.0) for mean, std in zip(means, stds)]
    max_label_anchor = max(label_anchors, default=1.0)
    ax.set_ylim(0, max(1.08, max_label_anchor + 0.06))

    for bar, mean, label_anchor in zip(bars, means, label_anchors):
        ax.annotate(
            f"{mean:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, label_anchor),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#111111",
            clip_on=False,
            zorder=5,
        )
    ax.set_title("Average QoS-Adjusted Composition Score by Mode")
    ax.set_ylabel("Average QoS-adjusted composition score")
    ax.set_xlabel("Mode")
    fig.tight_layout()
    return fig


def plot_functional_vs_qos(table: pd.DataFrame) -> plt.Figure:
    configure_matplotlib()
    modes = table["Mode"].astype(str).tolist()
    x = np.arange(len(modes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, table["Average_Functional_Coverage"], width, label="Functional coverage", color="#4C78A8")
    ax.bar(x + width / 2, table["Average_Normalized_QoS_Score"], width, label="Normalized QoS score", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_xlabel("Mode")
    ax.set_title("Functional Coverage and Normalized QoS by Mode")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metric_by_mode(table: pd.DataFrame, metric_col: str, ylabel: str, title: str) -> plt.Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    modes = table["Mode"].astype(str).tolist()
    ax.bar(modes, table[metric_col], color=[_mode_colors(modes)[mode] for mode in modes])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Mode")
    fig.tight_layout()
    return fig


def plot_ranking_similarity_heatmaps(matrices: dict[str, pd.DataFrame], selected_modes: list[str]) -> plt.Figure:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metric_order = ["spearman", "average_overlap", "rbo", "jaccard"]
    for ax, metric in zip(axes.flat, metric_order):
        matrix = matrices.get(metric, pd.DataFrame())
        if matrix.empty:
            ax.axis("off")
            ax.set_title(f"{RANKING_METRIC_LABELS[metric]} unavailable")
            continue
        available_modes = [mode for mode in selected_modes if mode in matrix.index and mode in matrix.columns]
        if not available_modes:
            ax.axis("off")
            ax.set_title(f"{RANKING_METRIC_LABELS[metric]} unavailable for selected modes")
            continue
        view = matrix.loc[available_modes, available_modes].astype(float)
        vmin, vmax = (-1, 1) if metric == "spearman" else (0, 1)
        image = ax.imshow(view.values, cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.set_title(RANKING_METRIC_LABELS[metric])
        ax.set_xticks(np.arange(len(view.columns)))
        ax.set_yticks(np.arange(len(view.index)))
        ax.set_xticklabels(view.columns, rotation=30, ha="right")
        ax.set_yticklabels(view.index)
        for i in range(len(view.index)):
            for j in range(len(view.columns)):
                value = view.iloc[i, j]
                label = "" if pd.isna(value) else f"{value:.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Ranking-Level Similarity and Mode Agreement", y=0.99)
    fig.tight_layout()
    return fig


def plot_figure_5_8_ranking_similarity_panel(ordered_matrices: dict[str, pd.DataFrame]) -> plt.Figure | None:
    available_metrics = [metric for metric in FIGURE_5_8_METRIC_ORDER if metric in ordered_matrices]
    if not available_metrics:
        return None

    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0), constrained_layout=True)
    image = None
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#F3F4F6")

    for ax, metric in zip(axes.flat, FIGURE_5_8_METRIC_ORDER):
        matrix = ordered_matrices.get(metric, pd.DataFrame())
        if matrix.empty:
            ax.axis("off")
            ax.set_title(f"{RANKING_METRIC_LABELS[metric]} unavailable", fontweight="bold")
            continue

        view = matrix.loc[FIGURE_5_8_MODE_ORDER, FIGURE_5_8_MODE_ORDER].astype(float)
        masked_values = np.ma.masked_invalid(view.to_numpy(dtype=float))
        image = ax.imshow(masked_values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(RANKING_METRIC_LABELS[metric], fontweight="bold", pad=8)
        ax.set_xticks(np.arange(len(FIGURE_5_8_MODE_ORDER)))
        ax.set_yticks(np.arange(len(FIGURE_5_8_MODE_ORDER)))
        ax.set_xticklabels(FIGURE_5_8_MODE_ORDER, rotation=32, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(FIGURE_5_8_MODE_ORDER)
        ax.tick_params(axis="both", length=0, labelsize=8.5)
        ax.set_xticks(np.arange(-0.5, len(FIGURE_5_8_MODE_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(FIGURE_5_8_MODE_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.15)
        ax.grid(False)

        for row_idx, row_mode in enumerate(FIGURE_5_8_MODE_ORDER):
            for col_idx, column_mode in enumerate(FIGURE_5_8_MODE_ORDER):
                value = view.loc[row_mode, column_mode]
                if pd.isna(value):
                    label = ""
                    text_color = "#111111"
                else:
                    label = f"{value:.3f}"
                    text_color = "white" if value >= 0.68 else "#111111"
                ax.text(
                    col_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold" if value >= 0.95 else "normal",
                    color=text_color,
                )

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88, fraction=0.036, pad=0.02)
        colorbar.set_label("Similarity", rotation=90)
        colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    return fig


def _wrapped_text(text: str, max_chars: int) -> str:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _add_concept_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: list[str],
    facecolor: str,
    edgecolor: str,
    wrap_chars: int,
    title_size: float = 11,
    body_size: float = 9,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.35,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018,
        y + height - 0.055,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#111827",
    )
    body_text = "\n\n".join(_wrapped_text(line, wrap_chars) for line in body)
    ax.text(
        x + 0.018,
        y + height - 0.118,
        body_text,
        ha="left",
        va="top",
        fontsize=body_size,
        color="#1F2937",
        linespacing=1.25,
    )


def _add_flow_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.55,
        color="#475569",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def plot_figure_5_10_hybrid_evidence_flow() -> plt.Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(13.6, 6.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        FIGURE_5_10_TITLE,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )

    _add_concept_box(
        ax,
        x=0.025,
        y=0.17,
        width=0.285,
        height=0.67,
        title="Observed mode behavior",
        body=[
            "no_qos: preserves functional relevance but has weaker QoS.",
            "qos_pure_llm: improves over no_qos, showing the value of QoS-aware LLM reasoning.",
            "qos_topsis: improves QoS in some dimensions but loses functional suitability.",
            "Ranking similarity: modes expose different planner-facing APIs.",
        ],
        facecolor="#F8FAFC",
        edgecolor="#334155",
        wrap_chars=38,
        body_size=8.4,
    )

    _add_concept_box(
        ax,
        x=0.355,
        y=0.31,
        width=0.18,
        height=0.39,
        title="Design implication",
        body=["Functional suitability must be preserved before QoS optimization."],
        facecolor="#FFF7ED",
        edgecolor="#C2410C",
        wrap_chars=24,
        body_size=9.6,
    )

    hybrid_x = 0.585
    hybrid_y = 0.13
    hybrid_w = 0.235
    hybrid_h = 0.75
    hybrid_patch = FancyBboxPatch(
        (hybrid_x, hybrid_y),
        hybrid_w,
        hybrid_h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.35,
        edgecolor="#047857",
        facecolor="#ECFDF5",
    )
    ax.add_patch(hybrid_patch)
    ax.text(
        hybrid_x + 0.018,
        hybrid_y + hybrid_h - 0.055,
        "Hybrid selection rule",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )
    step_boxes = [
        ("Step 1", "prioritize Functional Match Label = 1 APIs."),
        ("Step 2", "apply TOPSIS over response time, throughput, and availability."),
        ("Step 3", "pass functionally suitable, QoS-refined candidates to the planner."),
    ]
    step_height = 0.15
    step_width = hybrid_w - 0.044
    step_x = hybrid_x + 0.022
    step_ys = [0.605, 0.405, 0.205]
    for idx, ((step_label, step_text), step_y) in enumerate(zip(step_boxes, step_ys)):
        step_patch = FancyBboxPatch(
            (step_x, step_y),
            step_width,
            step_height,
            boxstyle="round,pad=0.01,rounding_size=0.014",
            linewidth=1.0,
            edgecolor="#34A07A",
            facecolor="#FFFFFF",
        )
        ax.add_patch(step_patch)
        ax.text(
            step_x + 0.014,
            step_y + step_height - 0.034,
            step_label,
            ha="left",
            va="top",
            fontsize=8.8,
            fontweight="bold",
            color="#065F46",
        )
        ax.text(
            step_x + 0.014,
            step_y + step_height - 0.072,
            _wrapped_text(step_text, 30),
            ha="left",
            va="top",
            fontsize=8.0,
            color="#1F2937",
            linespacing=1.15,
        )
        if idx < len(step_ys) - 1:
            _add_flow_arrow(ax, (hybrid_x + hybrid_w / 2, step_y - 0.005), (hybrid_x + hybrid_w / 2, step_ys[idx + 1] + step_height + 0.005))

    _add_concept_box(
        ax,
        x=0.86,
        y=0.33,
        width=0.12,
        height=0.35,
        title="Outcome",
        body=["qos_hybrid combines functional-first selection with QoS-aware refinement."],
        facecolor="#EFF6FF",
        edgecolor="#1D4ED8",
        wrap_chars=16,
        title_size=10.5,
        body_size=8.8,
    )

    _add_flow_arrow(ax, (0.31, 0.505), (0.355, 0.505))
    _add_flow_arrow(ax, (0.535, 0.505), (0.585, 0.505))
    _add_flow_arrow(ax, (0.82, 0.505), (0.86, 0.505))

    fig.tight_layout(pad=0.5)
    return fig


def plot_case_similarity_heatmap(matrix: pd.DataFrame, metric_label: str) -> plt.Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    values = matrix.astype(float)
    vmin, vmax = (-1, 1) if "Spearman" in metric_label else (0, 1)
    image = ax.imshow(values.values, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_title(f"Case-Level {metric_label}")
    ax.set_xticks(np.arange(len(values.columns)))
    ax.set_yticks(np.arange(len(values.index)))
    ax.set_xticklabels(values.columns, rotation=30, ha="right")
    ax.set_yticklabels(values.index)
    for i in range(len(values.index)):
        for j in range(len(values.columns)):
            value = values.iloc[i, j]
            ax.text(j, i, "" if pd.isna(value) else f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_overlap_by_depth(depth_df: pd.DataFrame, left_mode: str, right_mode: str) -> plt.Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(depth_df["depth"], depth_df["overlap_ratio"], marker="o", color="#4C78A8")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Overlap ratio")
    ax.set_title(f"Average Overlap by Depth: {left_mode} vs {right_mode}")
    fig.tight_layout()
    return fig


def plot_agreement_rate_by_query(agreement_df: pd.DataFrame) -> plt.Figure:
    configure_matplotlib()
    working = agreement_df.copy()
    working["top1_agreement_rate"] = (working["top1_agreement_status"] == "agreement").astype(float)
    summary = working.groupby("query_id", dropna=False)["top1_agreement_rate"].mean().reset_index()
    summary = summary.sort_values("query_id")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(summary["query_id"].astype(str), summary["top1_agreement_rate"], color="#54A24B")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Query ID")
    ax.set_ylabel("Top-1 agreement rate")
    ax.set_title("LLM QoS and TOPSIS Top-1 Agreement Rate by Query")
    fig.tight_layout()
    return fig


def plot_query_grouped_scores(composition_df: pd.DataFrame, selected_query_ids: list[str], selected_modes: list[str]) -> plt.Figure:
    configure_matplotlib()
    matrix, _best_df, _warnings = build_query_score_matrix(composition_df, selected_query_ids, selected_modes)
    modes = [mode for mode in selected_modes if mode in matrix.columns]
    pivot = matrix.set_index("Query_ID")[modes] if not matrix.empty and modes else pd.DataFrame(columns=modes)
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(modes))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    colors = _mode_colors(modes)
    for idx, mode in enumerate(modes):
        values = _as_numeric(pivot[mode]) if mode in pivot else pd.Series(dtype=float)
        ax.bar(x + (idx - (len(modes) - 1) / 2) * width, values, width, label=mode, color=colors[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=0, ha="center")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("QoS-Adjusted Composition Score")
    ax.set_xlabel("Query ID")
    ax.set_title("")
    if modes:
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    fig.tight_layout()
    return fig


def plot_winner_tie_heatmap(best_df: pd.DataFrame, selected_modes: list[str]) -> plt.Figure:
    configure_matplotlib()
    queries = sorted_query_ids(best_df["Query_ID"]) if not best_df.empty else []
    matrix = pd.DataFrame(0, index=queries, columns=selected_modes, dtype=int)
    for _, row in best_df.iterrows():
        query_id = str(row["Query_ID"])
        best_modes = [part.strip() for part in str(row.get("Best_Modes") or "").split(",") if part.strip()]
        value = 2 if len(best_modes) > 1 else 1
        for mode in best_modes:
            if mode in matrix.columns and query_id in matrix.index:
                matrix.loc[query_id, mode] = value
    cmap = ListedColormap(["#F3F4F6", "#54A24B", "#F58518"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.6, max(4.8, 0.38 * len(queries))))
    ax.imshow(matrix.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticklabels(matrix.index)
    for row_idx, query_id in enumerate(matrix.index):
        for col_idx, mode in enumerate(matrix.columns):
            value = int(matrix.loc[query_id, mode])
            label = "Best" if value == 1 else "Tie" if value == 2 else ""
            if label:
                ax.text(
                    col_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    color="#111827",
                    fontsize=9,
                    fontweight="bold",
                )
    ax.set_title("")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Query ID")
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    legend_labels = ["not best", "unique best", "tied best"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(idx)) for idx in range(3)]
    ax.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    return fig


def plot_per_query_score_by_mode(table: pd.DataFrame, query_id: str) -> plt.Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    modes = table["Mode"].astype(str).tolist()
    ax.bar(modes, _as_numeric(table[DEFAULT_SCORE_COL]), color=[_mode_colors(modes)[mode] for mode in modes])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Mode")
    ax.set_ylabel("QoS-adjusted composition score")
    ax.set_title(f"Final Score by Mode for {query_id}")
    fig.tight_layout()
    return fig


def plot_per_query_components(table: pd.DataFrame, query_id: str) -> plt.Figure:
    configure_matplotlib()
    modes = table["Mode"].astype(str).tolist()
    x = np.arange(len(modes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar(x - width / 2, _as_numeric(table["Functional_Coverage"]), width, label="Functional coverage", color="#4C78A8")
    ax.bar(x + width / 2, _as_numeric(table["Normalized_QoS_Score"]), width, label="Normalized QoS score", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_xlabel("Mode")
    ax.set_title(f"Functional Coverage and Normalized QoS by Mode for {query_id}")
    ax.legend()
    fig.tight_layout()
    return fig
