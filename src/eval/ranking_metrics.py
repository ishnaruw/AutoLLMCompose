from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
METRIC_NAMES = ["spearman", "average_overlap", "rbo", "jaccard"]
DEFAULT_RBO_P = 0.9
FALLBACK_K = 5
STRICT_SELECTED_MODES = "strict_selected_modes"
STRICT_ALL_MODES = "strict_all_modes"  # Backward-compatible alias.
PAIRWISE_AVAILABLE = "pairwise_available"  # Internal/debug only.
INCLUSION_POLICIES = [STRICT_SELECTED_MODES, PAIRWISE_AVAILABLE, STRICT_ALL_MODES]
DEFAULT_INCLUSION_POLICY = STRICT_SELECTED_MODES

REQUIRED_COLUMNS = {
    "query_id",
    "mode",
    "subtask_id",
    "api_id",
    "mode_rank",
    "functional_match_label",
}

PREFERRED_SHEETS = [
    "Ranked APIs",
    "Ranked API",
    "All Candidates",
    "Candidates",
    "Query",
]

CANONICAL_COLUMNS = {
    "query_id": {
        "query id",
        "query_id",
        "queryid",
        "query",
        "qid",
    },
    "mode": {
        "mode",
        "ranking mode",
    },
    "subtask_id": {
        "sub task",
        "sub_task",
        "subtask",
        "subtask id",
        "sub task id",
        "task",
        "task id",
    },
    "api_id": {
        "api id",
        "api_id",
        "api name",
        "api_name",
        "selected api",
        "selected_api",
        "selected api id",
        "selected api name",
        "tool",
        "tool name",
        "service",
        "service name",
        "endpoint",
        "endpoint name",
    },
    "mode_rank": {
        "mode rank",
        "mode_rank",
        "rank",
        "final rank",
        "final_rank",
        "final mode rank",
        "rank within mode",
    },
    "functional_match_label": {
        "functional match label",
        "functional_match_label",
        "functional match",
        "functional match 0 1",
        "functional_match_0_1",
        "relevant",
        "label",
    },
    "planner_selection_k": {
        "planner selection k",
        "planner_selection_k",
        "planner k",
        "planner_k",
        "selection k",
        "selected k",
        "k",
    },
    "selected_for_planner": {
        "selected for planner",
        "selected_for_planner",
        "planner selected",
        "used by planner",
        "selected",
    },
    "failure_flag": {
        "failure flag",
        "failure_flag",
        "failed",
        "invalid",
    },
    "failure_stage": {
        "failure stage",
        "failure_stage",
    },
    "failure_reason": {
        "failure reason",
        "failure_reason",
        "invalid reason",
        "exclusion reason",
    },
    "exclude_from_ranking_eval": {
        "exclude from ranking eval",
        "exclude_from_ranking_eval",
        "exclude ranking eval",
        "excluded from ranking eval",
        "ranking eval excluded",
    },
    "expected_api_count": {
        "expected api count",
        "expected_api_count",
    },
    "actual_api_count": {
        "actual api count",
        "actual_api_count",
    },
    "returned_api_count": {
        "returned api count",
        "returned_api_count",
    },
    "duplicate_api_ids": {
        "duplicate api ids",
        "duplicate_api_ids",
        "duplicated apis",
        "duplicated_apis",
    },
    "missing_api_ids": {
        "missing api ids",
        "missing_api_ids",
    },
    "unknown_api_ids": {
        "unknown api ids",
        "unknown_api_ids",
    },
    "is_hallucinated": {
        "is hallucinated 0 1",
        "is_hallucinated_0_1",
        "is hallucinated",
        "is_hallucinated",
    },
    "is_duplicated": {
        "is duplicated 0 1",
        "is_duplicated_0_1",
        "is duplicated",
        "is_duplicated",
    },
    "ranking_anomaly": {
        "ranking anomaly",
        "ranking_anomaly",
    },
    "ranking_anomaly_reason": {
        "ranking anomaly reason",
        "ranking_anomaly_reason",
    },
}

MODE_ALIASES = {
    "no qos": "no_qos",
    "no_qos": "no_qos",
    "noqos": "no_qos",
    "qos pure llm": "qos_pure_llm",
    "qos_pure_llm": "qos_pure_llm",
    "pure llm": "qos_pure_llm",
    "qos topsis": "qos_topsis",
    "qos_topsis": "qos_topsis",
    "topsis": "qos_topsis",
    "qos hybrid": "qos_hybrid",
    "qos_hybrid": "qos_hybrid",
    "hybrid": "qos_hybrid",
}

TRUE_VALUES = {"1", "true", "t", "yes", "y", "selected", "relevant", "match"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "not selected", "irrelevant", "nonmatch", "no match"}


@dataclass(frozen=True)
class RankingCase:
    case_id: str
    query_id: str
    subtask_id: str
    run_dir: str
    report_path: str
    k: int
    k_fallback_used: bool
    ranked_lists: Dict[str, List[str]]
    top_lists: Dict[str, List[str]]
    valid_modes: List[str]
    selected_modes: List[str]


@dataclass(frozen=True)
class EvaluationBundle:
    cases: List[RankingCase]
    matrices: Dict[str, pd.DataFrame]
    pairwise_counts: Dict[str, pd.DataFrame]
    pairwise_scores: pd.DataFrame
    warnings: List[str]
    raw_rows: pd.DataFrame
    invalid_cases: pd.DataFrame
    discovered_run_dirs: List[str]
    loaded_report_paths: List[str]
    inclusion_policy: str
    selected_modes: List[str]


def _normalize_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _preferred_sheet_weight(sheet_name: str) -> int:
    normalized = _normalize_name(sheet_name)
    for index, preferred in enumerate(PREFERRED_SHEETS):
        if normalized == _normalize_name(preferred):
            return len(PREFERRED_SHEETS) - index
    return 0


def _display_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _column_lookup() -> Dict[str, str]:
    return {
        _normalize_name(variant): canonical
        for canonical, variants in CANONICAL_COLUMNS.items()
        for variant in variants
    }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common report column variants to canonical snake_case names."""
    lookup = _column_lookup()
    rename: Dict[Any, str] = {}
    assigned: set[str] = set()

    for column in df.columns:
        canonical = lookup.get(_normalize_name(column))
        if canonical and canonical not in assigned:
            rename[column] = canonical
            assigned.add(canonical)

    return df.rename(columns=rename).copy()


def _canonical_mode(value: Any) -> str:
    raw = str(value).strip()
    normalized = _normalize_name(raw)
    return MODE_ALIASES.get(normalized, MODE_ALIASES.get(normalized.replace(" ", "_"), raw))


def _parse_binary(value: Any, default: int = 0) -> int:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) == 1 else 0
    if isinstance(value, (float, np.floating)):
        return 1 if float(value) == 1.0 else 0

    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return 1
    if text in FALSE_VALUES:
        return 0
    try:
        return 1 if float(text) == 1.0 else 0
    except ValueError:
        return default


def _parse_optional_binary(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return 1.0
    if text in FALSE_VALUES:
        return 0.0
    try:
        numeric = float(text)
    except ValueError:
        return np.nan
    if numeric in (0.0, 1.0):
        return numeric
    return np.nan


def _infer_query_id(path: Path) -> str:
    for part in [path.stem, *[parent.name for parent in path.parents]]:
        match = re.search(r"(q[0-9a-zA-Z]+)", part)
        if match:
            return match.group(1)
    return ""


def _infer_run_dir(report_path: Path) -> Path:
    if report_path.parent.name in {"evaluation", "functional_match_eval"}:
        return report_path.parent.parent
    return report_path.parent


def _looks_like_query_run_dir(path: Path) -> bool:
    return path.is_dir() and path.name.lower().startswith("q")


def discover_query_run_dirs(parent_runs_dir: str | Path) -> List[Path]:
    """Return immediate q* run folders under a parent directory.

    If the supplied path itself looks like a single query run folder, it is
    returned as the only entry. This makes the helper convenient for debugging.
    """
    parent = Path(parent_runs_dir).expanduser()
    if not parent.exists():
        return []

    children = sorted(path for path in parent.iterdir() if _looks_like_query_run_dir(path))
    if children:
        return children
    return [parent] if _looks_like_query_run_dir(parent) else []


def find_report_files(run_dir: str | Path) -> List[Path]:
    run_path = Path(run_dir).expanduser()
    if not run_path.exists():
        return []

    patterns = [
        "evaluation/query_*_candidate_api_rankings.xlsx",
        "functional_match_eval/query_*_candidate_api_rankings.xlsx",
        "query_*_candidate_api_rankings.xlsx",
        "evaluation/*candidate_api_rankings*.xlsx",
        "functional_match_eval/*candidate_api_rankings*.xlsx",
        "**/*candidate_api_rankings*.xlsx",
        "evaluation/*rank*.xlsx",
        "functional_match_eval/*rank*.xlsx",
        "**/*rank*.xlsx",
        "evaluation/*report*.xlsx",
        "functional_match_eval/*report*.xlsx",
        "**/*report*.xlsx",
        "evaluation/*.xlsx",
        "functional_match_eval/*.xlsx",
        "*.xlsx",
    ]

    files: List[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(run_path.glob(pattern)):
            lower_name = path.name.lower()
            if not path.is_file() or path in seen:
                continue
            if lower_name.startswith("~$") or "mode_anomal" in lower_name:
                continue
            files.append(path)
            seen.add(path)
    return files


def _sheet_score(report_path: Path, sheet_name: str) -> Tuple[int, pd.DataFrame]:
    header = pd.read_excel(report_path, sheet_name=sheet_name, nrows=0)
    normalized = normalize_columns(header)
    columns = set(normalized.columns)
    score = len(REQUIRED_COLUMNS & columns)
    score += _preferred_sheet_weight(sheet_name)
    if REQUIRED_COLUMNS <= columns:
        score += 100
    return score, normalized


def _read_ranked_sheet(report_path: Path) -> Tuple[pd.DataFrame, str]:
    excel = pd.ExcelFile(report_path)
    if not excel.sheet_names:
        raise ValueError("workbook has no sheets")

    scored = [
        (_sheet_score(report_path, sheet), sheet)
        for sheet in excel.sheet_names
    ]
    scored.sort(key=lambda item: item[0][0], reverse=True)
    best_score, best_sheet = scored[0]
    if best_score[0] == 0:
        raise ValueError("no sheet appears to contain ranking columns")

    df = pd.read_excel(report_path, sheet_name=best_sheet)
    return normalize_columns(df), best_sheet


def load_report_rows(report_path: str | Path, run_dir: str | Path | None = None) -> pd.DataFrame:
    """Load and normalize one completed AutoLLMCompose ranking/functional match Excel report."""
    path = Path(report_path).expanduser()
    df, sheet_name = _read_ranked_sheet(path)

    if "query_id" not in df.columns:
        inferred_query_id = _infer_query_id(path)
        if inferred_query_id:
            df["query_id"] = inferred_query_id

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns after normalization: {', '.join(missing)}")

    out = df.copy()
    out["source_sheet"] = sheet_name
    out["report_path"] = str(path)
    out["run_dir"] = str(Path(run_dir).expanduser() if run_dir else _infer_run_dir(path))
    out["query_id"] = out["query_id"].map(_display_id)
    out["mode"] = out["mode"].map(_canonical_mode)
    out["subtask_id"] = out["subtask_id"].map(_display_id)
    out["api_id"] = out["api_id"].map(_display_id)
    out["mode_rank"] = pd.to_numeric(out["mode_rank"], errors="coerce")
    out["functional_match_label"] = out["functional_match_label"].map(_parse_binary).astype(int)

    if "planner_selection_k" in out.columns:
        out["planner_selection_k"] = pd.to_numeric(out["planner_selection_k"], errors="coerce")

    if "selected_for_planner" in out.columns:
        out["selected_for_planner"] = out["selected_for_planner"].map(_parse_optional_binary)

    if "failure_flag" in out.columns:
        out["failure_flag"] = out["failure_flag"].map(_parse_binary).astype(int)
    else:
        out["failure_flag"] = 0

    for flag_col in ["is_hallucinated", "is_duplicated", "ranking_anomaly"]:
        if flag_col in out.columns:
            out[flag_col] = out[flag_col].map(_parse_binary).astype(int)
        else:
            out[flag_col] = 0

    if "exclude_from_ranking_eval" in out.columns:
        out["exclude_from_ranking_eval"] = out["exclude_from_ranking_eval"].map(_parse_binary).astype(int)
    else:
        out["exclude_from_ranking_eval"] = 0
    out["exclude_from_ranking_eval"] = out[["exclude_from_ranking_eval", "failure_flag"]].max(axis=1)

    out = out[out["mode"].isin(MODE_ORDER)]
    excluded = out["exclude_from_ranking_eval"] == 1
    valid_api = out["api_id"].ne("") & out["api_id"].str.lower().ne("nan")
    not_precision = out["api_id"].str.lower().ne("precision")
    out = out[excluded | out["mode_rank"].notna()]
    out = out[excluded | valid_api]
    out = out[excluded | not_precision]
    return out.reset_index(drop=True)


def load_parent_runs(parent_runs_dir: str | Path) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    warnings: List[str] = []
    frames: List[pd.DataFrame] = []
    loaded_report_paths: List[str] = []
    query_dirs = discover_query_run_dirs(parent_runs_dir)
    discovered_run_dirs = [str(path) for path in query_dirs]

    if not query_dirs:
        warnings.append(f"No query run folders were found under {Path(parent_runs_dir).expanduser()}.")

    for run_dir in query_dirs:
        reports = find_report_files(run_dir)
        if not reports:
            warnings.append(f"Skipped {run_dir}: no Excel ranking/functional match report found.")
            continue

        report_errors: List[str] = []
        for report in reports:
            try:
                rows = load_report_rows(report, run_dir=run_dir)
            except Exception as exc:  # noqa: BLE001 - keep UI warnings user-readable.
                report_errors.append(f"{report.name}: {exc}")
                continue

            if rows.empty:
                report_errors.append(f"{report.name}: no rows remained after mode/rank filtering")
                continue

            frames.append(rows)
            loaded_report_paths.append(str(report))
            break
        else:
            detail = f" Last error: {report_errors[-1]}" if report_errors else ""
            warnings.append(f"Skipped {run_dir}: no usable ranking report found.{detail}")

    if not frames:
        return pd.DataFrame(), warnings, discovered_run_dirs, loaded_report_paths

    return pd.concat(frames, ignore_index=True), warnings, discovered_run_dirs, loaded_report_paths


def _ordered_unique(values: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        text = str(value).strip()
        if text and text.lower() != "nan" and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _build_top_list(group: pd.DataFrame, k: int) -> List[str]:
    ordered = group.sort_values(["mode_rank", "api_id"], kind="mergesort")
    return [str(value).strip() for value in ordered["api_id"].tolist() if str(value).strip()][:k]


def _build_ranked_list(group: pd.DataFrame) -> List[str]:
    ordered = group.sort_values(["mode_rank", "api_id"], kind="mergesort")
    return [str(value).strip() for value in ordered["api_id"].tolist() if str(value).strip()]


def _format_case_name(query_id: str, subtask_id: str, run_dir: str) -> str:
    return f"{query_id}/subtask {subtask_id} ({Path(str(run_dir)).name})"


def _planner_k_values(case_df: pd.DataFrame) -> List[int]:
    if "planner_selection_k" not in case_df.columns:
        return []
    values: List[int] = []
    for value in case_df["planner_selection_k"].dropna().tolist():
        numeric = float(value)
        if numeric.is_integer():
            values.append(int(numeric))
    return sorted(set(values))


def _selected_counts_by_mode(case_df: pd.DataFrame) -> Dict[str, int]:
    if "selected_for_planner" not in case_df.columns:
        return {}
    selected = case_df[case_df["selected_for_planner"] == 1.0]
    if selected.empty:
        return {}
    return {
        mode: int(group["api_id"].nunique())
        for mode, group in selected.groupby("mode")
    }


def _normalize_inclusion_policy(inclusion_policy: str | None) -> str:
    policy = str(inclusion_policy or DEFAULT_INCLUSION_POLICY).strip()
    if policy == STRICT_ALL_MODES:
        return STRICT_SELECTED_MODES
    if policy not in INCLUSION_POLICIES:
        raise ValueError(
            f"Unsupported inclusion policy {policy!r}. Expected one of: {', '.join(INCLUSION_POLICIES)}"
        )
    return policy


def _normalize_selected_modes(selected_modes: Sequence[str] | None) -> List[str]:
    if selected_modes is None:
        return list(MODE_ORDER)
    normalized: List[str] = []
    for mode in selected_modes:
        canonical = _canonical_mode(mode)
        if canonical in MODE_ORDER and canonical not in normalized:
            normalized.append(canonical)
    return normalized or list(MODE_ORDER)


def _failure_details(excluded_rows: pd.DataFrame) -> List[str]:
    details: List[str] = []
    for _, row in excluded_rows.iterrows():
        mode = str(row.get("mode", ""))
        stage = str(row.get("failure_stage", ""))
        reason = str(row.get("failure_reason", ""))
        detail = "/".join(part for part in [mode, stage, reason] if part and part.lower() != "nan")
        if detail:
            details.append(detail)
    return details


def _evaluation_k(case_df: pd.DataFrame, case_name: str, warnings: List[str]) -> Tuple[int, bool, str]:
    hybrid = case_df[case_df["mode"] == "qos_hybrid"]
    if not hybrid.empty:
        k = int((hybrid["functional_match_label"] == 1).sum())
        if k > 0:
            return k, False, "qos_hybrid functional matches"
        warnings.append(
            f"{case_name}: qos_hybrid Functional Match (0/1) count was 0; using fallback K={FALLBACK_K}."
        )
        return FALLBACK_K, True, "fallback"

    planner_k = _planner_k_values(case_df)
    positive_planner_k = [value for value in planner_k if value > 0]
    if len(positive_planner_k) == 1:
        k = positive_planner_k[0]
        warnings.append(
            f"{case_name}: qos_hybrid is unavailable; using report Planner Selection K={k} "
            "for pairwise-available top-K metrics."
        )
        return k, False, "planner selection k"

    warnings.append(
        f"{case_name}: qos_hybrid is unavailable and no single Planner Selection K was available; "
        f"using fallback K={FALLBACK_K}."
    )
    return FALLBACK_K, True, "fallback"


def collect_invalid_case_rows(df: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        "run_dir",
        "report_path",
        "query_id",
        "subtask_id",
        "mode",
        "failure_stage",
        "failure_reason",
        "expected_api_count",
        "actual_api_count",
        "returned_api_count",
        "duplicate_api_ids",
        "missing_api_ids",
        "unknown_api_ids",
        "exclude_from_ranking_eval",
        "failure_flag",
    ]
    if df.empty or "exclude_from_ranking_eval" not in df.columns:
        return pd.DataFrame(columns=preferred_columns)
    invalid = df[df["exclude_from_ranking_eval"] == 1].copy()
    invalid = invalid.reindex(columns=preferred_columns)
    dedupe_columns = [
        "run_dir",
        "report_path",
        "query_id",
        "subtask_id",
        "mode",
        "failure_stage",
        "failure_reason",
    ]
    return invalid.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)


def build_ranking_cases(
    df: pd.DataFrame,
    inclusion_policy: str = DEFAULT_INCLUSION_POLICY,
    selected_modes: Sequence[str] | None = None,
) -> Tuple[List[RankingCase], List[str]]:
    """Build query/subtask ranking cases for the selected modes.

    The public/default evaluation rule is strict over selected modes: a case is
    retained only when every selected mode has a non-fatal ranking output. Metric
    functions later decide whether a partially malformed returned ranking can be
    used as-is for that specific metric.
    """
    inclusion_policy = _normalize_inclusion_policy(inclusion_policy)
    selected = _normalize_selected_modes(selected_modes)
    warnings: List[str] = []
    cases: List[RankingCase] = []
    if df.empty:
        return cases, warnings

    group_cols = ["run_dir", "report_path", "query_id", "subtask_id"]
    for (run_dir, report_path, query_id, subtask_id), case_df in df.groupby(group_cols, dropna=False):
        query_id = str(query_id)
        subtask_id = str(subtask_id)
        case_name = _format_case_name(query_id, subtask_id, str(run_dir))
        case_df = case_df[case_df["mode"].isin(selected)].copy()
        if case_df.empty:
            warnings.append(f"Skipped {case_name}: no rows for selected modes {', '.join(selected)}.")
            continue

        eval_df = case_df
        if "exclude_from_ranking_eval" in case_df.columns and (case_df["exclude_from_ranking_eval"] == 1).any():
            excluded_rows = case_df[case_df["exclude_from_ranking_eval"] == 1]
            details = _failure_details(excluded_rows)
            suffix = f": {', '.join(details)}" if details else ""
            if inclusion_policy == STRICT_SELECTED_MODES:
                warnings.append(
                    f"Excluded {case_name} from ranking evaluation because selected mode output was invalid{suffix}."
                )
                continue
            warnings.append(
                f"{case_name}: excluding invalid mode/subtask rows under internal pairwise_available policy{suffix}."
            )
            invalid_modes = set(excluded_rows["mode"].dropna().astype(str))
            eval_df = case_df[
                (case_df["exclude_from_ranking_eval"] != 1)
                & (~case_df["mode"].astype(str).isin(invalid_modes))
            ]

        modes = set(eval_df["mode"].dropna().astype(str))
        missing_modes = [mode for mode in selected if mode not in modes]
        if missing_modes and inclusion_policy == STRICT_SELECTED_MODES:
            warnings.append(f"Skipped {case_name}: missing selected modes {', '.join(missing_modes)}.")
            continue
        if missing_modes:
            warnings.append(
                f"{case_name}: missing selected modes under internal pairwise_available policy: {', '.join(missing_modes)}."
            )
        present_modes = [mode for mode in selected if mode in modes]
        if len(present_modes) < 2:
            warnings.append(f"Skipped {case_name}: fewer than two selected modes were available.")
            continue

        k, fallback_used, k_source = _evaluation_k(eval_df, case_name, warnings)

        planner_k = _planner_k_values(eval_df)
        if len(planner_k) > 1 or (planner_k and planner_k[0] != k):
            warnings.append(
                f"{case_name}: report Planner Selection K values {planner_k} differ from evaluation K={k}; "
                f"ranking evaluation uses K from {k_source}."
            )

        selected_counts = _selected_counts_by_mode(eval_df)
        if selected_counts:
            missing_selected = [mode for mode in present_modes if mode not in selected_counts]
            differing_selected = {
                mode: count
                for mode, count in selected_counts.items()
                if mode in present_modes and count != k
            }
            if missing_selected or differing_selected:
                warnings.append(
                    f"{case_name}: Selected for Planner counts do not match evaluation K={k}; "
                    f"counts={selected_counts or {}}. Metrics still use top K by Mode Rank."
                )

        ranked_lists: Dict[str, List[str]] = {}
        top_lists: Dict[str, List[str]] = {}
        too_short: List[str] = []
        anomaly_modes: List[str] = []
        for mode in present_modes:
            mode_rows = eval_df[eval_df["mode"] == mode]
            ranked_lists[mode] = _build_ranked_list(mode_rows)
            top_lists[mode] = ranked_lists[mode][:k]
            if len(top_lists[mode]) < k:
                too_short.append(f"{mode} ({len(top_lists[mode])}/{k})")
            if "ranking_anomaly" in mode_rows.columns and int(mode_rows["ranking_anomaly"].max() or 0) == 1:
                reason_values = []
                if "ranking_anomaly_reason" in mode_rows.columns:
                    reason_values = mode_rows["ranking_anomaly_reason"].dropna().unique().tolist()
                reasons = sorted(
                    str(value)
                    for value in reason_values
                    if str(value).strip() and str(value).lower() != "nan"
                )
                reason_text = f" ({', '.join(reasons)})" if reasons else ""
                anomaly_modes.append(f"{mode}{reason_text}")
        if too_short:
            warnings.append(
                f"{case_name}: selected mode ranking has fewer than K={k} returned APIs: {', '.join(too_short)}. "
                "Top-K metrics will ignore unsafe prefixes for the affected metric."
            )
        if anomaly_modes:
            warnings.append(
                f"{case_name}: ranking anomalies detected and left unmodified for selected modes: {', '.join(anomaly_modes)}."
            )

        reference_mode = present_modes[0]
        reference_set = set(ranked_lists[reference_mode]) if reference_mode in ranked_lists else set()
        mismatched_sets = [mode for mode in present_modes[1:] if set(ranked_lists[mode]) != reference_set]
        if mismatched_sets:
            warnings.append(
                f"{case_name}: full candidate sets differ across selected modes, so Spearman will be excluded "
                f"for this case. Mismatched modes: {', '.join(mismatched_sets)}."
            )

        cases.append(
            RankingCase(
                case_id=f"{Path(str(run_dir)).name}:{query_id}:subtask_{subtask_id}",
                query_id=query_id,
                subtask_id=subtask_id,
                run_dir=str(run_dir),
                report_path=str(report_path),
                k=k,
                k_fallback_used=fallback_used,
                ranked_lists=ranked_lists,
                top_lists=top_lists,
                valid_modes=present_modes,
                selected_modes=selected,
            )
        )

    return cases, warnings

def _metric_k(left: Sequence[str], right: Sequence[str], k: int | None) -> int:
    return int(k if k is not None else max(len(left), len(right)))


def _clip(value: float, lower: float, upper: float) -> float:
    if np.isnan(value):
        return lower
    return float(min(max(value, lower), upper))


def _has_duplicates(values: Sequence[str]) -> bool:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return len(cleaned) != len(set(cleaned))


def _top_k_prefix_is_safe(values: Sequence[str], k: int) -> bool:
    if k <= 0:
        return True
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if len(cleaned) < k:
        return False
    top = cleaned[:k]
    return len(top) == len(set(top))


def spearman_full(left: Sequence[str], right: Sequence[str]) -> float:
    """Standard Spearman correlation over a complete shared candidate set."""
    left_ranked = [str(value).strip() for value in left if str(value).strip()]
    right_ranked = [str(value).strip() for value in right if str(value).strip()]
    if _has_duplicates(left_ranked) or _has_duplicates(right_ranked):
        raise ValueError("standard Spearman requires duplicate-free rankings")
    left_set = set(left_ranked)
    right_set = set(right_ranked)
    if left_set != right_set:
        raise ValueError("standard Spearman requires both rankings to contain the same candidates")
    if len(left_set) < 2:
        return 1.0

    left_rank = {api: idx + 1 for idx, api in enumerate(left_ranked)}
    right_rank = {api: idx + 1 for idx, api in enumerate(right_ranked)}
    universe = sorted(left_set)
    x = np.array([left_rank[api] for api in universe], dtype=float)
    y = np.array([right_rank[api] for api in universe], dtype=float)
    statistic = spearmanr(x, y).statistic
    return _clip(float(statistic), -1.0, 1.0)

def spearman_union(left: Sequence[str], right: Sequence[str], k: int | None = None) -> float:
    """Backward-compatible alias for standard full-list Spearman.

    The optional k argument is accepted for older callers, but Spearman now
    intentionally uses the complete shared candidate ranking.
    """
    return spearman_full(left, right)


def average_overlap(left: Sequence[str], right: Sequence[str], k: int | None = None) -> float:
    k = _metric_k(left, right, k)
    if k <= 0:
        return 1.0

    if not _top_k_prefix_is_safe(left, k) or not _top_k_prefix_is_safe(right, k):
        raise ValueError("top-K metrics require each returned top-K prefix to contain K unique APIs")
    left_top = [str(value).strip() for value in left if str(value).strip()][:k]
    right_top = [str(value).strip() for value in right if str(value).strip()][:k]
    scores = []
    for depth in range(1, k + 1):
        overlap = len(set(left_top[:depth]) & set(right_top[:depth]))
        scores.append(overlap / float(depth))
    return _clip(float(np.mean(scores)), 0.0, 1.0)


def overlap_by_depth(left: Sequence[str], right: Sequence[str], k: int | None = None) -> pd.DataFrame:
    k = _metric_k(left, right, k)
    left_top = [str(value).strip() for value in left if str(value).strip()][:k]
    right_top = [str(value).strip() for value in right if str(value).strip()][:k]
    rows: List[Dict[str, Any]] = []
    for depth in range(1, k + 1):
        overlap = len(set(left_top[:depth]) & set(right_top[:depth]))
        rows.append(
            {
                "depth": depth,
                "overlap_count": overlap,
                "overlap_ratio": overlap / float(depth),
            }
        )
    return pd.DataFrame(rows)


def rbo_score(
    left: Sequence[str],
    right: Sequence[str],
    k: int | None = None,
    p: float = DEFAULT_RBO_P,
) -> float:
    """Finite extrapolated Rank-Biased Overlap for two top-K lists."""
    if not 0 < p < 1:
        raise ValueError("RBO p must be between 0 and 1.")

    k = _metric_k(left, right, k)
    if k <= 0:
        return 1.0

    if not _top_k_prefix_is_safe(left, k) or not _top_k_prefix_is_safe(right, k):
        raise ValueError("top-K metrics require each returned top-K prefix to contain K unique APIs")
    left_top = [str(value).strip() for value in left if str(value).strip()][:k]
    right_top = [str(value).strip() for value in right if str(value).strip()][:k]
    weighted = 0.0
    last_agreement = 0.0
    for depth in range(1, k + 1):
        last_agreement = len(set(left_top[:depth]) & set(right_top[:depth])) / float(depth)
        weighted += (p ** (depth - 1)) * last_agreement
    return _clip(float((1.0 - p) * weighted + (p**k) * last_agreement), 0.0, 1.0)


def jaccard_similarity(left: Sequence[str], right: Sequence[str], k: int | None = None) -> float:
    k = _metric_k(left, right, k)
    if not _top_k_prefix_is_safe(left, k) or not _top_k_prefix_is_safe(right, k):
        raise ValueError("top-K metrics require each returned top-K prefix to contain K unique APIs")
    left_set = set([str(value).strip() for value in left if str(value).strip()][:k])
    right_set = set([str(value).strip() for value in right if str(value).strip()][:k])
    union = left_set | right_set
    if not union:
        return 1.0
    return _clip(len(left_set & right_set) / float(len(union)), 0.0, 1.0)


def pairwise_metric(
    left: Sequence[str],
    right: Sequence[str],
    metric: str,
    k: int,
    p: float = DEFAULT_RBO_P,
) -> float:
    if metric == "spearman":
        return spearman_full(left, right)
    if metric == "average_overlap":
        return average_overlap(left, right, k)
    if metric == "rbo":
        return rbo_score(left, right, k, p=p)
    if metric == "jaccard":
        return jaccard_similarity(left, right, k)
    raise ValueError(f"Unsupported metric: {metric}")


def _case_metric_is_usable(case: RankingCase, metric: str) -> bool:
    selected = case.selected_modes or case.valid_modes
    if len(selected) < 2:
        return False
    if any(mode not in case.ranked_lists or mode not in case.top_lists for mode in selected):
        return False

    if metric == "spearman":
        ranked = {
            mode: [str(value).strip() for value in case.ranked_lists.get(mode, []) if str(value).strip()]
            for mode in selected
        }
        if any(len(values) < 2 or _has_duplicates(values) for values in ranked.values()):
            return False
        reference_set = set(ranked[selected[0]])
        return all(set(values) == reference_set for values in ranked.values())

    return all(_top_k_prefix_is_safe(case.top_lists.get(mode, []), case.k) for mode in selected)


def compute_case_matrices(case: RankingCase, p: float = DEFAULT_RBO_P) -> Dict[str, pd.DataFrame]:
    matrices: Dict[str, pd.DataFrame] = {}
    selected = case.selected_modes or case.valid_modes
    selected_set = set(selected)
    for metric in METRIC_NAMES:
        matrix = pd.DataFrame(index=MODE_ORDER, columns=MODE_ORDER, dtype=float)
        case_metric_usable = _case_metric_is_usable(case, metric)
        for left_idx, left_mode in enumerate(MODE_ORDER):
            for right_idx, right_mode in enumerate(MODE_ORDER):
                left_available = left_mode in case.ranked_lists and left_mode in case.top_lists
                right_available = right_mode in case.ranked_lists and right_mode in case.top_lists
                pair_in_selected_scope = left_mode in selected_set and right_mode in selected_set
                if not left_available or not right_available or not pair_in_selected_scope:
                    value = np.nan
                elif left_idx == right_idx:
                    value = 1.0 if case_metric_usable else np.nan
                elif not case_metric_usable:
                    value = np.nan
                else:
                    left_list = case.ranked_lists[left_mode] if metric == "spearman" else case.top_lists[left_mode]
                    right_list = case.ranked_lists[right_mode] if metric == "spearman" else case.top_lists[right_mode]
                    try:
                        value = pairwise_metric(left_list, right_list, metric, case.k, p=p)
                    except ValueError:
                        value = np.nan
                matrix.loc[left_mode, right_mode] = value
        matrices[metric] = matrix.astype(float)
    return matrices

def aggregate_matrices_with_counts(
    cases: Sequence[RankingCase],
    p: float = DEFAULT_RBO_P,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    if not cases:
        matrices = {
            metric: pd.DataFrame(np.eye(len(MODE_ORDER)), index=MODE_ORDER, columns=MODE_ORDER)
            for metric in METRIC_NAMES
        }
        counts = {
            metric: pd.DataFrame(0, index=MODE_ORDER, columns=MODE_ORDER, dtype=int)
            for metric in METRIC_NAMES
        }
        return matrices, counts

    sums = {
        metric: pd.DataFrame(0.0, index=MODE_ORDER, columns=MODE_ORDER)
        for metric in METRIC_NAMES
    }
    counts = {
        metric: pd.DataFrame(0, index=MODE_ORDER, columns=MODE_ORDER, dtype=int)
        for metric in METRIC_NAMES
    }
    for case in cases:
        for metric, matrix in compute_case_matrices(case, p=p).items():
            valid = matrix.notna()
            sums[metric] = sums[metric].add(matrix.fillna(0.0), fill_value=0.0)
            counts[metric] = counts[metric].add(valid.astype(int), fill_value=0).astype(int)

    matrices: Dict[str, pd.DataFrame] = {}
    for metric in METRIC_NAMES:
        denominator = counts[metric].replace(0, np.nan)
        matrices[metric] = sums[metric].divide(denominator).astype(float)
    return matrices, counts


def aggregate_matrices(cases: Sequence[RankingCase], p: float = DEFAULT_RBO_P) -> Dict[str, pd.DataFrame]:
    matrices, _ = aggregate_matrices_with_counts(cases, p=p)
    return matrices


def matrices_to_pairwise_table(
    matrices: Mapping[str, pd.DataFrame],
    counts: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric, matrix in matrices.items():
        for left_idx, left_mode in enumerate(MODE_ORDER):
            for right_mode in MODE_ORDER[left_idx + 1 :]:
                row = {
                    "metric": metric,
                    "mode_a": left_mode,
                    "mode_b": right_mode,
                    "score": float(matrix.loc[left_mode, right_mode]),
                }
                if counts is not None and metric in counts:
                    row["included_cases"] = int(counts[metric].loc[left_mode, right_mode])
                rows.append(row)
    columns = ["metric", "mode_a", "mode_b", "score"]
    if counts is not None:
        columns.append("included_cases")
    return pd.DataFrame(rows, columns=columns)


def evaluate_parent_runs(
    parent_runs_dir: str | Path,
    p: float = DEFAULT_RBO_P,
    inclusion_policy: str = DEFAULT_INCLUSION_POLICY,
    selected_modes: Sequence[str] | None = None,
) -> EvaluationBundle:
    inclusion_policy = _normalize_inclusion_policy(inclusion_policy)
    selected = _normalize_selected_modes(selected_modes)
    raw_rows, load_warnings, discovered_run_dirs, loaded_report_paths = load_parent_runs(parent_runs_dir)
    cases, case_warnings = build_ranking_cases(raw_rows, inclusion_policy=inclusion_policy, selected_modes=selected)
    matrices, pairwise_counts = aggregate_matrices_with_counts(cases, p=p)
    invalid_cases = collect_invalid_case_rows(raw_rows)
    return EvaluationBundle(
        cases=cases,
        matrices=matrices,
        pairwise_counts=pairwise_counts,
        pairwise_scores=matrices_to_pairwise_table(matrices, pairwise_counts),
        warnings=load_warnings + case_warnings,
        raw_rows=raw_rows,
        invalid_cases=invalid_cases,
        discovered_run_dirs=discovered_run_dirs,
        loaded_report_paths=loaded_report_paths,
        inclusion_policy=inclusion_policy,
        selected_modes=selected,
    )


def cases_to_frame(cases: Sequence[RankingCase]) -> pd.DataFrame:
    columns = [
        "case_id",
        "query_id",
        "subtask_id",
        "run_dir",
        "report_path",
        "k",
        "k_fallback_used",
        "ranked_count",
        "valid_modes",
        "selected_modes",
    ]
    rows = [
        {
            "case_id": case.case_id,
            "query_id": case.query_id,
            "subtask_id": case.subtask_id,
            "run_dir": case.run_dir,
            "report_path": case.report_path,
            "k": case.k,
            "k_fallback_used": case.k_fallback_used,
            "ranked_count": min((len(case.ranked_lists.get(mode, [])) for mode in case.valid_modes), default=0),
            "valid_modes": ", ".join(case.valid_modes),
            "selected_modes": ", ".join(case.selected_modes),
        }
        for case in cases
    ]
    return pd.DataFrame(rows, columns=columns)


def top_lists_to_frame(case: RankingCase) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for mode in case.valid_modes:
        for rank, api_id in enumerate(case.top_lists[mode], start=1):
            rows.append({"mode": mode, "rank": rank, "api_id": api_id})
    return pd.DataFrame(rows, columns=["mode", "rank", "api_id"])


def top_lists_to_wide_frame(case: RankingCase) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx in range(case.k):
        row: Dict[str, Any] = {"rank": idx + 1}
        for mode in MODE_ORDER:
            mode_top = case.top_lists.get(mode, [])
            row[mode] = mode_top[idx] if idx < len(mode_top) else ""
        rows.append(row)
    return pd.DataFrame(rows, columns=["rank", *MODE_ORDER])
