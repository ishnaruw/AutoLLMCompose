from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
METRIC_NAMES = ["spearman", "average_overlap", "rbo", "jaccard"]
DEFAULT_RBO_P = 0.9
FALLBACK_K = 5

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
        "api relevancy 0 1",
        "api_relevancy_0_1",
        "api relevancy",
        "api relevance 0 1",
        "relevancy 0 1",
        "relevance 0 1",
        "relevant",
        "relevance",
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
    top_lists: Dict[str, List[str]]


@dataclass(frozen=True)
class EvaluationBundle:
    cases: List[RankingCase]
    matrices: Dict[str, pd.DataFrame]
    pairwise_scores: pd.DataFrame
    warnings: List[str]
    raw_rows: pd.DataFrame
    discovered_run_dirs: List[str]
    loaded_report_paths: List[str]


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
    if report_path.parent.name in {"evaluation", "relevancy_eval"}:
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
        "evaluation/query_*_api_relevancy.xlsx",
        "relevancy_eval/query_*_api_relevancy.xlsx",
        "query_*_api_relevancy.xlsx",
        "evaluation/*api_relevancy*.xlsx",
        "relevancy_eval/*api_relevancy*.xlsx",
        "**/*api_relevancy*.xlsx",
        "evaluation/*rank*.xlsx",
        "relevancy_eval/*rank*.xlsx",
        "**/*rank*.xlsx",
        "evaluation/*report*.xlsx",
        "relevancy_eval/*report*.xlsx",
        "**/*report*.xlsx",
        "evaluation/*.xlsx",
        "relevancy_eval/*.xlsx",
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
    """Load and normalize one completed MAOF ranking/relevancy Excel report."""
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

    out = out[out["mode"].isin(MODE_ORDER)]
    out = out[out["mode_rank"].notna()]
    out = out[out["api_id"].ne("") & out["api_id"].str.lower().ne("nan")]
    out = out[out["api_id"].str.lower().ne("precision")]
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
            warnings.append(f"Skipped {run_dir}: no Excel ranking/relevancy report found.")
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
    return _ordered_unique(ordered["api_id"].tolist())[:k]


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


def build_ranking_cases(df: pd.DataFrame) -> Tuple[List[RankingCase], List[str]]:
    warnings: List[str] = []
    cases: List[RankingCase] = []
    if df.empty:
        return cases, warnings

    group_cols = ["run_dir", "report_path", "query_id", "subtask_id"]
    for (run_dir, report_path, query_id, subtask_id), case_df in df.groupby(group_cols, dropna=False):
        query_id = str(query_id)
        subtask_id = str(subtask_id)
        case_name = _format_case_name(query_id, subtask_id, str(run_dir))

        modes = set(case_df["mode"].dropna().astype(str))
        missing_modes = [mode for mode in MODE_ORDER if mode not in modes]
        if missing_modes:
            warnings.append(f"Skipped {case_name}: missing modes {', '.join(missing_modes)}.")
            continue

        hybrid = case_df[case_df["mode"] == "qos_hybrid"]
        k = int((hybrid["functional_match_label"] == 1).sum())
        fallback_used = k == 0
        if fallback_used:
            k = FALLBACK_K
            warnings.append(
                f"{case_name}: qos_hybrid Functional Match Label count was 0; using fallback K={FALLBACK_K}."
            )

        planner_k = _planner_k_values(case_df)
        if len(planner_k) > 1 or (planner_k and planner_k[0] != k):
            warnings.append(
                f"{case_name}: report Planner Selection K values {planner_k} differ from evaluation K={k}; "
                "ranking evaluation uses K derived from qos_hybrid functional matches."
            )

        selected_counts = _selected_counts_by_mode(case_df)
        if selected_counts:
            missing_selected = [mode for mode in MODE_ORDER if mode not in selected_counts]
            differing_selected = {
                mode: count
                for mode, count in selected_counts.items()
                if mode in MODE_ORDER and count != k
            }
            if missing_selected or differing_selected:
                warnings.append(
                    f"{case_name}: Selected for Planner counts do not match evaluation K={k}; "
                    f"counts={selected_counts or {}}. Metrics still use top K by Mode Rank."
                )

        top_lists: Dict[str, List[str]] = {}
        too_short: List[str] = []
        for mode in MODE_ORDER:
            mode_rows = case_df[case_df["mode"] == mode]
            top_lists[mode] = _build_top_list(mode_rows, k)
            if len(top_lists[mode]) < k:
                too_short.append(f"{mode} ({len(top_lists[mode])}/{k})")
        if too_short:
            warnings.append(f"Skipped {case_name}: not enough unique ranked APIs for K={k}: {', '.join(too_short)}.")
            continue

        cases.append(
            RankingCase(
                case_id=f"{Path(str(run_dir)).name}:{query_id}:subtask_{subtask_id}",
                query_id=query_id,
                subtask_id=subtask_id,
                run_dir=str(run_dir),
                report_path=str(report_path),
                k=k,
                k_fallback_used=fallback_used,
                top_lists=top_lists,
            )
        )

    return cases, warnings


def _metric_k(left: Sequence[str], right: Sequence[str], k: int | None) -> int:
    return int(k if k is not None else max(len(left), len(right)))


def _clip(value: float, lower: float, upper: float) -> float:
    if np.isnan(value):
        return lower
    return float(min(max(value, lower), upper))


def spearman_union(left: Sequence[str], right: Sequence[str], k: int | None = None) -> float:
    """Spearman correlation over the union, assigning missing APIs rank K + 1."""
    k = _metric_k(left, right, k)
    left_top = _ordered_unique(left)[:k]
    right_top = _ordered_unique(right)[:k]
    union = _ordered_unique([*left_top, *right_top])
    if len(union) < 2:
        return 1.0

    left_rank = {api: idx + 1 for idx, api in enumerate(left_top)}
    right_rank = {api: idx + 1 for idx, api in enumerate(right_top)}
    x = np.array([left_rank.get(api, k + 1) for api in union], dtype=float)
    y = np.array([right_rank.get(api, k + 1) for api in union], dtype=float)

    if np.std(x) == 0 or np.std(y) == 0:
        return 1.0 if np.array_equal(x, y) else 0.0

    return _clip(float(np.corrcoef(x, y)[0, 1]), -1.0, 1.0)


def average_overlap(left: Sequence[str], right: Sequence[str], k: int | None = None) -> float:
    k = _metric_k(left, right, k)
    if k <= 0:
        return 1.0

    left_top = _ordered_unique(left)[:k]
    right_top = _ordered_unique(right)[:k]
    scores = []
    for depth in range(1, k + 1):
        overlap = len(set(left_top[:depth]) & set(right_top[:depth]))
        scores.append(overlap / float(depth))
    return _clip(float(np.mean(scores)), 0.0, 1.0)


def overlap_by_depth(left: Sequence[str], right: Sequence[str], k: int | None = None) -> pd.DataFrame:
    k = _metric_k(left, right, k)
    left_top = _ordered_unique(left)[:k]
    right_top = _ordered_unique(right)[:k]
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

    left_top = _ordered_unique(left)[:k]
    right_top = _ordered_unique(right)[:k]
    weighted = 0.0
    last_agreement = 0.0
    for depth in range(1, k + 1):
        last_agreement = len(set(left_top[:depth]) & set(right_top[:depth])) / float(depth)
        weighted += (p ** (depth - 1)) * last_agreement
    return _clip(float((1.0 - p) * weighted + (p**k) * last_agreement), 0.0, 1.0)


def jaccard_similarity(left: Sequence[str], right: Sequence[str], k: int | None = None) -> float:
    k = _metric_k(left, right, k)
    left_set = set(_ordered_unique(left)[:k])
    right_set = set(_ordered_unique(right)[:k])
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
        return spearman_union(left, right, k)
    if metric == "average_overlap":
        return average_overlap(left, right, k)
    if metric == "rbo":
        return rbo_score(left, right, k, p=p)
    if metric == "jaccard":
        return jaccard_similarity(left, right, k)
    raise ValueError(f"Unsupported metric: {metric}")


def compute_case_matrices(case: RankingCase, p: float = DEFAULT_RBO_P) -> Dict[str, pd.DataFrame]:
    matrices: Dict[str, pd.DataFrame] = {}
    for metric in METRIC_NAMES:
        matrix = pd.DataFrame(index=MODE_ORDER, columns=MODE_ORDER, dtype=float)
        for left_idx, left_mode in enumerate(MODE_ORDER):
            for right_idx, right_mode in enumerate(MODE_ORDER):
                if left_idx == right_idx:
                    value = 1.0
                else:
                    value = pairwise_metric(
                        case.top_lists[left_mode],
                        case.top_lists[right_mode],
                        metric,
                        case.k,
                        p=p,
                    )
                matrix.loc[left_mode, right_mode] = value
        matrices[metric] = matrix.astype(float)
    return matrices


def aggregate_matrices(cases: Sequence[RankingCase], p: float = DEFAULT_RBO_P) -> Dict[str, pd.DataFrame]:
    if not cases:
        return {
            metric: pd.DataFrame(np.eye(len(MODE_ORDER)), index=MODE_ORDER, columns=MODE_ORDER)
            for metric in METRIC_NAMES
        }

    sums = {
        metric: pd.DataFrame(0.0, index=MODE_ORDER, columns=MODE_ORDER)
        for metric in METRIC_NAMES
    }
    for case in cases:
        for metric, matrix in compute_case_matrices(case, p=p).items():
            sums[metric] = sums[metric].add(matrix, fill_value=0.0)

    return {metric: matrix / float(len(cases)) for metric, matrix in sums.items()}


def matrices_to_pairwise_table(matrices: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric, matrix in matrices.items():
        for left_idx, left_mode in enumerate(MODE_ORDER):
            for right_mode in MODE_ORDER[left_idx + 1 :]:
                rows.append(
                    {
                        "metric": metric,
                        "mode_a": left_mode,
                        "mode_b": right_mode,
                        "score": float(matrix.loc[left_mode, right_mode]),
                    }
                )
    return pd.DataFrame(rows, columns=["metric", "mode_a", "mode_b", "score"])


def evaluate_parent_runs(parent_runs_dir: str | Path, p: float = DEFAULT_RBO_P) -> EvaluationBundle:
    raw_rows, load_warnings, discovered_run_dirs, loaded_report_paths = load_parent_runs(parent_runs_dir)
    cases, case_warnings = build_ranking_cases(raw_rows)
    matrices = aggregate_matrices(cases, p=p)
    return EvaluationBundle(
        cases=cases,
        matrices=matrices,
        pairwise_scores=matrices_to_pairwise_table(matrices),
        warnings=load_warnings + case_warnings,
        raw_rows=raw_rows,
        discovered_run_dirs=discovered_run_dirs,
        loaded_report_paths=loaded_report_paths,
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
        }
        for case in cases
    ]
    return pd.DataFrame(rows, columns=columns)


def top_lists_to_frame(case: RankingCase) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for mode in MODE_ORDER:
        for rank, api_id in enumerate(case.top_lists[mode], start=1):
            rows.append({"mode": mode, "rank": rank, "api_id": api_id})
    return pd.DataFrame(rows, columns=["mode", "rank", "api_id"])


def top_lists_to_wide_frame(case: RankingCase) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx in range(case.k):
        row: Dict[str, Any] = {"rank": idx + 1}
        for mode in MODE_ORDER:
            row[mode] = case.top_lists[mode][idx] if idx < len(case.top_lists[mode]) else ""
        rows.append(row)
    return pd.DataFrame(rows, columns=["rank", *MODE_ORDER])
