#!/usr/bin/env python3
"""Consolidate per-query composition evaluation artifacts into summary CSVs.

Given a completed AutoLLMCompose run folder containing qXX_* query folders, this
script writes:

  - all_15_query_composition_results.csv
  - aggregate_mode_scores.csv

By default the files are written to <run-folder>/summary.  The implementation is
intentionally deterministic: query folders are sorted by query id, mode rows use
the evaluator order when present, duplicate query folders are rejected, and every
output value is derived from saved evaluation artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


MODE_ORDER = ("no_qos", "qos_hybrid", "qos_pure_llm", "qos_topsis")
QUERY_DIR_RE = re.compile(r"^(q\d{2})_")
QUERY_ID_RE = re.compile(r"^q?(\d+)$", re.IGNORECASE)
TIE_TOLERANCE = 1e-12

DETAILED_COLUMNS = [
    "Query_ID",
    "Query_Text",
    "Run_Folder",
    "Mode",
    "Composition_Validity",
    "Invalid_Reason",
    "Planned_API_Count",
    "Covered_Subtask_Count",
    "Total_Subtask_Count",
    "Composition_Completeness",
    "Composition_Completeness_Gate",
    "Functional_Coverage",
    "Total_Response_Time_s",
    "Bottleneck_Throughput_kbps",
    "Average_Workflow_Availability",
    "Normalized_Response_Time_Score",
    "Normalized_Throughput_Score",
    "Normalized_Availability_Score",
    "Normalized_QoS_Score",
    "QoS_Adjusted_Composition_Score",
    "Source_File",
    "Planner_Output_File",
]

AGGREGATE_COLUMNS = [
    "Mode",
    "Query_Count",
    "Average_QoS_Adjusted_Composition_Score",
    "Std_QoS_Adjusted_Composition_Score",
    "Average_Composition_Validity",
    "Average_Composition_Completeness",
    "Average_Functional_Coverage",
    "Average_Normalized_QoS_Score",
    "Average_Total_Response_Time_s",
    "Average_Bottleneck_Throughput_kbps",
    "Average_Average_Workflow_Availability",
    "Complete_Composition_Count",
    "Invalid_Composition_Count",
    "Best_Query_Count",
]

AGGREGATE_AVERAGE_FIELDS = {
    "Average_QoS_Adjusted_Composition_Score": "QoS_Adjusted_Composition_Score",
    "Average_Composition_Validity": "Composition_Validity",
    "Average_Composition_Completeness": "Composition_Completeness",
    "Average_Functional_Coverage": "Functional_Coverage",
    "Average_Normalized_QoS_Score": "Normalized_QoS_Score",
    "Average_Total_Response_Time_s": "Total_Response_Time_s",
    "Average_Bottleneck_Throughput_kbps": "Bottleneck_Throughput_kbps",
    "Average_Average_Workflow_Availability": "Average_Workflow_Availability",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate consolidated composition-result CSVs from a completed "
            "AutoLLMCompose run folder."
        )
    )
    parser.add_argument("run_folder", type=Path, help="Folder containing qXX_* result folders.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to <run_folder>/summary.",
    )
    parser.add_argument(
        "--include-query-ids",
        help="Optional comma-separated query filter, e.g. q01,q02,q15.",
    )
    parser.add_argument(
        "--allow-missing-modes",
        action="store_true",
        help="Allow query rows that do not contain all canonical modes.",
    )
    return parser.parse_args()


def parse_query_id(value: str | int) -> str:
    if isinstance(value, int):
        number = value
    else:
        match = QUERY_ID_RE.fullmatch(str(value).strip())
        if not match:
            raise ValueError(f"Invalid query id {value!r}; use q01, 01, or 1.")
        number = int(match.group(1))
    if number <= 0:
        raise ValueError(f"Query id must be positive: {value!r}")
    return f"q{number:02d}"


def parse_query_ids(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    raw_values: Iterable[str]
    if isinstance(value, str):
        raw_values = value.split(",")
    else:
        raw_values = value
    query_ids: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        text = str(raw).strip()
        if not text:
            continue
        query_id = parse_query_id(text)
        if query_id not in seen:
            query_ids.append(query_id)
            seen.add(query_id)
    if not query_ids:
        raise ValueError("At least one query id is required when --include-query-ids is used.")
    return query_ids


def query_sort_key(query_id: str) -> tuple[int, str]:
    try:
        return (int(query_id[1:]), query_id)
    except (TypeError, ValueError):
        return (10_000, str(query_id))


def mode_sort_key(mode: str) -> tuple[int, str]:
    try:
        return (MODE_ORDER.index(mode), mode)
    except ValueError:
        return (len(MODE_ORDER), mode)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def resolve_run_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Run folder does not exist or is not a directory: {path}")
    return path.resolve()


def resolve_output_dir(run_root: Path, output_dir: str | Path | None) -> Path:
    if output_dir is None:
        path = run_root / "summary"
    else:
        path = Path(output_dir).expanduser()
        if not path.is_absolute():
            path = run_root / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def discover_query_runs(run_root: Path) -> dict[str, Path]:
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
            f"{query_id}: {', '.join(p.name for p in paths)}"
            for query_id, paths in sorted(duplicates.items(), key=lambda item: query_sort_key(item[0]))
        )
        raise ValueError(f"Ambiguous duplicate query folders under {run_root}: {details}")
    if not runs:
        raise ValueError(f"No qXX_* query folders found under {run_root}")
    return {query_id: runs[query_id] for query_id in sorted(runs, key=query_sort_key)}


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_meta(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "meta.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"meta.json must contain an object: {path}")
    return payload


def query_text_from_meta(meta: dict[str, Any], query_id: str) -> str:
    for key in ("user_goal", "query_text", "Query_Text", "goal", "query_title"):
        value = meta.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return query_id


def rows_path_from_run(run_dir: Path, query_id: str) -> Path:
    expected = run_dir / "evaluation" / f"query_{query_id}_composition_qos_eval_rows.json"
    if expected.exists():
        return expected

    eval_result_path = run_dir / "evaluation_result.json"
    if eval_result_path.exists():
        payload = read_json(eval_result_path)
        if isinstance(payload, dict):
            candidate = payload.get("composition_qos_eval_rows_json")
            if candidate:
                candidate_path = Path(str(candidate))
                if not candidate_path.is_absolute():
                    candidate_path = run_dir / candidate_path
                if candidate_path.exists():
                    return candidate_path

    raise FileNotFoundError(f"Missing composition evaluation rows for {query_id}: {expected}")


def coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def is_truthy_one(value: Any) -> bool:
    number = coerce_float(value)
    if number is None:
        return str(value).strip().lower() in {"true", "valid", "yes"}
    return math.isclose(number, 1.0, rel_tol=TIE_TOLERANCE, abs_tol=TIE_TOLERANCE)


def is_invalid(value: Any) -> bool:
    number = coerce_float(value)
    if number is None:
        return bool(str(value).strip())
    return not math.isclose(number, 1.0, rel_tol=TIE_TOLERANCE, abs_tol=TIE_TOLERANCE)


def format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    return format(float(value), ".12g")


def load_query_rows(
    run_root: Path,
    query_id: str,
    run_dir: Path,
    *,
    allow_missing_modes: bool = False,
) -> list[dict[str, Any]]:
    meta = load_meta(run_dir)
    query_text = query_text_from_meta(meta, query_id)
    path = rows_path_from_run(run_dir, query_id)
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Composition evaluation rows must be a JSON list: {path}")

    rows: list[dict[str, Any]] = []
    seen_modes: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"{path} row {index + 1} is not a JSON object")
        mode = str(raw.get("Mode") or raw.get("mode") or "").strip()
        if not mode:
            raise ValueError(f"{path} row {index + 1} is missing Mode")
        if mode in seen_modes:
            raise ValueError(f"{path} contains duplicate Mode row for {mode!r}")
        seen_modes.add(mode)

        row = {column: "" for column in DETAILED_COLUMNS}
        row.update(
            {
                "Query_ID": str(raw.get("Query_ID") or query_id),
                "Query_Text": query_text,
                "Run_Folder": run_dir.name,
                "Mode": mode,
                "Source_File": relative_path(path, run_root),
            }
        )
        for column in DETAILED_COLUMNS:
            if column in {"Query_Text", "Run_Folder", "Source_File"}:
                continue
            if column in raw:
                row[column] = raw[column]
        if not row["Planner_Output_File"]:
            planner_path = run_dir / mode / "4_planner.json"
            if planner_path.exists():
                row["Planner_Output_File"] = relative_path(planner_path, run_dir)
        rows.append(row)

    rows.sort(key=lambda row: mode_sort_key(str(row["Mode"])))
    missing_modes = [mode for mode in MODE_ORDER if mode not in seen_modes]
    has_canonical_mode = any(mode in seen_modes for mode in MODE_ORDER)
    if missing_modes and has_canonical_mode and not allow_missing_modes:
        raise ValueError(f"{path} is missing required modes: {', '.join(missing_modes)}")
    return rows


def mean_numeric(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    values = [value for value in (coerce_float(row.get(field)) for row in rows) if value is not None]
    if not values:
        return None
    return statistics.fmean(values)


def std_numeric(rows: Sequence[dict[str, Any]], field: str) -> float:
    values = [value for value in (coerce_float(row.get(field)) for row in rows) if value is not None]
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def best_modes_by_query(rows: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["Query_ID"]), []).append(row)

    winners: dict[str, set[str]] = {}
    for query_id, query_rows in grouped.items():
        scored = [
            (str(row["Mode"]), coerce_float(row.get("QoS_Adjusted_Composition_Score")))
            for row in query_rows
        ]
        scored = [(mode, score) for mode, score in scored if score is not None]
        if not scored:
            continue
        best_score = max(score for _, score in scored)
        winners[query_id] = {
            mode
            for mode, score in scored
            if math.isclose(score, best_score, rel_tol=TIE_TOLERANCE, abs_tol=TIE_TOLERANCE)
        }
    return winners


def aggregate_mode_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    modes = sorted({str(row["Mode"]) for row in rows}, key=mode_sort_key)
    winners_by_query = best_modes_by_query(rows)
    aggregate_rows: list[dict[str, Any]] = []

    for mode in modes:
        mode_rows = [row for row in rows if row["Mode"] == mode]
        if not mode_rows:
            continue
        out: dict[str, Any] = {
            "Mode": mode,
            "Query_Count": len({str(row["Query_ID"]) for row in mode_rows}),
            "Std_QoS_Adjusted_Composition_Score": format_number(
                std_numeric(mode_rows, "QoS_Adjusted_Composition_Score")
            ),
            "Complete_Composition_Count": sum(
                1
                for row in mode_rows
                if is_truthy_one(row.get("Composition_Completeness_Gate", row.get("Composition_Completeness")))
            ),
            "Invalid_Composition_Count": sum(1 for row in mode_rows if is_invalid(row.get("Composition_Validity"))),
            "Best_Query_Count": sum(1 for modes_for_query in winners_by_query.values() if mode in modes_for_query),
        }
        for aggregate_field, source_field in AGGREGATE_AVERAGE_FIELDS.items():
            out[aggregate_field] = format_number(mean_numeric(mode_rows, source_field))
        aggregate_rows.append({column: out.get(column, "") for column in AGGREGATE_COLUMNS})

    return aggregate_rows


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_report(
    output_dir: Path,
    run_root: Path,
    detailed_rows: Sequence[dict[str, Any]],
    aggregate_rows: Sequence[dict[str, Any]],
    *,
    all_query_ids: Sequence[str],
    included_query_ids: Sequence[str],
    excluded_query_ids: Sequence[str],
) -> dict[str, Any]:
    modes_found = sorted({str(row["Mode"]) for row in detailed_rows}, key=mode_sort_key)
    expected_modes_missing_by_query: dict[str, list[str]] = {}
    for query_id in included_query_ids:
        found = {str(row["Mode"]) for row in detailed_rows if row["Query_ID"] == query_id}
        missing = [mode for mode in MODE_ORDER if mode not in found]
        if missing and any(mode in found for mode in MODE_ORDER):
            expected_modes_missing_by_query[query_id] = missing

    tied_best_queries = []
    for query_id, modes in best_modes_by_query(detailed_rows).items():
        if len(modes) <= 1:
            continue
        query_rows = [row for row in detailed_rows if row["Query_ID"] == query_id]
        best_score = max(coerce_float(row.get("QoS_Adjusted_Composition_Score")) or float("-inf") for row in query_rows)
        tied_best_queries.append(
            {
                "Query_ID": query_id,
                "Best_Modes": sorted(modes, key=mode_sort_key),
                "Best_Score": format_number(best_score),
            }
        )
    tied_best_queries.sort(key=lambda row: query_sort_key(str(row["Query_ID"])))

    report = {
        "run_folder": str(run_root),
        "output_dir": str(output_dir),
        "all_15_query_composition_results_csv": str(output_dir / "all_15_query_composition_results.csv"),
        "aggregate_mode_scores_csv": str(output_dir / "aggregate_mode_scores.csv"),
        "total_query_runs_available_in_input": len(all_query_ids),
        "total_query_runs_found": len(included_query_ids),
        "total_metric_rows_written": len(detailed_rows),
        "query_ids_found": list(included_query_ids),
        "modes_found": modes_found,
        "expected_modes_missing_by_query": expected_modes_missing_by_query,
        "tied_best_queries": tied_best_queries,
        "query_ids_excluded_by_filter": list(excluded_query_ids),
        "aggregate_rows_written": len(aggregate_rows),
    }
    if excluded_query_ids:
        report["include_query_ids"] = list(included_query_ids)
        report["official_summary_note"] = (
            f"This official thesis summary includes {included_query_ids[0]}-{included_query_ids[-1]} only. "
            f"Additional query logs such as {', '.join(excluded_query_ids)} exist in the raw logs."
        )

    (output_dir / "summary_generation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def write_readme(output_dir: Path, report: dict[str, Any]) -> None:
    include_note = ""
    if report.get("include_query_ids"):
        include_note = (
            f"\nThis official thesis summary includes {report['query_ids_found'][0]}-"
            f"{report['query_ids_found'][-1]} only. Recreate with "
            f"`--include-query-ids {','.join(report['query_ids_found'])}`.\n"
        )
    text = f"""# Composition Result Summary

Generated from:

```text
{report["run_folder"]}
```

Files:

- `all_15_query_composition_results.csv`: one row per query and mode.
- `aggregate_mode_scores.csv`: one row per mode, derived from the detailed CSV.
- `summary_generation_report.json`: reproducibility metadata and warnings.
{include_note}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def consolidate(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    include_query_ids: Sequence[str] | str | None = None,
    *,
    allow_missing_modes: bool = False,
) -> dict[str, Any]:
    run_root = resolve_run_root(input_dir)
    out_dir = resolve_output_dir(run_root, output_dir)
    discovered = discover_query_runs(run_root)
    all_query_ids = list(discovered)

    include_filter = parse_query_ids(include_query_ids)
    if include_filter is None:
        selected_query_ids = all_query_ids
        excluded_query_ids: list[str] = []
    else:
        missing = [query_id for query_id in include_filter if query_id not in discovered]
        if missing:
            available = ", ".join(all_query_ids)
            raise ValueError(f"Requested query ids not found: {', '.join(missing)}. Available: {available}")
        selected_query_ids = include_filter
        excluded_query_ids = [query_id for query_id in all_query_ids if query_id not in set(include_filter)]

    detailed_rows: list[dict[str, Any]] = []
    for query_id in selected_query_ids:
        detailed_rows.extend(
            load_query_rows(
                run_root,
                query_id,
                discovered[query_id],
                allow_missing_modes=allow_missing_modes,
            )
        )

    aggregate_rows = aggregate_mode_rows(detailed_rows)
    detailed_csv = out_dir / "all_15_query_composition_results.csv"
    aggregate_csv = out_dir / "aggregate_mode_scores.csv"
    write_csv(detailed_csv, DETAILED_COLUMNS, detailed_rows)
    write_csv(aggregate_csv, AGGREGATE_COLUMNS, aggregate_rows)
    report = write_report(
        out_dir,
        run_root,
        detailed_rows,
        aggregate_rows,
        all_query_ids=all_query_ids,
        included_query_ids=selected_query_ids,
        excluded_query_ids=excluded_query_ids,
    )
    write_readme(out_dir, report)
    return {
        "detailed_csv": detailed_csv,
        "aggregate_csv": aggregate_csv,
        "report": report,
    }


def main() -> int:
    args = parse_args()
    result = consolidate(
        args.run_folder,
        args.output_dir,
        include_query_ids=args.include_query_ids,
        allow_missing_modes=args.allow_missing_modes,
    )
    print(result["detailed_csv"])
    print(result["aggregate_csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
