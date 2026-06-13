from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
MODE_INDEX = {name: idx for idx, name in enumerate(MODE_ORDER)}

def _query_sort_key(value: str) -> Tuple[int, str]:
    match = re.search(r"q(\d+)", str(value), flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), str(value)
    return 9999, str(value)


def _subtask_sort_key(value: Any) -> Tuple[int, str]:
    text = str(value)
    return (int(text), text) if text.isdigit() else (9999, text)


def _mode_sort_key(value: str) -> Tuple[int, str]:
    text = str(value)
    return MODE_INDEX.get(text, 9999), text


def _int_sort(values: Iterable[Any]) -> List[str]:
    def _key(value: Any) -> Tuple[int, str]:
        text = str(value)
        return (int(text), text) if text.isdigit() else (9999, text)

    return [str(v) for v in sorted(values, key=_key)]


def _safe_load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_evaluation_dir(run_dir: Path) -> Path | None:
    for candidate in ("evaluation", "functional_match_eval"):
        eval_dir = run_dir / candidate
        if eval_dir.exists():
            return eval_dir
    return None


def _find_rows_json(run_dir: Path) -> Path | None:
    eval_dir = _find_evaluation_dir(run_dir)
    if eval_dir is None:
        return None
    matches = sorted(eval_dir.glob("query_*_candidate_api_rankings_rows.json"))
    return matches[0] if matches else None


def _collect_group_summary(
    *,
    run_dir_name: str,
    query_id: str,
    subtask_id: str,
    mode: str,
    rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cleaned_api_ids = [str(row.get("Selected_API", "")).strip() for row in rows if str(row.get("Selected_API", "")).strip()]
    counts = Counter(cleaned_api_ids)
    duplicate_counts = {api_id: count for api_id, count in counts.items() if count > 1}
    duplicate_api_count = len(duplicate_counts)
    duplicate_row_count = sum(count - 1 for count in duplicate_counts.values())
    max_repeat_count = max(counts.values(), default=0)
    subtask_purpose = str(rows[0].get("Subtask_Purpose", "")) if rows else ""

    summary = {
        "Run_Dir": run_dir_name,
        "Query_ID": query_id,
        "Sub_Task": subtask_id,
        "Mode": mode,
        "Subtask_Purpose": subtask_purpose,
        "Candidate_Count": len(rows),
        "Unique_API_Count": len(counts),
        "Duplicate_API_Count": duplicate_api_count,
        "Duplicate_Row_Count": duplicate_row_count,
        "Max_Repeat_Count": max_repeat_count,
        "Has_Duplicates": "Y" if duplicate_counts else "N",
        "Duplicated_APIs": ", ".join(f"{api_id} x{count}" for api_id, count in sorted(duplicate_counts.items())),
    }

    duplicate_details: List[Dict[str, Any]] = []
    for api_id, repeat_count in sorted(duplicate_counts.items()):
        matching_rows = [row for row in rows if str(row.get("Selected_API", "")).strip() == api_id]
        mode_ranks = _int_sort(row.get("Mode Rank", "") for row in matching_rows)
        retrieved_ranks = _int_sort(row.get("Retrieved Rank", "") for row in matching_rows)
        functional_match_values = [str(row.get("Functional Match (0/1)", "")) for row in matching_rows]
        unique_comments = []
        seen_comments = set()
        for row in matching_rows:
            comment = str(row.get("Comments", "")).strip()
            if comment and comment not in seen_comments:
                seen_comments.add(comment)
                unique_comments.append(comment)

        duplicate_details.append(
            {
                "Run_Dir": run_dir_name,
                "Query_ID": query_id,
                "Sub_Task": subtask_id,
                "Mode": mode,
                "Subtask_Purpose": subtask_purpose,
                "Selected_API": api_id,
                "Repeat_Count": repeat_count,
                "Mode_Ranks": ", ".join(mode_ranks),
                "Retrieved_Ranks": ", ".join(retrieved_ranks),
                "Functional_Match_Values": ", ".join(functional_match_values),
                "Comments": " | ".join(unique_comments[:3]),
            }
        )

    return summary, duplicate_details


def _collect_duplicate_audit_for_runs(root_dir: Path, run_dirs: Iterable[Path]) -> Dict[str, Any]:
    summary_rows: List[Dict[str, Any]] = []
    duplicate_rows: List[Dict[str, Any]] = []
    missing_runs: List[str] = []

    for run_dir in run_dirs:
        rows_path = _find_rows_json(run_dir)
        if rows_path is None:
            missing_runs.append(run_dir.name)
            continue

        rows = _safe_load_json(rows_path)
        if not isinstance(rows, list):
            continue

        grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            query_id = str(row.get("Query_ID", "")).strip()
            subtask_id = str(row.get("Sub Task", "")).strip()
            mode = str(row.get("Mode", "")).strip()
            grouped[(query_id, subtask_id, mode)].append(row)

        for (query_id, subtask_id, mode), group_rows in sorted(
            grouped.items(),
            key=lambda item: (_query_sort_key(item[0][0]), _subtask_sort_key(item[0][1]), _mode_sort_key(item[0][2])),
        ):
            summary, details = _collect_group_summary(
                run_dir_name=run_dir.name,
                query_id=query_id,
                subtask_id=subtask_id,
                mode=mode,
                rows=group_rows,
            )
            summary_rows.append(summary)
            duplicate_rows.extend(details)

    query_overview: List[Dict[str, Any]] = []
    mode_overview: List[Dict[str, Any]] = []

    grouped_by_query: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grouped_by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped_by_query[str(row["Query_ID"])].append(row)
        grouped_by_mode[str(row["Mode"])].append(row)

    for query_id, rows in sorted(grouped_by_query.items(), key=lambda item: _query_sort_key(item[0])):
        query_overview.append(
            {
                "Key": query_id,
                "Groups_Checked": len(rows),
                "Groups_With_Duplicates": sum(1 for row in rows if row["Has_Duplicates"] == "Y"),
                "Duplicate_API_Count": sum(int(row["Duplicate_API_Count"]) for row in rows),
                "Duplicate_Row_Count": sum(int(row["Duplicate_Row_Count"]) for row in rows),
            }
        )

    for mode, rows in sorted(grouped_by_mode.items(), key=lambda item: _mode_sort_key(item[0])):
        mode_overview.append(
            {
                "Key": mode,
                "Groups_Checked": len(rows),
                "Groups_With_Duplicates": sum(1 for row in rows if row["Has_Duplicates"] == "Y"),
                "Duplicate_API_Count": sum(int(row["Duplicate_API_Count"]) for row in rows),
                "Duplicate_Row_Count": sum(int(row["Duplicate_Row_Count"]) for row in rows),
            }
        )

    overall = {
        "root_dir": str(root_dir),
        "run_count": len({row["Run_Dir"] for row in summary_rows}),
        "group_count": len(summary_rows),
        "groups_with_duplicates": sum(1 for row in summary_rows if row["Has_Duplicates"] == "Y"),
        "duplicate_api_count": sum(int(row["Duplicate_API_Count"]) for row in summary_rows),
        "duplicate_row_count": sum(int(row["Duplicate_Row_Count"]) for row in summary_rows),
        "missing_runs": missing_runs,
    }

    return {
        "overall": overall,
        "summary_rows": summary_rows,
        "duplicate_rows": duplicate_rows,
        "query_overview": query_overview,
        "mode_overview": mode_overview,
    }


def collect_duplicate_audit_for_run(run_dir: Path) -> Dict[str, Any]:
    return _collect_duplicate_audit_for_runs(run_dir, [run_dir])
