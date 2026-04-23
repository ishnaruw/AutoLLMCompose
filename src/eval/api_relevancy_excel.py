from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

COLUMNS = [
    "Query_ID",
    "Mode",
    "Sub Task",
    "Retrieved Rank",
    "Mode Rank",
    "Subtask_Purpose",
    "Selected_API",
    "Is Hallucinated? (0/1)",
    "Is Duplicated? (0/1)",
    "API Relevancy (0/1)",
    "QoS_RT",
    "QoS_TP",
    "QoS Availability",
    "Comments",
]

MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
MODE_INDEX = {name: idx for idx, name in enumerate(MODE_ORDER)}

HEADER_FILL = PatternFill(fill_type="solid", fgColor="D9EAF7")
def _subtask_sort_key(value: Any) -> Tuple[int, str]:
    text = str(value)
    return (int(text), text) if text.isdigit() else (9999, text)


def _mode_sort_key(value: Any) -> Tuple[int, str]:
    text = str(value)
    return MODE_INDEX.get(text, 9999), text


def _rank_sort_key(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 9999


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _append_sheet(
    ws,
    columns: List[str],
    rows: List[Dict[str, Any]],
    *,
    highlight_fn=None,
) -> None:
    ws.append(columns)
    for idx in range(1, len(columns) + 1):
        cell = ws.cell(row=1, column=idx)
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(vertical="top", wrap_text=True)

    for row in rows:
        ws.append([row.get(column, "") for column in columns])
        fill = highlight_fn(row) if highlight_fn else None
        if fill is not None:
            for idx in range(1, len(columns) + 1):
                ws.cell(row=ws.max_row, column=idx).fill = fill

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for column_cells in ws.columns:
        max_len = 0
        column_letter = column_cells[0].column_letter
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(value))
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        ws.column_dimensions[column_letter].width = min(max(max_len + 2, 12), 60)


def _build_precision_rows(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("Query_ID", "")), str(row.get("Sub Task", "")), str(row.get("Mode", "")))].append(row)

    precision_rows: List[Dict[str, Any]] = []
    for (query_id, subtask_id, mode), group in sorted(
        grouped.items(),
        key=lambda item: (_subtask_sort_key(item[0][1]), _mode_sort_key(item[0][2])),
    ):
        relevant = sum(1 for row in group if _safe_int(row.get("API Relevancy (0/1)")) == 1)
        total = len(group)
        precision_rows.append(
            {
                "Query_ID": query_id,
                "Sub Task": subtask_id,
                "Mode": mode,
                "Subtask_Purpose": str(group[0].get("Subtask_Purpose", "")) if group else "",
                "Relevant_Count": relevant,
                "Candidate_Count": total,
                "Precision": round(relevant / float(total or 1), 4),
            }
        )
    return precision_rows


def _aggregate_mode_rows(precision_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in precision_rows:
        grouped[str(row.get("Mode", ""))].append(row)

    out: List[Dict[str, Any]] = []
    for mode, rows in sorted(grouped.items(), key=lambda item: _mode_sort_key(item[0])):
        relevant = sum(_safe_int(row.get("Relevant_Count")) for row in rows)
        total = sum(_safe_int(row.get("Candidate_Count")) for row in rows)
        out.append(
            {
                "Mode": mode,
                "Subtasks": len(rows),
                "Relevant_Count": relevant,
                "Candidate_Count": total,
                "Precision": round(relevant / float(total or 1), 4),
            }
        )
    return out


def _aggregate_subtask_rows(precision_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in precision_rows:
        grouped[str(row.get("Sub Task", ""))].append(row)

    out: List[Dict[str, Any]] = []
    for subtask_id, rows in sorted(grouped.items(), key=lambda item: _subtask_sort_key(item[0])):
        relevant = sum(_safe_int(row.get("Relevant_Count")) for row in rows)
        total = sum(_safe_int(row.get("Candidate_Count")) for row in rows)
        out.append(
            {
                "Sub Task": subtask_id,
                "Subtask_Purpose": str(rows[0].get("Subtask_Purpose", "")) if rows else "",
                "Modes": len(rows),
                "Relevant_Count": relevant,
                "Candidate_Count": total,
                "Precision": round(relevant / float(total or 1), 4),
            }
        )
    return out


def write_relevancy_excel(
    rows: List[Dict[str, Any]],
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ranked_rows = sorted(
        rows,
        key=lambda r: (
            _subtask_sort_key(r.get("Sub Task")),
            _mode_sort_key(r.get("Mode")),
            _rank_sort_key(r.get("Mode Rank")),
        ),
    )
    precision_rows = _build_precision_rows(rows)
    mode_summary_rows = _aggregate_mode_rows(precision_rows)
    subtask_summary_rows = _aggregate_subtask_rows(precision_rows)

    total_relevant = sum(_safe_int(row.get("Relevant_Count")) for row in precision_rows)
    total_candidates = sum(_safe_int(row.get("Candidate_Count")) for row in precision_rows)

    wb = Workbook()

    overview_ws = wb.active
    overview_ws.title = "Overview"
    overview_rows = [
        {"Metric": "Query_ID", "Value": str(rows[0].get("Query_ID", "")) if rows else ""},
        {"Metric": "Total Ranked Rows", "Value": len(rows)},
        {"Metric": "Relevant Rows", "Value": total_relevant},
        {"Metric": "Overall Precision", "Value": round(total_relevant / float(total_candidates or 1), 4)},
        {"Metric": "Subtasks", "Value": len({str(row.get("Sub Task", "")) for row in rows})},
        {"Metric": "Modes", "Value": len({str(row.get("Mode", "")) for row in rows})},
    ]
    _append_sheet(overview_ws, ["Metric", "Value"], overview_rows)

    mode_ws = wb.create_sheet("Mode Summary")
    _append_sheet(
        mode_ws,
        [
            "Mode",
            "Subtasks",
            "Relevant_Count",
            "Candidate_Count",
            "Precision",
        ],
        mode_summary_rows,
    )

    subtask_ws = wb.create_sheet("Subtask Summary")
    _append_sheet(
        subtask_ws,
        [
            "Sub Task",
            "Subtask_Purpose",
            "Modes",
            "Relevant_Count",
            "Candidate_Count",
            "Precision",
        ],
        subtask_summary_rows,
    )

    precision_ws = wb.create_sheet("Precision")
    _append_sheet(
        precision_ws,
        [
            "Query_ID",
            "Sub Task",
            "Mode",
            "Subtask_Purpose",
            "Relevant_Count",
            "Candidate_Count",
            "Precision",
        ],
        precision_rows,
    )

    ranked_ws = wb.create_sheet("Ranked APIs")
    _append_sheet(
        ranked_ws,
        COLUMNS,
        ranked_rows,
    )

    wb.save(out_path)
    return out_path
