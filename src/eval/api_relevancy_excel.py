from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openpyxl import Workbook
from openpyxl.styles import Font

COLUMNS = [
    "Query_ID",
    "Mode",
    "Sub Task",
    "Retrieved Rank",
    "Mode Rank",
    "Subtask_Purpose",
    "Selected_API",
    "API Relevancy (0/1)",
    "QoS_RT",
    "QoS_TP",
    "QoS Availability",
    "Comments",
]

MODE_ORDER = ["no_qos", "qos_pure_llm", "qos_topsis", "qos_hybrid"]
MODE_INDEX = {name: idx for idx, name in enumerate(MODE_ORDER)}


def _subtask_sort_key(value: Any) -> Tuple[int, str]:
    text = str(value)
    return (int(text), text) if text.isdigit() else (9999, text)


def _rank_sort_key(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 9999


def write_relevancy_excel(rows: List[Dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Query"
    ws.append(COLUMNS)
    for c in range(1, len(COLUMNS) + 1):
        ws.cell(row=1, column=c).font = Font(bold=True)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("Sub Task")), str(row.get("Mode")))].append(row)

    subtasks = sorted({str(r.get("Sub Task")) for r in rows}, key=_subtask_sort_key)
    for subtask_id in subtasks:
        for mode in MODE_ORDER:
            group = grouped.get((subtask_id, mode), [])
            group.sort(key=lambda r: _rank_sort_key(r.get("Mode Rank")))
            if not group:
                continue
            for row in group:
                ws.append([
                    row.get("Query_ID"),
                    row.get("Mode"),
                    row.get("Sub Task"),
                    row.get("Retrieved Rank"),
                    row.get("Mode Rank"),
                    row.get("Subtask_Purpose"),
                    row.get("Selected_API"),
                    row.get("API Relevancy (0/1)"),
                    row.get("QoS_RT"),
                    row.get("QoS_TP"),
                    row.get("QoS Availability"),
                    row.get("Comments"),
                ])
            relevant = sum(1 for r in group if int(r.get("API Relevancy (0/1)") or 0) == 1)
            total = len(group)
            precision = round(relevant / float(total or 1), 4)
            ws.append([
                group[0].get("Query_ID"),
                mode,
                subtask_id,
                "",
                "",
                "",
                "Precision",
                precision,
                "",
                "",
                "",
                f"Precision = {relevant}/{total}",
            ])
            for c in range(1, len(COLUMNS) + 1):
                ws.cell(row=ws.max_row, column=c).font = Font(bold=True)
            ws.append([""] * len(COLUMNS))
    wb.save(out_path)
    return out_path
