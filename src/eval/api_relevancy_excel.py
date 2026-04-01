from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from openpyxl import Workbook
from openpyxl.styles import Font

COLUMNS = [
    "Query_ID",
    "Mode",
    "Sub Task",
    "Selected Rank",
    "Subtask_Purpose",
    "Selected_API",
    "Expected_Function",
    "API Relevancy (0/1)",
    "QoS_RT",
    "QoS_TP",
    "QoS Availability",
    "Comments",
]


def write_relevancy_excel(rows: List[Dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Query"

    ws.append(COLUMNS)
    for c in range(1, len(COLUMNS) + 1):
        ws.cell(row=1, column=c).font = Font(bold=True)

    for row in rows:
        ws.append(
            [
                row.get("Query_ID"),
                row.get("Mode"),
                row.get("Sub Task"),
                row.get("Selected Rank"),
                row.get("Subtask_Purpose"),
                row.get("Selected_API"),
                row.get("Expected_Function"),
                row.get("API Relevancy (0/1)"),
                row.get("QoS_RT"),
                row.get("QoS_TP"),
                row.get("QoS Availability"),
                row.get("Comments"),
            ]
        )

    wb.save(out_path)
    return out_path
