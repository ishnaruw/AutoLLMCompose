"""src/eval/export_excel.py

Generate a per-run Excel report summarizing MAOF pipeline outputs.

One workbook is produced per run directory (results/logs/<model>/<run_id>/).

Sheets:
  - Summary: high-level run information and key counts
  - Subtasks: per-subtask retrieval/ranking/planning/TOPSIS signals
  - Plans: flattened planner paths/steps (alternative workflows)
  - Candidates: top-N candidates passed to planner, per subtask
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


HEADER_FILL = PatternFill("solid", fgColor="1F4E79")  # dark blue
HEADER_FONT = Font(color="FFFFFF", bold=True)


def _autofit(ws, min_width: int = 10, max_width: int = 60) -> None:
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            v = "" if cell.value is None else str(cell.value)
            if len(v) > max_len:
                max_len = len(v)
        ws.column_dimensions[col_letter].width = max(min_width, min(max_width, max_len + 2))


def _style_header_row(ws, row: int = 1) -> None:
    for cell in ws[row]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"


def _safe_read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _first_step_api_by_subtask(plan: Dict[str, Any]) -> Dict[str, str]:
    """Return subtask_id -> first api_id used in the primary path."""
    out: Dict[str, str] = {}
    try:
        paths = plan.get("paths") or []
        if not isinstance(paths, list) or not paths:
            return out
        steps = (paths[0].get("steps") or [])
        if not isinstance(steps, list):
            return out
        for st in steps:
            sid = st.get("subtask_id")
            aid = st.get("api_id")
            if sid is None or not aid:
                continue
            sid_s = str(sid)
            if sid_s not in out:
                out[sid_s] = str(aid)
    except Exception:
        return out
    return out


def export_run_excel(
    *,
    run_dir: str | Path,
    out_xlsx: Optional[str | Path] = None,
    top_candidates_per_subtask: int = 12,
) -> Path:
    """Create an Excel report from a run folder.

    Expected files in run_dir:
      - meta.json
      - 0_decomposer.json
      - 1_retriever.json
      - 3_ranked_with_service.json
      - 4_planner.json
      - 5_topsis_eval.json
    """
    run_dir = Path(run_dir)
    meta = _safe_read_json(run_dir / "meta.json") or {}
    subtasks = _safe_read_json(run_dir / "0_decomposer.json") or []
    retrieved_by_subtask = _safe_read_json(run_dir / "1_retriever.json") or {}
    ranked_top = _safe_read_json(run_dir / "3_ranked_with_service.json") or []
    plan = _safe_read_json(run_dir / "4_planner.json") or {}
    topsis = _safe_read_json(run_dir / "5_topsis_eval.json") or {}

    if out_xlsx is None:
        out_xlsx = run_dir / "report.xlsx"
    out_xlsx = Path(out_xlsx)

    # Index helpers
    service_by_id: Dict[str, Dict[str, Any]] = {}
    for c in ranked_top if isinstance(ranked_top, list) else []:
        api_id = c.get("api_id")
        if api_id:
            service_by_id[str(api_id)] = c.get("service") or {}

    chosen_by_subtask = _first_step_api_by_subtask(plan if isinstance(plan, dict) else {})

    # TOPSIS: sid -> step info
    topsis_by_sub: Dict[str, Dict[str, Any]] = {}
    try:
        for st in (topsis.get("steps") or []):
            sid = st.get("subtask_id")
            if sid is not None:
                topsis_by_sub[str(sid)] = st
    except Exception:
        pass

    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # ---------------- Summary ----------------
    ws = wb.create_sheet("Summary")
    rows = [
        ("Run directory", str(run_dir)),
        ("Model tag", meta.get("model_tag")),
        ("Provider", meta.get("provider")),
        ("Model", meta.get("model")),
        ("with_qos", meta.get("with_qos")),
        ("Num subtasks", meta.get("num_subtasks", len(subtasks) if isinstance(subtasks, list) else None)),
        ("RAG index dir", meta.get("rag_index_dir") or "(env: MAOF_RAG_INDEX_DIR)"),
        ("RAG topK", meta.get("rag_topk") or "(env: MAOF_RAG_TOPK)"),
        ("Ranker candidates used", meta.get("ranker_candidates_used") or "25"),
        ("Selected per subtask", f"{meta.get('rag_keep_min', 8)}–{meta.get('rag_keep_max', 12)}"),
        ("Planner paths", len((plan.get("paths") or [])) if isinstance(plan, dict) else None),
        ("User goal", meta.get("user_goal")),
    ]
    ws.append(["Metric", "Value"])
    _style_header_row(ws)
    for k, v in rows:
        ws.append([k, v])
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 90
    ws["B"][0].alignment = Alignment(wrap_text=True)

    # ---------------- Subtasks ----------------
    ws = wb.create_sheet("Subtasks")
    headers = [
        "subtask_id",
        "subtask",
        "retrieved_k",
        "selected_n",
        "planner_api_primary",
        "topsis_status",
        "topsis_best_api_id",
        "planner_matches_topsis",
    ]
    ws.append(headers)
    _style_header_row(ws)

    subtasks_list = subtasks if isinstance(subtasks, list) else []
    for st in subtasks_list:
        sid = str(st.get("id"))
        desc = st.get("description")
        retrieved = retrieved_by_subtask.get(sid, []) if isinstance(retrieved_by_subtask, dict) else []
        retrieved_k = len(retrieved) if isinstance(retrieved, list) else 0
        selected = [c for c in ranked_top if str(c.get("subtask_id")) == sid]
        selected_n = len(selected)
        planner_api = chosen_by_subtask.get(sid, "")

        t = topsis_by_sub.get(sid, {})
        status = t.get("status", "")
        best_id = t.get("topsis_best_api_id") or t.get("best_api_id") or ""
        match = t.get("planner_chose_topsis_best")

        ws.append([sid, desc, retrieved_k, selected_n, planner_api, status, best_id, match])

    ws.column_dimensions["B"].width = 70
    for r in range(2, ws.max_row + 1):
        ws[f"B{r}"].alignment = Alignment(wrap_text=True, vertical="top")

    # ---------------- Plans ----------------
    ws = wb.create_sheet("Plans")
    headers = [
        "path_id",
        "path_score",
        "step",
        "subtask_id",
        "api_id",
        "api_name",
        "action",
        "why",
    ]
    ws.append(headers)
    _style_header_row(ws)

    paths = (plan.get("paths") or []) if isinstance(plan, dict) else []
    if isinstance(paths, list):
        for p in paths:
            pid = p.get("path_id")
            pscore = p.get("path_score")
            steps = p.get("steps") or []
            if not isinstance(steps, list):
                continue
            for st in steps:
                api_id = st.get("api_id")
                svc = service_by_id.get(str(api_id), {})
                api_name = svc.get("name") or svc.get("operation") or svc.get("title") or ""
                ws.append([
                    pid,
                    pscore,
                    st.get("step"),
                    st.get("subtask_id"),
                    api_id,
                    api_name,
                    st.get("action"),
                    st.get("why"),
                ])

    ws.column_dimensions["F"].width = 30
    ws.column_dimensions["G"].width = 45
    ws.column_dimensions["H"].width = 55
    for r in range(2, ws.max_row + 1):
        ws[f"G{r}"].alignment = Alignment(wrap_text=True, vertical="top")
        ws[f"H{r}"].alignment = Alignment(wrap_text=True, vertical="top")

    # ---------------- Candidates ----------------
    ws = wb.create_sheet("Candidates")
    headers = [
        "subtask_id",
        "rank",
        "api_id",
        "rag_score",
        "api_name",
        "rt_ms",
        "tp_rps",
        "availability",
        "category",
    ]
    ws.append(headers)
    _style_header_row(ws)

    # stable sort by subtask_id, then rank
    cands = ranked_top if isinstance(ranked_top, list) else []
    cands = sorted(cands, key=lambda x: (str(x.get("subtask_id")), int(x.get("rank", 10**9))))

    # cap per subtask
    per_sub_count: Dict[str, int] = {}
    for c in cands:
        sid = str(c.get("subtask_id"))
        per_sub_count.setdefault(sid, 0)
        if per_sub_count[sid] >= top_candidates_per_subtask:
            continue
        per_sub_count[sid] += 1

        api_id = c.get("api_id")
        svc = c.get("service") or {}
        qos = svc.get("qos") or {}
        api_name = svc.get("name") or svc.get("operation") or svc.get("title") or ""
        ws.append([
            sid,
            c.get("rank"),
            api_id,
            c.get("rag_score"),
            api_name,
            qos.get("rt_ms"),
            qos.get("tp_rps"),
            qos.get("availability"),
            svc.get("category"),
        ])

    _autofit(ws)

    # Final polish: make all sheets top-aligned by default
    for sheet in wb.worksheets:
        sheet.sheet_view.showGridLines = False

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_xlsx)
    return out_xlsx
