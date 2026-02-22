# src/eval/compare_selector_modes.py
"""
Compare PURE_LLM vs TOPSIS selector modes within a single MAOF run.

This evaluator answers:
- Did TOPSIS-based selection change which APIs were available to the planner?
- Did that change improve workflow-level QoS aggregates?

It operates only on logged run artifacts (no LLM calls).

Inputs expected in run_dir:
  - 0_decomposer.json
  - 4_selected_with_service_pure_llm.json
  - 4_selected_with_service_topsis.json
  - 5_planner_pure_llm.json
  - 5_planner_topsis.json

Outputs produced in run_dir:
  - selector_comparison.json
  - selector_comparison.csv
  - selector_comparison.xlsx
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openpyxl import Workbook


def _safe_read_json(p: Path) -> Any:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_qos(service: Dict[str, Any]) -> Dict[str, Any]:
    qos = (service or {}).get("qos") or {}
    return qos if isinstance(qos, dict) else {}


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


@dataclass
class PathQoS:
    mode: str
    path_id: str
    n_steps: int
    avg_rt_ms: Optional[float]
    avg_tp_rps: Optional[float]
    avg_availability: Optional[float]
    missing_rt: int
    missing_tp: int
    missing_avail: int
    composite: Optional[float]


def _collect_service_map(selected: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for c in selected or []:
        api_id = c.get("api_id")
        if not api_id:
            continue
        m[str(api_id)] = (c.get("service") or {}) if isinstance(c.get("service"), dict) else {}
    return m


def _extract_paths(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    paths = plan.get("paths") if isinstance(plan, dict) else None
    return paths if isinstance(paths, list) else []


def _avg(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _normalize_benefit(x: Optional[float], mn: float, mx: float) -> Optional[float]:
    if x is None:
        return None
    if mx <= mn:
        return 0.5
    return (x - mn) / (mx - mn)


def _normalize_cost(x: Optional[float], mn: float, mx: float) -> Optional[float]:
    # lower is better
    if x is None:
        return None
    if mx <= mn:
        return 0.5
    return (mx - x) / (mx - mn)


def _compute_composite(
    avg_rt_ms: Optional[float],
    avg_tp_rps: Optional[float],
    avg_availability: Optional[float],
    mins: Dict[str, float],
    maxs: Dict[str, float],
) -> Optional[float]:
    parts: List[float] = []
    rt = _normalize_cost(avg_rt_ms, mins["rt_ms"], maxs["rt_ms"])
    tp = _normalize_benefit(avg_tp_rps, mins["tp_rps"], maxs["tp_rps"])
    av = _normalize_benefit(avg_availability, mins["availability"], maxs["availability"])
    for v in (rt, tp, av):
        if v is not None:
            parts.append(float(v))
    if not parts:
        return None
    return sum(parts) / float(len(parts))


def _path_qos_for_mode(
    *,
    mode: str,
    plan: Dict[str, Any],
    service_by_id: Dict[str, Dict[str, Any]],
) -> List[PathQoS]:
    out: List[PathQoS] = []
    for p in _extract_paths(plan):
        pid = str(p.get("path_id") or "")
        steps = p.get("steps") or []
        if not isinstance(steps, list):
            continue

        rt_vals: List[float] = []
        tp_vals: List[float] = []
        av_vals: List[float] = []
        missing_rt = missing_tp = missing_av = 0

        for st in steps:
            api_id = str((st or {}).get("api_id") or "")
            svc = service_by_id.get(api_id, {})
            qos = _get_qos(svc)

            rt = _to_float(qos.get("rt_ms"))
            tp = _to_float(qos.get("tp_rps"))
            av = _to_float(qos.get("availability"))

            if rt is None:
                missing_rt += 1
            else:
                rt_vals.append(rt)

            if tp is None:
                missing_tp += 1
            else:
                tp_vals.append(tp)

            if av is None:
                missing_av += 1
            else:
                av_vals.append(av)

        out.append(PathQoS(
            mode=mode,
            path_id=pid or "(unknown)",
            n_steps=len(steps),
            avg_rt_ms=_avg(rt_vals),
            avg_tp_rps=_avg(tp_vals),
            avg_availability=_avg(av_vals),
            missing_rt=missing_rt,
            missing_tp=missing_tp,
            missing_avail=missing_av,
            composite=None,  # fill after we compute mins/maxs
        ))
    return out


def _mins_maxs(all_paths: List[PathQoS]) -> Tuple[Dict[str, float], Dict[str, float]]:
    def collect(attr: str) -> List[float]:
        vals = []
        for p in all_paths:
            v = getattr(p, attr)
            if v is not None:
                vals.append(float(v))
        return vals

    rt = collect("avg_rt_ms")
    tp = collect("avg_tp_rps")
    av = collect("avg_availability")

    # avoid empty
    mins = {
        "rt_ms": min(rt) if rt else 0.0,
        "tp_rps": min(tp) if tp else 0.0,
        "availability": min(av) if av else 0.0,
    }
    maxs = {
        "rt_ms": max(rt) if rt else 1.0,
        "tp_rps": max(tp) if tp else 1.0,
        "availability": max(av) if av else 1.0,
    }
    return mins, maxs


def write_comparison_reports(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)

    selected_pure = _safe_read_json(run_dir / "4_selected_with_service_pure_llm.json") or []
    selected_topsis = _safe_read_json(run_dir / "4_selected_with_service_topsis.json") or []

    plan_pure = _safe_read_json(run_dir / "5_planner_pure_llm.json") or {}
    plan_topsis = _safe_read_json(run_dir / "5_planner_topsis.json") or {}

    # maps for QoS lookup
    svc_pure = _collect_service_map(selected_pure)
    svc_topsis = _collect_service_map(selected_topsis)

    paths_pure = _path_qos_for_mode(mode="PURE_LLM", plan=plan_pure, service_by_id=svc_pure)
    paths_topsis = _path_qos_for_mode(mode="TOPSIS", plan=plan_topsis, service_by_id=svc_topsis)

    all_paths = paths_pure + paths_topsis
    mins, maxs = _mins_maxs(all_paths)

    # fill composite scores
    def with_composite(p: PathQoS) -> PathQoS:
        p.composite = _compute_composite(p.avg_rt_ms, p.avg_tp_rps, p.avg_availability, mins, maxs)
        return p

    paths_pure = [with_composite(p) for p in paths_pure]
    paths_topsis = [with_composite(p) for p in paths_topsis]

    # API overlap diagnostics (overall + per-subtask)
    def subtask_sets(selected: List[Dict[str, Any]]) -> Dict[str, set]:
        out: Dict[str, set] = {}
        for c in selected or []:
            sid = str(c.get("subtask_id"))
            out.setdefault(sid, set()).add(str(c.get("api_id")))
        return out

    pure_sets = subtask_sets(selected_pure)
    topsis_sets = subtask_sets(selected_topsis)

    per_sub_overlap = []
    for sid in sorted(set(pure_sets.keys()) | set(topsis_sets.keys()), key=lambda x: (len(x), x)):
        a = pure_sets.get(sid, set())
        b = topsis_sets.get(sid, set())
        inter = a & b
        union = a | b
        per_sub_overlap.append({
            "subtask_id": sid,
            "pure_count": len(a),
            "topsis_count": len(b),
            "overlap_count": len(inter),
            "jaccard": (len(inter) / len(union)) if union else 1.0,
        })

    def best_path(paths: List[PathQoS]) -> Optional[PathQoS]:
        scored = [p for p in paths if p.composite is not None]
        if not scored:
            return None
        return sorted(scored, key=lambda x: x.composite, reverse=True)[0]

    best_pure = best_path(paths_pure)
    best_topsis = best_path(paths_topsis)

    summary = {
        "metrics": {
            "composite_definition": "Normalize avg QoS per workflow across BOTH modes; composite = mean(rt_ms(cost), tp_rps(benefit), availability(benefit)) over available metrics.",
            "normalization_range": {"mins": mins, "maxs": maxs},
        },
        "selection_overlap": {
            "overall": {
                "pure_total": len({str(c.get("api_id")) for c in selected_pure}),
                "topsis_total": len({str(c.get("api_id")) for c in selected_topsis}),
            },
            "per_subtask": per_sub_overlap,
        },
        "planner": {
            "pure_paths": len(_extract_paths(plan_pure)),
            "topsis_paths": len(_extract_paths(plan_topsis)),
        },
        "best_paths": {
            "PURE_LLM": best_pure.__dict__ if best_pure else None,
            "TOPSIS": best_topsis.__dict__ if best_topsis else None,
        },
    }

    # rows for csv/xlsx
    rows = []
    for p in paths_pure + paths_topsis:
        rows.append(p.__dict__)

    # write json
    (run_dir / "selector_comparison.json").write_text(json.dumps({"summary": summary, "paths": rows}, indent=2), encoding="utf-8")

    # write csv
    import csv
    csv_path = run_dir / "selector_comparison.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # write simple xlsx
    xlsx_path = run_dir / "selector_comparison.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Comparison"

    if rows:
        headers = list(rows[0].keys())
        ws.append(headers)
        for r in rows:
            ws.append([r.get(h) for h in headers])

    wb.save(xlsx_path)

    return {"summary": summary, "paths": rows}
