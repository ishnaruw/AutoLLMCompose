from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from src.eval.topsis_eval import _extract_qos, _run_topsis_pydecision


def _selector_mode() -> str:
    mode = (os.getenv("MAOF_SELECTOR_MODE", "PURE_LLM") or "PURE_LLM").strip().upper()
    if mode in {"TOPSIS", "TOPSIS_ENABLED"}:
        return "TOPSIS"
    return "PURE_LLM"


def _get_qos(service: Dict[str, Any]) -> Dict[str, Any]:
    qos = (service or {}).get("qos") or {}
    return qos if isinstance(qos, dict) else {}


def _truncate(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    s = " ".join(s.strip().split())
    return s if len(s) <= n else (s[: n - 1] + "…")


def _coerce_json(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "{}"
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    return m.group(1) if m else "{}"


def _slim_candidate(c: Dict[str, Any], include_qos: bool, include_topsis: bool) -> Dict[str, Any]:
    service = c.get("service") or {}
    if not isinstance(service, dict):
        service = {}

    out: Dict[str, Any] = {
        "api_id": c.get("api_id"),
        "category": _truncate(service.get("category") or c.get("category"), 80),
        "tool_name": _truncate(service.get("tool_name") or service.get("_tool"), 120),
        "tool_description": _truncate(service.get("tool_description"), 220),
        "name": _truncate(service.get("name"), 120),
        "description": _truncate(service.get("description"), 240),
        "method": _truncate(service.get("method"), 16),
        "url": _truncate(service.get("url"), 140),
        "rag_score": c.get("rag_score"),
        "ranker_rank": c.get("rank"),
        "ranker_reason": _truncate(c.get("reason"), 120),
    }

    if include_qos:
        qos = _get_qos(service)
        slim_qos = {}
        for k in ("rt_ms", "tp_rps", "availability"):
            if k in qos:
                slim_qos[k] = qos.get(k)
        if slim_qos:
            out["qos"] = slim_qos
    if include_topsis and c.get("topsis_score") is not None:
        out["topsis_score"] = c.get("topsis_score")
    return out


def _compute_topsis_scores(candidates: List[Dict[str, Any]]) -> Tuple[Dict[str, float], int]:
    valid_rows: List[Tuple[str, List[float]]] = []
    for c in candidates:
        qos = _get_qos(c.get("service") or {})
        vec = _extract_qos(qos)
        if vec is None:
            continue
        valid_rows.append((str(c.get("api_id")), vec))

    if not valid_rows:
        return {}, 0

    X = np.asarray([v for _, v in valid_rows], dtype=float)
    scores, _ = _run_topsis_pydecision(X, [1.0, 1.0, 1.0])
    return {api_id: float(scores[i]) for i, (api_id, _) in enumerate(valid_rows)}, len(valid_rows)


def _attach_topsis_scores(candidates: List[Dict[str, Any]], topsis_scores: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in candidates:
        row = dict(c)
        api_id = str(row.get("api_id"))
        if api_id in topsis_scores:
            row["topsis_score"] = topsis_scores[api_id]
        out.append(row)
    return out


def _deterministic_topsis_select(
    *,
    selector_candidates: List[Dict[str, Any]],
    fallback_order: List[Dict[str, Any]],
    top_n: int,
    topsis_top_k: int,
    min_qos_candidates: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scoped = fallback_order[: max(topsis_top_k, top_n)]
    topsis_scores, qos_candidates = _compute_topsis_scores(scoped)
    trace: Dict[str, Any] = {
        "mode": "TOPSIS",
        "selected_by": "deterministic_topsis_then_fallback",
        "qos_candidates": qos_candidates,
        "topsis_top_k": topsis_top_k,
        "min_qos_candidates": min_qos_candidates,
    }

    if qos_candidates < min_qos_candidates:
        selected = []
        for c in fallback_order[:top_n]:
            row = dict(c)
            row.setdefault("selector_reason", "Fallback: insufficient valid QoS candidates for deterministic TOPSIS.")
            selected.append(row)
        trace["topsis_status"] = "insufficient_qos_candidates"
        return selected, trace

    ranked = _attach_topsis_scores(scoped, topsis_scores)
    ranked = sorted(
        ranked,
        key=lambda x: (
            -(float(x.get("topsis_score") or -1.0)),
            0 if x.get("rank") is not None else 1,
            int(x.get("rank") or 10**9),
            -(float(x.get("rag_score") or 0.0)),
        ),
    )

    selected: List[Dict[str, Any]] = []
    seen = set()
    for c in ranked:
        api_id = str(c.get("api_id"))
        if not api_id or api_id in seen:
            continue
        row = dict(c)
        row["selector_reason"] = "Selected by deterministic TOPSIS over functionally shortlisted candidates."
        selected.append(row)
        seen.add(api_id)
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for c in fallback_order:
            api_id = str(c.get("api_id"))
            if not api_id or api_id in seen:
                continue
            row = dict(c)
            if api_id in topsis_scores:
                row["topsis_score"] = topsis_scores[api_id]
            row.setdefault("selector_reason", "Fallback: filled remaining slots from rank/rag order after deterministic TOPSIS.")
            selected.append(row)
            seen.add(api_id)
            if len(selected) >= top_n:
                break

    trace["topsis_status"] = "ok"
    trace["scored_candidates"] = len(topsis_scores)
    return selected[:top_n], trace


def select_top_apis_for_subtask(
    *,
    llm_call: Callable[[str], str],
    user_query: str,
    subtask: Dict[str, Any],
    selector_candidates: List[Dict[str, Any]],
    mode_override: str | None = None,
    top_n: int = 10,
    topsis_top_k: int = 20,
    min_qos_candidates: int = 5,
    prompt_path: str = "prompts/selector_no_qos.md",
    debug_raw_path: str | None = None,
    debug_prompt_path: str | None = None,
    debug_parsed_path: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Selector over all retrieved candidates for one subtask.

    PURE_LLM: LLM-based selector over the broader retrieved set.
    TOPSIS: deterministic selector over the broader retrieved set.
    """
    mode = (mode_override.strip().upper() if isinstance(mode_override, str) and mode_override.strip() else _selector_mode())
    selector_candidates = [c for c in selector_candidates if isinstance(c, dict) and c.get("api_id")]

    fallback_order = sorted(
        selector_candidates,
        key=lambda x: (
            0 if x.get("rank") is not None else 1,
            int(x.get("rank") or 10**9),
            -(float(x.get("rag_score") or 0.0)),
        ),
    )

    trace: Dict[str, Any] = {
        "subtask_id": str(subtask.get("id", "unknown")),
        "mode": mode,
        "top_n": top_n,
        "topsis_top_k": topsis_top_k,
        "min_qos_candidates": min_qos_candidates,
        "selector_candidates": len(selector_candidates),
    }

    if mode == "TOPSIS":
        selected, topsis_trace = _deterministic_topsis_select(
            selector_candidates=selector_candidates,
            fallback_order=fallback_order,
            top_n=top_n,
            topsis_top_k=topsis_top_k,
            min_qos_candidates=min_qos_candidates,
        )
        trace.update(topsis_trace)
        if debug_parsed_path:
            p = Path(debug_parsed_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
        return selected, trace

    include_qos = any(_get_qos(c.get("service") or {}) for c in selector_candidates)
    selector_payload: List[Dict[str, Any]] = [_slim_candidate(c, include_qos=include_qos, include_topsis=False) for c in fallback_order]

    template = Path(prompt_path).read_text(encoding="utf-8")
    prompt = (
        template
        .replace("{user_query}", user_query)
        .replace("{subtask_json}", json.dumps(subtask, ensure_ascii=False))
        .replace("{top_n}", str(top_n))
        .replace("{candidates_json}", json.dumps(selector_payload, ensure_ascii=False))
    )

    if debug_prompt_path:
        p = Path(debug_prompt_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(prompt, encoding="utf-8")

    raw = llm_call(prompt)
    if debug_raw_path:
        p = Path(debug_raw_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(raw or "", encoding="utf-8")

    data = json.loads(_coerce_json(raw)) if raw else {}
    picked_raw = data.get("selected", [])
    picked: List[Dict[str, Any]] = []
    if isinstance(picked_raw, list):
        for r in picked_raw:
            if not isinstance(r, dict) or not r.get("api_id"):
                continue
            picked.append({"api_id": str(r.get("api_id")), "selector_reason": r.get("reason", "") or ""})

    if debug_parsed_path:
        p = Path(debug_parsed_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"selected": picked}, indent=2, ensure_ascii=False), encoding="utf-8")

    selected: List[Dict[str, Any]] = []
    seen = set()
    by_id = {str(c.get("api_id")): c for c in selector_candidates}
    for item in picked:
        api_id = item["api_id"]
        if api_id in seen or api_id not in by_id:
            continue
        row = dict(by_id[api_id])
        row["selector_reason"] = item.get("selector_reason", "")
        selected.append(row)
        seen.add(api_id)
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for c in fallback_order[:top_n]:
            api_id = str(c.get("api_id"))
            if api_id in seen:
                continue
            row = dict(c)
            row.setdefault("selector_reason", "Fallback: selector filled remaining slots from rank/rag order.")
            selected.append(row)
            seen.add(api_id)
            if len(selected) >= top_n:
                break

    trace["selected_by"] = "llm_selector_then_fallback"
    trace["prompt_path"] = prompt_path
    trace["picked_count"] = len(picked)
    return selected[:top_n], trace
