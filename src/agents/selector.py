# src/agents/selector.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np

from src.eval.topsis_eval import _extract_qos, _run_topsis_pydecision


def _selector_mode() -> str:
    """Return selector mode.

    Env:
      MAOF_SELECTOR_MODE = PURE_LLM | TOPSIS
    """
    mode = (os.getenv("MAOF_SELECTOR_MODE", "PURE_LLM") or "PURE_LLM").strip().upper()
    if mode in {"TOPSIS", "TOPSIS_ENABLED"}:
        return "TOPSIS"
    return "PURE_LLM"


def _get_qos(service: Dict[str, Any]) -> Dict[str, Any]:
    qos = (service or {}).get("qos") or {}
    return qos if isinstance(qos, dict) else {}


def select_top_apis_for_subtask(
    *,
    subtask_id: str,
    ranked_pool: List[Dict[str, Any]],
    mode_override: str | None = None,
    top_n: int = 8,
    topsis_top_k: int = 12,
    min_qos_candidates: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select final APIs for one subtask.

    Inputs:
      ranked_pool: ordered list (best first) from Ranker.
        Each element should contain:
          api_id, rank, rag_score, service (incl. qos)

    Modes:
      PURE_LLM:
        - select first top_n by rank order

      TOPSIS:
        - run TOPSIS on QoS over the top `topsis_top_k` items in ranked_pool
        - pick top_n by TOPSIS score
        - if QoS candidates are insufficient, fall back to rank order
        - always de-dup api_id in selected

    Returns:
      (selected_list, trace)
    """
    mode = (mode_override.strip().upper() if isinstance(mode_override, str) and mode_override.strip() else _selector_mode())
    ranked_pool = [c for c in ranked_pool if isinstance(c, dict) and c.get("api_id")]

    baseline = ranked_pool[:top_n]
    trace: Dict[str, Any] = {
        "subtask_id": subtask_id,
        "mode": mode,
        "top_n": top_n,
        "topsis_top_k": topsis_top_k,
        "min_qos_candidates": min_qos_candidates,
    }

    if mode != "TOPSIS":
        trace["selected_by"] = "rank_order"
        return baseline, trace

    pool = ranked_pool[: max(topsis_top_k, top_n)]

    # Build QoS matrix
    valid_rows: List[Tuple[str, List[float]]] = []
    for c in pool:
        qos = _get_qos(c.get("service") or {})
        vec = _extract_qos(qos)
        if vec is None:
            continue
        valid_rows.append((str(c["api_id"]), vec))

    trace["qos_candidates"] = len(valid_rows)

    if len(valid_rows) < min_qos_candidates:
        trace["topsis_status"] = "insufficient_qos_candidates"
        trace["selected_by"] = "fallback_rank_order"
        return baseline, trace

    X = np.asarray([v for (_, v) in valid_rows], dtype=float)
    w = [1.0, 1.0, 1.0]

    scores, ranking = _run_topsis_pydecision(X, w)

    topsis_order: List[Dict[str, Any]] = []
    for ridx in ranking:
        i = int(ridx)
        if 0 <= i < len(valid_rows):
            api_id = valid_rows[i][0]
            topsis_order.append({"api_id": api_id, "topsis_score": float(scores[i])})
    trace["topsis_ranking"] = topsis_order

    selected: List[Dict[str, Any]] = []
    seen = set()

    # 1) pick by TOPSIS
    for item in topsis_order:
        if len(selected) >= top_n:
            break
        api_id = str(item["api_id"])
        if api_id in seen:
            continue
        full = next((c for c in ranked_pool if str(c.get("api_id")) == api_id), None)
        if not full:
            continue
        full2 = dict(full)
        full2["topsis_score"] = item["topsis_score"]
        selected.append(full2)
        seen.add(api_id)

    # 2) fill remaining from baseline rank order
    if len(selected) < top_n:
        for c in baseline:
            if len(selected) >= top_n:
                break
            api_id = str(c.get("api_id"))
            if api_id in seen:
                continue
            selected.append(c)
            seen.add(api_id)

    trace["topsis_status"] = "ok"
    trace["selected_by"] = "topsis_then_fallback"
    return selected, trace
