# src/eval/topsis_eval.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import re

import numpy as np

# pyDecision TOPSIS
# Signature (per library source):
# topsis_method(dataset, weights, criterion_type, graph=True, verbose=True)
# criterion_type typically uses "min" for cost and "max" for benefit.
from pyDecision.algorithm.topsis import topsis_method


_SUBTASK_RE = re.compile(r"\[Subtask\s+(\d+)\]")

CRITERIA = ["rt_ms", "tp_rps", "availability"]
CRITERIA_TYPES = ["min", "max", "max"]  # cost, benefit, benefit


@dataclass
class CandidateRow:
    api_id: str
    ranker_score: float
    qos: Dict[str, Any]


def _extract_subtask_ids(reason: str) -> List[int]:
    if not reason:
        return []
    return [int(x) for x in _SUBTASK_RE.findall(reason)]


def _build_api_to_subtasks(picks: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    picks is saved as 1_retriever.json:
      [{"api_id": "...", "reason": "[Subtask 1] ... | Also relevant for subtask 2: ..."}, ...]
    """
    out: Dict[str, List[int]] = {}
    for p in picks:
        api_id = p.get("api_id")
        if not api_id:
            continue
        st = _extract_subtask_ids(p.get("reason", ""))
        # de-dup while keeping stable order
        seen = set()
        uniq = []
        for x in st:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        out[api_id] = uniq
    return out


def _select_primary_path(plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    paths = plan.get("paths") or []
    if not paths:
        return None
    # pick the highest path_score if present
    def score(p):
        try:
            return float(p.get("path_score", 0.0))
        except Exception:
            return 0.0
    paths_sorted = sorted(paths, key=score, reverse=True)
    return paths_sorted[0]


def _winsorize_matrix(X: np.ndarray, p_low: float = 5.0, p_high: float = 95.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Deterministic outlier mitigation.
    Clips each column to [p_low, p_high] percentiles and logs what happened.
    """
    X2 = X.copy()
    log: Dict[str, Any] = {"method": "winsorize", "p_low": p_low, "p_high": p_high, "per_column": []}
    for j in range(X.shape[1]):
        col = X[:, j]
        lo = np.percentile(col, p_low)
        hi = np.percentile(col, p_high)
        before = col.copy()
        col2 = np.clip(col, lo, hi)
        X2[:, j] = col2
        log["per_column"].append({
            "criterion": CRITERIA[j],
            "clip_low": float(lo),
            "clip_high": float(hi),
            "num_clipped": int(np.sum(before != col2)),
        })
    return X2, log


def _get_weights(weights: Optional[Dict[str, float]] = None) -> List[float]:
    """
    For now, keep weights simple and stable.
    You can later replace with your user-query weight extractor.
    """
    default = {"rt_ms": 0.4, "tp_rps": 0.3, "availability": 0.3}
    wmap = weights or default
    w = [float(wmap.get(k, 0.0)) for k in CRITERIA]
    s = sum(w)
    if s <= 0:
        return [1/3, 1/3, 1/3]
    return [x / s for x in w]


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _extract_qos(service_qos: Dict[str, Any]) -> Optional[List[float]]:
    """
    Expected qos like:
      {"rt_ms": 0.121, "tp_rps": 33.057, "availability": 1.0, "valid_qos": true}
    """
    if not isinstance(service_qos, dict):
        return None
    if service_qos.get("valid_qos") is False:
        return None

    rt = _safe_float(service_qos.get("rt_ms"))
    tp = _safe_float(service_qos.get("tp_rps"))
    av = _safe_float(service_qos.get("availability"))
    if rt is None or tp is None or av is None:
        return None
    return [rt, tp, av]


def _run_topsis_pydecision(X: np.ndarray, w: List[float]) -> Tuple[List[float], List[int]]:
    """
    Returns:
      - closeness scores aligned to rows
      - ranking order as indices (best first)
    """
    dataset = np.asarray(X, dtype=float)   # IMPORTANT: must be ndarray for pyDecision (.shape)
    weights = w
    criterion_type = CRITERIA_TYPES

    res = topsis_method(dataset, weights, criterion_type, graph=False, verbose=False)

    scores: Optional[List[float]] = None
    ranking: Optional[List[int]] = None

    if isinstance(res, tuple) and len(res) >= 2:
        a, b = res[0], res[1]
        if isinstance(a, (list, np.ndarray)) and isinstance(b, (list, np.ndarray)):
            scores = [float(x) for x in list(a)]
            ranking = [int(x) for x in list(b)]
    elif isinstance(res, (list, np.ndarray)):
        vals = list(res)
        if vals and all(isinstance(x, (int, np.integer)) for x in vals):
            ranking = [int(x) for x in vals]
        else:
            scores = [float(x) for x in vals]

    if scores is None:
        lin = (dataset[:, 1] + dataset[:, 2]) - dataset[:, 0]
        scores = [float(x) for x in lin.tolist()]

    if ranking is None:
        ranking = list(np.argsort(np.array(scores))[::-1])

    if ranking and min(ranking) == 1 and max(ranking) == dataset.shape[0]:
        ranking = [r - 1 for r in ranking]

    return scores, ranking



def evaluate_topsis_mode1(
    *,
    subtasks: List[Dict[str, Any]],
    ranked_top: List[Dict[str, Any]],
    plan: Dict[str, Any],
    top_k_candidates: int = 12,
    min_qos_candidates: int = 5,
    weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Mode 1 TOPSIS evaluation over the per-subtask candidate set.

    Inputs:
      - subtasks: decomposed subtasks [{"id":..., "description":...}, ...]
      - ranked_top: list of candidates passed to planner, each including:
            {"api_id", "subtask_id", "rank", "score", "rag_score", "service": {...}}
      - plan: planner output JSON that includes steps with subtask_id and api_id

    For each subtask:
      - consider up to top_k_candidates candidates from ranked_top for that subtask
      - extract QoS metrics (rt_ms, tp_rps, availability) from service.qos when present
      - run TOPSIS if enough valid QoS candidates exist
      - report whether planner chose the TOPSIS best

    Note: Ranker score is not used here. TOPSIS uses QoS metrics only.
    """
    weights = weights or [1.0, 1.0, 1.0]

    # Build plan selection: subtask_id -> first api_id used for that subtask
    chosen_by_subtask: Dict[str, str] = {}
    try:
        paths = plan.get("paths", []) or []
        if paths:
            steps = paths[0].get("steps", []) or []
            for st in steps:
                sid = st.get("subtask_id")
                aid = st.get("api_id")
                if sid is None or not aid:
                    continue
                sid_s = str(sid)
                if sid_s not in chosen_by_subtask:
                    chosen_by_subtask[sid_s] = aid
    except Exception:
        pass

    # Group candidates by subtask_id
    cand_by_sub: Dict[str, List[Dict[str, Any]]] = {}
    for c in ranked_top:
        sid = c.get("subtask_id")
        if sid is None:
            continue
        sid_s = str(sid)
        cand_by_sub.setdefault(sid_s, []).append(c)

    # For each subtask, run TOPSIS if possible
    eval_steps: List[Dict[str, Any]] = []
    for sub in subtasks:
        sid_s = str(sub.get("id", "unknown"))
        cands = cand_by_sub.get(sid_s, [])[:top_k_candidates]

        # Extract QoS rows
        rows: List[CandidateRow] = []
        for c in cands:
            api_id = c.get("api_id")
            service = c.get("service") or {}
            qos = service.get("qos") or {}
            if not api_id or not isinstance(qos, dict):
                continue
            rows.append(CandidateRow(api_id=str(api_id), ranker_score=float(c.get("score", 0.0) or 0.0), qos=qos))

        # Build TOPSIS matrix
        dataset = []
        kept_ids = []
        for r in rows:
            rt = r.qos.get("rt_ms")
            tp = r.qos.get("tp_rps")
            av = r.qos.get("availability")
            try:
                rt_f = float(rt)
                tp_f = float(tp)
                av_f = float(av)
            except Exception:
                continue
            dataset.append([rt_f, tp_f, av_f])
            kept_ids.append(r.api_id)

        if len(dataset) < min_qos_candidates:
            eval_steps.append({
                "subtask_id": sid_s,
                "subtask": sub.get("description", ""),
                "status": "insufficient_qos_candidates",
                "qos_candidates": len(dataset),
                "planner_choice": chosen_by_subtask.get(sid_s),
            })
            continue

        ds = np.array(dataset, dtype=float)

        # Winsorize outliers (p5-p95) per metric to reduce dominance of extreme values
        def winsorize(col: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> np.ndarray:
            lo = np.percentile(col, p_lo)
            hi = np.percentile(col, p_hi)
            return np.clip(col, lo, hi)

        ds_w = ds.copy()
        for j in range(ds.shape[1]):
            ds_w[:, j] = winsorize(ds_w[:, j])

        # Run TOPSIS (robust to pyDecision return-shape changes)
        try:
            scores, ranking = _run_topsis_pydecision(ds_w, [float(x) for x in weights])
        except Exception as e:
            eval_steps.append({
                "subtask_id": sid_s,
                "subtask": sub.get("description", ""),
                "status": "topsis_error",
                "error": str(e),
            })
            continue

        if not scores:
            eval_steps.append({
                "subtask_id": sid_s,
                "subtask": sub.get("description", ""),
                "status": "topsis_error",
                "error": "empty topsis scores",
            })
            continue

        best_idx = int(np.argmax(np.array(scores, dtype=float)))
        topsis_best = kept_ids[best_idx]
        planner_choice = chosen_by_subtask.get(sid_s)

        eval_steps.append({
            "subtask_id": sid_s,
            "subtask": sub.get("description", ""),
            "status": "ok",
            "qos_candidates": len(dataset),
            "topsis_best": topsis_best,
            "planner_choice": planner_choice,
            "planner_matches_topsis": (planner_choice == topsis_best) if planner_choice else None,
            "topsis_ranking": [
                {"api_id": kept_ids[i], "topsis_score": float(scores[i])}
                for i in (ranking if ranking else list(np.argsort(-np.array(scores, dtype=float))))
            ],
        })

    return {
        "mode": "mode1_per_subtask_candidates",
        "criteria": CRITERIA,
        "criteria_types": CRITERIA_TYPES,
        "weights": weights,
        "top_k_candidates": top_k_candidates,
        "min_qos_candidates": min_qos_candidates,
        "steps": eval_steps,
    }
