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
    picks: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    ranked_top: List[Dict[str, Any]],
    plan: Dict[str, Any],
    top_k_candidates: int = 12,
    min_qos_candidates: int = 5,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Mode 1 evaluator:
      Candidate set per microservice = ranker-scored APIs that retriever marked relevant for that subtask.
      TOPSIS runs only on that candidate set (8 to 12 typical).

    Outputs a JSON-serializable report.
    """
    api_to_subtasks = _build_api_to_subtasks(picks)

    # build ranker lookup: api_id -> (ranker_score, ranker_rank)
    ranked_sorted = sorted(ranked, key=lambda r: float(r.get("score", 0.0)), reverse=True)
    ranker_pos = {r["api_id"]: i for i, r in enumerate(ranked_sorted)}
    ranker_score = {r["api_id"]: float(r.get("score", 0.0) or 0.0) for r in ranked_sorted}

    # service lookup for qos
    id_to_service = {r.get("api_id"): (r.get("service") or {}) for r in ranked_top}
    id_to_qos = {api_id: (svc.get("qos") if isinstance(svc, dict) else None) for api_id, svc in id_to_service.items()}

    primary_path = _select_primary_path(plan)
    if not primary_path:
        return {
            "mode": "mode1_ranker_candidates",
            "error": "No paths found in planner output",
        }

    steps = primary_path.get("steps") or []

    w = _get_weights(weights)

    microservice_reports: List[Dict[str, Any]] = []
    for step in steps:
        step_num = step.get("step")
        step_api = step.get("api_id")
        subtask_id = step.get("subtask_id")

        # Build candidate pool for this microservice
        candidates_ids: List[str] = []
        if isinstance(subtask_id, int):
            for api_id, st_ids in api_to_subtasks.items():
                if subtask_id in st_ids:
                    candidates_ids.append(api_id)
        else:
            # fallback: global top-k candidates
            candidates_ids = [r["api_id"] for r in ranked_sorted[:top_k_candidates]]

        # intersect with ranker outputs and keep ranker order
        candidates_ids = [api_id for api_id in ranked_sorted if api_id["api_id"] in set(candidates_ids)]
        candidates_ids = [x["api_id"] for x in candidates_ids[:top_k_candidates]]

        # collect candidate rows with valid QoS
        rows: List[CandidateRow] = []
        dropped: List[Dict[str, Any]] = []
        for api_id in candidates_ids:
            qos_obj = id_to_qos.get(api_id)
            vec = _extract_qos(qos_obj if isinstance(qos_obj, dict) else {})
            if vec is None:
                dropped.append({"api_id": api_id, "reason": "missing_or_invalid_qos"})
                continue
            rows.append(CandidateRow(api_id=api_id, ranker_score=ranker_score.get(api_id, 0.0), qos=qos_obj))

        report: Dict[str, Any] = {
            "step": step_num,
            "subtask_id": subtask_id,
            "planner_choice_api_id": step_api,
            "candidate_source": "subtask_scoped" if isinstance(subtask_id, int) else "global_fallback",
            "num_candidates_requested": len(candidates_ids),
            "num_candidates_valid_qos": len(rows),
            "num_candidates_dropped": len(dropped),
            "dropped": dropped[:50],  # cap
        }

        if len(rows) < min_qos_candidates:
            report["status"] = "insufficient_qos_candidates"
            microservice_reports.append(report)
            continue

        # build matrix X in fixed order
        X = np.array([_extract_qos(r.qos) for r in rows], dtype=float)

        # outlier mitigation
        X2, outlier_log = _winsorize_matrix(X, p_low=5.0, p_high=95.0)
        report["outlier_mitigation"] = outlier_log

        # run topsis
        scores, order = _run_topsis_pydecision(X2, w)

        # order: indices best-first
        ranked_rows = [rows[i] for i in order]
        top5 = ranked_rows[:5]

        def _rank_of(api_id: str) -> Optional[int]:
            for idx, r in enumerate(ranked_rows, start=1):
                if r.api_id == api_id:
                    return idx
            return None

        planner_rank = _rank_of(step_api) if step_api else None
        planner_hit5 = bool(planner_rank is not None and planner_rank <= 5)

        # ranker_top1 among this candidate set is the first by ranker score/order
        # since candidates_ids were in ranker order, use first valid-qos row that appears in that order
        ranker_top1 = None
        valid_set = {r.api_id for r in rows}
        for api_id in candidates_ids:
            if api_id in valid_set:
                ranker_top1 = api_id
                break

        ranker_rank = _rank_of(ranker_top1) if ranker_top1 else None
        ranker_hit5 = bool(ranker_rank is not None and ranker_rank <= 5)

        # Best by TOPSIS is ranked_rows[0]
        topsis_best = ranked_rows[0].api_id if ranked_rows else None

        report.update({
            "status": "ok",
            "weights": {"rt_ms": w[0], "tp_rps": w[1], "availability": w[2]},
            "topsis_best_api_id": topsis_best,
            "topsis_top5_api_ids": [r.api_id for r in top5],
            "planner_hit5": planner_hit5,
            "planner_rank": planner_rank,
            "ranker_top1_api_id": ranker_top1,
            "ranker_hit5": ranker_hit5,
            "ranker_rank": ranker_rank,
        })

        microservice_reports.append(report)

    # Path-level aggregation
    ok = [m for m in microservice_reports if m.get("status") == "ok"]
    def avg(xs):
        xs = [x for x in xs if x is not None]
        return float(sum(xs) / len(xs)) if xs else None

    path_summary = {
        "num_steps": len(microservice_reports),
        "num_ok": len(ok),
        "num_insufficient_qos": len([m for m in microservice_reports if m.get("status") != "ok"]),
        "planner_hit5_rate": avg([1.0 if m.get("planner_hit5") else 0.0 for m in ok]),
        "ranker_hit5_rate": avg([1.0 if m.get("ranker_hit5") else 0.0 for m in ok]),
        "planner_avg_rank": avg([m.get("planner_rank") for m in ok]),
        "ranker_avg_rank": avg([m.get("ranker_rank") for m in ok]),
    }

    return {
        "mode": "mode1_ranker_candidates",
        "primary_path_id": primary_path.get("path_id"),
        "path_score": primary_path.get("path_score"),
        "microservices": microservice_reports,
        "path_summary": path_summary,
    }
