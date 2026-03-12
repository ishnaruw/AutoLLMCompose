# src/eval/topsis_eval.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import contextlib
import io
import numpy as np
from pyDecision.algorithm import topsis_method


def _extract_qos(qos: Dict[str, Any]) -> Optional[List[float]]:
    """Return [rt_ms, tp_rps, availability] if all are valid positive numeric values."""
    try:
        rt_ms = qos.get("rt_ms")
        tp_rps = qos.get("tp_rps")
        availability = qos.get("availability")
        if rt_ms is None or tp_rps is None or availability is None:
            return None
        rt_ms = float(rt_ms)
        tp_rps = float(tp_rps)
        availability = float(availability)
        if rt_ms <= 0 or tp_rps <= 0 or availability <= 0:
            return None
        return [rt_ms, tp_rps, availability]
    except Exception:
        return None


def _run_topsis_pydecision(X: np.ndarray, weights: List[float]) -> Tuple[np.ndarray, List[int]]:
    """Run standard TOPSIS using pyDecision.

    Criteria directions:
      rt_ms        -> cost (min)
      tp_rps       -> benefit (max)
      availability -> benefit (max)

    Supports pyDecision variants that return either:
      - (scores, ranking)
      - scores only (ndarray), in which case ranking is derived descending by score
    Suppresses console printing from pyDecision.
    """
    criterion_types = ["min", "max", "max"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = topsis_method(X, weights, criterion_types, graph=False)

    if isinstance(result, tuple) and len(result) == 2:
        scores, ranking = result
        scores = np.asarray(scores, dtype=float).reshape(-1)
        ranking_arr = np.asarray(ranking).reshape(-1)
        # pyDecision may return 0-based indices or 1-based labels; normalize to 0-based indices
        if ranking_arr.size and ranking_arr.min() >= 1 and ranking_arr.max() <= len(scores):
            ranking_idx = [int(x) - 1 for x in ranking_arr.tolist()]
        else:
            ranking_idx = [int(x) for x in ranking_arr.tolist()]
        return scores, ranking_idx

    if isinstance(result, np.ndarray):
        scores = np.asarray(result, dtype=float).reshape(-1)
        ranking_idx = list(np.argsort(-scores))
        return scores, ranking_idx

    # Some versions may return list-like scores
    try:
        scores = np.asarray(result, dtype=float).reshape(-1)
        ranking_idx = list(np.argsort(-scores))
        return scores, ranking_idx
    except Exception as e:
        raise ValueError(f"Unexpected TOPSIS return format: {type(result)}") from e
