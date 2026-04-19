from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List


def _parse_llm_qos_scores_no_rank(raw: str) -> Dict[str, float]:
    """
    Extract QoS scores from LLM output without ranking.
    Used for batched processing where global ranking happens after all batches.
    Returns api_id -> score
    """
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    items = data.get("qos_scored", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        return {}
    
    out: Dict[str, float] = {}
    for item in items:
        if isinstance(item, dict) and item.get("api_id") and item.get("qos_score") is not None:
            api_id = str(item.get("api_id"))
            score = float(item.get("qos_score", 0))
            out[api_id] = score
    return out


def _parse_llm_qos_scores(raw: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract QoS scores from LLM output and assign ranks locally.
    Used for non-batched processing (single LLM call).
    Returns api_id -> {qos_llm_score, qos_llm_rank}
    """
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    items = data.get("qos_scored", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        return {}
    
    # Sort by qos_score descending (higher score = better QoS)
    valid_items = [item for item in items if isinstance(item, dict) and item.get("api_id") and item.get("qos_score") is not None]
    valid_items.sort(key=lambda x: float(x.get("qos_score", 0)), reverse=True)
    
    out: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(valid_items, start=1):
        api_id = str(item.get("api_id"))
        score = float(item.get("qos_score", 0))
        out[api_id] = {
            "qos_llm_score": score,
            "qos_llm_rank": idx,
        }
    return out


def score_qos_llm(
    llm_call: Callable[[str], str],
    *,
    candidates: List[Dict[str, Any]],
    prompt_path: str = "prompts/qos_score_llm.md",
    debug_raw_path: str | None = None,
    batch_size: int | None = 15,
) -> Dict[str, Dict[str, Any]]:
    """
    LLM-only QoS scoring with optional adaptive batching.
    
    Args:
        llm_call: Function to invoke LLM
        candidates: List of API candidates with QoS metrics
        prompt_path: Path to QoS scoring prompt
        debug_raw_path: Optional path to save debug output
        batch_size: Batch size for adaptive batching (None or 0 = no batching, default 15 = batch mode)
    
    Returns:
        api_id -> {qos_llm_score, qos_llm_rank}
    
    If batch_size is None or 0: processes all candidates in one LLM call (original behavior).
    If batch_size > 0: processes candidates in batches, then ranks globally for better quality.
    """
    template = Path(prompt_path).read_text(encoding="utf-8")
    
    # Non-batched: single LLM call for all candidates
    if not batch_size or batch_size <= 0:
        payload = []
        for c in candidates:
            payload.append(
                {
                    "api_id": str(c.get("api_id", "")),
                    "rt_ms": c.get("rt_ms"),
                    "tp_rps": c.get("tp_rps"),
                    "availability": c.get("availability"),
                }
            )

        prompt = template.replace("{candidates_json}", json.dumps(payload, ensure_ascii=False))
        raw = llm_call(prompt)

        if debug_raw_path:
            p = Path(debug_raw_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(raw or "", encoding="utf-8")

        parsed = _parse_llm_qos_scores(raw)
        if parsed:
            return parsed

        total = len(payload) or 1
        return {
            item["api_id"]: {
                "qos_llm_rank": idx,
                "qos_llm_score": float(total - idx + 1) / float(total),
            }
            for idx, item in enumerate(payload, start=1)
            if item.get("api_id")
        }
    
    # Batched: process in chunks, collect scores, then rank globally
    all_scores: Dict[str, float] = {}
    
    for batch_idx, i in enumerate(range(0, len(candidates), batch_size)):
        batch = candidates[i : i + batch_size]
        batch_payload = []
        for c in batch:
            batch_payload.append(
                {
                    "api_id": str(c.get("api_id", "")),
                    "rt_ms": c.get("rt_ms"),
                    "tp_rps": c.get("tp_rps"),
                    "availability": c.get("availability"),
                }
            )
        
        prompt = template.replace("{candidates_json}", json.dumps(batch_payload, ensure_ascii=False))
        raw = llm_call(prompt)
        
        if debug_raw_path:
            p = Path(debug_raw_path)
            batch_file = p.parent / f"{p.stem}_batch{batch_idx}{p.suffix}"
            batch_file.parent.mkdir(parents=True, exist_ok=True)
            batch_file.write_text(raw or "", encoding="utf-8")
        
        # Extract scores only (no local ranking; global ranking happens after all batches)
        parsed = _parse_llm_qos_scores_no_rank(raw)
        all_scores.update(parsed)
    
    if not all_scores:
        # Fallback: uniform scoring if no valid results
        total = len(candidates) or 1
        return {
            item["api_id"]: {
                "qos_llm_rank": idx,
                "qos_llm_score": float(total - idx + 1) / float(total),
            }
            for idx, item in enumerate(candidates, start=1)
            if item.get("api_id")
        }
    
    # Sort globally by score and reassign ranks
    sorted_items = sorted(
        all_scores.items(),
        key=lambda x: float(x[1]),
        reverse=True
    )
    
    result: Dict[str, Dict[str, Any]] = {}
    for idx, (api_id, score) in enumerate(sorted_items, start=1):
        result[api_id] = {
            "qos_llm_score": score,
            "qos_llm_rank": idx,
        }
    
    return result
