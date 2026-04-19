from __future__ import annotations

from typing import Any, Dict, List, Tuple


def select_ranked_relevant_apis(
    *,
    ranked_candidates: List[Dict[str, Any]],
    relevancy_map: Dict[str, Dict[str, Any]],
    fallback_top_n: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deterministic selector:
    - keep only relevant APIs in the mode's rank order
    - if none are relevant, fall back to top N ranked APIs for that mode
    """
    selected: List[Dict[str, Any]] = []
    for item in ranked_candidates:
        api_id = str(item.get("api_id", "")).strip()
        if not api_id:
            continue
        rel = relevancy_map.get(api_id, {}).get("relevant", 0)
        if int(rel) == 1:
            row = dict(item)
            row["selection_reason"] = "Relevant API selected deterministically."
            row["fallback_used"] = False
            selected.append(row)

    trace: Dict[str, Any] = {
        "selected_api_ids": [str(x.get("api_id")) for x in selected],
        "fallback_used": False,
        "fallback_top_n": int(fallback_top_n),
    }

    if selected:
        return selected, trace

    fallback: List[Dict[str, Any]] = []
    for item in ranked_candidates[: max(0, int(fallback_top_n))]:
        row = dict(item)
        row["selection_reason"] = f"Fallback: no relevant APIs found, using top {int(fallback_top_n)} ranked APIs."
        row["fallback_used"] = True
        fallback.append(row)

    trace["selected_api_ids"] = [str(x.get("api_id")) for x in fallback]
    trace["fallback_used"] = True
    return fallback, trace
