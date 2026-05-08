from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def format_candidate_id(index: int) -> str:
    """Return a stable short candidate ID for a one-based prompt position."""
    return f"C{int(index):02d}"


def assign_candidate_ids(
    candidates: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    """
    Add C01-style IDs in the current candidate order.

    Returns:
      - candidates with candidate_id inserted first
      - candidate_id -> api_id
      - api_id -> first candidate_id
    """
    enriched: List[Dict[str, Any]] = []
    candidate_id_to_api_id: Dict[str, str] = {}
    api_id_to_candidate_id: Dict[str, str] = {}

    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            candidate = {}
        candidate_id = format_candidate_id(index)
        api_id = str(candidate.get("api_id") or "").strip()
        row: Dict[str, Any] = {"candidate_id": candidate_id}
        row.update({key: value for key, value in candidate.items() if key != "candidate_id"})
        enriched.append(row)
        if api_id:
            candidate_id_to_api_id[candidate_id] = api_id
            api_id_to_candidate_id.setdefault(api_id, candidate_id)

    return enriched, candidate_id_to_api_id, api_id_to_candidate_id
