# src/agents/ranker.py
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List


def _truncate(s: Any, n: int) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def _slim_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a candidate to a compact representation to avoid context overflow.

    The RAG index metadata can be large. For ranking we only need a few fields.
    """
    comp = c.get("compressed") or {}
    if not isinstance(comp, dict):
        comp = {}

    service = c.get("service") or {}
    if not isinstance(service, dict):
        service = {}

    # Common functional fields
    name = comp.get("name") or service.get("name") or comp.get("operation") or comp.get("title")
    summary = comp.get("summary") or comp.get("description") or comp.get("desc") or service.get("description")
    method = comp.get("method") or service.get("method")
    path = comp.get("path") or comp.get("endpoint") or comp.get("url") or service.get("url")
    category = comp.get("category") or service.get("category") or c.get("category")
    tool_name = comp.get("tool_name") or service.get("tool_name") or service.get("_tool")
    tool_description = comp.get("tool_description") or service.get("tool_description")

    # Params can explode prompt size; keep only names.
    params = comp.get("params") or comp.get("parameters")
    param_names: List[str] = []
    if isinstance(params, list):
        for p in params[:30]:
            if isinstance(p, dict) and p.get("name"):
                param_names.append(str(p.get("name")))
            elif isinstance(p, str):
                param_names.append(p)
    elif isinstance(params, dict):
        # sometimes parameters is a dict keyed by name
        param_names = [str(k) for k in list(params.keys())[:30]]

    # QoS fields: include only if present (and compact).
    qos: Dict[str, Any] = {}
    service_qos = service.get("qos") if isinstance(service.get("qos"), dict) else {}
    for k in [
        "availability",
        "reliability",
        "throughput",
        "tp_rps",
        "rt_ms",
        "response_time_ms",
        "latency_ms",
        "p95_latency_ms",
    ]:
        if k in comp:
            qos[k] = comp.get(k)
        elif k in c:
            qos[k] = c.get(k)
        elif k in service:
            qos[k] = service.get(k)
        elif k in service_qos:
            qos[k] = service_qos.get(k)

    slim: Dict[str, Any] = {
        "api_id": c.get("api_id"),
        "rag_score": c.get("rag_score"),
        "category": category,
        "tool_name": _truncate(tool_name, 120),
        "tool_description": _truncate(tool_description, 240),
        "name": _truncate(name, 120),
        "summary": _truncate(summary, 240),
        "method": _truncate(method, 16),
        "path": _truncate(path, 140),
    }
    if param_names:
        slim["param_names"] = param_names
    if qos:
        slim["qos"] = qos
    return slim


def _coerce_json(s: str) -> str:
    """Return a valid JSON string or {} if we cannot parse."""
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


def rank_subtask(
    llm_call: Callable[[str], str],
    *,
    user_query: str,
    subtask: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    prompt_path: str = "prompts/ranker.md",
    debug_raw_path: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Rank candidates for ONE subtask.

    Input candidates should already include any relevant catalog fields, including:
      - api_id
      - rag_score (weak hint)
      - service / compressed fields
      - qos fields if present

    Prompt (prompts/ranker.md) must return strict JSON:
      {
        "ranked": [
          {"api_id": "...", "reason": "..."},
          ...
        ]
      }

    Returns the ranked list in order (best first).
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    # --- Context safety ---
    # RAG retrieval can return large "compressed" payloads. Sending all topK
    # (e.g., 60) candidates verbatim can exceed local model context windows.
    # We:
    #   1) take the top N by rag_score
    #   2) slim each candidate down to only essential fields
    from src.config import CONFIG

    MAX_RANK_CANDIDATES = CONFIG.ranker_max_candidates
    # Ranker outputs a stable pool size for the Selector.
    RANKER_POOL_N = CONFIG.ranker_pool_n
    cand_sorted = sorted(
        (c for c in candidates if isinstance(c, dict) and c.get("api_id")),
        key=lambda x: float(x.get("rag_score") or 0.0),
        reverse=True,
    )
    cand_trimmed = cand_sorted[:MAX_RANK_CANDIDATES]
    cand_slim = [_slim_candidate(c) for c in cand_trimmed]

    prompt = (
        template
        .replace("{user_query}", user_query)
        .replace("{subtask_json}", json.dumps(subtask, ensure_ascii=False))
        .replace("{candidates_json}", json.dumps(cand_slim, ensure_ascii=False))
    )

    resp_raw = llm_call(prompt)
    if debug_raw_path:
        from pathlib import Path
        Path(debug_raw_path).parent.mkdir(parents=True, exist_ok=True)
        Path(debug_raw_path).write_text(resp_raw or "", encoding="utf-8")

    resp = _coerce_json(resp_raw)
    data = json.loads(resp) if resp else {}

    ranked_raw = data.get("ranked", [])
    ranked: List[Dict[str, Any]] = []
    if isinstance(ranked_raw, list):
        for r in ranked_raw:
            if not isinstance(r, dict):
                continue
            api_id = r.get("api_id")
            if not api_id:
                continue
            ranked.append(
                {
                    "api_id": api_id,
                    "reason": r.get("reason", "") or "",
                }
            )

    # If model failed, fall back to rag_score ordering (descending)
    if not ranked:
        ranked = [
            {"api_id": c.get("api_id"), "reason": "Fallback: rag_score ordering."}
            for c in cand_trimmed
            if c.get("api_id")
        ]

    # Enforce a stable pool size: append missing IDs (rag_score order) if the
    # model returned too few items.
    ranked_ids = {str(r.get("api_id")) for r in ranked if r.get("api_id")}
    if len(ranked) < RANKER_POOL_N:
        for c in cand_trimmed:
            cid = c.get("api_id")
            if not cid:
                continue
            cid_s = str(cid)
            if cid_s in ranked_ids:
                continue
            ranked.append({"api_id": cid, "reason": "Appended to fill pool size."})
            ranked_ids.add(cid_s)
            if len(ranked) >= RANKER_POOL_N:
                break

    return ranked[:RANKER_POOL_N]
