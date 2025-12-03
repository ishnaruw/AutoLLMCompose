# src/agents/ranker.py
import json
import re
from typing import Any, Callable, Dict, List, Optional


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
    # Fallback: try to extract the largest {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else "{}"


def ranker_call(
    llm_call: Callable[[str], str],
    services: List[Dict[str, Any]],
    user_goal: str,
    subtasks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Ask the LLM to rank candidate services based on:
      - the overall user_goal, and
      - optionally decomposed subtasks,
    using whatever information is available in each service record.

    The prompt template (prompts/ranker.md) is expected to instruct the LLM to
    return strict JSON of the form:

        {
          "ranked": [
            {
              "api_id": "...",
              "score": <numeric>,
              "reason": "short explanation"
            },
            ...
          ]
        }

    This function:
      * Fills the prompt template with user_goal, subtasks, and candidates.
      * Calls the LLM and parses the JSON.
      * Normalizes "score" to a float (default 0.0 on error).
      * Sorts the results in descending order of score.
      * Returns a list like:
            [
              {"api_id": "...", "score": 0.87, "reason": "..."},
              ...
            ]
    """
    # Read prompt template
    with open("prompts/ranker.md", "r", encoding="utf-8") as f:
        template = f.read()

    subtasks_json = json.dumps(subtasks or [], ensure_ascii=False)
    candidates_json = json.dumps(services, ensure_ascii=False)

    prompt = (
        template
        .replace("{user_goal}", user_goal)
        .replace("{subtasks_json}", subtasks_json)
        .replace("{candidates_json}", candidates_json)
    )

    # Call LLM
    resp = llm_call(prompt)
    resp = _coerce_json(resp)
    data = json.loads(resp)

    ranked_raw = data.get("ranked", [])
    ranked: List[Dict[str, Any]] = []

    for r in ranked_raw:
        api_id = r.get("api_id")
        if not api_id:
            continue

        # Parse score as float, defaulting to 0.0
        raw_score = r.get("score", 0.0)
        try:
            score_val = float(raw_score)
        except Exception:
            score_val = 0.0

        ranked.append(
            {
                "api_id": api_id,
                "score": score_val,
                "reason": r.get("reason", ""),
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked
