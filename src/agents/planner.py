# src/agents/planner.py
import json
import re


def _coerce_json(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "{}"
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else "{}"


def planner_call(llm_call, user_goal: str, ranked_top, subtasks=None):
    """
    Compose an orchestration plan from ranked candidates.

    Inputs:
      - user_goal: the original natural language goal.
      - ranked_top: list of candidate APIs with scores and their catalog entries,
        for example:
            [
              {
                "api_id": "...",
                "score": <number>,
                "service": {
                  "api_id": "...",
                  "description": "...",
                  "category": "...",
                  "qos": { ... },  # arbitrary fields if present
                  ...
                }
              },
              ...
            ]
      - subtasks: optional list of decomposed subtasks.

    Behavior:
      - Builds a compact JSON payload with api_id, score, and the full service
        entry as seen in the catalog.
      - Fills the planner prompt template with:
          * user_goal
          * subtasks (JSON)
          * ranked candidates (compact JSON)
      - Calls the LLM and parses the returned JSON plan.
      - Returns the parsed dictionary directly.
    """
    compact = []
    for r in ranked_top:
        compact.append({
            "api_id": r.get("api_id"),
            "score": float(r.get("score", 0) or 0),
            "service": r.get("service", {}),
        })

    subtasks_json = json.dumps(subtasks or [], ensure_ascii=False)
    ranked_json = json.dumps(compact, ensure_ascii=False)

    with open("prompts/planner.md", "r", encoding="utf-8") as f:
        tmpl = f.read()

    prompt = (
        tmpl
        .replace("{user_goal}", user_goal)
        .replace("{subtasks_json}", subtasks_json)
        .replace("{ranked_compact}", ranked_json)
    )

    resp = llm_call(prompt)
    resp = _coerce_json(resp)
    return json.loads(resp)
