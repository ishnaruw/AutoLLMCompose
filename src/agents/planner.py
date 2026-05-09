# src/agents/planner.py
import json

from src.core.json_parsing import parse_llm_json


def planner_call(llm_call, user_goal: str, ranked_top, subtasks=None, prompt_path: str = "prompts/planner.md"):
    """
    Compose one or more alternative orchestration paths from ranked candidates.

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
      - subtasks: optional list of decomposed subtasks, for example:
            [
              {"id": 1, "description": "..."},
              ...
            ]

    Expected LLM output (see prompts/planner.md for schema):
      {
        "paths": [
          {
            "path_id": <int>,
            "path_score": <number>,
            "summary": "...",
            "steps": [
              {
                "step": <int>,
                "api_id": "...",
                "subtask_id": <int or null>,
                "action": "...",
                "why": "...",
                "score": <number>,
                "qos": <null or object copied from service.qos>
              },
              ...
            ],
            "subtask_coverage": [
              {
                "subtask_id": <int>,
                "description": "...",
                "steps": [<int, ...>],
                "coverage": "full" | "partial" | "missing"
              },
              ...
            ]
          },
          ...
        ],
        "selected_api_ids": [...],
        "overall_rationale": "..."
      }

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
            "rank": r.get("rank"),
            "subtask_id": r.get("subtask_id"),
            "rag_score": r.get("rag_score"),
            "service": r.get("service", {}),
        })

    subtasks_json = json.dumps(subtasks or [], ensure_ascii=False)
    ranked_json = json.dumps(compact, ensure_ascii=False)

    with open(prompt_path, "r", encoding="utf-8") as f:
        tmpl = f.read()

    prompt = (
        tmpl
        .replace("{user_goal}", user_goal)
        .replace("{subtasks_json}", subtasks_json)
        .replace("{ranked_compact}", ranked_json)
    )

    def _parse_plan(text: str):
        parsed = parse_llm_json(text)
        if parsed.error:
            raise ValueError(parsed.error)
        if not isinstance(parsed.value, dict):
            raise ValueError({"reason": "wrong_json_type", "expected_type": "object", "actual_type": type(parsed.value).__name__})
        return parsed.value

    resp = llm_call(prompt)
    plan = _parse_plan(resp)

    # Enforce 3 paths: retry once with a short corrective instruction.
    try:
        paths = plan.get("paths") or []
    except Exception:
        paths = []

    if not isinstance(paths, list) or len(paths) < 3:
        retry_prompt = prompt + "\n\nIMPORTANT: Return EXACTLY 3 paths in 'paths' with path_id = 1, 2, 3. Output JSON only."
        resp2 = llm_call(retry_prompt)
        plan2 = _parse_plan(resp2)
        try:
            paths2 = plan2.get("paths") or []
        except Exception:
            paths2 = []
        plan = plan2 if isinstance(paths2, list) and len(paths2) >= 1 else plan

    # If still fewer than 3 paths, pad by cloning the best path into valid alternatives.
    try:
        paths = plan.get("paths") or []
    except Exception:
        paths = []

    if not isinstance(paths, list):
        paths = []
    if paths and len(paths) < 3:
        base = paths[0]
        for pid in range(len(paths) + 1, 4):
            clone = json.loads(json.dumps(base))  # deep copy
            clone["path_id"] = pid
            clone["summary"] = f"Alternative plan {pid} (fallback)"
            # Slightly lower score to keep ordering deterministic
            try:
                clone["path_score"] = float(clone.get("path_score", 0.0)) - (pid * 0.01)
            except Exception:
                clone["path_score"] = 0.0
            paths.append(clone)
        plan["paths"] = paths[:3]

    return plan
