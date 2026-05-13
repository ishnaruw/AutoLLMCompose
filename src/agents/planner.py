# src/agents/planner.py
import json

from src.core.json_parsing import parse_llm_json
from src.core.output_schemas import PlannerOutput, validate_output_schema


def planner_call(llm_call, user_goal: str, ranked_top, subtasks=None, prompt_path: str = "prompts/planner.md"):
    """
    Compose one final orchestration plan from ranked candidates.

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
        "primary_plan": {
          "plan_id": <int>,
          "summary": "...",
          "steps": [
            {
              "step": <int>,
              "api_id": "...",
              "subtask_id": <int or null>,
              "action": "...",
              "input_from_previous_step": "...",
              "output_to_next_step": "...",
              "why": "...",
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
        _schema, schema_issue = validate_output_schema(PlannerOutput, parsed.value)
        if schema_issue:
            raise ValueError(schema_issue)
        return parsed.value

    resp = llm_call(prompt)
    return _parse_plan(resp)
