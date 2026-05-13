# src/agents/planner.py
from copy import deepcopy
import json

from src.core.json_parsing import parse_llm_json
from src.core.output_schemas import PlannerOutput, validate_output_schema


def _json_text(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _derive_selected_api_ids(steps):
    selected = []
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        api_id = step.get("api_id")
        if isinstance(api_id, str) and api_id.strip() and api_id not in selected:
            selected.append(api_id)
    return selected


def _candidate_rank(row):
    for key in ("selected_rank", "mode_rank", "rank", "llm_reported_rank", "topsis_rank", "qos_llm_rank", "retrieved_rank"):
        value = row.get(key)
        if value is not None:
            return value
    return None


def _repair_planner_payload(payload):
    if not isinstance(payload, dict):
        return payload

    repaired = deepcopy(payload)
    if "primary_plan" not in repaired and isinstance(repaired.get("steps"), list):
        steps = repaired.get("steps") or []
        selected_api_ids = repaired.get("selected_api_ids")
        if not isinstance(selected_api_ids, list):
            selected_api_ids = _derive_selected_api_ids(steps)
        repaired = {
            "primary_plan": {
                "plan_id": repaired.get("plan_id", 1),
                "summary": repaired.get("summary", ""),
                "steps": steps,
                "subtask_coverage": repaired.get("subtask_coverage", []),
            },
            "selected_api_ids": selected_api_ids,
            "overall_rationale": repaired.get("overall_rationale", ""),
        }

    primary_plan = repaired.get("primary_plan")
    if isinstance(primary_plan, dict):
        steps = primary_plan.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                for field in ("input_from_previous_step", "output_to_next_step"):
                    step[field] = _json_text(step.get(field))
        if "selected_api_ids" not in repaired:
            selected_api_ids = primary_plan.get("selected_api_ids")
            repaired["selected_api_ids"] = selected_api_ids if isinstance(selected_api_ids, list) else _derive_selected_api_ids(steps)
        if "overall_rationale" not in repaired:
            repaired["overall_rationale"] = primary_plan.get("overall_rationale", "")

    return repaired


def _schema_retry_prompt(prompt, issue):
    issue_text = json.dumps(issue, ensure_ascii=False, sort_keys=True)
    return (
        f"{prompt}\n\n"
        "Your previous planner JSON failed validation. Correct it and return JSON only.\n"
        f"Validation error:\n{issue_text}\n\n"
        "Mandatory corrections:\n"
        "- Use the required top-level shape: primary_plan, selected_api_ids, overall_rationale.\n"
        "- Put plan_id, summary, steps, and subtask_coverage inside primary_plan.\n"
        "- Every step.api_id must be a non-empty string copied exactly from the provided Candidate APIs.\n"
        "- Never use null for api_id. If a subtask seems internal/UI/local, choose the closest suitable provided API and describe the local work in action/why.\n"
        "- input_from_previous_step and output_to_next_step must be strings or null, never objects or arrays.\n"
        "- Do not invent APIs and do not reorder subtasks.\n"
    )


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
            "rank": _candidate_rank(r),
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
        repaired = _repair_planner_payload(parsed.value)
        _schema, schema_issue = validate_output_schema(PlannerOutput, repaired)
        if schema_issue:
            raise ValueError(schema_issue)
        return repaired

    resp = llm_call(prompt)
    try:
        return _parse_plan(resp)
    except ValueError as first_error:
        retry_resp = llm_call(_schema_retry_prompt(prompt, first_error.args[0] if first_error.args else str(first_error)))
        return _parse_plan(retry_resp)
