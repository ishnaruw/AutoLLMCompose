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


def _selection_order(row):
    for key in ("selection_order", "selected_rank"):
        value = row.get(key)
        if value is not None:
            return value
    return _candidate_rank(row)


_NO_QOS_SERVICE_KEYS = {
    "qos",
    "rt_ms",
    "tp_rps",
    "availability",
    "qos_score",
    "qos_rank",
    "topsis_score",
    "topsis_rank",
    "qos_llm_score",
    "qos_llm_rank",
}

_PLANNER_SERVICE_KEYS = (
    "api_id",
    "name",
    "category",
    "description",
    "tool_name",
    "tool_description",
    "method",
    "url",
    "endpoint_details",
)

_PLANNER_QOS_KEYS = (
    "qos",
    "rt_ms",
    "tp_rps",
    "availability",
    "qos_score",
    "qos_rank",
    "topsis_score",
    "topsis_rank",
    "qos_llm_score",
    "qos_llm_rank",
)


def _has_value(value):
    return value is not None and value != "" and value != [] and value != {}


def _first_value(*values):
    for value in values:
        if _has_value(value):
            return value
    return None


def _compact_endpoint_details(service, enrichment):
    endpoint_details = _first_value(
        service.get("endpoint_details"),
        service.get("toolbench_endpoint_details"),
        enrichment.get("endpoint_details"),
    )
    if not isinstance(endpoint_details, dict):
        return None

    compact = {}
    for key in ("required_parameters", "optional_parameters"):
        value = endpoint_details.get(key)
        if isinstance(value, list):
            compact[key] = value
    return compact if compact else None


def _service_for_planner(row, *, strip_qos: bool):
    service = row.get("service", {})
    if not isinstance(service, dict):
        return {}

    enrichment = service.get("toolbench_enrichment")
    enrichment = enrichment if isinstance(enrichment, dict) else {}

    service_copy = {}
    for key in _PLANNER_SERVICE_KEYS:
        if key == "tool_name":
            value = _first_value(
                service.get("tool_name"),
                service.get("toolbench_tool_name"),
                enrichment.get("tool_name"),
            )
        elif key == "tool_description":
            value = _first_value(
                service.get("tool_description"),
                service.get("toolbench_tool_description"),
                enrichment.get("tool_description"),
            )
        elif key == "description":
            value = _first_value(
                service.get("description"),
                service.get("toolbench_endpoint_description"),
                enrichment.get("endpoint_description"),
            )
        elif key == "method":
            value = _first_value(service.get("method"), enrichment.get("endpoint_method"))
        elif key == "url":
            value = _first_value(service.get("url"), enrichment.get("endpoint_url"))
        elif key == "endpoint_details":
            value = _compact_endpoint_details(service, enrichment)
        else:
            value = service.get(key)

        if _has_value(value):
            service_copy[key] = deepcopy(value)

    for key in _PLANNER_QOS_KEYS:
        value = service.get(key)
        if _has_value(value):
            service_copy[key] = deepcopy(value)

    if strip_qos:
        for key in _NO_QOS_SERVICE_KEYS:
            service_copy.pop(key, None)
    return service_copy


def _repair_planner_payload(payload):
    if not isinstance(payload, dict):
        return payload

    repaired = deepcopy(payload)
    if "primary_plan" not in repaired and isinstance(repaired.get("steps"), list):
        execution_workflow = repaired.get("execution_workflow")
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
            "execution_workflow": execution_workflow,
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
        "- Use the required top-level shape: primary_plan, execution_workflow, selected_api_ids, overall_rationale.\n"
        "- Put plan_id, summary, steps, and subtask_coverage inside primary_plan.\n"
        "- Put type and steps inside execution_workflow.\n"
        "- Each execution_workflow step must include step, api_id, method, url, required_parameters, optional_parameters, depends_on, input_mapping, output_mapping, and expected_output.\n"
        "- Copy method, url, and endpoint parameter details from the matching Candidate API service when available.\n"
        "- Every step.api_id must be a non-empty string copied exactly from the provided Candidate APIs.\n"
        "- Never use null for api_id. If a subtask seems internal/UI/local, choose the closest suitable provided API and describe the local work in action/why.\n"
        "- input_from_previous_step and output_to_next_step must be strings or null, never objects or arrays.\n"
        "- Do not invent APIs and do not reorder subtasks.\n"
    )


def planner_call(llm_call, user_goal: str, ranked_top, subtasks=None, prompt_path: str = "prompts/planner_qos.md"):
    """
    Compose one final orchestration plan from selected candidates.

    Inputs:
      - user_goal: the original natural language goal.
      - ranked_top: selected candidate APIs with planner selection order and
        their catalog metadata,
        for example:
            [
              {
                "api_id": "...",
                "subtask_id": 1,
                "selection_order": 1,
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

    Expected LLM output (see prompts/planner_qos.md for schema):
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
        "execution_workflow": {
          "type": "sequential",
          "steps": [
            {
              "step": <int>,
              "api_id": "...",
              "subtask_id": <int or null>,
              "method": "...",
              "url": "...",
              "required_parameters": [...],
              "optional_parameters": [...],
              "depends_on": [<int, ...>],
              "input_mapping": "...",
              "output_mapping": "...",
              "expected_output": "..."
            },
            ...
          ]
        },
        "selected_api_ids": [...],
        "overall_rationale": "..."
      }

    Behavior:
      - Builds a compact JSON payload with api_id, subtask_id,
        selection_order, and the service entry as seen in the catalog.
      - Fills the planner prompt template with:
          * user_goal
          * subtasks (JSON)
          * selected candidates (compact JSON)
      - Calls the LLM and parses the returned JSON plan.
      - Returns the parsed dictionary directly.
    """
    compact = []
    strip_qos = str(prompt_path).endswith("planner_no_qos.md")
    for r in ranked_top:
        compact.append({
            "api_id": r.get("api_id"),
            "subtask_id": r.get("subtask_id"),
            "selection_order": _selection_order(r),
            "service": _service_for_planner(r, strip_qos=strip_qos),
        })

    subtasks_json = json.dumps(subtasks or [], ensure_ascii=False)
    selected_candidates_json = json.dumps(compact, ensure_ascii=False)

    with open(prompt_path, "r", encoding="utf-8") as f:
        tmpl = f.read()

    prompt = (
        tmpl
        .replace("{user_goal}", user_goal)
        .replace("{subtasks_json}", subtasks_json)
        .replace("{selected_candidates_json}", selected_candidates_json)
        .replace("{ranked_compact}", selected_candidates_json)
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
