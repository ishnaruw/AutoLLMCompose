# src/agents/planner.py
from copy import deepcopy
import json
import re

from src.core.json_parsing import parse_llm_json
from src.core.output_schemas import PlannerOutput, validate_output_schema
from src.core.run_logging import log_line, log_warning_event


def _json_text(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _compact_json_text(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _json_text(value)


def _derive_selected_api_ids(steps):
    selected = []
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            continue
        api_id = step.get("api_id")
        if isinstance(api_id, str) and api_id.strip() and api_id not in selected:
            selected.append(api_id)
    return selected


def _normalize_subtask_id(value):
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _subtask_text_by_id(subtasks):
    by_id = {}
    for subtask in subtasks if isinstance(subtasks, list) else []:
        if not isinstance(subtask, dict):
            continue
        subtask_id = _normalize_subtask_id(subtask.get("id") or subtask.get("subtask_id"))
        if not subtask_id:
            continue
        description = subtask.get("description") or subtask.get("task") or subtask.get("title") or ""
        by_id[subtask_id] = str(description)
    return by_id


_MULTI_API_SUBTASK_PATTERNS = (
    r"\bmultiple\s+apis?\b",
    r"\bmore\s+than\s+one\s+apis?\b",
    r"\btwo\s+apis?\b",
    r"\bboth\s+apis?\b",
    r"\bcombine\s+(?:data|results|outputs)\s+from\b",
    r"\baggregate\s+(?:data|results|outputs)\s+from\b",
    r"\bcross[-\s]?reference\b",
)

_CLEAR_MULTI_API_JUSTIFICATION_PATTERNS = (
    r"\bexplicitly\s+requires\s+multiple\s+apis?\b",
    r"\brequires\s+multiple\s+apis?\b",
    r"\brequires\s+more\s+than\s+one\s+apis?\b",
    r"\bneeds\s+more\s+than\s+one\s+apis?\b",
    r"\bmust\s+use\s+multiple\s+apis?\b",
    r"\bno\s+single\s+apis?\s+(?:can|covers|provides|supports)\b",
    r"\bone\s+apis?\s+(?:cannot|does\s+not)\b",
)


def _matches_any_pattern(text, patterns):
    normalized = str(text or "").lower()
    return any(re.search(pattern, normalized) for pattern in patterns)


def _has_clear_multi_api_justification(subtask_text, steps):
    if _matches_any_pattern(subtask_text, _MULTI_API_SUBTASK_PATTERNS):
        return True

    fragments = [subtask_text]
    for step in steps:
        if not isinstance(step, dict):
            continue
        for field in ("action", "why", "input_from_previous_step", "output_to_next_step"):
            value = step.get(field)
            if value is not None:
                fragments.append(str(value))
    return _matches_any_pattern(" ".join(fragments), _CLEAR_MULTI_API_JUSTIFICATION_PATTERNS)


def _over_composed_subtasks(plan, subtasks):
    if not isinstance(plan, dict):
        return []

    primary_plan = plan.get("primary_plan")
    if not isinstance(primary_plan, dict):
        return []

    steps = primary_plan.get("steps")
    if not isinstance(steps, list):
        return []

    by_subtask = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        subtask_id = _normalize_subtask_id(step.get("subtask_id"))
        api_id = step.get("api_id")
        if not subtask_id or not isinstance(api_id, str) or not api_id.strip():
            continue
        by_subtask.setdefault(subtask_id, []).append(step)

    subtask_descriptions = _subtask_text_by_id(subtasks)
    over_composed = []
    for subtask_id, subtask_steps in by_subtask.items():
        api_ids = [
            step.get("api_id").strip()
            for step in subtask_steps
            if isinstance(step.get("api_id"), str) and step.get("api_id").strip()
        ]
        if len(api_ids) <= 1:
            continue
        if _has_clear_multi_api_justification(subtask_descriptions.get(subtask_id, ""), subtask_steps):
            continue
        over_composed.append({
            "subtask_id": subtask_id,
            "step_count": len(api_ids),
            "api_ids": api_ids,
        })
    return over_composed


def _log_over_composition(over_composed, prompt_path):
    payload = {
        "event_type": "planner_over_composition_detected",
        "warning_type": "planner_over_composition_detected",
        "prompt_path": str(prompt_path),
        "over_composed_subtasks": over_composed,
    }
    log_line(f"[planner] warning: planner_over_composition_detected {json.dumps(over_composed, ensure_ascii=False)}")
    log_warning_event(payload)


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


def _candidate_qos_values(row):
    qos_values = row.get("qos_values")
    if isinstance(qos_values, dict):
        return qos_values

    service = row.get("service")
    service = service if isinstance(service, dict) else {}
    qos = service.get("qos")
    qos = qos if isinstance(qos, dict) else {}
    return {
        "rt_s": row.get("rt_s", qos.get("rt_s")),
        "tp_kbps": row.get("tp_kbps", qos.get("tp_kbps")),
        "availability": row.get("availability", qos.get("availability")),
    }


def _compact_candidate_for_ablation(row, *, strip_qos: bool):
    compact = {
        "api_id": row.get("api_id"),
        "subtask_id": row.get("subtask_id"),
        "selection_order": _selection_order(row),
        "mode_rank": row.get("mode_rank"),
        "rank_source": row.get("rank_source"),
        "short_rank_reason": row.get("short_rank_reason") or row.get("reason"),
        "service": _service_for_planner(row, strip_qos=strip_qos),
        "qos_values": None if strip_qos else _candidate_qos_values(row),
        "functional_match_label": row.get("functional_match_label"),
        "functional_refiner_reason": row.get("functional_refiner_reason"),
    }
    for key in (
        "selected_by_view",
        "pareto_status",
        "balanced_relative_qos_score",
        "topsis_rank",
        "topsis_score",
        "hybrid_pool_strategy",
    ):
        if row.get(key) is not None:
            compact[key] = row.get(key)
    return compact


_PLANNER_MODE_RULES_PLACEHOLDER = "{planner_candidate_mode_rules}"


def _planner_candidate_mode_rules(planner_candidate_mode: str, planner_top_n_cap: int, *, strip_qos: bool) -> str:
    mode = str(planner_candidate_mode or "fixed_one").strip()
    if mode == "top_n_ablation":
        evidence_fields = (
            "- Compact candidate evidence fields may include short_rank_reason, rank_source, mode_rank, "
            "selection_order, functional_match_label, and functional_refiner_reason.\n"
        )
        qos_choice = (
            "- Prefer selection_order = 1 unless another candidate clearly improves functional fit, "
            "step compatibility, or workflow-level QoS.\n"
            "- Compare workflow-level QoS using total response time = sum(rt_s), "
            "bottleneck throughput = min(tp_kbps), and average availability = mean(availability).\n"
            "- Do not choose a locally strong API if it weakens the full workflow.\n"
        )
        if strip_qos:
            qos_choice = (
                "- Prefer selection_order = 1 unless another candidate clearly improves functional fit "
                "or step compatibility.\n"
                "- QoS fields are omitted in this prompt; do not infer hidden QoS values.\n"
            )
        else:
            evidence_fields = (
                "- Compact candidate evidence fields may include qos_values, short_rank_reason, rank_source, "
                "mode_rank, selection_order, functional_match_label, and functional_refiner_reason.\n"
            )
        return (
            "Planner candidate mode: top_n_ablation.\n"
            f"- The planner received up to {planner_top_n_cap} ranked alternatives per subtask.\n"
            "- Provided APIs are ranked alternatives for each subtask.\n"
            "- Choose exactly one API per subtask from that subtask's provided candidates.\n"
            f"{qos_choice}"
            f"{evidence_fields}"
            "- qos_hybrid metadata meanings when present: selected_by_view identifies the ranking view "
            "that surfaced the candidate; pareto_status shows whether it is Pareto-preferred or part of "
            "the expanded pool; balanced_relative_qos_score is the balanced QoS comparison score; "
            "topsis_rank and topsis_score provide TOPSIS ordering and closeness; hybrid_pool_strategy "
            "describes how the hybrid candidate pool was formed.\n"
            "- If you choose a candidate whose selection_order is greater than 1, include "
            "planner_override_reason on the primary_plan step and execution_workflow step for that subtask. "
            "Base the reason on functional fit, step compatibility, workflow-level QoS, or the candidate "
            "evidence fields.\n"
            "- selected_api_ids must contain only the chosen API IDs, one per subtask.\n"
        )

    return (
        "Planner candidate mode: fixed_one.\n"
        "- The selected APIs are fixed by the selection stage.\n"
        "- There is exactly one selected API per subtask.\n"
        "- Use that API for its subtask and preserve subtask order.\n"
        "- Do not replace, re-rank, or substitute APIs.\n"
        "- The planner only composes the fixed APIs into a coherent workflow.\n"
    )


def _insert_planner_candidate_mode_rules(template: str, rules: str) -> str:
    if _PLANNER_MODE_RULES_PLACEHOLDER in template:
        return template.replace(_PLANNER_MODE_RULES_PLACEHOLDER, rules.rstrip())

    for marker in ("\nCandidate APIs:", "\nCandidates:", "\nRules:"):
        index = template.find(marker)
        if index >= 0:
            return f"{template[:index]}\n{rules.rstrip()}\n{template[index:]}"

    return f"{rules.rstrip()}\n\n{template}"


_NO_QOS_SERVICE_KEYS = {
    "qos",
    "rt_s",
    "tp_kbps",
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
    "rt_s",
    "tp_kbps",
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

    execution_workflow = repaired.get("execution_workflow")
    if isinstance(execution_workflow, dict):
        steps = execution_workflow.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                for field in ("input_mapping", "output_mapping"):
                    value = step.get(field)
                    if value is None:
                        step[field] = ""
                    elif isinstance(value, (dict, list, tuple)):
                        step[field] = _compact_json_text(value)

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
        "- execution_workflow.steps[*].input_mapping must be a string, never null.\n"
        "- execution_workflow.steps[*].output_mapping must be a string, never null.\n"
        "- For the first step, use \"none\" or \"\" for input_mapping instead of null.\n"
        "- Use \"none\" or \"\" for output_mapping if no explicit output mapping is needed.\n"
        "- Copy method, url, and endpoint parameter details from the matching Candidate API service when available.\n"
        "- Every step.api_id must be a non-empty string copied exactly from the provided Candidate APIs.\n"
        "- Never use null for api_id. If a subtask seems internal/UI/local, choose the closest suitable provided API and describe the local work in action/why.\n"
        "- input_from_previous_step and output_to_next_step must be strings or null, never objects or arrays.\n"
        "- Do not invent APIs and do not reorder subtasks.\n"
    )


def _over_composition_retry_prompt(prompt, over_composed):
    issue_text = json.dumps(over_composed, ensure_ascii=False, sort_keys=True)
    return (
        f"{prompt}\n\n"
        "Your previous plan used multiple APIs for the same subtask without necessity. "
        "Revise the workflow so each subtask uses exactly one API unless multiple APIs are explicitly required.\n"
        f"Over-composed subtasks:\n{issue_text}\n\n"
        "Return exactly one primary plan as JSON only.\n"
    )


def planner_call(
    llm_call,
    user_goal: str,
    ranked_top,
    subtasks=None,
    prompt_path: str = "prompts/planner_qos.md",
    planner_candidate_mode: str = "fixed_one",
    planner_top_n_cap: int = 1,
):
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
        if planner_candidate_mode == "top_n_ablation":
            compact.append(_compact_candidate_for_ablation(r, strip_qos=strip_qos))
        else:
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

    mode_rules = _planner_candidate_mode_rules(
        planner_candidate_mode,
        planner_top_n_cap,
        strip_qos=strip_qos,
    )
    tmpl = _insert_planner_candidate_mode_rules(tmpl, mode_rules)

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

    def _call_and_parse_with_schema_retry(base_prompt):
        resp = llm_call(base_prompt)
        try:
            return _parse_plan(resp)
        except ValueError as first_error:
            retry_resp = llm_call(_schema_retry_prompt(base_prompt, first_error.args[0] if first_error.args else str(first_error)))
            return _parse_plan(retry_resp)

    plan = _call_and_parse_with_schema_retry(prompt)
    over_composed = _over_composed_subtasks(plan, subtasks)
    if over_composed:
        _log_over_composition(over_composed, prompt_path)
        plan = _call_and_parse_with_schema_retry(_over_composition_retry_prompt(prompt, over_composed))
    return plan
