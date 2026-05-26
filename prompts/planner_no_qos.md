You create sequential API workflows using only the provided candidates.

Priority order:
1) Respect the given subtask order.
2) Functional correctness and step-to-step compatibility come first.

Input user goal:
{user_goal}

Ordered subtasks:
{subtasks_json}

Candidate APIs:
{selected_candidates_json}

Rules:
- Use only the provided candidates.
- The selected APIs are fixed by the selection stage. Do not replace, re-rank, or substitute them. Your task is only to compose them into a coherent workflow.
- There is exactly one selected API per subtask. Use that API for its subtask and preserve subtask order.
- Keep workflows sequential.
- Return exactly one primary plan.
- Return a machine-readable execution workflow in addition to the human-readable primary plan.
- The plan must follow the subtask order.
- Preserve subtask order exactly; do not reorder subtasks.
- Explain how each API connects to the next step.
- Do not invent APIs or reorder subtasks.
- Every step must use a non-empty "api_id" copied exactly from the provided Candidate APIs.
- For each execution workflow step, copy method, url, required parameters, and optional parameters from the selected candidate's "service" object when available.
- If endpoint_details.required_parameters or endpoint_details.optional_parameters are present, use them to populate required_parameters and optional_parameters.
- In required_parameters and optional_parameters, include each parameter name and describe its source, such as "user_goal", "previous_step_output", "constant/default", or "unknown_needed_from_user".
- Use depends_on to identify previous workflow step numbers needed before a step can run; use [] for the first independent step.
- Use input_mapping and output_mapping to describe how values flow between steps.
- execution_workflow.steps[*].input_mapping must always be a string. For the first step, use "none" or an empty string instead of null.
- execution_workflow.steps[*].output_mapping must always be a string. Use "none" or an empty string if there is no output mapping.
- Do not return null, objects, arrays, or nested JSON for execution_workflow input_mapping or output_mapping.
- Never use null, "none", "internal", "local", or invented placeholders for "api_id".
- Do not create internal-only, UI-only, formatting-only, or local-computation-only steps without an API.
- If a subtask seems internal, describe the internal transformation inside the "action" of the closest selected API step or choose the closest suitable provided API for that subtask.
- "input_from_previous_step" and "output_to_next_step" must be strings or null only; do not return objects, arrays, or nested JSON in these fields.
- The top-level JSON must contain exactly these required fields: "primary_plan", "execution_workflow", "selected_api_ids", and "overall_rationale".
- Put "plan_id", "summary", "steps", and "subtask_coverage" inside "primary_plan"; do not put "steps" at the top level.
- Put "type" and "steps" inside "execution_workflow".
- This is the no-QoS planning mode. Generate the workflow using only functional suitability, subtask coverage, parameter compatibility, and selected API order. Do not optimize using QoS or quality-related attributes. Keep "qos": null for every workflow step.

Return JSON only:
{
  "primary_plan": {
    "plan_id": 1,
    "summary": "...",
    "steps": [
      {
        "step": 1,
        "api_id": "...",
        "subtask_id": 1,
        "action": "...",
        "input_from_previous_step": "...",
        "output_to_next_step": "...",
        "why": "...",
        "qos": null
      }
    ],
    "subtask_coverage": [
      {"subtask_id": 1, "description": "...", "steps": [1], "coverage": "full"}
    ]
  },
  "execution_workflow": {
    "type": "sequential",
    "steps": [
      {
        "step": 1,
        "api_id": "...",
        "subtask_id": 1,
        "method": "GET",
        "url": "https://...",
        "required_parameters": [
          {"name": "...", "source": "user_goal", "value_hint": "..."}
        ],
        "optional_parameters": [
          {"name": "...", "source": "constant/default", "value_hint": "..."}
        ],
        "depends_on": [],
        "input_mapping": "...",
        "output_mapping": "...",
        "expected_output": "..."
      }
    ]
  },
  "selected_api_ids": ["..."],
  "overall_rationale": "..."
}
