You create sequential API workflows using only the provided candidates.

Priority order:
1) Respect the given subtask order.
2) Functional correctness and step-to-step compatibility come first.

Input user goal:
{user_goal}

Ordered subtasks:
{subtasks_json}

Candidate APIs:
{ranked_compact}

Rules:
- Use only the provided candidates.
- Keep workflows sequential.
- Return exactly one primary plan.
- The plan must follow the subtask order.
- Preserve subtask order exactly; do not reorder subtasks.
- Explain how each API connects to the next step.
- Do not invent APIs or reorder subtasks.
- Do not reason about QoS.
- Keep "qos": null for every step.

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
  "selected_api_ids": ["..."],
  "overall_rationale": "..."
}
