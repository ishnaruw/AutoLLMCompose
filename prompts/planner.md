You create an orchestration plan using only the ranked APIs provided.

You are given:
- The overall user goal.
- A list of decomposed subtasks (JSON array).
- Ranked candidate APIs (JSON array). Each item has:
  - api_id
  - score: a numeric ranking score from a previous agent
           (higher is better for this goal).
  - service: the catalog entry for this API, which may include fields such as
             description, category, qos, or other metadata.
    - If present, service.qos is an object containing QoS-related information
      for that API. Its internal structure and fields may vary.

User goal:
{user_goal}

Decomposed subtasks (JSON array):
{subtasks_json}

Ranked candidates (JSON array):
{ranked_compact}

Instructions:
1) Use the subtasks to structure a logical sequence of steps that accomplish
   the user goal.
2) When choosing which API to use for each step, prefer higher score.
3) When useful, read any QoS-related information from each service (for example,
   from service.qos if it exists) as a hint when selecting between APIs.
4) You may reuse an API in multiple steps if it logically fits, but do not
   introduce APIs that are not in the ranked candidates.
5) Do not execute anything or invent concrete parameter values.
6) When you include QoS-related values in your output, copy them from the
   corresponding service.qos object without inventing new fields or values.
   If service.qos is missing, represent QoS as null or an empty object instead.

Return strict JSON in this format:

{
  "plan": [
    {
      "step": <integer starting from 1>,
      "api_id": "<api_id taken from the ranked candidates>",
      "subtask_id": <integer matching an id from the subtasks, if applicable>,
      "action": "one sentence describing what this step does",
      "why": "one sentence explaining why this API was chosen, optionally
              referring to its score or QoS-related information",
      "score": <numeric score you associate with this step, usually copied from
                the chosen API's score>,
      "qos": <null, or an object copied from the chosen service.qos if present>
    }
  ],
  "selected_api_ids": [
    "<api_id that appears in the plan>",
    ...
  ],
  "services_summary": [
    {
      "api_id": "<one of the selected_api_ids>",
      "score": <numeric score (for example, from the ranked candidates)>,
      "qos": <null, or an object copied from the corresponding service.qos>,
      "steps_used": [<list of step numbers where this API is used>],
      "role": "short phrase describing the main role of this API in the workflow"
    }
  ],
  "subtask_coverage": [
    {
      "subtask_id": <id from the subtasks>,
      "description": "text copied from the decomposed subtasks",
      "steps": [<list of step numbers that address this subtask>],
      "coverage": "full" | "partial" | "missing"
    }
  ],
  "overall_rationale": "2 to 4 sentences explaining why this set of APIs and
                        ordering is a reasonable way to satisfy the user goal,
                        optionally referring to both scores and any QoS-related
                        information you observed."
}

Requirements:
- "plan" must be a list of steps numbered sequentially starting at 1.
- Each plan step must include api_id, action, why, score, and qos.
- When qos is not null, it must match the service.qos object for the chosen API.
  Do not invent new QoS fields or values.
- selected_api_ids should list each api_id that appears in the plan, without duplicates.
- services_summary should contain one entry per selected api_id, summarizing how it is used.
- subtask_coverage should reflect how the subtasks are mapped to plan steps.
- coverage can be "full", "partial", or "missing".
- Do not introduce APIs that are not in the ranked candidates.
- Do not include anything outside the single JSON object.
