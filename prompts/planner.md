You create one or more alternative orchestration plans using only the ranked APIs provided.

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
1) Use the subtasks to design several alternative end-to-end plans ("paths")
   that could satisfy the user goal.
2) Each path is a sequence of steps. Each step must use one of the ranked APIs.
3) Prefer higher-scoring APIs when it makes sense, but you may explore
   different trade-offs across paths (for example, fewer calls vs. more calls,
   different API choices, or different ways to cover subtasks).
4) When useful, read any QoS-related information from each service entry
   (for example, from service.qos if it exists) as a hint when selecting
   between APIs.
5) Do not introduce APIs that are not in the ranked candidates.
6) Do not execute anything or invent concrete parameter values.
7) When you include QoS-related values in your output, copy them from the
   corresponding service.qos object without inventing new fields or values.
   If service.qos is missing, represent QoS as null or an empty object instead.
8) Assign each path an overall path_score that reflects how well that path
   satisfies the user goal and subtasks (higher is better for this goal).

Return strict JSON in this format:

{
  "paths": [
    {
      "path_id": <integer starting from 1>,
      "path_score": <numeric overall score for this path>,
      "summary": "one or two sentences describing the main idea of this path",
      "steps": [
        {
          "step": <integer starting from 1 within this path>,
          "api_id": "<api_id taken from the ranked candidates>",
          "subtask_id": <integer matching an id from the subtasks, if applicable>,
          "action": "one sentence describing what this step does",
          "why": "one sentence explaining why this API was chosen for this step, "
                  "optionally referring to its score or QoS-related information",
          "score": <numeric score you associate with this step, usually copied from
                    the chosen API's score>,
          "qos": <null, or an object copied from the chosen service.qos if present>
        }
      ],
      "subtask_coverage": [
        {
          "subtask_id": <id from the subtasks>,
          "description": "text copied from the decomposed subtasks",
          "steps": [<list of step numbers in this path that address this subtask>],
          "coverage": "full" | "partial" | "missing"
        }
      ]
    }
  ],
  "selected_api_ids": [
    "<api_id that appears in at least one path>",
    ...
  ],
  "overall_rationale": "2 to 4 sentences explaining why these paths are reasonable
                        ways to satisfy the user goal, and how they differ (for
                        example, in coverage, simplicity, or QoS-related trade-offs)."
}

Requirements:
- Produce between 3 and 5 paths when enough suitable APIs are available; if
  there are fewer obvious options, it is acceptable to return fewer paths.
- In each path, steps must be numbered sequentially starting at 1.
- Each step must include api_id, action, why, score, and qos.
- When qos is not null, it must match the service.qos object for the chosen API
  as given in the input ranked candidates.
- path_score must be a numeric value where higher means better for this goal.
- selected_api_ids should list each api_id that appears in any path, without duplicates.
- subtask_coverage for each path should reflect how that path maps steps to
  subtasks. The same subtask may be covered differently in different paths.
- Do not introduce APIs that are not in the ranked candidates.
- Do not include anything outside the single JSON object.
