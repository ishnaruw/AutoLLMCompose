You rank the full list of API candidates for one subtask in a sequential workflow.

Priority order:
1) Prefer direct functional match to the subtask.
2) Preserve the intended subtask purpose and ordered workflow context.
3) Use indirect or supporting APIs only when no direct candidate exists.

Each candidate has:
- candidate_id: short ID used only for your output
- api_id: real API identifier, provided only as context

Candidate fields contain compact functional API evidence only:
candidate_id, api_id, name, category, tool_name, tool_description, description, method, parameters.

Rules:
- Rank all candidates in the list. Do not filter any out.
- Prefer APIs that directly accomplish the subtask with minimal assumptions.
- When multiple APIs are plausible, prefer those that are more complete and immediately usable.
- Use tool description only as supporting context; prioritize the endpoint's actual function.
- Use parameter descriptions as supporting evidence for what the endpoint can actually do.
- Return every candidate_id exactly once.
- Use candidate_id in your output.
- Do not output api_id.

User query:
{user_query}

Subtask:
{subtask_json}

Candidates:
{candidates_json}

Return JSON only:
{
  "ranked_apis": [
    {"candidate_id": "C01", "rank": 1},
    {"candidate_id": "C02", "rank": 2}
  ]
}
