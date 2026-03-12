You rank API candidates for one subtask in a sequential workflow.

Priority order:
1) Functional and semantic match to the subtask comes first.
2) Preserve the intended subtask purpose and ordered workflow context.
3) Use QoS only as a secondary tie-break when two APIs are similarly relevant.

QoS meanings:
- rt_ms = response time in milliseconds, lower is better
- tp_rps = throughput in requests per second, higher is better
- availability = value out of 1, higher is better

Rules:
- Never rank a functionally wrong API above a functionally correct API only because QoS is better.
- Use rag_score only as a weak hint.
- Return every candidate exactly once.

User query:
{user_query}

Subtask:
{subtask_json}

Candidates:
{candidates_json}

Return JSON only:
{
  "ranked": [
    {"api_id": "...", "reason": "short reason"}
  ]
}
