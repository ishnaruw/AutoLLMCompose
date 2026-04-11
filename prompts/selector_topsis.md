You select the final {top_n} APIs for one subtask in a sequential workflow.

Priority order:
1) Prefer direct functional match first.
2) Prefer APIs that can be immediately used to satisfy the subtask with minimal extra assumptions.
3) If no API directly fulfills the subtask, prefer APIs that retrieve the core data required for that subtask.
4) Among functionally valid and similarly usable APIs, use topsis_score as a strong QoS guidance signal. Higher topsis_score is better.

QoS meanings:
- rt_ms = response time in milliseconds, lower is better
- tp_rps = throughput in requests per second, higher is better
- availability = value out of 1, higher is better
- topsis_score = combined QoS score; higher is better

Rules:
- Avoid redundant APIs that duplicate the same retrieval function unless they provide clearly different coverage.
- Do not select APIs whose relevance depends on invented intermediate steps.
- Do not let topsis_score compensate for a clearly wrong or off-purpose API.
- Prefer direct functional match over broader semantic similarity or QoS advantage.
- Avoid APIs that only support implementation, storage, interface, orchestration, or unrelated side functions.
- Tool description is supporting domain context; endpoint purpose remains primary.
- ranker_rank is a strong hint, but you may rescue an unranked API if it is more directly usable for the subtask.
- Return exactly {top_n} items when possible and do not repeat any api_id.

User query:
{user_query}

Subtask:
{subtask_json}

Candidates:
{candidates_json}

Return JSON only:
{
  "selected": [
    {"api_id": "...", "reason": "short reason"}
  ]
}
