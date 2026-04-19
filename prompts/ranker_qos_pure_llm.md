You rank the full list of API candidates for one subtask in a sequential workflow.

Ranking objective:
Rank candidates by jointly considering:
1) Functional suitability for the subtask and its ordered workflow context
2) QoS guidance derived from a separate QoS-only scoring step
3) QoS quality across response time, throughput, and availability

Functional guidance:
- Prefer APIs that directly accomplish the subtask.
- Prefer APIs that can satisfy the subtask with minimal extra assumptions.
- If no API directly fulfills the subtask, prefer APIs that provide the essential data or capability needed to support completing that subtask.

QoS meanings:
- rt_ms = response time in milliseconds, lower is better
- tp_rps = throughput in requests per second, higher is better
- availability = value out of 1, higher is better
- qos_llm_rank = QoS rank from a separate QoS-only LLM scoring step, where 1 is best
- qos_llm_score = QoS score from a separate QoS-only LLM scoring step, where higher is better
- If no special weighting is provided, treat the three QoS metrics equally.

Rules:
- Rank all candidates in the list. Do not filter any out.
- Prioritize QoS over functionality: APIs with better qos_llm_rank (lower number) should be ranked higher, even if functionally less direct.
- Among APIs with similar QoS ranks, prefer better functional suitability.
- Do not rank a functionally relevant API with a much worse qos_llm_rank above another functionally relevant API with a much better qos_llm_rank.
- APIs with stronger overall QoS should rank higher than other similarly relevant or reasonably usable alternatives.
- Do not let excellent QoS elevate an API that is clearly unrelated to the subtask.
- If QoS values are missing, treat that API as weak or uncertain from an operational perspective.
- Use tool description as supporting domain context, but keep endpoint purpose primary.
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
