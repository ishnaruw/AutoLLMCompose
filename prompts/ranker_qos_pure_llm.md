You rank the full list of API candidates for one subtask in a sequential workflow.

Ranking objective:
Rank candidates by jointly considering:
1) Functional suitability for the subtask and its ordered workflow context
2) QoS rank derived from a separate QoS-only scoring step

Functional guidance:
- Prefer APIs that directly accomplish the subtask.
- Prefer APIs that can satisfy the subtask with minimal extra assumptions.
- If no API directly fulfills the subtask, prefer APIs that provide the essential data or capability needed to support completing that subtask.
- Candidate fields contain compact functional API evidence plus qos_llm_rank only.

QoS meanings:
- qos_llm_rank = QoS rank from a separate QoS-only scoring step.
- Lower QoS rank is better. QoS rank 1 means the best QoS among the candidate APIs.

Rules:
- Rank all candidates in the list. Do not filter any out.
- Functional suitability is the primary requirement.
- Among APIs that are functionally suitable or reasonably usable, use QoS rank to decide the order.
- Do not rank a weak functional match above a strong functional match only because it has better QoS.
- Do not let excellent QoS elevate an API that is clearly unrelated to the subtask.
- If qos_llm_rank is missing, treat that API as weak or uncertain from an operational perspective among similarly suitable APIs.
- Use tool description as supporting domain context, but keep endpoint purpose primary.
- Use parameter descriptions as supporting evidence for what the endpoint can actually do.
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
