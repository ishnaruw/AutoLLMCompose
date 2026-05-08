You rank the full list of API candidates for one subtask in a sequential workflow.

Ranking objective:
Rank candidates by jointly considering:
1) Functional suitability for the subtask and its ordered workflow context
2) QoS rank derived from a separate QoS-only scoring step

Functional guidance:
- Prefer APIs that directly accomplish the subtask.
- Prefer APIs that can satisfy the subtask with minimal extra assumptions.
- If no API directly fulfills the subtask, prefer APIs that provide the essential data or capability needed to support completing that subtask.
- Each candidate has candidate_id, the short ID used only for your output, and api_id, the real API identifier provided only as context.
- Candidate fields contain compact functional API evidence plus qos_llm_rank only.
- Candidate fields are: candidate_id, api_id, name, category, tool_name, tool_description, description, method, parameters, qos_llm_rank.

QoS meanings:
- qos_llm_rank = QoS rank from a separate QoS-only scoring step.
- Lower QoS rank is better. QoS rank 1 means the best QoS among the candidate APIs.

Two-phase ranking process:
Step 1: Determine whether each API is functionally plausible for the subtask.
Step 2: Rank functionally plausible APIs using QoS rank as a major ordering factor.
Step 3: Place clearly unrelated APIs below functionally plausible APIs.

Rules:
- Rank all candidates in the list. Do not filter any out.
- Functional suitability is a required gate.
- First identify APIs that can reasonably satisfy the subtask.
- Among functionally plausible APIs, QoS rank must materially affect ordering.
- Do not treat QoS rank as a minor tie-breaker.
- Prefer APIs with better QoS rank unless there is a clear functional reason not to.
- Do not let excellent QoS elevate an API that is clearly unrelated to the subtask.
- If qos_llm_rank is missing, treat that API as weak or uncertain from an operational perspective among similarly suitable APIs.
- Use tool description as supporting domain context, but keep endpoint purpose primary.
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
    {
      "candidate_id": "C01",
      "rank": 1
    },
    {
      "candidate_id": "C02",
      "rank": 2
    }
  ]
}
