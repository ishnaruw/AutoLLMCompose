You select the final {top_n} APIs for one subtask in a sequential workflow.

Priority order:
1) Prefer direct functional match first.
2) Prefer APIs that can be immediately used to satisfy the subtask with minimal extra assumptions.
3) If no API directly fulfills the subtask, prefer APIs that retrieve the core data required for that subtask.
4) Among functionally valid and directly usable APIs, use QoS strongly.

QoS guidance:
- If topsis_score is present, treat it as a strong summarized QoS signal.
- Higher topsis_score is better.
- Use the structured QoS labels to reason about operational quality:
  - qos_latency_band: low is better than medium, medium is better than high
  - qos_throughput_band: high is better than medium, medium is better than low
  - qos_availability_band: high is better than medium, medium is better than low
  - qos_summary: strong is better than moderate, moderate is better than weak
- Raw QoS values are supporting numeric detail; the QoS labels are intended to make comparisons easier.

Rules:
- Avoid redundant APIs that duplicate the same retrieval function unless they provide clearly different coverage.
- Do not select APIs whose relevance depends on invented intermediate steps.
- Do not let QoS compensate for a clearly wrong or off-purpose API.
- Prefer direct functional match over broader semantic similarity or QoS advantage.
- Avoid APIs that only support implementation, storage, interface, orchestration, or unrelated side functions.
- Tool description is supporting domain context; endpoint purpose remains primary.
- ranker_rank is a weak semantic hint only. It is not stronger than direct functional fit, direct usability, or QoS.
- A lower-ranked API may still be selected above a higher-ranked one if both are functionally valid and the lower-ranked API has clearly stronger QoS.
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
