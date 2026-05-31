You rank the full list of API candidates for one subtask in a sequential workflow.

Ranking objective:
Rank candidates by considering:
1) Functional suitability for the subtask and its ordered workflow context
2) QoS evidence only after functional suitability is established

Functional guidance:
- Functional match is the first priority.
- Prefer APIs that directly accomplish the subtask.
- Prefer APIs that can satisfy the subtask with minimal extra assumptions.
- If no API directly fulfills the subtask, prefer APIs that provide the essential data or capability needed to support completing that subtask.
- Each candidate has candidate_id, the short ID used only for your output, and api_id, the real API identifier provided only as context.
- Candidate fields contain compact functional API evidence and QoS evidence when available.
- The compact ranker payload normally contains: candidate_id, api_id, name, category, tool_name, tool_description, description, method, parameters, and qos_llm_rank.
- Some payloads may also include title, summary, path, param_names, qos_llm_score, rt_s, tp_kbps, or availability. Use these fields only if they are present.
- Use endpoint-level fields as primary functional evidence. Use tool_name, tool_description, category, parameters, and param_names as supporting evidence.

QoS meanings:
- qos_llm_rank = QoS rank from a separate QoS-only scoring step.
- Lower QoS rank is better. QoS rank 1 means the best QoS among the candidate APIs.
- qos_llm_score, if present, is a QoS score where higher is better.
- rt_s, if present, is response time in seconds where lower is better.
- tp_kbps, if present, is throughput where higher is better.
- availability, if present, is availability where higher is better.

Functional tiers:
Internally place each candidate into one functional tier before using QoS:
- Tier 1: Direct functional match for the subtask.
- Tier 2: Plausible supporting match that provides essential data or capability for the subtask.
- Tier 3: Partial or weak match that may help but does not fully satisfy the subtask.
- Tier 4: Wrong-domain or unrelated API.

Use these tiers only for internal ranking judgment; do not output tier labels.
Then use qos_llm_rank, qos_llm_score, rt_s, tp_kbps, and availability only within the same or nearly equivalent functional tier. QoS rank is a major ordering signal only among APIs with comparable functional suitability. QoS should reorder candidates within the same functional tier, not across clearly different functional tiers.

Two-phase ranking process:
Step 1: Determine each API's functional tier for the subtask.
Step 2: Rank candidates by functional tier, preserving clear functional superiority.
Step 3: Within the same or nearly equivalent functional tier, use QoS evidence as a major ordering signal.
Step 4: Place wrong-domain or unrelated APIs below functionally relevant APIs regardless of QoS.

Wrong-domain guardrails:
- Do not rank a domain-specific API as a generic solution unless the endpoint clearly supports the requested generic action.
- Shipment notification APIs are not generic notification APIs unless they can send arbitrary user-defined alerts.
- OTP APIs are not generic alert or message APIs unless they support arbitrary message delivery.
- Cannabis-shop or niche venue APIs are not general restaurant or venue discovery APIs.
- Retirement calculators are not loan or amortization calculators unless they expose the required loan repayment functionality.
- Climate-change, news, or weather-report APIs are not current local weather APIs unless they provide the requested weather data.

Rules:
- Rank all candidates in the list. Do not filter any out.
- Functional suitability is a required gate.
- First identify APIs that can reasonably satisfy the subtask.
- A high-QoS API must never outrank a clearly better functional match.
- Missing required functionality, wrong domain, or inability to perform the subtask cannot be compensated by QoS.
- Among candidates with comparable functional suitability, QoS rank must materially affect ordering.
- Prefer APIs with better QoS rank only within the same or nearly equivalent functional tier.
- Do not let excellent QoS elevate an API that is clearly unrelated, wrong-domain, or unable to perform the subtask.
- If qos_llm_rank is missing, treat that API as weak or uncertain from an operational perspective among similarly suitable APIs.
- Use tool description as supporting domain context, but keep endpoint purpose primary.
- Use parameters or param_names, when present, as supporting evidence for what the endpoint can actually do.
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
  "ranked": [
    {"candidate_id": "C01"},
    {"candidate_id": "C02"}
  ]
}

{llm_output_contract}
