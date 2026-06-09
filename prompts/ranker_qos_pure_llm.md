You rank the full list of API candidates for one subtask in a sequential workflow.

Rank candidates by considering:
1) Functional suitability to the subtask.
2) QoS ranking evidence only after functional suitability.

Functional guidance:
- Functional match is the first priority.
- QoS is used only after functional suitability is established.
- Prefer APIs that directly accomplish the subtask.
- Prefer APIs that can satisfy the subtask with minimal extra assumptions.
- If no API directly fulfills the subtask, prefer APIs that provide the essential data or capability needed to support completing that subtask.
- Each candidate has candidate_id, the short ID used only for your output, and api_id, the real API identifier provided only as context.
- Candidate fields contain compact functional API evidence, functional refinement evidence when available, and QoS evidence when available.
- Candidate fields are: candidate_id, api_id, name, category, tool_name, tool_description, description, method, parameters, functional_match_label, functional_match_reason, qos_llm_rank, qos_llm_score, rt_s, tp_kbps, availability.
- Use name, description, method, and parameters as primary evidence for what the endpoint can actually do. Use tool_name, tool_description, and category as supporting domain context.
- If functional_match_label is present, label 1 means the API was independently judged functionally suitable for the subtask; label 0 means it was independently judged unsuitable.
- Treat functional_match_label = 1 as the functional gate before QoS. Do not rank a functional_match_label = 0 candidate above any functional_match_label = 1 candidate.
- Use functional_match_reason as supporting evidence for why the functional label was assigned.

QoS meanings:
- qos_llm_rank = QoS rank from a separate QoS-only scoring step.
- Lower qos_llm_rank is better.
- qos_llm_score, if present, is higher-is-better.
- rt_s is response time; lower is better.
- tp_kbps is throughput; higher is better.
- availability is higher-is-better.

Functional tiers:
Internally place each candidate into one functional tier before using QoS:
- Tier 1: Direct functional match. The API directly performs the requested subtask action and matches the expected domain.
- Tier 2: Plausible supporting match. The API does not directly perform the full action, but can reasonably support the subtask with minimal assumptions.
- Tier 3: Partial or weak match. The API is only loosely related, missing key action or domain requirements, or requires strong assumptions.
- Tier 4: Wrong-domain or unrelated API. The API belongs to a different domain, performs a different action, or cannot satisfy the subtask.

Use these tiers only for internal ranking judgment; do not output tier labels.
Rank Tier 1 before Tier 2, Tier 2 before Tier 3, and Tier 3 before Tier 4.
Use qos_llm_rank, qos_llm_score, rt_s, tp_kbps, and availability only within the same or nearly equivalent functional tier.
QoS rank and QoS metrics are major ordering signals only among APIs with comparable functional suitability.
QoS should reorder candidates within the same functional tier, not across clearly different functional tiers.
When functional suitability differs clearly, prefer the stronger functional match even if its QoS is weaker.
Do not promote Tier 3 or Tier 4 APIs above Tier 1 APIs because of QoS.

Two-phase ranking process:
Step 1: Determine each API's functional tier for the subtask.
Step 2: Rank stronger functional tiers ahead of weaker tiers.
Step 3: Within the same or nearly equivalent functional tier, use QoS evidence as a strong tie-breaker.
Step 4: Place wrong-domain or unrelated APIs below functionally relevant APIs regardless of QoS.

Wrong-domain guardrails:
- Domain-specific APIs should not be ranked as generic solutions unless they clearly support the requested generic action.
- Statistics, metadata, dashboard, or reporting endpoints are not scan/check/validation endpoints unless they perform the requested scan/check/validation.
- Shipment notification APIs are not generic notification APIs unless they support arbitrary user-defined alerts or messages.
- OTP APIs are not generic alert or message APIs unless they support arbitrary message delivery.
- Cannabis-shop, vegetarian-only, or niche venue APIs are not general restaurant/venue discovery APIs unless the user requested that niche.
- Retirement calculators are not loan or amortization calculators unless they expose the required loan repayment functionality.
- Climate-change, news, or weather-report APIs are not current local weather APIs unless they provide requested current forecast/weather data.
- Crypto-only APIs are not stock-market APIs unless they explicitly support stocks.
- Generic financial data APIs should not outrank domain-specific stock price APIs unless they clearly support the requested stock price fields.
- Coupon listing APIs should not outrank coupon-link APIs when the subtask specifically asks for active coupon links.

Required-action guardrail:
- The API must satisfy the main action verb in the subtask before QoS is considered.
- Main action verbs include scan, validate, check, send, fetch, retrieve, search, book, predict, classify, and return links.
- If the endpoint only returns stats, metadata, lists, or generic content, do not treat it as satisfying a stronger action such as scan, validate, predict, or retrieve links.
- Judge functionality at the endpoint level, not only from the parent API/tool name. The endpoint-level name, title, summary, path, method, and param_names must show that the endpoint performs the requested action.
- Do not infer endpoint capability from the parent API/tool name alone.
- A list, catalog, metadata, stats, supported-items, or reporting endpoint does not satisfy stronger actions such as prediction, classification, analysis, scan, validation, booking, alerting, or message delivery.
- This remains true even if the parent API name contains words like prediction, AI, security, notification, weather, finance, or recommendation.
- If the subtask asks to predict or classify stock trends, an endpoint that only lists cryptocurrencies, supported assets, symbols, metadata, or catalog items is not a valid prediction/classification endpoint.
- Required action verbs must be satisfied before QoS is considered.

Rules:
- Rank all candidates in the list. Do not filter any out.
- Functional suitability is a required gate.
- First identify APIs that can reasonably satisfy the subtask.
- If functional_match_label is present and any candidate has label 1, the top-ranked candidate must have label 1.
- Among label 1 candidates, use endpoint-level functional evidence first and QoS rank/metrics as strong ordering signals.
- A high-QoS API must never outrank a clearly better functional match.
- Missing required functionality, wrong domain, wrong endpoint action, or inability to perform the subtask cannot be compensated by QoS.
- Among candidates with comparable functional suitability, QoS rank and QoS metrics must materially affect ordering.
- Prefer APIs with better QoS rank only within the same or nearly equivalent functional tier.
- Do not let excellent QoS elevate an API that is clearly unrelated, wrong-domain, or unable to perform the subtask.
- If qos_llm_rank is missing, treat that API as weak or uncertain from an operational perspective among similarly suitable APIs.
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
