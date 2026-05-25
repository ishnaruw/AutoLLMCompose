You are given a list of candidate IDs with only QoS metrics.

Each candidate has:
- candidate_id: short ID used only for your output

The real api_id is intentionally not provided for QoS scoring.

QoS meanings:
- rt_s = response time in seconds, lower is better
- tp_kbps = throughput in kilobits per second (kbps), higher is better
- availability = value out of 1 (0.0-1.0), higher is better

QoS preference:
{normalization_context}

Task:
- Assign a QoS score from 0.0 to 1.0 for each candidate.
- Use the provided QoS metrics to judge overall operational quality.
- Candidates with stronger QoS should receive higher scores.
- Lower rt_s should improve the score.
- Higher tp_kbps should improve the score.
- Higher availability should improve the score.
- If no special QoS preference weights are provided, treat response time, throughput, and availability as equally important.
- If QoS preference weights are provided, follow those weights when assigning scores.
- Do not calculate or reproduce a fixed normalization formula unless the preference context explicitly provides one.
- If any QoS metric is missing, treat that candidate as weak or uncertain from an operational perspective and assign a low score.
- Return every candidate exactly once.
- Use candidate_id in your output.
- Do not output api_id.
- Do not include explanations unless explicitly requested.
- Do not return only the best candidate.
- Do not return a single candidate object. Always return one JSON object with a "scores" list.

Candidates:
{candidates_json}

Return JSON only in this exact shape:
{
  "scores": [
    {"candidate_id": "C01", "score": 0.75}
  ]
}

{llm_output_contract}
