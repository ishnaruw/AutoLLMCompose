You are given a list of candidate IDs with only QoS metrics.

Each candidate has:
- candidate_id: short ID used only for your output

The real api_id is intentionally not provided for QoS scoring.

QoS meanings:
- rt_ms = response time in milliseconds, lower is better (cost criterion)
- tp_rps = throughput in requests per second, higher is better (benefit criterion)
- availability = value out of 1 (0.0-1.0), higher is better (benefit criterion)

Task:
- Normalize each metric to a 0.0-1.0 scale:
  * rt_ms: lower is better. Calculate: 1.0 - (api_rt / max_rt_in_list). Clamp to [0, 1].
  * tp_rps: higher is better. Calculate: api_tp / max_tp_in_list. Clamp to [0, 1].
  * availability: already in [0, 1] range.
- Compute composite QoS score by averaging the three normalized metrics: (norm_rt + norm_tp + norm_availability) / 3.0
- This produces a score from 0.0 (worst overall QoS) to 1.0 (best overall QoS).
- Round scores to 4 decimal places.
- If any metric is missing, treat that candidate as having very poor QoS (score 0.0).
- Do not assign the same score to different candidates.
- Return each candidate_id exactly once.
- Use candidate_id in your output.
- Do not output api_id.

Candidates:
{candidates_json}

Return JSON only:
{
  "qos_scored": [
    {"candidate_id": "C01", "qos_score": 0.75},
    {"candidate_id": "C02", "qos_score": 0.50}
  ]
}
