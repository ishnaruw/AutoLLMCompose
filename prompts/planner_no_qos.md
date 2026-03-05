You create multiple alternative orchestration plans using only the provided candidate APIs.

You are given:
- The overall user goal (original query).
- A list of decomposed subtasks (JSON array).
- Candidate APIs (JSON array). Each item has:
  - api_id
  - subtask_id: which subtask this API is intended for
  - rank: the within-subtask rank (1 is best)
  - score: a derived score (higher is better)
  - rag_score: semantic similarity score from retrieval (for context only)
  - service: the full catalog entry for this API, which may include fields such as
             description, category, endpoints, or other metadata.

User goal:
{user_goal}

Decomposed subtasks (JSON array):
{subtasks_json}

Candidate APIs (JSON array):
{ranked_compact}

Instructions:
1) Use the subtasks to design a valid multi-step workflow.
2) For each subtask, prefer higher-ranked candidates (rank 1 first), but only if they actually fit the needed function.
3) Use only APIs from the candidate list. Never invent APIs.
4) Return EXACTLY 3 alternative paths. Use path_id = 1, 2, 3.
   - Path 1 should be the best overall plan.
   - Paths 2 and 3 should be valid alternatives (fallbacks), for example using different APIs where possible.
5) Return strict JSON as specified below. Output JSON only.

Return strict JSON in this format:

{
  "paths": [
    {
      "path_id": 1,
      "path_score": 0.0,
      "summary": "short summary of this plan",
      "steps": [
        {
          "step": 1,
          "api_id": "string",
          "subtask_id": 1,
          "action": "what you would do with this API",
          "why": "brief reasoning"
        }
      ],
      "subtask_coverage": [
        {
          "subtask_id": 1,
          "description": "string",
          "steps": [1],
          "coverage": "full"
        }
      ]
    }
  ],
  "selected_api_ids": ["..."],
  "overall_rationale": "brief overall reasoning"
}

Requirements:
- Return exactly 3 items in paths, with path_id = 1, 2, 3.
- Ensure each step's api_id exists in the candidates.
- subtask_id should refer to an id from the subtasks list.
- Keep responses concise and valid JSON only.
