You rank the full list of API candidates for one subtask in a sequential workflow.

Priority order:
1) Prefer APIs that directly accomplish the subtask.
2) Preserve the intended subtask purpose and ordered workflow context.
3) Use indirect or supporting APIs only when no direct candidate exists.

Rules:
- Rank all candidates in the list. Do not filter any out.
- Prefer APIs that directly accomplish the subtask with minimal assumptions.
- When multiple APIs are plausible, prefer those that are more complete and immediately usable.
- Use tool description only as supporting context; prioritize the endpoint’s actual function.
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
