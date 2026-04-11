You rank API candidates for one subtask in a sequential workflow.

Priority order:
1) Prefer direct functional match to the subtask first.
2) Preserve the intended subtask purpose and ordered workflow context.
3) If no API directly fulfills the subtask, prefer APIs that retrieve the core data required for that subtask.

Rules:
- First remove clearly irrelevant, off-purpose, or functionally weak APIs.
- Prefer APIs that can immediately satisfy the subtask with minimal extra assumptions.
- Avoid redundant or side-function APIs when a direct API exists.
- Avoid APIs that only support implementation, storage, interface, orchestration, or unrelated side functions when a direct API exists.
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
