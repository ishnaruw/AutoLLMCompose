You normalize decomposed subtasks so they are suitable for API-based service retrieval.

User goal:
{user_goal}

Original subtasks:
{original_subtasks_json}

Your job:
Refine the subtasks so that each one is a clear API-callable, user-facing function.

Instructions:
1) Keep only subtasks that directly support explicit user-visible functions in the goal.
2) Remove any subtask that exists only as a hidden prerequisite, dependency, preparation step, storage step, infrastructure step, interface step, or support step unless it is explicitly required by the goal.
3) If a subtask is phrased as comparison, ranking, analysis, evaluation, recommendation logic, or other internal processing, rewrite it as retrieving or performing the direct user-facing function instead.
4) The final subtasks must be directly traceable to functions stated in the user goal.
5) Do not add new functionality that is not present in the user goal.
6) You may rewrite, remove, or merge subtasks, but do not expand the workflow unnecessarily.
7) Keep the output concise, API-oriented, and aligned with the goal.

Return strict JSON in this format:

{
  "subtasks": [
    {
      "id": 1,
      "description": "short phrase describing this subtask"
    }
  ]
}

Rules:
- Focus on the final user-visible functions, not intermediate support steps.
- Do not invent APIs or parameters.
- Do not include anything outside the JSON object.
- Keep the subtasks in a logical order.
- Number subtasks starting from 1.