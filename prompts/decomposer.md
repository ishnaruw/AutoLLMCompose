You decompose a user goal into ordered subtasks.

User goal:
{user_goal}

Instructions:
1) Break the goal into 2 to 5 clear subtasks that could reasonably be handled by APIs.
2) Each subtask should be a meaningful unit of work (for example, a separate action or query).
3) Order the subtasks in a logical execution sequence.

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
- Do not invent APIs or parameters.
- Do not include anything outside the JSON object.
- Number subtasks starting from 1.