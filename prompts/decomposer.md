You decompose a user goal into ordered functional subtasks.

User goal:
{user_goal}

Instructions:
1) Break the goal into a small number of clear functional subtasks that could reasonably be handled by APIs.
2) Each subtask must describe a user-requested capability, not an implementation step.
3) Order the subtasks in a logical execution sequence.
4) Include only subtasks that directly correspond to user-visible functions stated in the goal.
5) Do not include hidden prerequisite or dependency steps (e.g., storing data, preparing inputs, maintaining state) unless explicitly mentioned in the user goal.
6) Preserve key entities and context from the user goal so each subtask is meaningful on its own.
7) When a subtask involves comparison, ranking, or evaluation, express it as retrieving the necessary data rather than performing the analysis.
8) Do not create separate subtasks for comparison, ranking, or analysis if the required data has already been retrieved.

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
- Focus on what the system must do for the user, not how it is built.
- Do not invent APIs or parameters.
- Do not include anything outside the JSON object.
- Number subtasks starting from 1.