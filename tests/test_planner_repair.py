from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agents.planner import planner_call


PROMPT_TEMPLATE = """Goal: {user_goal}
Subtasks: {subtasks_json}
Candidates: {ranked_compact}
"""


def _execution_workflow(api_id: str = "api_a") -> dict:
    return {
        "type": "sequential",
        "steps": [
            {
                "step": 1,
                "api_id": api_id,
                "subtask_id": 1,
                "method": "GET",
                "url": "https://example.test/api",
                "required_parameters": [{"name": "q", "source": "user_goal"}],
                "optional_parameters": [],
                "depends_on": [],
                "input_mapping": "Use the user goal as q.",
                "output_mapping": "Return the API result.",
                "expected_output": "API result",
            }
        ],
    }


class PlannerRepairTests(unittest.TestCase):
    def _prompt_file(self) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        with tmp:
            tmp.write(PROMPT_TEMPLATE)
        return Path(tmp.name)

    def test_repairs_top_level_plan_and_stringifies_connector_fields(self) -> None:
        prompt_path = self._prompt_file()
        response = {
            "plan_id": 1,
            "summary": "Use API",
            "steps": [
                {
                    "step": 1,
                    "api_id": "api_a",
                    "subtask_id": 1,
                    "action": "Call API",
                    "input_from_previous_step": {"source": "user"},
                    "output_to_next_step": ["api_a result"],
                    "why": "Matches the subtask",
                    "qos": None,
                }
            ],
            "subtask_coverage": [],
            "execution_workflow": _execution_workflow(),
            "selected_api_ids": ["api_a"],
            "overall_rationale": "Best fit",
        }

        output = planner_call(
            llm_call=lambda _prompt: json.dumps(response),
            user_goal="test",
            ranked_top=[{"api_id": "api_a"}],
            subtasks=[{"id": 1, "description": "Call API"}],
            prompt_path=str(prompt_path),
        )

        step = output["primary_plan"]["steps"][0]
        self.assertEqual(output["primary_plan"]["plan_id"], 1)
        self.assertEqual(step["input_from_previous_step"], '{"source": "user"}')
        self.assertEqual(step["output_to_next_step"], '["api_a result"]')

    def test_retries_once_for_unrepaired_schema_failure(self) -> None:
        prompt_path = self._prompt_file()
        prompts = []
        responses = [
            {
                "primary_plan": {
                    "plan_id": 1,
                    "summary": "Bad API id",
                    "steps": [
                        {
                            "step": 1,
                            "api_id": None,
                            "subtask_id": 1,
                            "action": "Internal work",
                            "input_from_previous_step": None,
                            "output_to_next_step": "result",
                            "why": "No API needed",
                            "qos": None,
                        }
                    ],
                    "subtask_coverage": [],
                },
                "execution_workflow": _execution_workflow(api_id="api_a"),
                "selected_api_ids": [],
                "overall_rationale": "Bad shape",
            },
            {
                "primary_plan": {
                    "plan_id": 1,
                    "summary": "Fixed API id",
                    "steps": [
                        {
                            "step": 1,
                            "api_id": "api_a",
                            "subtask_id": 1,
                            "action": "Use the closest provided API",
                            "input_from_previous_step": None,
                            "output_to_next_step": "result",
                            "why": "Uses a provided API",
                            "qos": None,
                        }
                    ],
                    "subtask_coverage": [],
                },
                "execution_workflow": _execution_workflow(api_id="api_a"),
                "selected_api_ids": ["api_a"],
                "overall_rationale": "Fixed",
            },
        ]

        def fake_llm(prompt: str) -> str:
            prompts.append(prompt)
            return json.dumps(responses.pop(0))

        output = planner_call(
            llm_call=fake_llm,
            user_goal="test",
            ranked_top=[{"api_id": "api_a"}],
            subtasks=[{"id": 1, "description": "Call API"}],
            prompt_path=str(prompt_path),
        )

        self.assertEqual(len(prompts), 2)
        self.assertIn("failed validation", prompts[1])
        self.assertEqual(output["primary_plan"]["steps"][0]["api_id"], "api_a")

    def test_planner_prompt_uses_selected_rank_when_rank_is_missing(self) -> None:
        prompt_path = self._prompt_file()
        prompts = []
        response = {
            "primary_plan": {
                "plan_id": 1,
                "summary": "Use API",
                "steps": [
                    {
                        "step": 1,
                        "api_id": "api_a",
                        "subtask_id": 1,
                        "action": "Call API",
                        "input_from_previous_step": None,
                        "output_to_next_step": "result",
                        "why": "Matches",
                        "qos": None,
                    }
                ],
                "subtask_coverage": [],
            },
            "execution_workflow": _execution_workflow(),
            "selected_api_ids": ["api_a"],
            "overall_rationale": "Best fit",
        }

        planner_call(
            llm_call=lambda prompt: prompts.append(prompt) or json.dumps(response),
            user_goal="test",
            ranked_top=[{"api_id": "api_a", "selected_rank": 2, "mode_rank": 7}],
            subtasks=[{"id": 1, "description": "Call API"}],
            prompt_path=str(prompt_path),
        )

        self.assertIn('"rank": 2', prompts[0])
        self.assertNotIn('"rank": null', prompts[0])


if __name__ == "__main__":
    unittest.main()
