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

    def test_planner_prompt_uses_selection_order(self) -> None:
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
            ranked_top=[{"api_id": "api_a", "selection_order": 3, "selected_rank": 2, "mode_rank": 7}],
            subtasks=[{"id": 1, "description": "Call API"}],
            prompt_path=str(prompt_path),
        )

        self.assertIn('"selection_order": 3', prompts[0])
        self.assertNotIn('"rank"', prompts[0])
        self.assertNotIn('"selected_rank"', prompts[0])

    def test_planner_prompt_omits_rag_score(self) -> None:
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
            ranked_top=[{"api_id": "api_a", "selected_rank": 1, "rag_score": 0.99}],
            subtasks=[{"id": 1, "description": "Call API"}],
            prompt_path=str(prompt_path),
        )

        self.assertNotIn("rag_score", prompts[0])

    def test_planner_prompt_omits_noisy_candidate_fields(self) -> None:
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
            ranked_top=[
                {
                    "api_id": "api_a",
                    "score": 0.5,
                    "rank": 1,
                    "rag_score": 0.99,
                    "functional_match": 1,
                    "functional_reason": "matches",
                    "Comments": "comment",
                    "selection_order": 1,
                    "service": {"api_id": "api_a"},
                }
            ],
            subtasks=[{"id": 1, "description": "Call API"}],
            prompt_path=str(prompt_path),
        )

        for field in ['"score"', '"rank"', '"rag_score"', '"functional_match"', '"functional_reason"', '"Comments"']:
            self.assertNotIn(field, prompts[0])

    def test_no_qos_planner_prompt_strips_service_qos_fields(self) -> None:
        prompt_path = self._prompt_file()
        no_qos_path = prompt_path.with_name(f"{prompt_path.name}_planner_no_qos.md")
        no_qos_path.write_text(PROMPT_TEMPLATE, encoding="utf-8")
        prompts = []
        service = {
            "api_id": "api_a",
            "qos": {"rt_s": 1},
            "rt_s": 1,
            "tp_kbps": 2,
            "availability": 0.9,
            "qos_score": 0.8,
            "qos_rank": 1,
            "topsis_score": 0.7,
            "topsis_rank": 2,
            "qos_llm_score": 0.6,
            "qos_llm_rank": 3,
        }
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
            ranked_top=[{"api_id": "api_a", "selection_order": 1, "service": service}],
            subtasks=[{"id": 1, "description": "Call API"}],
            prompt_path=str(no_qos_path),
        )

        for field in ['"qos"', '"rt_s"', '"tp_kbps"', '"availability"', '"qos_score"', '"qos_rank"', '"topsis_score"', '"topsis_rank"', '"qos_llm_score"', '"qos_llm_rank"']:
            self.assertNotIn(field, prompts[0])
        self.assertIn("qos", service)

    def test_planner_prompt_compacts_service_metadata(self) -> None:
        prompt_path = self._prompt_file()
        prompts = []
        service = {
            "_file": "api_a.json",
            "_tool": "api_a_tool",
            "api_id": "api_a",
            "name": "Lookup",
            "category": "Weather",
            "description": "Endpoint description",
            "tool_description": "Tool description",
            "method": "GET",
            "url": "https://example.test/api",
            "endpoint_details": {
                "required_parameters": [{"name": "lat", "type": "NUMBER"}],
                "optional_parameters": [{"name": "units", "type": "STRING"}],
                "unused_detail": "drop me",
            },
            "toolbench_tool_name": "Repeated tool name",
            "toolbench_tool_description": "Repeated tool description",
            "toolbench_endpoint_description": "Repeated endpoint description",
            "toolbench_enrichment": {
                "endpoint_url": "https://example.test/from-enrichment",
                "endpoint_method": "POST",
                "tool_description": "Repeated enrichment description",
                "endpoint_description": "Repeated enrichment endpoint",
                "status": "matched",
                "toolbench_relative_path": "Weather/api_a.json",
            },
            "qos": {"availability": 0.99, "rt_s": 100, "tp_kbps": 10},
        }
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
                        "qos": {"availability": 0.99, "rt_s": 100, "tp_kbps": 10},
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
            ranked_top=[{"api_id": "api_a", "selection_order": 1, "service": service}],
            subtasks=[{"id": 1, "description": "Call API"}],
            prompt_path=str(prompt_path),
        )

        self.assertIn('"api_id": "api_a"', prompts[0])
        self.assertIn('"method": "GET"', prompts[0])
        self.assertIn('"url": "https://example.test/api"', prompts[0])
        self.assertIn('"required_parameters": [{"name": "lat", "type": "NUMBER"}]', prompts[0])
        self.assertIn('"optional_parameters": [{"name": "units", "type": "STRING"}]', prompts[0])
        self.assertIn('"qos": {"availability": 0.99, "rt_s": 100, "tp_kbps": 10}', prompts[0])

        for field in [
            '"_file"',
            '"_tool"',
            '"toolbench_tool_name"',
            '"toolbench_tool_description"',
            '"toolbench_endpoint_description"',
            '"toolbench_enrichment"',
            '"toolbench_relative_path"',
            '"unused_detail"',
        ]:
            self.assertNotIn(field, prompts[0])


if __name__ == "__main__":
    unittest.main()
