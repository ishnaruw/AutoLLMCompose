from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.composition_qos_eval import evaluate_composition_qos


class CompositionQosEvalTests(unittest.TestCase):
    def _write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_product_availability_and_invalid_score_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            query_dir = Path(tmp)
            eval_dir = query_dir / "evaluation"
            self._write_json(
                query_dir / "0_decomposer.json",
                [
                    {"id": 1, "description": "First API call"},
                    {"id": 2, "description": "Second API call"},
                ],
            )
            selected_rows = {
                "api_a": {"api_id": "api_a", "service": {"qos": {"rt_ms": 10, "tp_rps": 50, "availability": 0.9}}},
                "api_b": {"api_id": "api_b", "service": {"qos": {"rt_ms": 20, "tp_rps": 40, "availability": 0.8}}},
            }
            self._write_json(query_dir / "no_qos" / "3_selected_s1.json", [selected_rows["api_a"]])
            self._write_json(query_dir / "no_qos" / "3_selected_s2.json", [selected_rows["api_b"]])
            self._write_json(
                query_dir / "no_qos" / "4_planner.json",
                {
                    "primary_plan": {
                        "plan_id": 1,
                        "summary": "Valid plan",
                        "steps": [
                            {
                                "step": 1,
                                "api_id": "api_a",
                                "subtask_id": 1,
                                "action": "Call A",
                                "input_from_previous_step": None,
                                "output_to_next_step": "a result",
                                "why": "Matches",
                                "qos": None,
                            },
                            {
                                "step": 2,
                                "api_id": "api_b",
                                "subtask_id": 2,
                                "action": "Call B",
                                "input_from_previous_step": "a result",
                                "output_to_next_step": "b result",
                                "why": "Matches",
                                "qos": None,
                            },
                        ],
                        "subtask_coverage": [],
                    },
                    "selected_api_ids": ["api_a", "api_b"],
                    "overall_rationale": "Valid",
                },
            )
            self._write_json(
                eval_dir / "query_qx_candidate_api_rankings_rows.json",
                [
                    {
                        "Mode": "no_qos",
                        "Sub Task": "1",
                        "Selected_API": "api_a",
                        "Functional Match (0/1)": 1,
                        "QoS_RT": 10,
                        "QoS_TP": 50,
                        "QoS Availability": 0.9,
                    },
                    {
                        "Mode": "no_qos",
                        "Sub Task": "2",
                        "Selected_API": "api_b",
                        "Functional Match (0/1)": 1,
                        "QoS_RT": 20,
                        "QoS_TP": 40,
                        "QoS Availability": 0.8,
                    },
                ],
            )

            result = evaluate_composition_qos(query_dir=query_dir, query_id="qx", output_dir=eval_dir)
            rows_by_mode = {row["Mode"]: row for row in result["rows"]}

            self.assertEqual(rows_by_mode["no_qos"]["Composition_Validity"], 1)
            self.assertEqual(rows_by_mode["no_qos"]["Workflow_Availability"], 0.72)
            self.assertEqual(rows_by_mode["qos_pure_llm"]["Composition_Validity"], 0)
            self.assertEqual(rows_by_mode["qos_pure_llm"]["QoS_Adjusted_Composition_Score"], 0.0)


if __name__ == "__main__":
    unittest.main()
