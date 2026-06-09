from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_figure_5_6.py"
SPEC = importlib.util.spec_from_file_location("generate_figure_5_6", SCRIPT_PATH)
assert SPEC is not None
figure_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = figure_module
SPEC.loader.exec_module(figure_module)


class GenerateFigure56Tests(unittest.TestCase):
    def _write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_summary(self, path: Path, *, q02_scores: dict[str, float] | None = None) -> None:
        q02_scores = q02_scores or {
            "qos_hybrid": 0.91,
            "qos_pure_llm": 0.88,
            "no_qos": 0.84,
            "qos_topsis": 0.40,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "Query_ID",
            "Query_Text",
            "Run_Folder",
            "Mode",
            "QoS_Adjusted_Composition_Score",
            "Planner_Output_File",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for query_id in figure_module.QUERY_ORDER:
                for mode in figure_module.MODE_ORDER:
                    writer.writerow(
                        {
                            "Query_ID": query_id,
                            "Query_Text": f"Goal for {query_id}",
                            "Run_Folder": f"{query_id}_run",
                            "Mode": mode,
                            "QoS_Adjusted_Composition_Score": q02_scores[mode] if query_id == "q02" else "0.5",
                            "Planner_Output_File": f"{mode}/4_planner.json",
                        }
                    )

    def _write_q02_sources(self, logs_root: Path, mode: str = "qos_hybrid") -> None:
        run_dir = logs_root / "q02_run"
        self._write_json(
            run_dir / "0_decomposer.json",
            [
                {"id": 1, "description": "Find nearby restaurants"},
                {"id": 2, "description": "Fetch restaurant details"},
                {"id": 3, "description": "Email recommendations"},
            ],
        )
        self._write_json(
            run_dir / mode / "4_planner.json",
            {
                "selected_api_ids": ["api_search", "api_details", "api_email"],
                "execution_workflow": {
                    "type": "sequential",
                    "steps": [
                        {
                            "step": 1,
                            "subtask_id": "1",
                            "api_id": "api_search",
                            "method": "GET",
                            "url": "https://example.test/search",
                            "input_mapping": "none",
                            "output_mapping": "restaurants",
                            "expected_output": "Restaurant list",
                        },
                        {
                            "step": 2,
                            "subtask_id": "2",
                            "api_id": "api_details",
                            "method": "GET",
                            "url": "https://example.test/details",
                            "input_mapping": "output_of_step_1",
                            "output_mapping": "details",
                            "expected_output": "Restaurant details",
                        },
                        {
                            "step": 3,
                            "subtask_id": "3",
                            "api_id": "api_email",
                            "method": "POST",
                            "url": "https://example.test/email",
                            "input_mapping": "output_of_step_2",
                            "output_mapping": "email_status",
                            "expected_output": "Email status",
                        },
                    ],
                },
            },
        )

    def test_loads_best_q02_mode_flow_from_official_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_path = tmp / "all_15_query_composition_results.csv"
            logs_root = tmp / "logs"
            self._write_summary(summary_path)
            self._write_q02_sources(logs_root)

            flow = figure_module.load_planned_flow(summary_path, logs_root, requested_mode=None, expected_steps=3)

            self.assertEqual(flow.mode, "qos_hybrid")
            self.assertAlmostEqual(flow.score, 0.91)
            self.assertEqual([subtask.subtask_id for subtask in flow.subtasks], ["1", "2", "3"])
            self.assertEqual([step.api_id for step in flow.steps], ["api_search", "api_details", "api_email"])

    def test_requires_explicit_mode_when_q02_best_score_is_tied(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_path = tmp / "all_15_query_composition_results.csv"
            self._write_summary(
                summary_path,
                q02_scores={
                    "qos_hybrid": 0.91,
                    "qos_pure_llm": 0.91,
                    "no_qos": 0.84,
                    "qos_topsis": 0.40,
                },
            )
            q02_rows, _query_text = figure_module.load_official_q02_rows(summary_path)

            with self.assertRaisesRegex(ValueError, "tied best modes"):
                figure_module.choose_mode_row(q02_rows, requested_mode=None)

    def test_rejects_planner_steps_that_do_not_match_decomposer_subtasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            planner_path = tmp / "4_planner.json"
            self._write_json(
                planner_path,
                {
                    "selected_api_ids": ["api_search", "api_details", "api_email"],
                    "execution_workflow": {
                        "steps": [
                            {
                                "step": 1,
                                "subtask_id": "1",
                                "api_id": "api_search",
                                "method": "GET",
                                "url": "https://example.test/search",
                                "input_mapping": "none",
                                "output_mapping": "restaurants",
                                "expected_output": "Restaurant list",
                            },
                            {
                                "step": 2,
                                "subtask_id": "99",
                                "api_id": "api_details",
                                "method": "GET",
                                "url": "https://example.test/details",
                                "input_mapping": "output_of_step_1",
                                "output_mapping": "details",
                                "expected_output": "Restaurant details",
                            },
                            {
                                "step": 3,
                                "subtask_id": "3",
                                "api_id": "api_email",
                                "method": "POST",
                                "url": "https://example.test/email",
                                "input_mapping": "output_of_step_2",
                                "output_mapping": "email_status",
                                "expected_output": "Email status",
                            },
                        ]
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "workflow subtask IDs"):
                figure_module.parse_planner_steps(
                    planner_path,
                    "qos_hybrid",
                    subtask_ids=["1", "2", "3"],
                    expected_steps=3,
                )


if __name__ == "__main__":
    unittest.main()
