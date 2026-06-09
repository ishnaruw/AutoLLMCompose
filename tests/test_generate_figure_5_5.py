from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_figure_5_5.py"
SPEC = importlib.util.spec_from_file_location("generate_figure_5_5", SCRIPT_PATH)
assert SPEC is not None
figure_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = figure_module
SPEC.loader.exec_module(figure_module)


class GenerateFigure55Tests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_summary(self, path: Path) -> None:
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
                            "QoS_Adjusted_Composition_Score": "0.9",
                            "Planner_Output_File": f"{mode}/4_planner.json",
                        }
                    )

    def _planner_payload(self, api_ids: list[str]) -> dict:
        return {
            "selected_api_ids": api_ids,
            "execution_workflow": {
                "type": "sequential",
                "steps": [
                    {
                        "step": index,
                        "api_id": api_id,
                        "method": "GET",
                        "url": f"https://example.test/{api_id}",
                    }
                    for index, api_id in enumerate(api_ids, start=1)
                ],
            },
        }

    def test_loads_q02_mode_paths_from_official_summary_and_planners(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_path = tmp / "all_15_query_composition_results.csv"
            logs_root = tmp / "logs"
            self._write_summary(summary_path)
            for mode in figure_module.MODE_ORDER:
                self._write_json(
                    logs_root / "q02_run" / mode / "4_planner.json",
                    self._planner_payload([f"{mode}_api_1", f"{mode}_api_2", f"{mode}_api_3"]),
                )

            q02_rows, query_text = figure_module.load_official_q02_rows(summary_path)
            mode_paths = figure_module.load_mode_paths(q02_rows, logs_root, expected_steps=3)

            self.assertEqual(query_text, "Goal for q02")
            self.assertEqual([mode_path.mode for mode_path in mode_paths], list(figure_module.MODE_ORDER))
            self.assertEqual(len(mode_paths), 4)
            self.assertEqual([step.api_id for step in mode_paths[0].steps], ["qos_hybrid_api_1", "qos_hybrid_api_2", "qos_hybrid_api_3"])

    def test_rejects_planner_sequence_that_disagrees_with_selected_api_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            planner_path = tmp / "4_planner.json"
            payload = self._planner_payload(["api_a", "api_b", "api_c"])
            payload["execution_workflow"]["steps"][1]["api_id"] = "different_api"
            self._write_json(planner_path, payload)

            with self.assertRaisesRegex(ValueError, "selected_api_ids sequence does not match"):
                figure_module.extract_planner_steps("no_qos", planner_path, expected_steps=3)


if __name__ == "__main__":
    unittest.main()
