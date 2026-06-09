from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "consolidate_composition_results.py"
SPEC = importlib.util.spec_from_file_location("consolidate_composition_results", SCRIPT_PATH)
assert SPEC is not None
consolidate_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(consolidate_module)


class ConsolidateCompositionResultsTests(unittest.TestCase):
    def _write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_query_run(self, root: Path, query_id: str, rows) -> None:
        run_dir = root / f"{query_id}_20260531T000000"
        self._write_json(run_dir / "meta.json", {"query_id": query_id, "user_goal": f"Goal for {query_id}"})
        self._write_json(
            run_dir / "evaluation" / f"query_{query_id}_composition_qos_eval_rows.json",
            [{"Query_ID": query_id, **row} for row in rows],
        )

    def test_consolidates_rows_aggregates_scores_and_counts_tied_best_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "logs"
            output_dir = tmp / "summaries"
            self._write_query_run(
                input_dir,
                "q01",
                [
                    self._row("alpha", score=0.5, coverage=0.5),
                    self._row("beta", score=0.7, coverage=1.0),
                ],
            )
            self._write_query_run(
                input_dir,
                "q02",
                [
                    self._row("alpha", score=0.8, coverage=1.0),
                    self._row("beta", score=0.8, coverage=1.0),
                ],
            )

            result = consolidate_module.consolidate(input_dir, output_dir)

            report = result["report"]
            self.assertEqual(report["total_query_runs_found"], 2)
            self.assertEqual(report["total_metric_rows_written"], 4)
            self.assertEqual(report["query_ids_found"], ["q01", "q02"])
            self.assertEqual(report["modes_found"], ["alpha", "beta"])
            self.assertEqual(report["expected_modes_missing_by_query"], {})
            self.assertEqual(
                report["tied_best_queries"],
                [{"Query_ID": "q02", "Best_Modes": ["alpha", "beta"], "Best_Score": "0.8"}],
            )

            with (output_dir / "all_15_query_composition_results.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["Query_ID"], "q01")
            self.assertEqual(rows[0]["Query_Text"], "Goal for q01")
            self.assertEqual(rows[0]["Source_File"], "q01_20260531T000000/evaluation/query_q01_composition_qos_eval_rows.json")

            with (output_dir / "aggregate_mode_scores.csv").open(newline="", encoding="utf-8") as handle:
                aggregates = {row["Mode"]: row for row in csv.DictReader(handle)}
            self.assertAlmostEqual(float(aggregates["alpha"]["Average_QoS_Adjusted_Composition_Score"]), 0.65)
            self.assertAlmostEqual(float(aggregates["beta"]["Average_QoS_Adjusted_Composition_Score"]), 0.75)
            self.assertEqual(aggregates["alpha"]["Best_Query_Count"], "1")
            self.assertEqual(aggregates["beta"]["Best_Query_Count"], "2")

    def test_include_query_ids_filters_rows_aggregates_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "logs"
            output_dir = tmp / "summaries"
            for index, query_id in enumerate(("q01", "q02", "q03"), start=1):
                self._write_query_run(
                    input_dir,
                    query_id,
                    [
                        self._row("alpha", score=index / 10, coverage=1.0),
                        self._row("beta", score=(index + 1) / 10, coverage=1.0),
                    ],
                )

            result = consolidate_module.consolidate(input_dir, output_dir, include_query_ids=["q01", "q02"])

            report = result["report"]
            self.assertEqual(report["total_query_runs_available_in_input"], 3)
            self.assertEqual(report["total_query_runs_found"], 2)
            self.assertEqual(report["total_metric_rows_written"], 4)
            self.assertEqual(report["query_ids_found"], ["q01", "q02"])
            self.assertEqual(report["include_query_ids"], ["q01", "q02"])
            self.assertEqual(report["query_ids_excluded_by_filter"], ["q03"])
            self.assertIn("q01-q02 only", report["official_summary_note"])
            self.assertIn("q03 exist in the raw logs", report["official_summary_note"])

            with (output_dir / "all_15_query_composition_results.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["Query_ID"] for row in rows}, {"q01", "q02"})

            with (output_dir / "aggregate_mode_scores.csv").open(newline="", encoding="utf-8") as handle:
                aggregates = {row["Mode"]: row for row in csv.DictReader(handle)}
            self.assertEqual(aggregates["alpha"]["Query_Count"], "2")
            self.assertAlmostEqual(float(aggregates["alpha"]["Average_QoS_Adjusted_Composition_Score"]), 0.15)
            self.assertAlmostEqual(float(aggregates["beta"]["Average_QoS_Adjusted_Composition_Score"]), 0.25)

            readme = (output_dir / "README.md").read_text(encoding="utf-8")
            self.assertIn("--include-query-ids q01,q02", readme)
            self.assertIn("This official thesis summary includes q01-q02 only.", readme)

    def _row(self, mode: str, score: float, coverage: float) -> dict:
        return {
            "Mode": mode,
            "Composition_Validity": 1,
            "Invalid_Reason": "",
            "Planned_API_Count": 1,
            "Covered_Subtask_Count": int(coverage),
            "Total_Subtask_Count": 1,
            "Composition_Completeness": coverage,
            "Composition_Completeness_Gate": coverage,
            "Functional_Coverage": coverage,
            "Total_Response_Time_s": 1.0,
            "Bottleneck_Throughput_kbps": 10.0,
            "Average_Workflow_Availability": 0.99,
            "Normalized_Response_Time_Score": 1.0,
            "Normalized_Throughput_Score": 1.0,
            "Normalized_Availability_Score": 1.0,
            "Normalized_QoS_Score": 0.9,
            "QoS_Adjusted_Composition_Score": score,
        }


if __name__ == "__main__":
    unittest.main()
