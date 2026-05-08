from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("openpyxl") is not None, "openpyxl is required")
class ModeAnomalyReportTests(unittest.TestCase):
    def test_invalid_ranked_rows_are_reported_as_mode_anomalies(self) -> None:
        from src.eval.mode_anomaly_report import collect_ranking_anomaly_audit_for_run, build_mode_anomaly_rows

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "q02_20260507T164427"
            ranked_dir = run_dir / "qos_pure_llm"
            ranked_dir.mkdir(parents=True)
            (ranked_dir / "2_ranked_s2.json").write_text(
                json.dumps(
                    [
                        {
                            "api_id": "api_a",
                            "candidate_id": "C01",
                            "failure_flag": True,
                            "exclude_from_ranking_eval": True,
                            "failure_stage": "llm_ranking",
                            "failure_reason": "duplicate_candidate_ids_after_retries",
                            "ranking_anomaly": True,
                            "ranking_anomaly_reason": "duplicate_candidate_ids_after_retries",
                            "duplicate_candidate_ids": ["C01"],
                            "duplicate_api_ids": ["api_a"],
                            "expected_api_count": 2,
                            "actual_api_count": 1,
                            "returned_api_count": 2,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            audit = collect_ranking_anomaly_audit_for_run(run_dir, query_id="q02")
            rows = build_mode_anomaly_rows(None, None, audit)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Query_ID"], "q02")
        self.assertEqual(rows[0]["Subtask"], "2")
        self.assertEqual(rows[0]["Mode"], "qos_pure_llm")
        self.assertEqual(rows[0]["Invalid_Case"], 1)
        self.assertEqual(rows[0]["Failure_Reason"], "duplicate_candidate_ids_after_retries")
        self.assertEqual(rows[0]["Duplicate_API_IDs"], "api_a")


if __name__ == "__main__":
    unittest.main()
