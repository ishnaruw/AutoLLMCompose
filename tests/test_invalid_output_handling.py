from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.agents.qos_scorer_llm import InvalidQosScoringOutput, score_qos_llm
from src.agents.ranker import InvalidRankingOutput, rank_subtask
from src.core.run_logging import clear_run_log, configure_run_log, log_invalid_case_event


class InvalidOutputHandlingTests(unittest.TestCase):
    def test_invalid_cases_log_is_separate_from_errors_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_log = Path(tmpdir) / "run.log"
            configure_run_log(run_log)
            try:
                log_invalid_case_event(
                    {
                        "query_id": "q01",
                        "subtask_id": "1",
                        "mode": "no_qos",
                        "failure_stage": "llm_ranking",
                        "failure_reason": "duplicate_ranked_apis_after_retries",
                    }
                )
            finally:
                clear_run_log()

            invalid_log = Path(tmpdir) / "invalid_cases.log"
            self.assertTrue(invalid_log.exists())
            self.assertFalse((Path(tmpdir) / "errors.log").exists())
            event = json.loads(invalid_log.read_text(encoding="utf-8").strip())
            self.assertEqual(event["level"], "warning")
            self.assertEqual(event["event_type"], "invalid_evaluation_case")
            self.assertTrue(event["exclude_from_ranking_eval"])

    def test_duplicate_ranking_output_raises_invalid_metadata_after_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = Path(tmpdir) / "ranker.md"
            prompt.write_text("{user_query}\n{subtask_json}\n{candidates_json}", encoding="utf-8")

            def duplicate_llm(_: str) -> str:
                return json.dumps(
                    {
                        "ranked_apis": [
                            {"api_id": "api_a", "rank": 1},
                            {"api_id": "api_a", "rank": 2},
                        ]
                    }
                )

            with self.assertRaises(InvalidRankingOutput) as raised:
                rank_subtask(
                    duplicate_llm,
                    user_query="find weather data",
                    subtask={"id": "1", "description": "weather"},
                    candidates=[{"api_id": "api_a"}, {"api_id": "api_b"}],
                    prompt_path=str(prompt),
                    max_validation_retries=1,
                )

        metadata = raised.exception.metadata
        self.assertEqual(metadata["failure_stage"], "llm_ranking")
        self.assertEqual(metadata["failure_reason"], "duplicate_ranked_apis_after_retries")
        self.assertTrue(metadata["exclude_from_ranking_eval"])
        self.assertEqual(metadata["duplicate_api_ids"], ["api_a"])

    def test_incomplete_ranking_output_raises_invalid_metadata_after_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = Path(tmpdir) / "ranker.md"
            prompt.write_text("{user_query}\n{subtask_json}\n{candidates_json}", encoding="utf-8")

            def incomplete_llm(_: str) -> str:
                return json.dumps({"ranked_apis": [{"api_id": "api_a", "rank": 1}]})

            with self.assertRaises(InvalidRankingOutput) as raised:
                rank_subtask(
                    incomplete_llm,
                    user_query="find weather data",
                    subtask={"id": "1", "description": "weather"},
                    candidates=[{"api_id": "api_a"}, {"api_id": "api_b"}],
                    prompt_path=str(prompt),
                    max_validation_retries=1,
                )

        metadata = raised.exception.metadata
        self.assertEqual(metadata["failure_stage"], "llm_ranking")
        self.assertEqual(metadata["failure_reason"], "incomplete_ranked_api_list_after_retries")
        self.assertTrue(metadata["exclude_from_ranking_eval"])
        self.assertEqual(metadata["missing_api_ids"], ["api_b"])

    def test_incomplete_qos_scores_raise_invalid_metadata_after_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = Path(tmpdir) / "qos.md"
            prompt.write_text("{candidates_json}", encoding="utf-8")

            def incomplete_qos_llm(_: str) -> str:
                return json.dumps({"qos_scored": [{"api_id": "api_a", "qos_score": 0.8}]})

            with self.assertRaises(InvalidQosScoringOutput) as raised:
                score_qos_llm(
                    incomplete_qos_llm,
                    candidates=[
                        {"api_id": "api_a", "rt_ms": 10, "tp_rps": 10, "availability": 0.99},
                        {"api_id": "api_b", "rt_ms": 20, "tp_rps": 5, "availability": 0.95},
                    ],
                    prompt_path=str(prompt),
                    batch_size=0,
                    max_validation_retries=1,
                )

        metadata = raised.exception.metadata
        self.assertEqual(metadata["failure_stage"], "qos_llm_scoring")
        self.assertEqual(metadata["failure_reason"], "incomplete_qos_scores_after_retries")
        self.assertTrue(metadata["exclude_from_ranking_eval"])
        self.assertEqual(metadata["missing_api_ids"], ["api_b"])


if __name__ == "__main__":
    unittest.main()
