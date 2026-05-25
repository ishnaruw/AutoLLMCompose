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
        self.assertEqual([row["api_id"] for row in metadata["_invalid_ranked_items"]], ["api_a", "api_a"])

    def test_retry_success_is_logged_as_structured_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_log = Path(tmpdir) / "run.log"
            configure_run_log(run_log)
            prompt = Path(tmpdir) / "ranker.md"
            prompt.write_text("{user_query}\n{subtask_json}\n{candidates_json}", encoding="utf-8")
            calls = 0

            def retry_then_valid_llm(_: str) -> str:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return json.dumps(
                        {
                            "ranked_apis": [
                                {"api_id": "api_a", "rank": 1},
                                {"api_id": "api_a", "rank": 2},
                            ]
                        }
                    )
                return json.dumps(
                    {
                        "ranked_apis": [
                            {"api_id": "api_a", "rank": 1},
                            {"api_id": "api_b", "rank": 2},
                        ]
                    }
                )

            try:
                ranked = rank_subtask(
                    retry_then_valid_llm,
                    user_query="find weather data",
                    subtask={"id": "1", "description": "weather"},
                    candidates=[{"api_id": "api_a"}, {"api_id": "api_b"}],
                    prompt_path=str(prompt),
                    max_validation_retries=1,
                )
            finally:
                clear_run_log()

            retry_log = Path(tmpdir) / "retry_outcomes.log"
            event = json.loads(retry_log.read_text(encoding="utf-8").strip())

        self.assertEqual([row["api_id"] for row in ranked], ["api_a", "api_b"])
        self.assertEqual(event["event_type"], "llm_validation_retry_outcome")
        self.assertEqual(event["outcome"], "success")
        self.assertEqual(event["stage"], "llm_ranking")
        self.assertEqual(event["invalid_attempts"], 1)

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
                        {"api_id": "api_a", "rt_s": 10, "tp_kbps": 10, "availability": 0.99},
                        {"api_id": "api_b", "rt_s": 20, "tp_kbps": 5, "availability": 0.95},
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

    def test_candidate_id_ranking_output_maps_to_api_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = Path(tmpdir) / "ranker.md"
            prompt.write_text("{user_query}\n{subtask_json}\n{candidates_json}", encoding="utf-8")

            def candidate_id_llm(_: str) -> str:
                return json.dumps(
                    {
                        "ranked_apis": [
                            {"candidate_id": "C02", "rank": 2},
                            {"candidate_id": "C01", "rank": 1},
                        ]
                    }
                )

            ranked = rank_subtask(
                candidate_id_llm,
                user_query="find weather data",
                subtask={"id": "1", "description": "weather"},
                candidates=[{"api_id": "api_a"}, {"api_id": "api_b"}],
                prompt_path=str(prompt),
                max_validation_retries=0,
            )

        self.assertEqual([row["candidate_id"] for row in ranked], ["C01", "C02"])
        self.assertEqual([row["api_id"] for row in ranked], ["api_a", "api_b"])
        self.assertEqual([row["llm_reported_rank"] for row in ranked], [1, 2])

    def test_duplicate_candidate_id_ranking_output_raises_candidate_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = Path(tmpdir) / "ranker.md"
            prompt.write_text("{user_query}\n{subtask_json}\n{candidates_json}", encoding="utf-8")

            def duplicate_candidate_llm(_: str) -> str:
                return json.dumps(
                    {
                        "ranked_apis": [
                            {"candidate_id": "C01", "rank": 1},
                            {"candidate_id": "C01", "rank": 2},
                        ]
                    }
                )

            with self.assertRaises(InvalidRankingOutput) as raised:
                rank_subtask(
                    duplicate_candidate_llm,
                    user_query="find weather data",
                    subtask={"id": "1", "description": "weather"},
                    candidates=[{"api_id": "api_a"}, {"api_id": "api_b"}],
                    prompt_path=str(prompt),
                    max_validation_retries=1,
                )

        metadata = raised.exception.metadata
        self.assertEqual(metadata["failure_reason"], "duplicate_candidate_ids_after_retries")
        self.assertEqual(metadata["duplicate_candidate_ids"], ["C01"])
        self.assertEqual(metadata["duplicate_api_ids"], ["api_a"])
        self.assertEqual([row["candidate_id"] for row in metadata["_invalid_ranked_items"]], ["C01", "C01"])
        self.assertEqual([row["api_id"] for row in metadata["_invalid_ranked_items"]], ["api_a", "api_a"])

    def test_duplicate_rank_output_raises_invalid_metadata_after_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = Path(tmpdir) / "ranker.md"
            prompt.write_text("{user_query}\n{subtask_json}\n{candidates_json}", encoding="utf-8")

            def duplicate_rank_llm(_: str) -> str:
                return json.dumps(
                    {
                        "ranked_apis": [
                            {"candidate_id": "C01", "rank": 1},
                            {"candidate_id": "C02", "rank": 1},
                        ]
                    }
                )

            with self.assertRaises(InvalidRankingOutput) as raised:
                rank_subtask(
                    duplicate_rank_llm,
                    user_query="find weather data",
                    subtask={"id": "1", "description": "weather"},
                    candidates=[{"api_id": "api_a"}, {"api_id": "api_b"}],
                    prompt_path=str(prompt),
                    max_validation_retries=1,
                )

        metadata = raised.exception.metadata
        self.assertEqual(metadata["failure_stage"], "llm_ranking")
        self.assertEqual(metadata["failure_reason"], "duplicate_rank_values_after_retries")
        self.assertEqual(metadata["expected_rank_count"], 2)
        self.assertEqual(metadata["actual_rank_count"], 1)
        self.assertTrue(metadata["exclude_from_ranking_eval"])

    def test_qos_scorer_prompts_candidate_id_only_and_maps_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = Path(tmpdir) / "qos.md"
            prompt.write_text("{candidates_json}", encoding="utf-8")
            seen_prompts: list[str] = []

            def candidate_qos_llm(prompt_text: str) -> str:
                seen_prompts.append(prompt_text)
                return json.dumps(
                    {
                        "qos_scored": [
                            {"candidate_id": "C01", "qos_score": 0.8},
                            {"candidate_id": "C02", "qos_score": 0.5},
                        ]
                    }
                )

            scores = score_qos_llm(
                candidate_qos_llm,
                candidates=[
                    {"api_id": "api_a", "rt_s": 10, "tp_kbps": 10, "availability": 0.99},
                    {"api_id": "api_b", "rt_s": 20, "tp_kbps": 5, "availability": 0.95},
                ],
                prompt_path=str(prompt),
                batch_size=0,
                max_validation_retries=0,
            )

        self.assertNotIn('"api_id"', seen_prompts[0])
        self.assertIn('"candidate_id": "C01"', seen_prompts[0])
        self.assertEqual(scores["api_a"]["qos_llm_rank"], 1)
        self.assertEqual(scores["api_b"]["qos_llm_rank"], 2)

    def test_qos_scorer_accepts_judgment_scores_that_do_not_match_formula_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = Path(tmpdir) / "qos.md"
            prompt.write_text("{normalization_context}\n{candidates_json}", encoding="utf-8")
            seen_prompts: list[str] = []

            def judgment_qos_llm(prompt_text: str) -> str:
                seen_prompts.append(prompt_text)
                return json.dumps(
                    {
                        "scores": [
                            {"candidate_id": "C01", "score": 0.1},
                            {"candidate_id": "C02", "score": 0.2},
                        ]
                    }
                )

            scores = score_qos_llm(
                judgment_qos_llm,
                candidates=[
                    {"api_id": "api_a", "rt_s": 10, "tp_kbps": 10, "availability": 0.99},
                    {"api_id": "api_b", "rt_s": 20, "tp_kbps": 5, "availability": 0.95},
                ],
                prompt_path=str(prompt),
                batch_size=0,
                max_validation_retries=0,
            )

        self.assertEqual(len(seen_prompts), 1)
        self.assertIn('"weights_provided": false', seen_prompts[0])
        self.assertNotIn('"max_rt_s"', seen_prompts[0])
        self.assertEqual(scores["api_b"]["qos_llm_rank"], 1)
        self.assertEqual(scores["api_a"]["qos_llm_score"], 0.1)

    def test_qos_scorer_retry_rejects_bare_candidate_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = Path(tmpdir) / "qos.md"
            prompt.write_text("{normalization_context}\n{candidates_json}", encoding="utf-8")
            seen_prompts: list[str] = []

            def retry_qos_llm(prompt_text: str) -> str:
                seen_prompts.append(prompt_text)
                if len(seen_prompts) == 1:
                    return json.dumps({"candidate_id": "C01", "score": 0.83})
                return json.dumps(
                    {
                        "scores": [
                            {"candidate_id": "C01", "score": 0.83},
                            {"candidate_id": "C02", "score": 0.4833},
                        ]
                    }
                )

            scores = score_qos_llm(
                retry_qos_llm,
                candidates=[
                    {"api_id": "api_a", "rt_s": 10, "tp_kbps": 10, "availability": 0.99},
                    {"api_id": "api_b", "rt_s": 20, "tp_kbps": 5, "availability": 0.95},
                ],
                prompt_path=str(prompt),
                batch_size=0,
                max_validation_retries=1,
            )

        self.assertEqual(len(seen_prompts), 2)
        self.assertIn('"weights_provided": false', seen_prompts[0])
        self.assertIn('never return a bare {"candidate_id": ..., "score": ...} object', seen_prompts[1])
        self.assertIn('["C01", "C02"]', seen_prompts[1])
        self.assertEqual(scores["api_a"]["qos_llm_rank"], 1)
        self.assertEqual(scores["api_b"]["qos_llm_rank"], 2)

    def test_qos_scorer_formula_mismatch_raises_invalid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = Path(tmpdir) / "qos.md"
            prompt.write_text("{normalization_context}\n{candidates_json}", encoding="utf-8")

            def mismatched_qos_llm(_: str) -> str:
                return json.dumps(
                    {
                        "scores": [
                            {"candidate_id": "C01", "score": 0.1},
                            {"candidate_id": "C02", "score": 0.2},
                        ]
                    }
                )

            with self.assertRaises(InvalidQosScoringOutput) as raised:
                score_qos_llm(
                    mismatched_qos_llm,
                    candidates=[
                        {"api_id": "api_a", "rt_s": 10, "tp_kbps": 10, "availability": 0.99},
                        {"api_id": "api_b", "rt_s": 20, "tp_kbps": 5, "availability": 0.95},
                    ],
                    prompt_path=str(prompt),
                    batch_size=0,
                    max_validation_retries=0,
                    validate_formula=True,
                )

        metadata = raised.exception.metadata
        self.assertEqual(metadata["failure_stage"], "qos_llm_scoring")
        self.assertEqual(metadata["failure_reason"], "qos_score_formula_mismatch")
        self.assertEqual(metadata["mismatched_candidate_ids"], ["C01", "C02"])
        self.assertEqual(metadata["expected_candidate_count"], 2)

    def test_qos_scorer_formula_audit_logs_mismatch_without_rejecting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_log = Path(tmpdir) / "run.log"
            configure_run_log(run_log)
            prompt = Path(tmpdir) / "qos.md"
            prompt.write_text("{normalization_context}\n{candidates_json}", encoding="utf-8")

            def mismatched_qos_llm(_: str) -> str:
                return json.dumps(
                    {
                        "scores": [
                            {"candidate_id": "C01", "score": 0.1},
                            {"candidate_id": "C02", "score": 0.2},
                        ]
                    }
                )

            try:
                scores = score_qos_llm(
                    mismatched_qos_llm,
                    candidates=[
                        {"api_id": "api_a", "rt_s": 10, "tp_kbps": 10, "availability": 0.99},
                        {"api_id": "api_b", "rt_s": 20, "tp_kbps": 5, "availability": 0.95},
                    ],
                    prompt_path=str(prompt),
                    batch_size=0,
                    max_validation_retries=0,
                    formula_audit=True,
                )
            finally:
                clear_run_log()

            audit_log_text = run_log.read_text(encoding="utf-8")

        self.assertEqual(scores["api_b"]["qos_llm_rank"], 1)
        self.assertIn("formula audit mismatch", audit_log_text)

    def test_qos_scorer_none_batch_size_scores_all_candidates_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = Path(tmpdir) / "qos.md"
            prompt.write_text("{candidates_json}", encoding="utf-8")
            seen_prompts: list[str] = []

            def batched_qos_llm(prompt_text: str) -> str:
                seen_prompts.append(prompt_text)
                return json.dumps(
                    {
                        "scores": [
                            {"candidate_id": "C01", "score": 0.3},
                            {"candidate_id": "C02", "score": 0.2},
                            {"candidate_id": "C03", "score": 0.1},
                        ]
                    }
                )

            scores = score_qos_llm(
                batched_qos_llm,
                candidates=[
                    {"api_id": "api_a", "rt_s": 10, "tp_kbps": 10, "availability": 0.99},
                    {"api_id": "api_b", "rt_s": 20, "tp_kbps": 5, "availability": 0.95},
                    {"api_id": "api_c", "rt_s": 30, "tp_kbps": 4, "availability": 0.9},
                ],
                prompt_path=str(prompt),
                batch_size=None,
                max_validation_retries=0,
            )

        self.assertEqual(len(seen_prompts), 1)
        self.assertIn('"candidate_id": "C03"', seen_prompts[0])
        self.assertEqual(scores["api_a"]["qos_llm_rank"], 1)


if __name__ == "__main__":
    unittest.main()
