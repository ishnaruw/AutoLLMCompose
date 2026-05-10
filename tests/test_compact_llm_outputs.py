from __future__ import annotations

import json
import unittest

from src.agents.qos_scorer_llm import _parse_qos_score_output
from src.eval.functional_match_eval import _parse_results_with_issue


class CompactLlmOutputTests(unittest.TestCase):
    def test_qos_scorer_accepts_compact_scores_schema(self) -> None:
        raw = json.dumps({"scores": [{"api_id": "api_a", "score": 0.82}, {"api_id": "api_b", "score": 0.41}]})

        scores, issue = _parse_qos_score_output(raw, ["api_a", "api_b"])

        self.assertIsNone(issue)
        self.assertEqual(scores, {"api_a": 0.82, "api_b": 0.41})

    def test_qos_scorer_recovers_scorers_key_with_complete_candidate_scores(self) -> None:
        raw = json.dumps({"scorers": [{"candidate_id": "C01", "score": 0.82}, {"candidate_id": "C02", "score": 0.41}]})

        scores, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNone(issue)
        self.assertEqual(scores, {"api_a": 0.82, "api_b": 0.41})

    def test_qos_scorer_recovers_scor_key_with_complete_candidate_scores(self) -> None:
        raw = json.dumps({"scor": [{"candidate_id": "C01", "score": 0.82}, {"candidate_id": "C02", "score": 0.41}]})

        scores, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNone(issue)
        self.assertEqual(scores, {"api_a": 0.82, "api_b": 0.41})

    def test_qos_scorer_recovers_candidates_key_with_complete_candidate_scores(self) -> None:
        raw = json.dumps({"candidates": [{"candidate_id": "C01", "score": 0.82}, {"candidate_id": "C02", "score": 0.41}]})

        scores, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNone(issue)
        self.assertEqual(scores, {"api_a": 0.82, "api_b": 0.41})

    def test_qos_scorer_recovery_accepts_legacy_qos_score_field(self) -> None:
        raw = json.dumps({"scorers": [{"candidate_id": "C01", "qos_score": 0.82}, {"candidate_id": "C02", "qos_score": 0.41}]})

        scores, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNone(issue)
        self.assertEqual(scores, {"api_a": 0.82, "api_b": 0.41})

    def test_qos_scorer_recovery_rejects_multiple_list_fields_as_ambiguous(self) -> None:
        raw = json.dumps(
            {
                "scorers": [{"candidate_id": "C01", "score": 0.82}],
                "candidates": [{"candidate_id": "C02", "score": 0.41}],
            }
        )

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "ambiguous_score_list_key")

    def test_qos_scorer_recovery_rejects_missing_score(self) -> None:
        raw = json.dumps({"scorers": [{"candidate_id": "C01"}, {"candidate_id": "C02", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "missing_score")

    def test_qos_scorer_recovery_rejects_missing_candidate_id(self) -> None:
        raw = json.dumps({"scorers": [{"score": 0.82}, {"candidate_id": "C02", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "missing_required_key")

    def test_qos_scorer_recovery_rejects_incomplete_candidate_list(self) -> None:
        raw = json.dumps({"scorers": [{"candidate_id": "C01", "score": 0.82}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "incomplete_qos_scores")
        self.assertEqual(issue["missing_candidate_ids"], ["C02"])

    def test_qos_scorer_recovery_rejects_duplicate_candidate_ids(self) -> None:
        raw = json.dumps({"scorers": [{"candidate_id": "C01", "score": 0.82}, {"candidate_id": "C01", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "duplicate_candidate_ids")

    def test_qos_scorer_recovery_rejects_unknown_candidate_ids(self) -> None:
        raw = json.dumps({"scorers": [{"candidate_id": "C01", "score": 0.82}, {"candidate_id": "C99", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "unknown_candidate_ids")

    def test_qos_scorer_recovery_rejects_out_of_range_score(self) -> None:
        raw = json.dumps({"scorers": [{"candidate_id": "C01", "score": 1.2}, {"candidate_id": "C02", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "invalid_score_range")

    def test_qos_scorer_recovery_rejects_non_numeric_score(self) -> None:
        raw = json.dumps({"scorers": [{"candidate_id": "C01", "score": "high"}, {"candidate_id": "C02", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "invalid_score_value")

    def test_qos_scorer_recovery_rejects_nan_score(self) -> None:
        raw = json.dumps({"scorers": [{"candidate_id": "C01", "score": float("nan")}, {"candidate_id": "C02", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "invalid_score_range")

    def test_qos_scorer_rejects_invalid_score_values(self) -> None:
        raw = json.dumps({"scores": [{"api_id": "api_a", "score": 1.2}, {"api_id": "api_b", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"])

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "invalid_score_range")

    def test_qos_scorer_rejects_negative_score_values(self) -> None:
        raw = json.dumps({"scores": [{"api_id": "api_a", "score": -0.1}, {"api_id": "api_b", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"])

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "invalid_score_range")

    def test_qos_scorer_rejects_non_numeric_score_values(self) -> None:
        raw = json.dumps({"scores": [{"api_id": "api_a", "score": "high"}, {"api_id": "api_b", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"])

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "invalid_score_value")

    def test_qos_scorer_rejects_missing_score(self) -> None:
        raw = json.dumps({"scores": [{"api_id": "api_a"}, {"api_id": "api_b", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"])

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "missing_score")

    def test_qos_scorer_rejects_unknown_api_id(self) -> None:
        raw = json.dumps({"scores": [{"api_id": "api_a", "score": 0.8}, {"api_id": "api_x", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"])

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "unknown_api_ids")

    def test_qos_scorer_rejects_duplicate_candidate_ids(self) -> None:
        raw = json.dumps(
            {
                "scores": [
                    {"candidate_id": "C01", "score": 0.8},
                    {"candidate_id": "C01", "score": 0.41},
                ]
            }
        )

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "duplicate_candidate_ids")
        self.assertEqual(issue["duplicate_candidate_ids"], ["C01"])

    def test_qos_scorer_rejects_unknown_candidate_ids(self) -> None:
        raw = json.dumps(
            {
                "scores": [
                    {"candidate_id": "C01", "score": 0.8},
                    {"candidate_id": "C99", "score": 0.41},
                ]
            }
        )

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "unknown_candidate_ids")
        self.assertEqual(issue["unknown_candidate_ids"], ["C99"])

    def test_qos_scorer_rejects_bare_item_without_scores_key(self) -> None:
        raw = json.dumps({"candidate_id": "C01", "score": 0.83})

        scores, issue = _parse_qos_score_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertEqual(scores, {})
        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "missing_required_key")
        self.assertEqual(issue["expected_key"], "scores")
        self.assertEqual(issue["actual_candidate_count"], 0)

    def test_functional_match_accepts_compact_matches_schema(self) -> None:
        raw = json.dumps({"matches": [{"api_id": "api_a", "label": 1}, {"api_id": "api_b", "label": 0}]})

        parsed, issue = _parse_results_with_issue(raw, ["api_a", "api_b"])

        self.assertIsNone(issue)
        self.assertEqual(parsed["api_a"]["functional_match"], 1)
        self.assertEqual(parsed["api_b"]["functional_match"], 0)
        self.assertEqual(parsed["api_a"]["comment"], "")

    def test_functional_match_accepts_reason_as_optional_comment(self) -> None:
        raw = json.dumps(
            {
                "matches": [
                    {"candidate_id": "C01", "label": 1, "reason": "direct endpoint fit"},
                    {"candidate_id": "C02", "label": 0},
                ]
            }
        )

        parsed, issue = _parse_results_with_issue(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNone(issue)
        self.assertEqual(parsed["api_a"]["comment"], "direct endpoint fit")
        self.assertEqual(parsed["api_b"]["comment"], "")

    def test_functional_match_rejects_invalid_label_values(self) -> None:
        raw = json.dumps({"matches": [{"api_id": "api_a", "label": 2}, {"api_id": "api_b", "label": 0}]})

        _, issue = _parse_results_with_issue(raw, ["api_a", "api_b"])

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "invalid_label_value")

    def test_functional_match_rejects_missing_label(self) -> None:
        raw = json.dumps({"matches": [{"api_id": "api_a"}, {"api_id": "api_b", "label": 0}]})

        _, issue = _parse_results_with_issue(raw, ["api_a", "api_b"])

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "missing_label")


if __name__ == "__main__":
    unittest.main()
