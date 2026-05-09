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

    def test_qos_scorer_rejects_invalid_score_values(self) -> None:
        raw = json.dumps({"scores": [{"api_id": "api_a", "score": 1.2}, {"api_id": "api_b", "score": 0.41}]})

        _, issue = _parse_qos_score_output(raw, ["api_a", "api_b"])

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "invalid_score_range")

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
