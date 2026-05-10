from __future__ import annotations

import math
import unittest

from pydantic import ValidationError

from src.core.output_schemas import (
    DecompositionOutput,
    FunctionalMatchOutput,
    PlannerOutput,
    QoSScoreOutput,
    RankedCandidatesOutput,
)


class OutputSchemaTests(unittest.TestCase):
    def test_ranked_candidates_accepts_compact_api_id_items(self) -> None:
        output = RankedCandidatesOutput.model_validate({"ranked": [{"api_id": "api_a"}, {"api_id": "api_b"}]})

        self.assertEqual([item.api_id for item in output.ranked], ["api_a", "api_b"])

    def test_ranked_candidates_accepts_compact_candidate_id_items(self) -> None:
        output = RankedCandidatesOutput.model_validate({"ranked": [{"candidate_id": "C01"}]})

        self.assertEqual(output.ranked[0].candidate_id, "C01")

    def test_ranked_candidates_accepts_optional_reason_fields(self) -> None:
        output = RankedCandidatesOutput.model_validate(
            {
                "ranked": [
                    {
                        "api_id": "api_a",
                        "reason": "direct fit",
                        "functional_reason": "matches operation",
                        "qos_reason": "fast enough",
                    }
                ]
            }
        )

        self.assertEqual(output.ranked[0].reason, "direct fit")
        self.assertEqual(output.ranked[0].functional_reason, "matches operation")

    def test_qos_scores_accepts_valid_scores(self) -> None:
        output = QoSScoreOutput.model_validate(
            {"scores": [{"api_id": "api_a", "score": 0.9}, {"api_id": "api_b", "score": 0}]}
        )

        self.assertEqual([item.score for item in output.scores], [0.9, 0.0])

    def test_qos_scores_accepts_optional_explanation_fields(self) -> None:
        output = QoSScoreOutput.model_validate(
            {"scores": [{"candidate_id": "C01", "score": 0.5, "reason": "balanced", "explanation": "ok"}]}
        )

        self.assertEqual(output.scores[0].reason, "balanced")
        self.assertEqual(output.scores[0].explanation, "ok")

    def test_functional_match_accepts_valid_labels(self) -> None:
        output = FunctionalMatchOutput.model_validate(
            {"matches": [{"api_id": "api_a", "label": 1}, {"api_id": "api_b", "label": 0}]}
        )

        self.assertEqual([item.label for item in output.matches], [1, 0])

    def test_functional_match_accepts_optional_reason_fields(self) -> None:
        output = FunctionalMatchOutput.model_validate(
            {"matches": [{"candidate_id": "C01", "label": 1, "reason": "endpoint matches"}]}
        )

        self.assertEqual(output.matches[0].reason, "endpoint matches")

    def test_decomposition_accepts_current_structure(self) -> None:
        output = DecompositionOutput.model_validate(
            {"subtasks": [{"id": 1, "description": "Find route"}, {"id": 2, "goal": "Book ticket"}]}
        )

        self.assertEqual(len(output.subtasks), 2)

    def test_planner_accepts_current_structure(self) -> None:
        output = PlannerOutput.model_validate(
            {
                "paths": [
                    {
                        "path_id": 1,
                        "path_score": 0.8,
                        "summary": "Use best APIs",
                        "steps": [
                            {
                                "step": 1,
                                "api_id": "api_a",
                                "subtask_id": 1,
                                "action": "Call API",
                                "why": "Matches subtask",
                                "qos": None,
                            }
                        ],
                        "subtask_coverage": [
                            {"subtask_id": 1, "description": "Find route", "steps": [1], "coverage": "full"}
                        ],
                    }
                ],
                "selected_api_ids": ["api_a"],
                "overall_rationale": "Best fit",
            }
        )

        self.assertEqual(output.paths[0].steps[0].api_id, "api_a")

    def test_ranked_output_missing_ranked_key_is_invalid(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            RankedCandidatesOutput.model_validate({"items": [{"api_id": "api_a"}]})

        self.assertEqual(raised.exception.errors()[0]["type"], "missing")

    def test_ranked_item_missing_id_is_invalid(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            RankedCandidatesOutput.model_validate({"ranked": [{"reason": "no id"}]})

        self.assertEqual(raised.exception.errors()[0]["type"], "missing_required_key")

    def test_qos_item_missing_score_is_invalid(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            QoSScoreOutput.model_validate({"scores": [{"api_id": "api_a"}]})

        self.assertEqual(raised.exception.errors()[0]["type"], "missing_score")

    def test_qos_score_outside_range_is_invalid(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            QoSScoreOutput.model_validate({"scores": [{"api_id": "api_a", "score": 1.2}]})

        self.assertEqual(raised.exception.errors()[0]["type"], "invalid_score_range")

    def test_qos_score_nonnumeric_or_nan_is_invalid(self) -> None:
        with self.assertRaises(ValidationError) as nonnumeric:
            QoSScoreOutput.model_validate({"scores": [{"api_id": "api_a", "score": "high"}]})
        with self.assertRaises(ValidationError) as nan_score:
            QoSScoreOutput.model_validate({"scores": [{"api_id": "api_a", "score": math.nan}]})

        self.assertEqual(nonnumeric.exception.errors()[0]["type"], "invalid_score_value")
        self.assertEqual(nan_score.exception.errors()[0]["type"], "invalid_score_range")

    def test_functional_match_missing_label_is_invalid(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            FunctionalMatchOutput.model_validate({"matches": [{"api_id": "api_a"}]})

        self.assertEqual(raised.exception.errors()[0]["type"], "missing_label")

    def test_functional_match_invalid_label_is_invalid(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            FunctionalMatchOutput.model_validate({"matches": [{"api_id": "api_a", "label": 2}]})

        self.assertEqual(raised.exception.errors()[0]["type"], "invalid_label_value")

    def test_wrong_top_level_type_is_invalid(self) -> None:
        with self.assertRaises(ValidationError) as raised:
            RankedCandidatesOutput.model_validate([{"api_id": "api_a"}])

        self.assertEqual(raised.exception.errors()[0]["type"], "model_type")


if __name__ == "__main__":
    unittest.main()
