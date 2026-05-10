from __future__ import annotations

import math
import unittest

from src.core.json_parsing import (
    coerce_finite_score,
    normalize_binary_label,
    normalize_llm_payload,
    parse_llm_json,
    recover_scores_key_from_single_list,
    strip_markdown_fences,
    validate_expected_ids,
)


class JsonParsingUtilityTests(unittest.TestCase):
    def test_parse_plain_json_object(self) -> None:
        parsed = parse_llm_json('{"ranked": [{"api_id": "api_a"}]}')

        self.assertIsNone(parsed.error)
        self.assertEqual(parsed.value["ranked"][0]["api_id"], "api_a")

    def test_strips_markdown_fence(self) -> None:
        self.assertEqual(strip_markdown_fences('```json\n{"ok": true}\n```'), '{"ok": true}')

    def test_parse_markdown_fenced_json(self) -> None:
        parsed = parse_llm_json('```json\n{"ranked": [{"api_id": "api_a"}]}\n```')

        self.assertIsNone(parsed.error)
        self.assertEqual(parsed.value["ranked"][0]["api_id"], "api_a")

    def test_parse_json_with_text_before_and_after(self) -> None:
        parsed = parse_llm_json('Here is the JSON:\n{"scores": [{"api_id": "api_a", "score": 0.8}]}\nThanks')

        self.assertIsNone(parsed.error)
        self.assertEqual(parsed.value["scores"][0]["score"], 0.8)

    def test_parse_list_only_output(self) -> None:
        parsed = parse_llm_json('[{"api_id": "api_a"}, {"api_id": "api_b"}]')

        self.assertIsNone(parsed.error)
        self.assertIsInstance(parsed.value, list)

    def test_empty_response_error(self) -> None:
        parsed = parse_llm_json("")

        self.assertEqual(parsed.error["reason"], "empty_response")

    def test_malformed_json_error(self) -> None:
        parsed = parse_llm_json('{"ranked": [')

        self.assertEqual(parsed.error["reason"], "invalid_json")

    def test_normalizes_legacy_keys(self) -> None:
        payload = {"qos_scored": [{"api_id": "api_a", "qos_score": 0.8}]}

        items, issue = normalize_llm_payload(payload, "scores", aliases={"qos_scored": "scores"})

        self.assertIsNone(issue)
        self.assertEqual(items[0]["api_id"], "api_a")

    def test_missing_required_key_error(self) -> None:
        _, issue = normalize_llm_payload({"items": []}, "ranked", aliases={"ranked_apis": "ranked"})

        self.assertEqual(issue["reason"], "missing_required_key")

    def test_recovers_scores_from_single_unexpected_list_key(self) -> None:
        payload = {"scorers": [{"candidate_id": "C01", "score": 0.8}]}

        recovered, issue = recover_scores_key_from_single_list(payload)

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["scores"], payload["scorers"])
        self.assertEqual(issue["reason"], "normalized_unexpected_scores_key")
        self.assertEqual(issue["original_key"], "scorers")

    def test_score_recovery_rejects_multiple_list_keys(self) -> None:
        payload = {
            "scorers": [{"candidate_id": "C01", "score": 0.8}],
            "candidates": [{"candidate_id": "C02", "score": 0.6}],
        }

        recovered, issue = recover_scores_key_from_single_list(payload)

        self.assertIsNone(recovered)
        self.assertEqual(issue["reason"], "ambiguous_score_list_key")

    def test_score_recovery_ignores_non_score_like_single_list(self) -> None:
        payload = {"notes": [{"candidate_id": "C01", "label": 1}]}

        recovered, issue = recover_scores_key_from_single_list(payload)

        self.assertIsNone(recovered)
        self.assertIsNone(issue)

    def test_wrong_json_type_error(self) -> None:
        _, issue = normalize_llm_payload("not an object", "ranked")

        self.assertEqual(issue["reason"], "wrong_json_type")

    def test_validate_expected_ids_accepts_exact_set_once(self) -> None:
        issue = validate_expected_ids(["api_a", "api_b"], ["api_a", "api_b"])

        self.assertIsNone(issue)

    def test_validate_expected_ids_rejects_duplicate(self) -> None:
        issue = validate_expected_ids(["api_a", "api_a"], ["api_a", "api_b"])

        self.assertEqual(issue["reason"], "duplicate_api_id")
        self.assertEqual(issue["duplicate_api_ids"], ["api_a"])

    def test_validate_expected_ids_rejects_unknown(self) -> None:
        issue = validate_expected_ids(["api_a", "api_x"], ["api_a", "api_b"])

        self.assertEqual(issue["reason"], "unknown_api_id")
        self.assertEqual(issue["unknown_api_ids"], ["api_x"])

    def test_validate_expected_ids_rejects_missing(self) -> None:
        issue = validate_expected_ids(["api_a"], ["api_a", "api_b"])

        self.assertEqual(issue["reason"], "incomplete_api_list")
        self.assertEqual(issue["missing_api_ids"], ["api_b"])

    def test_coerce_finite_score_rejects_nan_and_out_of_range(self) -> None:
        self.assertEqual(coerce_finite_score(math.nan)[1], "invalid_score_range")
        self.assertEqual(coerce_finite_score(1.01)[1], "invalid_score_range")

    def test_coerce_finite_score_rejects_missing_and_nonnumeric(self) -> None:
        self.assertEqual(coerce_finite_score(None)[1], "missing_score")
        self.assertEqual(coerce_finite_score("high")[1], "invalid_score_value")

    def test_normalize_binary_label_rejects_invalid_and_missing(self) -> None:
        self.assertEqual(normalize_binary_label(2)[1], "invalid_label_value")
        self.assertEqual(normalize_binary_label(None)[1], "missing_label")


if __name__ == "__main__":
    unittest.main()
