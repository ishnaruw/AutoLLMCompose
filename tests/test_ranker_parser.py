from __future__ import annotations

import json
import unittest

from src.agents.ranker import _parse_ranked_output


class RankerParserTests(unittest.TestCase):
    def test_accepts_minimal_ranked_apis_schema(self) -> None:
        raw = json.dumps(
            {
                "ranked_apis": [
                    {"api_id": "api_a", "rank": 1},
                    {"api_id": "api_b", "rank": 2},
                ]
            }
        )

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"])

        self.assertIsNone(issue)
        self.assertEqual([item["api_id"] for item in ranked], ["api_a", "api_b"])
        self.assertEqual([item["llm_reported_rank"] for item in ranked], [1, 2])

    def test_accepts_legacy_ranked_key(self) -> None:
        raw = json.dumps(
            {
                "ranked": [
                    {"api_id": "api_a", "rank": 1, "reason": "functional match"},
                    {"api_id": "api_b", "rank": 2, "reason": "less direct"},
                ]
            }
        )

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"])

        self.assertIsNone(issue)
        self.assertEqual([item["api_id"] for item in ranked], ["api_a", "api_b"])

    def test_accepts_compact_ranked_order_without_rank_or_reason(self) -> None:
        raw = json.dumps({"ranked": [{"api_id": "api_b"}, {"api_id": "api_a"}]})

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"])

        self.assertIsNone(issue)
        self.assertEqual([item["api_id"] for item in ranked], ["api_b", "api_a"])
        self.assertEqual([item["llm_reported_rank"] for item in ranked], [1, 2])
        self.assertEqual([item["reason"] for item in ranked], ["", ""])

    def test_accepts_markdown_fenced_compact_ranked_output(self) -> None:
        raw = '```json\n{"ranked": [{"api_id": "api_b"}, {"api_id": "api_a"}]}\n```'

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"])

        self.assertIsNone(issue)
        self.assertEqual([item["api_id"] for item in ranked], ["api_b", "api_a"])

    def test_accepts_text_around_compact_ranked_output(self) -> None:
        raw = 'Here is the JSON:\n{"ranked": [{"api_id": "api_b"}, {"api_id": "api_a"}]}\nDone.'

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"])

        self.assertIsNone(issue)
        self.assertEqual([item["api_id"] for item in ranked], ["api_b", "api_a"])

    def test_accepts_list_only_ranked_output(self) -> None:
        raw = json.dumps([{"api_id": "api_b"}, {"api_id": "api_a"}])

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"])

        self.assertIsNone(issue)
        self.assertEqual([item["api_id"] for item in ranked], ["api_b", "api_a"])

    def test_accepts_ranking_alias(self) -> None:
        raw = json.dumps({"ranking": [{"api_id": "api_a"}, {"api_id": "api_b"}]})

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"])

        self.assertIsNone(issue)
        self.assertEqual([item["api_id"] for item in ranked], ["api_a", "api_b"])

    def test_missing_ranked_key_is_clear_error(self) -> None:
        raw = json.dumps({"items": [{"api_id": "api_a"}, {"api_id": "api_b"}]})

        _, issue = _parse_ranked_output(raw, ["api_a", "api_b"])

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "missing_required_key")

    def test_accepts_compact_ranked_candidate_ids_without_rank_or_reason(self) -> None:
        raw = json.dumps({"ranked": [{"candidate_id": "C02"}, {"candidate_id": "C01"}]})

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNone(issue)
        self.assertEqual([item["candidate_id"] for item in ranked], ["C02", "C01"])
        self.assertEqual([item["api_id"] for item in ranked], ["api_b", "api_a"])
        self.assertEqual([item["llm_reported_rank"] for item in ranked], [1, 2])

    def test_compact_api_id_schema_backfills_candidate_ids_when_available(self) -> None:
        raw = json.dumps({"ranked": [{"api_id": "api_b"}, {"api_id": "api_a"}]})

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNone(issue)
        self.assertEqual([item["candidate_id"] for item in ranked], ["C02", "C01"])

    def test_accepts_old_reason_fields_without_requiring_them(self) -> None:
        raw = json.dumps(
            {
                "ranked_apis": [
                    {
                        "api_id": "api_a",
                        "rank": 1,
                        "functional_reason": "directly supports task",
                        "qos_reason": "best QoS rank",
                    },
                    {
                        "api_id": "api_b",
                        "rank": 2,
                        "functional_reason": "usable fallback",
                        "qos_reason": "lower QoS",
                    },
                ]
            }
        )

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"])

        self.assertIsNone(issue)
        self.assertEqual(ranked[0]["functional_reason"], "directly supports task")
        self.assertEqual(ranked[0]["qos_reason"], "best QoS rank")

    def test_duplicate_ids_remain_invalid(self) -> None:
        raw = json.dumps(
            {
                "ranked_apis": [
                    {"api_id": "api_a", "rank": 1},
                    {"api_id": "api_a", "rank": 2},
                ]
            }
        )

        _, issue = _parse_ranked_output(raw, ["api_a", "api_b"])

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "duplicate_ranked_apis")

    def test_maps_candidate_ids_back_to_api_ids(self) -> None:
        raw = json.dumps(
            {
                "ranked_apis": [
                    {"candidate_id": "C02", "rank": 2},
                    {"candidate_id": "C01", "rank": 1},
                ]
            }
        )

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNone(issue)
        self.assertEqual([item["candidate_id"] for item in ranked], ["C01", "C02"])
        self.assertEqual([item["api_id"] for item in ranked], ["api_a", "api_b"])
        self.assertEqual([item["llm_reported_rank"] for item in ranked], [1, 2])

    def test_duplicate_candidate_ids_are_invalid_with_api_mapping(self) -> None:
        raw = json.dumps(
            {
                "ranked_apis": [
                    {"candidate_id": "C01", "rank": 1},
                    {"candidate_id": "C01", "rank": 2},
                ]
            }
        )

        _, issue = _parse_ranked_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "duplicate_candidate_ids")
        self.assertEqual(issue["duplicate_candidate_ids"], ["C01"])
        self.assertEqual(issue["duplicate_api_ids"], ["api_a"])

    def test_legacy_api_ids_are_accepted_with_candidate_mapping(self) -> None:
        raw = json.dumps(
            {
                "ranked_apis": [
                    {"api_id": "api_b", "rank": 2},
                    {"api_id": "api_a", "rank": 1},
                ]
            }
        )

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNone(issue)
        self.assertEqual([item["api_id"] for item in ranked], ["api_a", "api_b"])

    def test_duplicate_rank_values_are_invalid(self) -> None:
        raw = json.dumps(
            {
                "ranked_apis": [
                    {"candidate_id": "C01", "rank": 1},
                    {"candidate_id": "C02", "rank": 1},
                ]
            }
        )

        _, issue = _parse_ranked_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "duplicate_rank_values")
        self.assertEqual(issue["duplicate_rank_values"], [1])

    def test_missing_rank_value_is_invalid(self) -> None:
        raw = json.dumps(
            {
                "ranked_apis": [
                    {"candidate_id": "C01", "rank": 1},
                    {"candidate_id": "C02"},
                ]
            }
        )

        _, issue = _parse_ranked_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "missing_rank_values")
        self.assertEqual(issue["missing_rank_candidate_ids"], ["C02"])

    def test_non_integer_rank_value_is_invalid(self) -> None:
        raw = json.dumps(
            {
                "ranked_apis": [
                    {"candidate_id": "C01", "rank": 1},
                    {"candidate_id": "C02", "rank": "second"},
                ]
            }
        )

        _, issue = _parse_ranked_output(raw, ["api_a", "api_b"], {"C01": "api_a", "C02": "api_b"})

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "non_integer_rank_values")
        self.assertEqual(issue["non_integer_rank_values"], ["second"])

    def test_rank_sequence_missing_number_is_invalid(self) -> None:
        raw = json.dumps(
            {
                "ranked_apis": [
                    {"candidate_id": "C01", "rank": 1},
                    {"candidate_id": "C02", "rank": 3},
                    {"candidate_id": "C03", "rank": 4},
                ]
            }
        )

        _, issue = _parse_ranked_output(
            raw,
            ["api_a", "api_b", "api_c"],
            {"C01": "api_a", "C02": "api_b", "C03": "api_c"},
        )

        self.assertIsNotNone(issue)
        self.assertEqual(issue["reason"], "rank_values_out_of_range")
        self.assertEqual(issue["missing_rank_values"], [2])
        self.assertEqual(issue["rank_values_out_of_range"], [4])


if __name__ == "__main__":
    unittest.main()
