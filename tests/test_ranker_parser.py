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
                    {"api_id": "api_a", "reason": "functional match"},
                    {"api_id": "api_b", "reason": "less direct"},
                ]
            }
        )

        ranked, issue = _parse_ranked_output(raw, ["api_a", "api_b"])

        self.assertIsNone(issue)
        self.assertEqual([item["api_id"] for item in ranked], ["api_a", "api_b"])

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


if __name__ == "__main__":
    unittest.main()
