from __future__ import annotations

import json
import unittest

from src.agents.decomposer import _parse_subtasks


class DecomposerPostprocessTests(unittest.TestCase):
    def test_folds_local_workflow_subtasks_into_previous_api_step(self) -> None:
        raw = json.dumps(
            {
                "subtasks": [
                    {"id": 1, "description": "Fetch top news articles"},
                    {"id": 2, "description": "Compose SMS digest"},
                    {"id": 3, "description": "Send SMS digest"},
                ]
            }
        )

        subtasks = _parse_subtasks(raw, "fallback")

        self.assertEqual(len(subtasks), 2)
        self.assertEqual(subtasks[0]["id"], 1)
        self.assertIn("Fetch top news articles", subtasks[0]["description"])
        self.assertIn("Compose SMS digest", subtasks[0]["description"])
        self.assertEqual(subtasks[1]["id"], 2)

    def test_preserves_api_backed_summarization_subtask(self) -> None:
        raw = json.dumps(
            {
                "subtasks": [
                    {"id": 1, "description": "Fetch top news articles"},
                    {"id": 2, "description": "Summarize articles using a text-summarization API"},
                    {"id": 3, "description": "Send SMS digest"},
                ]
            }
        )

        subtasks = _parse_subtasks(raw, "fallback")

        self.assertEqual([s["id"] for s in subtasks], [1, 2, 3])
        self.assertEqual(subtasks[1]["description"], "Summarize articles using a text-summarization API")

    def test_folds_daily_workflow_scheduling_into_previous_api_step(self) -> None:
        raw = json.dumps(
            {
                "subtasks": [
                    {"id": 1, "description": "Fetch top news articles using a news aggregation API"},
                    {"id": 2, "description": "Summarize the fetched articles using a text-summarization API"},
                    {"id": 3, "description": "Send the summarized digest via an SMS messaging API"},
                    {"id": 4, "description": "Schedule the fetch-summarize-send workflow to run daily using a scheduling API"},
                ]
            }
        )

        subtasks = _parse_subtasks(raw, "fallback")

        self.assertEqual(len(subtasks), 3)
        self.assertEqual([s["id"] for s in subtasks], [1, 2, 3])
        self.assertIn("Send the summarized digest", subtasks[2]["description"])
        self.assertIn("Schedule the fetch-summarize-send workflow", subtasks[2]["description"])

    def test_preserves_weather_display_as_api_backed_information_need(self) -> None:
        raw = json.dumps({"subtasks": [{"id": 1, "description": "Display current weather info"}]})

        subtasks = _parse_subtasks(raw, "fallback")

        self.assertEqual(subtasks, [{"id": 1, "description": "Display current weather info"}])

    def test_drops_leading_domain_inventory_subtask(self) -> None:
        raw = json.dumps(
            {
                "subtasks": [
                    {"id": 1, "description": "Fetch the list of domains to monitor via a configuration or inventory API"},
                    {"id": 2, "description": "Check each provided domain using a domain threat intelligence API"},
                    {"id": 3, "description": "Send SMS alerts to administrators"},
                ]
            }
        )

        subtasks = _parse_subtasks(raw, "fallback")

        self.assertEqual(len(subtasks), 2)
        self.assertEqual(subtasks[0]["id"], 1)
        self.assertEqual(subtasks[0]["description"], "Check each provided domain using a domain threat intelligence API")
        self.assertEqual(subtasks[1]["id"], 2)

    def test_folds_downstream_blocking_handoff_into_scan_step(self) -> None:
        raw = json.dumps(
            {
                "subtasks": [
                    {"id": 1, "description": "Scan the URL for malware using a URL scanning API"},
                    {"id": 2, "description": "Send aggregated scan results to the downstream blocking service via an API"},
                ]
            }
        )

        subtasks = _parse_subtasks(raw, "fallback")

        self.assertEqual(len(subtasks), 1)
        self.assertIn("Scan the URL for malware", subtasks[0]["description"])
        self.assertIn("downstream blocking service", subtasks[0]["description"])

    def test_folds_price_baseline_comparison_even_when_it_mentions_api(self) -> None:
        raw = json.dumps(
            {
                "subtasks": [
                    {"id": 1, "description": "Search for the target product on each eCommerce platform using product search APIs"},
                    {"id": 2, "description": "Fetch current pricing details for the located product listings via price retrieval APIs"},
                    {
                        "id": 3,
                        "description": "Check the fetched prices against stored baseline values to detect any price drops using a price-monitoring/check API",
                    },
                    {"id": 4, "description": "Send an alert through a notification API when a price drop is detected"},
                ]
            }
        )

        subtasks = _parse_subtasks(raw, "fallback")

        self.assertEqual(len(subtasks), 3)
        self.assertEqual([s["id"] for s in subtasks], [1, 2, 3])
        self.assertIn("Fetch current pricing details", subtasks[1]["description"])
        self.assertIn("Check the fetched prices against stored baseline values", subtasks[1]["description"])
        self.assertIn("Send an alert", subtasks[2]["description"])


if __name__ == "__main__":
    unittest.main()
