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

    def test_preserves_weather_display_as_api_backed_information_need(self) -> None:
        raw = json.dumps({"subtasks": [{"id": 1, "description": "Display current weather info"}]})

        subtasks = _parse_subtasks(raw, "fallback")

        self.assertEqual(subtasks, [{"id": 1, "description": "Display current weather info"}])


if __name__ == "__main__":
    unittest.main()
