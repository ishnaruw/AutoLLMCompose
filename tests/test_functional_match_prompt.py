from __future__ import annotations

import unittest

from src.eval.functional_match_eval import _zero_match_retry_attempted
from src.eval.functional_match_prompt import build_llm_prompt


class FunctionalMatchPromptTests(unittest.TestCase):
    def test_prompt_always_requires_short_reasons(self) -> None:
        prompt = build_llm_prompt(
            query_id="q01",
            main_task="Fetch weather and send alerts",
            subtask_id="1",
            subtask_description="Fetch weather data",
            api_entries=[{"candidate_id": "C01", "api_id": "weather_api", "description": "Weather forecast"}],
        )

        self.assertIn('"reason"', prompt)
        self.assertIn("reason is required", prompt)
        self.assertIn("weather forecast/current-weather APIs", prompt)

    def test_zero_match_retry_prompt_adds_recheck_guidance(self) -> None:
        prompt = build_llm_prompt(
            query_id="q01",
            main_task="Fetch weather and send alerts",
            subtask_id="1",
            subtask_description="Fetch weather data",
            api_entries=[{"candidate_id": "C01", "api_id": "weather_api", "description": "Weather forecast"}],
            zero_match_retry=True,
        )

        self.assertIn("zero-match recheck", prompt)
        self.assertIn("Do not force a match", prompt)

    def test_zero_match_retry_attempt_flag_requires_all_apis(self) -> None:
        results = {
            "api_a": {"functional_match": 0, "zero_match_retry_attempted": True},
            "api_b": {"functional_match": 0},
        }
        apis = [{"api_id": "api_a"}, {"api_id": "api_b"}]

        self.assertFalse(_zero_match_retry_attempted(results, apis))

        results["api_b"]["zero_match_retry_attempted"] = True
        self.assertTrue(_zero_match_retry_attempted(results, apis))


if __name__ == "__main__":
    unittest.main()
