from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.core import retry
from src.llm.backends import FailoverBackend, groq_experiment_model_pool


class _FakeBackend:
    provider = "groq"

    def __init__(self, model_name: str, outcomes: list[object]) -> None:
        self.model_name = model_name
        self.outcomes = list(outcomes)
        self.calls = 0

    def chat_json(self, **_kwargs: object) -> str:
        self.calls += 1
        if not self.outcomes:
            raise AssertionError(f"Unexpected call to {self.model_name}")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)


class GroqFailoverBackendTests(unittest.TestCase):
    def test_default_groq_multi_pool_starts_with_largest_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            pool = groq_experiment_model_pool()

        self.assertEqual(pool[0], "llama-3.3-70b-versatile")
        self.assertIn("openai/gpt-oss-20b", pool)

    def test_custom_groq_multi_pool_is_prioritized_by_known_request_limit(self) -> None:
        with patch.dict(
            os.environ,
            {
                "GROQ_MULTI_MODELS": "llama-3.1-8b-instant,openai/gpt-oss-20b,llama-3.3-70b-versatile",
            },
            clear=True,
        ):
            pool = groq_experiment_model_pool()

        self.assertEqual(
            pool,
            ["llama-3.3-70b-versatile", "openai/gpt-oss-20b", "llama-3.1-8b-instant"],
        )

    def test_large_prompt_skips_models_that_cannot_fit_by_known_limit(self) -> None:
        small = _FakeBackend("qwen/qwen3-32b", ['{"bad": true}'])
        medium = _FakeBackend("openai/gpt-oss-20b", ['{"bad": true}'])
        large = _FakeBackend("llama-3.3-70b-versatile", ['{"ok": true}'])
        backend = FailoverBackend(provider="groq", backends=[small, medium, large])

        with patch.dict(os.environ, {"GROQ_MULTI_COMPLETION_TOKEN_RESERVE": "2500"}, clear=False):
            out = backend.chat_json("", "x" * 26000)

        self.assertEqual(out, '{"ok": true}')
        self.assertEqual(small.calls, 0)
        self.assertEqual(medium.calls, 0)
        self.assertEqual(large.calls, 1)

    def test_temporary_tpm_limit_retries_same_model_before_failover(self) -> None:
        err = RuntimeError(
            "Error code: 429 - rate_limit_exceeded on tokens per minute. "
            "Please try again in 0.1s."
        )
        primary = _FakeBackend("llama-3.3-70b-versatile", [err, '{"ok": true}'])
        fallback = _FakeBackend("openai/gpt-oss-20b", ['{"fallback": true}'])
        backend = FailoverBackend(provider="groq", backends=[primary, fallback])

        old_sleep = retry.time.sleep
        old_uniform = retry.random.uniform
        retry.time.sleep = lambda _seconds: None
        retry.random.uniform = lambda _start, _end: 0.0
        try:
            out = backend.chat_json("", "small prompt")
        finally:
            retry.time.sleep = old_sleep
            retry.random.uniform = old_uniform

        self.assertEqual(out, '{"ok": true}')
        self.assertEqual(primary.calls, 2)
        self.assertEqual(fallback.calls, 0)
        self.assertEqual(backend.failover_events(), [])

    def test_provider_confirmed_prompt_too_large_switches_model_without_retrying(self) -> None:
        too_large = RuntimeError(
            "Error code: 413 - Request too large for model `qwen/qwen3-32b`: "
            "Limit 6000, Requested 9203, please reduce your message size"
        )
        primary = _FakeBackend("qwen/qwen3-32b", [too_large])
        fallback = _FakeBackend("llama-3.3-70b-versatile", ['{"ok": true}'])
        backend = FailoverBackend(provider="groq", backends=[primary, fallback])

        out = backend.chat_json("", "small prompt")

        self.assertEqual(out, '{"ok": true}')
        self.assertEqual(primary.calls, 1)
        self.assertEqual(fallback.calls, 1)
        self.assertEqual(backend.failover_events()[0]["reason"], "prompt_too_large")


if __name__ == "__main__":
    unittest.main()
