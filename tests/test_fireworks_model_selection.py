from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.config import CONFIG
from src.llm.backends import (
    _resolve_fireworks_model_name,
    fireworks_model_options,
    make_backend,
)


class _FakeBackend:
    def __init__(self, model_name: str, outcomes: list[object] | None = None) -> None:
        self.model_name = model_name
        self.outcomes = list(outcomes or ['{"ok": true}'])
        self.calls = 0

    def active_model_name(self) -> str:
        return self.model_name

    def model_pool(self) -> list[str]:
        return [self.model_name]

    def multi_model_mode(self) -> bool:
        return False

    def failover_events(self) -> list[dict]:
        return []

    def chat_json(self, **_kwargs: object) -> str:
        self.calls += 1
        if not self.outcomes:
            raise AssertionError(f"Unexpected call to {self.model_name}")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)


class FireworksModelSelectionTests(unittest.TestCase):
    def test_deepseek_v4_pro_alias_resolves_to_fireworks_model_path(self) -> None:
        self.assertEqual(
            _resolve_fireworks_model_name("deepseek-v4-pro"),
            "accounts/fireworks/models/deepseek-v4-pro",
        )

    def test_gpt_oss_120b_alias_resolves_to_fireworks_model_path(self) -> None:
        self.assertEqual(
            _resolve_fireworks_model_name("gpt-oss-120b"),
            "accounts/fireworks/models/gpt-oss-120b",
        )

    def test_deepseek_v32_alias_resolves_to_fireworks_model_path(self) -> None:
        self.assertEqual(
            _resolve_fireworks_model_name("deepseek-v3.2"),
            "accounts/fireworks/models/deepseek-v3p2",
        )

    def test_fireworks_options_include_gpt_oss_120b(self) -> None:
        old_models = os.environ.get("FIREWORKS_MODELS")
        old_model = os.environ.get("FIREWORKS_MODEL")
        try:
            os.environ.pop("FIREWORKS_MODELS", None)
            os.environ.pop("FIREWORKS_MODEL", None)

            self.assertIn("accounts/fireworks/models/gpt-oss-120b", fireworks_model_options())
        finally:
            if old_models is None:
                os.environ.pop("FIREWORKS_MODELS", None)
            else:
                os.environ["FIREWORKS_MODELS"] = old_models
            if old_model is None:
                os.environ.pop("FIREWORKS_MODEL", None)
            else:
                os.environ["FIREWORKS_MODEL"] = old_model

    def test_fireworks_default_first_choice_is_gpt_oss_120b(self) -> None:
        old_models = os.environ.get("FIREWORKS_MODELS")
        old_model = os.environ.get("FIREWORKS_MODEL")
        try:
            os.environ.pop("FIREWORKS_MODELS", None)
            os.environ.pop("FIREWORKS_MODEL", None)

            self.assertEqual(
                fireworks_model_options()[0],
                "accounts/fireworks/models/gpt-oss-120b",
            )
        finally:
            if old_models is None:
                os.environ.pop("FIREWORKS_MODELS", None)
            else:
                os.environ["FIREWORKS_MODELS"] = old_models
            if old_model is None:
                os.environ.pop("FIREWORKS_MODEL", None)
            else:
                os.environ["FIREWORKS_MODEL"] = old_model

    def test_fireworks_defaults_do_not_include_known_unavailable_deepseek_v32(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("accounts/fireworks/models/deepseek-v3p2", fireworks_model_options())
            self.assertNotIn("accounts/fireworks/models/deepseek-v3p1", fireworks_model_options())
            self.assertNotIn("accounts/fireworks/models/llama-v3p1-8b-instruct", fireworks_model_options())

    def test_make_backend_uses_only_explicit_fireworks_model(self) -> None:
        with patch.dict(
            os.environ,
            {"FIREWORKS_MODELS": "accounts/fireworks/models/gpt-oss-120b"},
            clear=True,
        ), patch(
            "src.llm.backends.FireworksBackend",
            side_effect=lambda model: _FakeBackend(_resolve_fireworks_model_name(model)),
        ):
            backend = make_backend(provider="fireworks", model="deepseek-v3p2")

        self.assertFalse(backend.multi_model_mode())
        self.assertEqual(backend.active_model_name(), "accounts/fireworks/models/deepseek-v3p2")
        self.assertEqual(backend.model_pool(), ["accounts/fireworks/models/deepseek-v3p2"])
        self.assertEqual(backend.failover_events(), [])

    def test_large_model_qos_batch_default_sends_all_candidates(self) -> None:
        self.assertEqual(CONFIG.qos_llm_batch_size, 0)


if __name__ == "__main__":
    unittest.main()
