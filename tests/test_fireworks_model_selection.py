from __future__ import annotations

import os
import unittest

from src.config import CONFIG
from src.llm.backends import _resolve_fireworks_model_name, fireworks_model_options


class FireworksModelSelectionTests(unittest.TestCase):
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

    def test_large_model_qos_batch_default_sends_all_candidates(self) -> None:
        self.assertEqual(CONFIG.qos_llm_batch_size, 0)


if __name__ == "__main__":
    unittest.main()
