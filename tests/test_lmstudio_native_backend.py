from __future__ import annotations

import json
import os
import unittest

from src.llm import backends


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class LMStudioNativeBackendTests(unittest.TestCase):
    def test_qwen_provider_uses_native_chat_payload(self) -> None:
        captured: dict = {}
        old_urlopen = backends.request.urlopen
        old_model = os.environ.get("LMSTUDIO_QWEN_MODEL")
        old_url = os.environ.get("LMSTUDIO_QWEN_CHAT_URL")
        os.environ["LMSTUDIO_QWEN_MODEL"] = "qwen2.5-3b-instruct.gguf"
        os.environ["LMSTUDIO_QWEN_CHAT_URL"] = "http://localhost:1234/api/v1/chat"

        def fake_urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
            captured["url"] = req.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeResponse({"response": '{"ok": true}'})

        backends.request.urlopen = fake_urlopen
        try:
            backend = backends.make_backend(provider="lmstudio_qwen")
            out = backend.chat_json(
                "System rules.",
                "User prompt.",
                temperature=0.0,
                force_json=True,
                max_tokens=2500,
                timeout_seconds=12,
            )
        finally:
            backends.request.urlopen = old_urlopen
            if old_model is None:
                os.environ.pop("LMSTUDIO_QWEN_MODEL", None)
            else:
                os.environ["LMSTUDIO_QWEN_MODEL"] = old_model
            if old_url is None:
                os.environ.pop("LMSTUDIO_QWEN_CHAT_URL", None)
            else:
                os.environ["LMSTUDIO_QWEN_CHAT_URL"] = old_url

        self.assertEqual(json.loads(out), {"ok": True})
        self.assertEqual(captured["url"], "http://localhost:1234/api/v1/chat")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(captured["payload"]["model"], "qwen2.5-3b-instruct.gguf")
        self.assertEqual(captured["payload"]["input"], "User prompt.")
        self.assertIn("System rules.", captured["payload"]["system_prompt"])
        self.assertNotIn("max_tokens", captured["payload"])


if __name__ == "__main__":
    unittest.main()
