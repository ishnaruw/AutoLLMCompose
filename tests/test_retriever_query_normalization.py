from __future__ import annotations

import unittest

from src.agents.retriever import _normalize_retrieval_query


class RetrieverQueryNormalizationTests(unittest.TestCase):
    def test_boosts_email_delivery_queries(self) -> None:
        query = "Send the curated restaurant recommendations via email using an email delivery API"

        normalized = _normalize_retrieval_query(query)

        self.assertTrue(normalized.startswith("send email email delivery api email notification api"))
        self.assertIn(query, normalized)

    def test_preserves_multichannel_notification_queries(self) -> None:
        query = "Send an alert (e.g., email, SMS, push) through a notification API"

        normalized = _normalize_retrieval_query(query)

        self.assertEqual(normalized, query)


if __name__ == "__main__":
    unittest.main()
