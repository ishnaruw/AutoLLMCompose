from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src.core.api_formatting import normalize_api_for_ranking
from src.tools import fetch_services
from src.tools.build_enriched_catalog import build_enriched_catalog


class ApiEnrichmentTests(unittest.TestCase):
    def test_normalizer_uses_precomputed_enrichment_without_toolbench_lookup(self) -> None:
        api = {
            "api_id": "weather_forecast",
            "name": "Catalog Weather",
            "category": "Weather",
            "description": "Short catalog description",
            "method": "GET",
            "toolbench_tool_name": "ToolBench Weather",
            "toolbench_tool_description": "Full ToolBench weather service description",
            "toolbench_endpoint_description": "Get a forecast for a location",
            "endpoint_details": {
                "required_parameters": [
                    {"name": "location", "description": "City name or latitude and longitude"}
                ],
                "optional_parameters": [],
            },
            "toolbench_enrichment": {
                "status": "matched",
                "tool_file_found": True,
                "endpoint_found": True,
            },
        }

        with patch("src.core.api_formatting._load_tool_json", side_effect=AssertionError("unexpected lookup")):
            normalized = normalize_api_for_ranking(api, subtask_text="show weather forecast")

        self.assertEqual(normalized["tool_name"], "ToolBench Weather")
        self.assertEqual(normalized["tool_description"], "Full ToolBench weather service description")
        self.assertEqual(normalized["description"], "Get a forecast for a location")
        self.assertEqual(
            normalized["parameters"],
            [{"name": "location", "description": "City name or latitude and longitude"}],
        )

    def test_builder_materializes_only_compact_toolbench_endpoint_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tool_dir = root / "Weather"
            tool_dir.mkdir(parents=True)
            (tool_dir / "weather_tool.json").write_text(
                json.dumps(
                    {
                        "tool_name": "Weather Tool",
                        "tool_description": "Detailed weather API collection",
                        "api_list": [
                            {
                                "name": "Forecast",
                                "url": "https://example.test/forecast",
                                "method": "GET",
                                "description": "Get weather forecast data",
                                "required_parameters": [
                                    {"name": "city", "description": "City to forecast", "type": "STRING"}
                                ],
                                "optional_parameters": [
                                    {"name": "units", "description": "Metric or imperial", "type": "STRING"}
                                ],
                                "code": "large generated sample should not be copied",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            input_path = root / "catalog.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "api_id": "weather_tool_forecast",
                        "category": "Weather",
                        "_file": "weather_tool.json",
                        "name": "Forecast",
                        "method": "GET",
                        "url": "https://example.test/forecast",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "enriched.jsonl"

            summary = build_enriched_catalog(input_path, output_path, toolbench_root=root)

            self.assertEqual(summary["records"], 1)
            self.assertEqual(summary["endpoint_found"], 1)
            row = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["toolbench_enrichment"]["status"], "matched")
            self.assertEqual(row["toolbench_endpoint_description"], "Get weather forecast data")
            self.assertEqual(row["endpoint_details"]["required_parameters"][0]["name"], "city")
            self.assertNotIn("code", json.dumps(row))

    def test_catalog_loader_merges_qos_overlay_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            enriched_path = root / "api_repo.enriched.jsonl"
            base_path = root / "api_repo.tooldesc.jsonl"
            qos_path = root / "api_qos.jsonl"
            rows = [
                {"api_id": "api_a", "category": "Weather", "description": "Forecast"},
                {"api_id": "api_b", "category": "Weather", "description": "Alerts"},
            ]
            enriched_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            base_path.write_text(enriched_path.read_text(encoding="utf-8"), encoding="utf-8")
            qos_path.write_text(
                json.dumps({"api_id": "api_a", "qos": {"rt_ms": 1.2, "tp_rps": 3.4, "availability": 0.99}})
                + "\n",
                encoding="utf-8",
            )
            config = SimpleNamespace(
                catalog_enriched_path=enriched_path,
                catalog_path=base_path,
                catalog_no_qos_path=base_path,
                catalog_with_qos_path=root / "missing_with_qos.jsonl",
                api_qos_path=qos_path,
            )

            with patch.object(fetch_services, "CONFIG", config):
                without_qos = fetch_services.load_catalog_records(with_qos=False)
                with_qos = fetch_services.load_catalog_records(with_qos=True)

        self.assertNotIn("qos", without_qos[0])
        self.assertEqual(with_qos[0]["qos"]["availability"], 0.99)
        self.assertNotIn("qos", with_qos[1])


if __name__ == "__main__":
    unittest.main()
