from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.config import CONFIG
from src.core.api_formatting import _match_endpoint

DEFAULT_TOOLBENCH_TOOLS_ROOT = Path(
    os.getenv(
        "TOOLBENCH_TOOLS_ROOT",
        "/Users/ishwaryapns/Documents/Thesis/ToolBench/data/toolenv/tools",
    )
)
ENRICHMENT_SCHEMA_VERSION = 1


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def _count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _catalog_locator(record: Dict[str, Any]) -> Tuple[str, str]:
    compressed = _as_dict(record.get("compressed"))
    service = _as_dict(record.get("service"))
    category = _first_text(compressed.get("category"), service.get("category"), record.get("category"))
    file_name = _first_text(
        service.get("_file"),
        record.get("_file"),
        compressed.get("_file"),
        service.get("file_name"),
        record.get("file_name"),
        compressed.get("file_name"),
    )
    return category, file_name


def _compact_parameters(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        compact: Dict[str, Any] = {}
        for key in ("name", "description", "type", "default"):
            if item.get(key) is not None:
                compact[key] = item.get(key)
        if compact.get("name"):
            out.append(compact)
    return out


def _load_tool_json(
    *,
    toolbench_root: Path,
    category: str,
    file_name: str,
    cache: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    key = (category, file_name)
    if key in cache:
        return cache[key]

    path = toolbench_root / category / file_name
    if not path.exists():
        cache[key] = {}
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        cache[key] = {}
        return {}

    cache[key] = data if isinstance(data, dict) else {}
    return cache[key]


def enrich_record(
    record: Dict[str, Any],
    *,
    toolbench_root: Path,
    cache: Dict[Tuple[str, str], Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    row = dict(record)
    row.pop("qos", None)
    cache = cache if cache is not None else {}
    category, file_name = _catalog_locator(row)
    enrichment: Dict[str, Any] = {
        "schema_version": ENRICHMENT_SCHEMA_VERSION,
        "source": "toolbench_toolenv",
        "category": category,
        "file_name": file_name,
        "toolbench_relative_path": f"{category}/{file_name}" if category and file_name else "",
        "tool_file_found": False,
        "endpoint_found": False,
        "status": "missing_category_or_file",
    }

    if not category or not file_name:
        row["toolbench_enrichment"] = enrichment
        return row

    tool_json = _load_tool_json(
        toolbench_root=toolbench_root,
        category=category,
        file_name=file_name,
        cache=cache,
    )
    if not tool_json:
        enrichment["status"] = "tool_file_missing_or_invalid"
        row["toolbench_enrichment"] = enrichment
        return row

    enrichment["tool_file_found"] = True
    tool_name = _first_text(tool_json.get("tool_name"), tool_json.get("name"), tool_json.get("title"))
    tool_description = _first_text(tool_json.get("tool_description"))
    if tool_name:
        row["toolbench_tool_name"] = tool_name
        enrichment["tool_name"] = tool_name
    if tool_description:
        row["toolbench_tool_description"] = tool_description
        enrichment["tool_description"] = tool_description

    compressed = _as_dict(row.get("compressed"))
    service = _as_dict(row.get("service"))
    endpoint = _match_endpoint(tool_json, row, compressed, service)
    if not endpoint:
        enrichment["status"] = "endpoint_not_found"
        row["toolbench_enrichment"] = enrichment
        return row

    required_parameters = _compact_parameters(endpoint.get("required_parameters"))
    optional_parameters = _compact_parameters(endpoint.get("optional_parameters"))
    endpoint_description = _first_text(endpoint.get("description"))
    endpoint_details = {
        "required_parameters": required_parameters,
        "optional_parameters": optional_parameters,
    }

    row["toolbench_endpoint_description"] = endpoint_description
    row["endpoint_details"] = endpoint_details
    enrichment.update(
        {
            "endpoint_found": True,
            "endpoint_name": _first_text(endpoint.get("name")),
            "endpoint_method": _first_text(endpoint.get("method")),
            "endpoint_url": _first_text(endpoint.get("url")),
            "endpoint_description": endpoint_description,
            "required_parameter_count": len(required_parameters),
            "optional_parameter_count": len(optional_parameters),
            "status": "matched",
        }
    )
    row["toolbench_enrichment"] = enrichment
    return row


def build_enriched_catalog(input_path: Path, output_path: Path, *, toolbench_root: Path) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input catalog not found: {input_path}")

    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    tool_file_found = 0
    endpoint_found = 0

    def records() -> Iterable[Dict[str, Any]]:
        nonlocal tool_file_found, endpoint_found
        for record in iter_jsonl(input_path):
            enriched = enrich_record(record, toolbench_root=toolbench_root, cache=cache)
            info = _as_dict(enriched.get("toolbench_enrichment"))
            status_counts[str(info.get("status") or "unknown")] += 1
            if info.get("tool_file_found"):
                tool_file_found += 1
            if info.get("endpoint_found"):
                endpoint_found += 1
            yield enriched

    record_count = write_jsonl(output_path, records())
    return {
        "input": str(input_path),
        "output": str(output_path),
        "records": record_count,
        "tool_file_found": tool_file_found,
        "endpoint_found": endpoint_found,
        "status_counts": dict(sorted(status_counts.items())),
    }


def build_functional_catalog(input_path: Path, output_path: Path) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input catalog not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        return {
            "input": str(input_path),
            "output": str(output_path),
            "records": _count_jsonl(input_path),
            "reused_existing": True,
        }

    def records() -> Iterable[Dict[str, Any]]:
        for record in iter_jsonl(input_path):
            row = dict(record)
            row.pop("qos", None)
            row.pop("toolbench_enrichment", None)
            row.pop("toolbench_tool_name", None)
            row.pop("toolbench_tool_description", None)
            row.pop("toolbench_endpoint_description", None)
            yield row

    record_count = write_jsonl(output_path, records())
    return {
        "input": str(input_path),
        "output": str(output_path),
        "records": record_count,
    }


def build_qos_overlay(input_path: Path, output_path: Path) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"QoS input catalog not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        qos_count = sum(
            1
            for record in iter_jsonl(input_path)
            if str(record.get("api_id") or "").strip() and isinstance(record.get("qos"), dict)
        )
        return {
            "input": str(input_path),
            "output": str(output_path),
            "records": qos_count,
            "qos_records": qos_count,
            "reused_existing": True,
        }

    qos_count = 0

    def records() -> Iterable[Dict[str, Any]]:
        nonlocal qos_count
        for record in iter_jsonl(input_path):
            api_id = str(record.get("api_id") or "").strip()
            qos = record.get("qos")
            if not api_id or not isinstance(qos, dict):
                continue
            qos_count += 1
            yield {"api_id": api_id, "qos": qos}

    record_count = write_jsonl(output_path, records())
    return {
        "input": str(input_path),
        "output": str(output_path),
        "records": record_count,
        "qos_records": qos_count,
    }


def build_all(
    *,
    toolbench_root: Path,
    catalog_input: Path,
    qos_input: Path,
    catalog_output: Path,
    enriched_output: Path,
    qos_output: Path,
    manifest_output: Path,
) -> Dict[str, Any]:
    functional_catalog = build_functional_catalog(catalog_input, catalog_output)
    enriched_catalog = build_enriched_catalog(catalog_output, enriched_output, toolbench_root=toolbench_root)
    qos_overlay = build_qos_overlay(qos_input, qos_output)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": ENRICHMENT_SCHEMA_VERSION,
        "toolbench_root": str(toolbench_root),
        "functional_catalog": functional_catalog,
        "enriched_catalog": enriched_catalog,
        "qos_overlay": qos_overlay,
        "runtime_mapping": {
            "with_qos_false": "load enriched_catalog only",
            "with_qos_true": "load enriched_catalog and merge qos_overlay by api_id",
        },
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact ToolBench-enriched AutoLLMCompose catalog snapshots.")
    parser.add_argument("--toolbench-root", type=Path, default=DEFAULT_TOOLBENCH_TOOLS_ROOT)
    parser.add_argument(
        "--catalog-input",
        type=Path,
        default=CONFIG.catalog_path,
    )
    parser.add_argument(
        "--qos-input",
        type=Path,
        default=CONFIG.api_qos_path,
    )
    parser.add_argument("--catalog-output", type=Path, default=CONFIG.catalog_path)
    parser.add_argument("--enriched-output", type=Path, default=CONFIG.catalog_enriched_path)
    parser.add_argument("--qos-output", type=Path, default=CONFIG.api_qos_path)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/processed/api_catalog_sample_balanced/enrichment_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_all(
        toolbench_root=args.toolbench_root,
        catalog_input=args.catalog_input,
        qos_input=args.qos_input,
        catalog_output=args.catalog_output,
        enriched_output=args.enriched_output,
        qos_output=args.qos_output,
        manifest_output=args.manifest_output,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
