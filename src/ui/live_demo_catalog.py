from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

try:
    import streamlit as st
except Exception:  # pragma: no cover - keeps helper importable outside Streamlit.
    st = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = PROJECT_ROOT / "data/processed/api_catalog_sample_balanced"
DEFAULT_CATALOG_PATHS = [
    CATALOG_DIR / "api_repo.enriched.jsonl",
    CATALOG_DIR / "api_repo.tooldesc.jsonl",
]


def _cache_data(func: Callable) -> Callable:
    if st is None:
        return func
    return st.cache_data(show_spinner=False)(func)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _endpoint_path(url_text: Any) -> str:
    text = _clean(url_text)
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        suffix = f"?{parsed.query}" if parsed.query else ""
        return (parsed.path or "/") + suffix
    return text


def _param_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _normalized_key(value: Any) -> str:
    return _clean(value).lower()


def normalize_catalog_entry(raw: dict[str, Any], source_path: str) -> dict[str, Any]:
    enrichment = raw.get("toolbench_enrichment") if isinstance(raw.get("toolbench_enrichment"), dict) else {}
    details = raw.get("endpoint_details") if isinstance(raw.get("endpoint_details"), dict) else {}
    endpoint_url = _first_present(raw.get("url"), enrichment.get("endpoint_url"), raw.get("endpoint_url"))
    endpoint_name = _first_present(enrichment.get("endpoint_name"), raw.get("name"), raw.get("endpoint_name"))
    tool_name = _first_present(raw.get("tool_name"), raw.get("toolbench_tool_name"), enrichment.get("tool_name"), raw.get("_tool"))
    short_description = _first_present(raw.get("description"), enrichment.get("endpoint_description"), raw.get("toolbench_endpoint_description"))
    full_description = _first_present(
        raw.get("tool_description"),
        raw.get("toolbench_tool_description"),
        enrichment.get("tool_description"),
        short_description,
    )
    api_id = _clean(raw.get("api_id") or raw.get("endpoint_id") or raw.get("id"))
    return {
        "api_id": api_id,
        "tool_name": _clean(tool_name),
        "api_name": _clean(endpoint_name),
        "endpoint_name": _clean(endpoint_name),
        "display_name": " / ".join(part for part in [_clean(tool_name), _clean(endpoint_name)] if part),
        "category": _clean(_first_present(raw.get("category"), enrichment.get("category"))),
        "http_method": _clean(_first_present(raw.get("method"), enrichment.get("endpoint_method"))),
        "endpoint_path": _endpoint_path(endpoint_url),
        "endpoint_url": _clean(endpoint_url),
        "short_description": _clean(short_description),
        "full_description": _clean(full_description),
        "required_parameters": _param_list(details.get("required_parameters") or raw.get("required_parameters")),
        "optional_parameters": _param_list(details.get("optional_parameters") or raw.get("optional_parameters")),
        "request_schema": _first_present(raw.get("request_schema"), raw.get("requestBody"), raw.get("input_schema")),
        "response_schema": _first_present(raw.get("response_schema"), raw.get("responses"), raw.get("output_schema")),
        "source_file": source_path,
        "catalog_file": _clean(raw.get("_file") or enrichment.get("file_name")),
        "toolbench_relative_path": _clean(enrichment.get("toolbench_relative_path")),
        "raw": raw,
    }


def _catalog_aliases(meta: dict[str, Any]) -> set[str]:
    aliases = {
        meta.get("api_id"),
        meta.get("endpoint_name"),
        meta.get("api_name"),
        meta.get("endpoint_url"),
        meta.get("endpoint_path"),
    }
    return {_normalized_key(alias) for alias in aliases if _clean(alias)}


@_cache_data
def load_api_catalog(path_text: str = "") -> dict[str, Any]:
    requested = [Path(path_text).expanduser()] if path_text else DEFAULT_CATALOG_PATHS
    warnings: list[str] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    loaded_sources: list[str] = []

    for raw_path in requested:
        path = raw_path if raw_path.is_absolute() else (PROJECT_ROOT / raw_path).resolve()
        if not path.exists():
            warnings.append(f"Catalog file not found: {path}")
            continue
        loaded_sources.append(str(path))
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            warnings.append(f"Could not read catalog file {path}: {exc}")
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            meta = normalize_catalog_entry(raw, str(path))
            api_id = meta.get("api_id")
            if not api_id or api_id in rows_by_id:
                continue
            rows_by_id[api_id] = meta
            for alias in _catalog_aliases(meta):
                aliases.setdefault(alias, api_id)
        if rows_by_id:
            break

    return {
        "by_id": rows_by_id,
        "aliases": aliases,
        "warnings": warnings,
        "sources": loaded_sources,
    }


def find_api_metadata(catalog: dict[str, Any], api_id_or_alias: Any) -> dict[str, Any] | None:
    text = _clean(api_id_or_alias)
    if not text:
        return None
    by_id = catalog.get("by_id") if isinstance(catalog.get("by_id"), dict) else {}
    if text in by_id:
        return by_id[text]
    alias_id = catalog.get("aliases", {}).get(_normalized_key(text)) if isinstance(catalog.get("aliases"), dict) else None
    if alias_id:
        return by_id.get(alias_id)
    return None
