# src/tools/fetch_services.py
from pathlib import Path
import json
from typing import Optional, List, Dict, Any

from src.config import CONFIG

CATALOG_WITH_QOS_TOOLDESC = Path("data/processed/api_catalog_sample_balanced/misc/api_repo.with_qos.tooldesc.jsonl")
CATALOG_NO_QOS_TOOLDESC = Path("data/processed/api_catalog_sample_balanced/misc/api_repo.no_qos.tooldesc.jsonl")
CATALOG_WITH_QOS = Path("data/processed/api_catalog_sample_balanced/misc/deprecated_api_repo.with_qos.jsonl")
CATALOG_NO_QOS = Path("data/processed/api_catalog_sample_balanced/misc/deprecated_api_repo.no_qos.jsonl")


def catalog_path(with_qos: bool, *, prefer_enriched: bool = True) -> Path:
    candidates = []
    if prefer_enriched:
        candidates.append(CONFIG.catalog_enriched_path)
    candidates.extend([CONFIG.catalog_path, CONFIG.catalog_no_qos_path, CATALOG_NO_QOS_TOOLDESC, CATALOG_NO_QOS])

    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _legacy_qos_by_id() -> Dict[str, Dict[str, Any]]:
    path = CONFIG.catalog_with_qos_path if CONFIG.catalog_with_qos_path.exists() else CATALOG_WITH_QOS_TOOLDESC
    if not path.exists():
        path = CATALOG_WITH_QOS
    if not path.exists():
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        api_id = str(row.get("api_id") or "").strip()
        qos = row.get("qos")
        if api_id and isinstance(qos, dict):
            out[api_id] = dict(qos)
    return out


def load_qos_by_id(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    path = path or CONFIG.api_qos_path
    if not path.exists():
        return _legacy_qos_by_id()

    out: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        api_id = str(row.get("api_id") or "").strip()
        qos = row.get("qos")
        if api_id and isinstance(qos, dict):
            out[api_id] = dict(qos)
    return out


def load_catalog_records(*, with_qos: bool = False, prefer_enriched: bool = True) -> List[Dict[str, Any]]:
    path = catalog_path(with_qos=False, prefer_enriched=prefer_enriched)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path.resolve()}")

    items = [dict(row) for row in iter_jsonl(path)]
    if not with_qos:
        for item in items:
            item.pop("qos", None)
        return items

    qos_by_id = load_qos_by_id()
    for item in items:
        api_id = str(item.get("api_id") or "").strip()
        if api_id in qos_by_id:
            item["qos"] = qos_by_id[api_id]
    return items


def load_catalog_map(*, with_qos: bool = False, prefer_enriched: bool = True) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("api_id")): item
        for item in load_catalog_records(with_qos=with_qos, prefer_enriched=prefer_enriched)
        if str(item.get("api_id") or "").strip()
    }


def fetch_services(
    category: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
    with_qos: bool = True,
) -> List[Dict[str, Any]]:
    items = load_catalog_records(with_qos=with_qos)
    if category is not None:
        items = [r for r in items if r.get("category") == category]

    return items[offset : offset + limit]


def compress_service(s: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "api_id": s["api_id"],
        "name": s.get("name", ""),
        "description": (s.get("toolbench_endpoint_description") or s.get("description", "") or "")[:200],
        "category": s.get("category"),
        "method": s.get("method", ""),
        "url": s.get("url", ""),
        "tool_name": (s.get("toolbench_tool_name") or s.get("tool_name", "") or "")[:120],
        "tool_description": (s.get("toolbench_tool_description") or s.get("tool_description", "") or "")[:300],
    }
