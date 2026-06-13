# src/tools/fetch_services.py
from pathlib import Path
import json
from typing import Optional, List, Dict, Any

from src.config import CONFIG


def catalog_path(with_qos: bool = False, *, prefer_enriched: bool = True) -> Path:
    """Return the canonical functional catalog path.

    ``with_qos`` is retained for callers that ask for a no-QoS view, but QoS is
    now stored in CONFIG.api_qos_path and merged by load_catalog_records().
    """
    candidates = []
    if prefer_enriched:
        candidates.append(CONFIG.catalog_enriched_path)
    candidates.append(CONFIG.catalog_path)

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


def load_qos_by_id(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    path = path or CONFIG.api_qos_path
    if not path.exists():
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        api_id = str(row.get("api_id") or "").strip()
        qos = row.get("qos")
        if api_id and isinstance(qos, dict):
            out[api_id] = dict(qos)
    return out


def load_catalog_records(*, with_qos: bool = False, prefer_enriched: bool = True) -> List[Dict[str, Any]]:
    path = catalog_path(prefer_enriched=prefer_enriched)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path.resolve()}")

    items = [dict(row) for row in iter_jsonl(path)]
    if not with_qos:
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
