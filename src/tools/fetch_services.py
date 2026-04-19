from pathlib import Path
import json
from typing import Optional, List, Dict, Any

CATALOG_WITH_QOS_TOOLDESC = Path("data/processed/api_catalog_sample_balanced/api_repo.with_qos.tooldesc.jsonl")
CATALOG_NO_QOS_TOOLDESC = Path("data/processed/api_catalog_sample_balanced/api_repo.no_qos.tooldesc.jsonl")
CATALOG_WITH_QOS = Path("data/processed/api_catalog_sample_balanced/api_repo.with_qos.jsonl")
CATALOG_NO_QOS = Path("data/processed/api_catalog_sample_balanced/api_repo.no_qos.jsonl")


def _catalog_path(with_qos: bool) -> Path:
    preferred = CATALOG_WITH_QOS_TOOLDESC if with_qos else CATALOG_NO_QOS_TOOLDESC
    fallback = CATALOG_WITH_QOS if with_qos else CATALOG_NO_QOS
    return preferred if preferred.exists() else fallback


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def fetch_services(
    category: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
    with_qos: bool = True,
) -> List[Dict[str, Any]]:
    path = _catalog_path(with_qos)

    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path.resolve()}")

    if category is None:
        items = list(iter_jsonl(path))
    else:
        items = [r for r in iter_jsonl(path) if r.get("category") == category]

    return items[offset : offset + limit]


def load_catalog_map(with_qos: bool) -> Dict[str, Dict[str, Any]]:
    path = _catalog_path(with_qos)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path.resolve()}")
    out: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        api_id = str(row.get("api_id", "")).strip()
        if api_id:
            out[api_id] = row
    return out


def compress_service(s: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "api_id": s["api_id"],
        "name": s.get("name", ""),
        "description": (s.get("description", "") or "")[:200],
        "category": s.get("category"),
        "method": s.get("method", ""),
        "url": s.get("url", ""),
        "tool_name": (s.get("tool_name", "") or "")[:120],
        "tool_description": (s.get("tool_description", "") or "")[:300],
    }
