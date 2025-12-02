# src/tools/fetch_services.py
from pathlib import Path
import json
from typing import Optional, List, Dict, Any

# point to your API catalogs

CATALOG_WITH_QOS = Path("data/processed/api_catalog_sample_balanced/api_repo.with_qos.jsonl")
CATALOG_NO_QOS = Path("data/processed/api_catalog_sample_balanced/api_repo.no_qos.jsonl")


def iter_jsonl(path: Path):
    """Yield JSON objects line by line from a .jsonl file."""
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
    """
    Fetch services from the API catalog.

    Behavior:
    - If category is None: return services from *all* categories.
    - If category is a string: return only services whose record["category"] == category.
    - Supports pagination with offset & limit.
    """
    path = CATALOG_WITH_QOS if with_qos else CATALOG_NO_QOS

    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path.resolve()}")

    # If category is None → do not filter
    if category is None:
        items = list(iter_jsonl(path))
    else:
        items = [r for r in iter_jsonl(path) if r.get("category") == category]

    # Pagination
    return items[offset : offset + limit]
