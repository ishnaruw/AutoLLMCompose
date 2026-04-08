from __future__ import annotations

import json
from typing import Any, Dict, List


def build_llm_prompt(
    *,
    query_id: str,
    main_task: str,
    subtask_id: str,
    subtask_description: str,
    api_entries: List[Dict[str, Any]],
) -> str:
    compact_entries = []
    for a in api_entries:
        compact: Dict[str, Any] = {
            "api_id": a.get("api_id"),
            "name": a.get("name"),
            "category": a.get("category"),
            "description": a.get("description"),
            "method": a.get("method"),
            "url": a.get("url"),
        }
        extra = a.get("endpoint_details") or {}
        if extra:
            compact["endpoint_details"] = extra
        compact_entries.append(compact)

    return (
        "You are evaluating whether APIs are functionally relevant to one subtask.\n"
        "Return ONLY one JSON object.\n"
        "Do not omit any API.\n"
        "For every input api_id, you must return exactly one output item.\n"
        "Use the same api_id string exactly as given.\n"
        "Do not change field names.\n"
        "Do not add markdown.\n"
        "Do not add explanation outside JSON.\n"
        "Judge only functional relevance to the subtask. Ignore QoS.\n"
        "Use the API description and compact endpoint details when available.\n\n"
        f"Query ID: {query_id}\n"
        f"Main Task: {main_task}\n"
        f"Subtask ID: {subtask_id}\n"
        f"Subtask Description: {subtask_description}\n\n"
        "Output format:\n"
        "{\n"
        '  "results": [\n'
        '    {"api_id": "...", "relevant": 0, "comment": "..."},\n'
        '    {"api_id": "...", "relevant": 1, "comment": "..."}\n'
        "  ]\n"
        "}\n\n"
        "Important rules:\n"
        "- relevant must be 0 or 1\n"
        "- return one item for every api_id\n"
        "- keep comments short\n"
        "- judge only functional relevance\n\n"
        f"APIs:\n{json.dumps(compact_entries, ensure_ascii=False)}"
    )
