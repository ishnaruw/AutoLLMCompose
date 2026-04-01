from __future__ import annotations

import json
from typing import Any, Dict, List


def build_llm_prompt(
    *,
    query_id: str,
    main_task: str,
    subtask_id: str,
    subtask_description: str,
    expected_function: str,
    api_entries: List[Dict[str, Any]],
) -> str:
    compact_entries = []
    for a in api_entries:
        compact_entries.append(
            {
                "api_id": a.get("api_id"),
                "name": a.get("name"),
                "category": a.get("category"),
                "description": a.get("description"),
                "method": a.get("method"),
                "url": a.get("url"),
            }
        )

    schema = {
        "results": [
            {"api_id": "<string>", "relevant": 0, "comment": "<short reason>"}
        ]
    }

    return (
        "You are evaluating whether APIs are functionally relevant to a specific subtask.\n"
        "Be conservative. If an API cannot clearly help accomplish the subtask, mark it as 0.\n"
        "Use functional capability, not category similarity alone.\n"
        "Do not rank APIs. Do not infer QoS. Do not add extra text.\n"
        "Return strict JSON only.\n\n"
        f"Query ID: {query_id}\n"
        f"Main Task: {main_task}\n"
        f"Subtask ID: {subtask_id}\n"
        f"Subtask Description: {subtask_description}\n"
        f"Expected Function: {expected_function}\n\n"
        "For each API, output:\n"
        "- relevant: 1 if the API can realistically help accomplish the subtask\n"
        "- relevant: 0 otherwise\n"
        "- comment: one short reason\n\n"
        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"APIs:\n{json.dumps(compact_entries, ensure_ascii=False)}"
    )
