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
        entry = {
            "api_id": a.get("api_id"),
            "name": a.get("name"),
            "category": a.get("category"),
            "tool_name": a.get("tool_name"),
            "tool_description": a.get("tool_description"),
            "description": a.get("description"),
            "method": a.get("method"),
            "url": a.get("url"),
        }
        if a.get("endpoint_details"):
            entry["endpoint_details"] = a.get("endpoint_details")
        compact_entries.append(entry)

    return (
        "You are evaluating whether APIs are functionally relevant to one subtask in an API-selection experiment.\n"
        "Your job is to judge whether each API is a good functional match for the subtask, not whether it is loosely related by topic words.\n"
        "Return ONLY one JSON object.\n"
        "Do not omit any API.\n"
        "For every input api_id, you must return exactly one output item.\n"
        "Use the same api_id string exactly as given.\n"
        "Do not change field names.\n"
        "Do not add markdown.\n"
        "Do not add explanation outside JSON.\n"
        "Judge only functional relevance to the subtask.\n"
        "Use both endpoint details and tool-level context when available.\n"
        "If the API belongs to the wrong domain or dataset, mark it not relevant even if some keywords overlap.\n"
        f"Query ID: {query_id}\n"
        f"Main Task: {main_task}\n"
        f"Subtask ID: {subtask_id}\n"
        f"Subtask Description: {subtask_description}"
        "Output format: \n"
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
        "- prioritize actual function and domain fit over keyword overlap\n"
        "- tool_description can reveal the true purpose of the API and should be used\n"
        "- subject lists, topic lists, scripture topics, generic education content, or unrelated datasets are not course suggestion APIs unless they explicitly support retrieving or recommending real courses\n\n"
        f"APIs:\n{json.dumps(compact_entries, ensure_ascii=False)}"
    )
