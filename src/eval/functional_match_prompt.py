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
            "candidate_id": a.get("candidate_id"),
            "api_id": a.get("api_id"),
            "name": a.get("name"),
            "category": a.get("category"),
            "tool_name": a.get("tool_name"),
            "tool_description": a.get("tool_description"),
            "description": a.get("description"),
            "method": a.get("method"),
            "parameters": a.get("parameters", []),
        }
        compact_entries.append(entry)

    return (
        "You are evaluating whether APIs are functionally suitable for one subtask in an API-selection experiment.\n"
        "Your job is to judge whether each API is a good functional match for the subtask, not whether it is loosely related by topic words.\n"
        "Return ONLY one JSON object.\n"
        "Do not omit any API.\n"
        "Each candidate has candidate_id, the short ID used only for your output, and api_id, the real API identifier provided only as context.\n"
        "For every input candidate_id, you must return exactly one output item.\n"
        "Use candidate_id in your output. Do not output api_id.\n"
        "Do not change field names.\n"
        "Do not add markdown.\n"
        "Do not add explanation outside JSON.\n"
        "Judge only functional suitability for the subtask.\n"
        "Use endpoint name, description, method, compact parameter descriptions, and tool-level context when available.\n"
        "If the API belongs to the wrong domain or dataset, set relevant to 0 even if some keywords overlap.\n"
        f"Query ID: {query_id}\n"
        f"Main Task: {main_task}\n"
        f"Subtask ID: {subtask_id}\n"
        f"Subtask Description: {subtask_description}\n"
        "Output format: \n"
        "{\n"
        '  "results": [\n'
        '    {"candidate_id": "C01", "relevant": 0, "comment": "..."},\n'
        '    {"candidate_id": "C02", "relevant": 1, "comment": "..."}\n'
        "  ]\n"
        "}\n\n"
        "Important rules:\n"
        "- relevant must be 0 or 1\n"
        "- return one item for every candidate_id exactly once\n"
        "- do not output api_id\n"
        "- keep comments short\n"
        "- prioritize actual function and domain fit over keyword overlap\n"
        "- tool_description can reveal the true purpose of the API and should be used\n"
        "- parameters contain only compact name/description pairs and should be used as functional evidence\n"
        "- subject lists, topic lists, scripture topics, generic education content, or unrelated datasets are not course suggestion APIs unless they explicitly support retrieving or recommending real courses\n\n"
        f"APIs:\n{json.dumps(compact_entries, ensure_ascii=False)}"
    )
