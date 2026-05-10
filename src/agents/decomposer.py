# src/agents/decomposer.py
from typing import Callable, Dict, List, Any

from src.core.json_parsing import parse_llm_json
from src.core.output_schemas import DecompositionOutput, validate_output_schema


def _parse_subtasks(resp: str, fallback_goal: str) -> List[Dict[str, Any]]:
    parsed = parse_llm_json(resp)
    data = parsed.value if isinstance(parsed.value, dict) and parsed.error is None else {}
    if data:
        _schema, schema_issue = validate_output_schema(DecompositionOutput, data)
        if schema_issue:
            data = {}

    subtasks_raw = data.get("subtasks", [])
    subtasks: List[Dict[str, Any]] = []

    for idx, st in enumerate(subtasks_raw, start=1):
        if not isinstance(st, dict):
            continue
        desc = (st.get("description") or st.get("goal") or "").strip()
        if not desc:
            continue
        subtasks.append(
            {
                "id": idx,
                "description": desc,
            }
        )

    if not subtasks:
        subtasks = [{"id": 1, "description": fallback_goal.strip()}]
    return subtasks


def decompose_goal(
    llm_call: Callable[[str], str],
    user_goal: str,
) -> List[Dict[str, Any]]:
    """
    Use an LLM to decompose the user goal into ordered user-facing subtasks.
    """
    with open("prompts/decomposer.md", "r", encoding="utf-8") as f:
        tmpl = f.read()

    prompt = tmpl.replace("{user_goal}", user_goal)
    resp = llm_call(prompt)
    return _parse_subtasks(resp, user_goal)
