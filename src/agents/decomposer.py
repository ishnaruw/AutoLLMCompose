# src/agents/decomposer.py
import json
import re
from typing import Callable, Dict, List, Any


def _coerce_json(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "{}"
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else "{}"


def _parse_subtasks(resp: str, fallback_goal: str) -> List[Dict[str, Any]]:
    resp = _coerce_json(resp)
    try:
        data = json.loads(resp)
    except Exception:
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
