# src/agents/decomposer.py
import json
import re
from typing import Any, Callable, Dict, List


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


def decompose_goal(
    llm_call: Callable[[str], str],
    user_goal: str,
) -> List[Dict[str, Any]]:
    """
    Use an LLM to decompose the user goal into subtasks.

    Expects JSON:
    {
      "subtasks": [
        {"id": 1, "description": "..."},
        ...
      ]
    }

    Falls back to a single subtask equal to the original goal if parsing fails.
    """
    with open("prompts/decomposer.md", "r", encoding="utf-8") as f:
        tmpl = f.read()

    prompt = tmpl.replace("{user_goal}", user_goal)

    resp = llm_call(prompt)
    resp = _coerce_json(resp)
    data = json.loads(resp)

    subtasks_raw = data.get("subtasks", [])
    subtasks: List[Dict[str, Any]] = []

    for idx, st in enumerate(subtasks_raw, start=1):
        desc = (st.get("description") or st.get("goal") or "").strip()
        if not desc:
            continue
        subtasks.append(
            {
                "id": st.get("id", idx),
                "description": desc,
            }
        )

    if not subtasks:
        subtasks = [
            {
                "id": 1,
                "description": user_goal.strip(),
            }
        ]

    return subtasks
