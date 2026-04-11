# src/agents/subtask_normalizer.py
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


def normalize_subtasks(
    llm_call: Callable[[str], str],
    user_goal: str,
    original_subtasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Normalize decomposed subtasks into concise API-retrieval-oriented subtasks.
    Falls back to the original subtasks if parsing fails.
    """
    with open("prompts/subtask_normalizer.md", "r", encoding="utf-8") as f:
        tmpl = f.read()

    original_json = json.dumps({"subtasks": original_subtasks}, ensure_ascii=False, indent=2)
    prompt = tmpl.replace("{user_goal}", user_goal).replace("{original_subtasks_json}", original_json)

    resp = llm_call(prompt)
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
        subtasks.append({"id": idx, "description": desc})

    if not subtasks:
        subtasks = [
            {"id": idx, "description": str(st.get("description", "")).strip()}
            for idx, st in enumerate(original_subtasks, start=1)
            if str(st.get("description", "")).strip()
        ]

    if not subtasks:
        subtasks = [{"id": 1, "description": user_goal.strip()}]

    return subtasks
