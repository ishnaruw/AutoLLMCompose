"""Goal decomposition agent helpers."""

import re
from typing import Callable, Dict, List, Any

from src.core.json_parsing import parse_llm_json
from src.core.output_schemas import DecompositionOutput, validate_output_schema


_INTERNAL_ONLY_PATTERNS = (
    r"\baggregate\b",
    r"\bcombine\b",
    r"\bcompare\b",
    r"\bcompose\b",
    r"\bformat\b",
    r"\brank\b",
    r"\branking\b",
    r"\bsort\b",
    r"\bpresent\b",
    r"\bcapture selected\b",
    r"\bcapture user\b",
    r"\bselect from\b",
    r"\bdecide\b",
    r"\bdecision\b",
    r"\bdetermine whether\b",
    r"\bupdate dashboard\b",
    r"\bdashboard update\b",
)

_ALWAYS_INTERNAL_PATTERNS = (
    r"\bfetch (?:the )?list of domains to monitor\b",
    r"\bretrieve (?:the )?list of domains to monitor\b",
    r"\bget (?:the )?list of domains to monitor\b",
    r"\bconfiguration or inventory api\b",
    r"\bdownstream blocking service\b",
    r"\baggregated scan results\b",
    r"\breturn(?:ing)? (?:the )?scan results\b",
)

_EXPLICIT_API_BACKED_TERMS = (
    " api",
    "apis",
    "endpoint",
    "external service",
    "web service",
    "using an llm",
    "using a llm",
    "using an ai",
    "using a model",
    "text-summarization",
    "summarization api",
    "generation api",
)


def _looks_internal_subtask(description: str) -> bool:
    """Return True for local workflow work that should not stand alone."""
    text = re.sub(r"\s+", " ", str(description or "").strip().lower())
    if not text:
        return False
    if any(re.search(pattern, text) for pattern in _ALWAYS_INTERNAL_PATTERNS):
        return True
    if any(term in text for term in _EXPLICIT_API_BACKED_TERMS):
        return False
    return any(re.search(pattern, text) for pattern in _INTERNAL_ONLY_PATTERNS)


def _renumber_subtasks(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            **subtask,
            "id": idx,
        }
        for idx, subtask in enumerate(subtasks, start=1)
    ]


def _fold_internal_subtasks(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold obvious local-only subtasks into the previous API-backed subtask."""
    folded: List[Dict[str, Any]] = []
    for subtask in subtasks:
        desc = str(subtask.get("description") or "").strip()
        if not desc:
            continue
        if _looks_internal_subtask(desc):
            if not folded:
                continue
            previous = str(folded[-1].get("description") or "").strip()
            folded[-1]["description"] = f"{previous}; include local workflow step: {desc}"
            continue
        folded.append({"id": len(folded) + 1, "description": desc})
    return _renumber_subtasks(folded or subtasks)


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
    return _fold_internal_subtasks(subtasks)


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
