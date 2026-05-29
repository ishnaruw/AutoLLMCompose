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
    r"\blog\b",
    r"\blogging\b",
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
    r"\bfinal recommendation\b",
    r"\brecommendation\b",
    r"\bupdate dashboard\b",
    r"\bdashboard update\b",
)

_ALWAYS_INTERNAL_PATTERNS = (
    r"\brecord (?:the )?blocking decision\b",
    r"\blocal decision\b",
    r"\bblocking decision\b",
    r"\bdecide whether to block\b",
    r"\bfinal decision\b",
    r"\brecord decision\b",
    r"\brecord (?:the )?.*decision\b",
    r"\brecord(?:ing)? (?:the )?.*decision\b",
    r"\blog (?:the )?.*decision\b",
    r"\b(?:create|make|produce|generate) (?:the )?(?:final )?decision\b",
    r"\bsecurity[- ]policy update\b",
    r"\bupdate policy\b",
    r"\b(?:security[- ]?)?policy (?:update|decision)\b",
    r"\bupdate (?:the )?(?:security[- ]?)?policy\b",
    r"\b(?:final )?recommendation\b",
    r"\bfetch (?:the )?list of domains to monitor\b",
    r"\bretrieve (?:the )?list of domains to monitor\b",
    r"\bget (?:the )?list of domains to monitor\b",
    r"\bconfiguration or inventory api\b",
    r"\bdownstream blocking service\b",
    r"\baggregated scan results\b",
    r"\breturn(?:ing)? (?:the )?scan results\b",
    r"\bschedule\b.*\bworkflow\b",
    r"\bworkflow\b.*\bschedule\b",
    r"\brun\b.*\bdaily\b",
    r"\bdaily\b.*\bschedule\b",
    r"\bcron\b.*\bworkflow\b",
    r"\brecurring\b.*\bworkflow\b",
    r"\b(?:check|compare)\b.*\b(?:fetched|current|retrieved)\b.*\bprices?\b.*\b(?:stored|baseline|previous|previously recorded|historical|recorded)\b",
    r"\b(?:check|compare)\b.*\b(?:stored|baseline|previous|previously recorded|historical|recorded)\b.*\b(?:fetched|current|retrieved)\b.*\bprices?\b",
    r"\b(?:detect|identify)\b.*\bprice drops?\b.*\b(?:stored|baseline|previous|previously recorded|recorded|threshold)\b",
    r"\b(?:stored|baseline|previous|previously recorded|recorded|threshold)\b.*\b(?:detect|identify)\b.*\bprice drops?\b",
)

_DROP_INTERNAL_PATTERNS = (
    r"\brecord (?:the )?blocking decision\b",
    r"\bblocking decision\b",
    r"\blocal decision\b",
    r"\bfinal decision\b",
    r"\bdecide whether to block\b",
    r"\brecord decision\b",
    r"\brecord (?:the )?.*decision\b",
    r"\blog (?:the )?.*decision\b",
    r"\bsecurity[- ]policy update\b",
    r"\bupdate policy\b",
    r"\b(?:security[- ]?)?policy (?:update|decision)\b",
    r"\bupdate (?:the )?(?:security[- ]?)?policy\b",
    r"\bdownstream blocking service\b",
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


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _explicit_goal_external_action(goal: str, description: str) -> bool:
    """Return True when the user explicitly asks an internal-looking action to call an external API."""
    goal_text = _normalize_text(goal)
    desc_text = _normalize_text(description)
    if not goal_text:
        return False

    external_api = (
        r"(?:api|apis|endpoint|external service|web service|"
        r"policy[- ]management api|security[- ]policy api)"
    )
    action_terms = [
        "aggregate",
        "score",
        "scoring",
        "log",
        "logging",
        "format",
        "recommend",
        "recommendation",
        "record",
        "decision",
        "policy",
    ]
    relevant_actions = [term for term in action_terms if term in desc_text]
    if not relevant_actions:
        return False

    action_alt = "|".join(re.escape(term) for term in relevant_actions)
    return bool(
        re.search(rf"\b(?:call|use|using|via|through|with)\b[^.]*\b{external_api}\b[^.]*\b(?:{action_alt})\b", goal_text)
        or re.search(rf"\b(?:{action_alt})\b[^.]*\b(?:call|use|using|via|through|with)\b[^.]*\b{external_api}\b", goal_text)
    )


def _should_drop_internal_subtask(description: str, *, user_goal: str = "") -> bool:
    """Return True when an internal subtask should be removed from retrieval text entirely."""
    text = _normalize_text(description)
    if not text:
        return False
    if _explicit_goal_external_action(user_goal, description):
        return False
    return any(re.search(pattern, text) for pattern in _DROP_INTERNAL_PATTERNS)


def _looks_internal_subtask(description: str, *, user_goal: str = "") -> bool:
    """Return True for local workflow work that should not stand alone."""
    text = _normalize_text(description)
    if not text:
        return False
    if _explicit_goal_external_action(user_goal, description):
        return False
    if _should_drop_internal_subtask(description, user_goal=user_goal):
        return True
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


def _fold_internal_subtasks(subtasks: List[Dict[str, Any]], *, user_goal: str = "") -> List[Dict[str, Any]]:
    """Fold or drop local-only subtasks before retrieval."""
    folded: List[Dict[str, Any]] = []
    for subtask in subtasks:
        desc = str(subtask.get("description") or "").strip()
        if not desc:
            continue
        if _should_drop_internal_subtask(desc, user_goal=user_goal):
            continue
        if _looks_internal_subtask(desc, user_goal=user_goal):
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
    return _fold_internal_subtasks(subtasks, user_goal=fallback_goal)


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
