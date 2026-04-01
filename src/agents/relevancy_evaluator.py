from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from src.eval.api_relevancy_prompt import build_llm_prompt


@dataclass
class RelevancyEvaluatorAgent:
    system_message: str = (
        "You are a strict API relevance evaluator agent. "
        "Decide only whether an API is functionally relevant to a subtask. "
        "Ignore QoS for relevance decisions. Return strict JSON only."
    )

    def evaluate_batch(
        self,
        *,
        llm_call: Callable[[str, str, str], str],
        query_id: str,
        main_task: str,
        subtask_id: str,
        subtask_description: str,
        expected_function: str,
        api_entries: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        prompt = build_llm_prompt(
            query_id=query_id,
            main_task=main_task,
            subtask_id=subtask_id,
            subtask_description=subtask_description,
            expected_function=expected_function,
            api_entries=api_entries,
        )
        raw = llm_call("relevancy_evaluator", self.system_message, prompt)
        return self._parse_results(raw, [str(a.get("api_id")) for a in api_entries])

    @staticmethod
    def _parse_results(text: str, expected_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(text)
        except Exception:
            return {}

        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        expected = set(expected_ids)
        for r in results:
            if not isinstance(r, dict):
                continue
            api_id = str(r.get("api_id", "")).strip()
            rel = r.get("relevant")
            if not api_id or api_id not in expected or rel not in (0, 1):
                continue
            out[api_id] = {
                "relevant": int(rel),
                "comment": str(r.get("comment", "")).strip()[:200],
            }
        return out
