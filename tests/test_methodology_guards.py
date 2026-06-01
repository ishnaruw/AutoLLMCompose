from __future__ import annotations

import json
from dataclasses import replace

from src.agents.ranker import rank_subtask
from src.core.run_logging import clear_run_log, configure_run_log
import src.driver.run_autogen_pipeline as pipeline
from src.driver.run_autogen_pipeline import _deterministic_select_and_plan


def _planner_payload(api_ids_by_subtask):
    steps = []
    execution_steps = []
    selected_api_ids = []
    for idx, (subtask_id, api_id) in enumerate(api_ids_by_subtask, start=1):
        selected_api_ids.append(api_id)
        steps.append(
            {
                "step": idx,
                "api_id": api_id,
                "subtask_id": subtask_id,
                "action": f"Call {api_id}",
                "input_from_previous_step": None if idx == 1 else "previous result",
                "output_to_next_step": "result",
                "why": "Uses the selected API.",
                "qos": None,
            }
        )
        execution_steps.append(
            {
                "step": idx,
                "api_id": api_id,
                "subtask_id": subtask_id,
                "method": "GET",
                "url": f"https://example.test/{api_id}",
                "required_parameters": [],
                "optional_parameters": [],
                "depends_on": [] if idx == 1 else [idx - 1],
                "input_mapping": "none" if idx == 1 else "previous result",
                "output_mapping": "result",
                "expected_output": "result",
            }
        )
    return {
        "primary_plan": {
            "plan_id": 1,
            "summary": "Use fixed selected APIs.",
            "steps": steps,
            "subtask_coverage": [],
        },
        "execution_workflow": {
            "type": "sequential",
            "steps": execution_steps,
        },
        "selected_api_ids": selected_api_ids,
        "overall_rationale": "The planner composes the fixed selected APIs.",
    }


def test_no_qos_ranker_prompt_does_not_instruct_qos_reasoning():
    prompt = open("prompts/ranker_no_qos.md", "r", encoding="utf-8").read().lower()

    forbidden = [
        "response time",
        "throughput",
        "availability",
        "topsis",
        "qos rank",
        "normalized qos",
        "service quality",
    ]
    for phrase in forbidden:
        assert phrase not in prompt


def test_no_qos_ranker_input_strips_qos_fields():
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps({"ranked": [{"candidate_id": "C01"}, {"candidate_id": "C02"}]})

    rank_subtask(
        llm,
        user_query="Find weather data",
        subtask={"id": "1", "description": "Retrieve forecast weather"},
        candidates=[
            {
                "api_id": "api_a",
                "rt_s": 1.0,
                "tp_kbps": 2.0,
                "availability": 0.99,
                "qos_score": 0.8,
                "qos_rank": 1,
                "topsis_score": 0.7,
                "topsis_rank": 2,
                "qos_llm_score": 0.6,
                "qos_llm_rank": 3,
                "service": {
                    "api_id": "api_a",
                    "name": "Weather lookup",
                    "qos": {"rt_s": 1.0, "tp_kbps": 2.0, "availability": 0.99},
                },
            },
            {
                "api_id": "api_b",
                "service": {
                    "api_id": "api_b",
                    "name": "Weather forecast",
                    "qos": {"rt_s": 3.0, "tp_kbps": 4.0, "availability": 0.95},
                },
            },
        ],
        prompt_path="prompts/ranker_no_qos.md",
        use_compact_api_evidence=False,
        max_validation_retries=0,
    )

    prompt = prompts[0]
    for field in [
        '"qos"',
        '"rt_s"',
        '"tp_kbps"',
        '"availability"',
        '"qos_score"',
        '"qos_rank"',
        '"topsis_score"',
        '"topsis_rank"',
        '"qos_llm_score"',
        '"qos_llm_rank"',
    ]:
        assert field not in prompt


def test_qos_pure_llm_prompt_requires_functional_before_qos():
    prompt = open("prompts/ranker_qos_pure_llm.md", "r", encoding="utf-8").read().lower()

    assert "functional suitability is a required gate" in prompt
    assert "functional match is the first priority" in prompt
    assert "qos should reorder candidates within the same functional tier" in prompt
    assert "qos_llm_rank" in prompt


def test_qos_pure_llm_input_includes_functional_refinement_labels_and_enforces_first():
    prompts: list[str] = []
    responses = [
        {"ranked": [{"candidate_id": "C01"}, {"candidate_id": "C02"}]},
        {"ranked": [{"candidate_id": "C02"}, {"candidate_id": "C01"}]},
    ]

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(responses.pop(0))

    ranked = rank_subtask(
        llm,
        user_query="Analyze stock trends and generate signals",
        subtask={"id": "1", "description": "Analyze stock trends and generate buy/sell signals"},
        candidates=[
            {
                "api_id": "weak_fast",
                "name": "Stock signal support",
                "description": "Provides weak stock signal support",
                "functional_match_label": 0,
                "qos_llm_rank": 1,
                "qos_llm_score": 0.9,
            },
            {
                "api_id": "weak_slow",
                "name": "Analyze news",
                "description": "Analyzes generic news over a period",
                "functional_match_label": 1,
                "qos_llm_rank": 20,
                "qos_llm_score": 0.4,
            },
        ],
        prompt_path="prompts/ranker_qos_pure_llm.md",
        use_compact_api_evidence=True,
        include_qos_rank=True,
        include_functional_match_label=True,
        enforce_functional_first=True,
        max_validation_retries=1,
    )

    assert [row["api_id"] for row in ranked] == ["weak_slow", "weak_fast"]
    assert len(prompts) == 2
    assert '"functional_match_label": 1' in prompts[0]
    assert '"qos_llm_rank": 1' in prompts[0]


def test_zero_functional_retrieval_retry_query_expands_stock_signal_terms():
    query = pipeline._zero_functional_retrieval_retry_query(
        "Build a tool that tracks stock prices, analyzes trends using ML, and emails buy/sell signals to users.",
        {
            "id": 2,
            "description": "Analyze stock trends and generate buy/sell signals using an external machine learning model API",
        },
    )

    assert "stock market technical indicator" in query
    assert "candlestick pattern" in query
    assert "buy sell signal" in query
    assert "Analyze stock trends" in query


def test_zero_functional_retrieval_retry_merge_prioritizes_retry_candidates():
    original = [
        {"api_id": "old_crypto_prediction", "rag_score": 0.91},
        {"api_id": "duplicate_stock_signal", "rag_score": 0.88},
        {"api_id": "old_generic_news", "rag_score": 0.87},
    ]
    retry = [
        {"api_id": "new_stock_signal", "rag_score": 0.93},
        {"api_id": "duplicate_stock_signal", "rag_score": 0.92},
    ]

    merged = pipeline._merge_retrieval_retry_candidates(original, retry, max_candidates=4)

    assert [row["api_id"] for row in merged] == [
        "new_stock_signal",
        "duplicate_stock_signal",
        "old_crypto_prediction",
        "old_generic_news",
    ]
    assert [row["retrieved_rank"] for row in merged] == [1, 2, 3, 4]
    assert merged[0]["retrieval_retry_source"] == "zero_functional_match_retry"
    assert merged[2]["retrieval_retry_source"] == "initial_retrieval"


def test_fixed_one_planner_retries_when_selected_apis_are_swapped_by_subtask(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CONFIG", replace(pipeline.CONFIG, planner_candidate_mode="fixed_one"))
    configure_run_log(tmp_path / "run.log")
    prompts: list[str] = []
    responses = [
        _planner_payload([("1", "api_b"), ("2", "api_a")]),
        _planner_payload([("1", "api_a"), ("2", "api_b")]),
    ]

    def fake_llm(_role, _system, prompt):
        prompts.append(prompt)
        return json.dumps(responses.pop(0))

    try:
        result = _deterministic_select_and_plan(
            "qos_pure_llm",
            [{"id": 1, "description": "first"}, {"id": 2, "description": "second"}],
            {
                "1": [{"api_id": "api_a", "mode_rank": 1, "functional_match_label": 1}],
                "2": [{"api_id": "api_b", "mode_rank": 1, "functional_match_label": 1}],
            },
            {"1": 1, "2": 1},
            fake_llm,
            "Compose two APIs.",
            tmp_path,
            "prompts/planner_qos.md",
        )
    finally:
        clear_run_log()

    assert len(prompts) == 2
    assert "fixed_one mode received exactly one selected API per subtask" in prompts[1]
    assert [step["api_id"] for step in result["planner"]["primary_plan"]["steps"]] == ["api_a", "api_b"]

    warning_event = json.loads((tmp_path / "warnings.log").read_text(encoding="utf-8").strip())
    assert warning_event["event_type"] == "planner_fixed_selection_retry"
