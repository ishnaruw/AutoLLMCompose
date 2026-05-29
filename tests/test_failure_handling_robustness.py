import json
from dataclasses import replace

from src.agents.planner import _repair_planner_payload
from src.agents.ranker import _exception_reason, _failure_metadata
from src.core.retry import classify_retryable_error
import src.driver.run_autogen_pipeline as pipeline
from src.driver.run_autogen_pipeline import (
    PlannerPrecheckFailure,
    _deterministic_select_and_plan,
    _write_invalid_ranked,
)


def test_ranker_classifies_fireworks_upstream_reset_as_transport_error():
    exc = RuntimeError(
        "Request timed out. upstream connect error or disconnect/reset before headers. "
        "reset reason: connection termination."
    )

    assert _exception_reason(exc) == "llm_transport_error"


def test_ranker_transport_failure_reason_stays_stable_after_retries():
    metadata = _failure_metadata(
        {"reason": "llm_transport_error", "error": "upstream connect error"},
        after_retries=True,
    )

    assert metadata["failure_reason"] == "llm_transport_error"
    assert metadata["retry_exhausted"] is True
    assert metadata["error"] == "upstream connect error"


def test_retry_treats_provider_reset_as_retryable_network_transient():
    should_retry, reason = classify_retryable_error(
        RuntimeError("disconnect/reset before headers; reset reason: connection termination")
    )

    assert should_retry is True
    assert reason == "network_transient"


def test_planner_repair_normalizes_execution_mapping_fields_only():
    payload = {
        "primary_plan": {
            "plan_id": 1,
            "summary": "summary",
            "steps": [
                {
                    "step": 1,
                    "api_id": "api-1",
                    "subtask_id": 1,
                    "action": "Call API",
                    "input_from_previous_step": None,
                    "output_to_next_step": {"keep": "schema-allows-null-or-string"},
                    "why": "why",
                    "qos": None,
                }
            ],
            "subtask_coverage": [],
        },
        "execution_workflow": {
            "type": "sequential",
            "steps": [
                {
                    "step": 1,
                    "api_id": "api-1",
                    "input_mapping": None,
                    "output_mapping": {"field": ["value"]},
                }
            ],
        },
        "selected_api_ids": ["api-1"],
        "overall_rationale": "rationale",
    }

    repaired = _repair_planner_payload(payload)
    step = repaired["execution_workflow"]["steps"][0]

    assert step["api_id"] == "api-1"
    assert step["input_mapping"] == ""
    assert step["output_mapping"] == '{"field":["value"]}'


def test_invalid_ranked_transport_artifact_uses_transport_reason(tmp_path):
    rows = _write_invalid_ranked(
        tmp_path / "qos_pure_llm",
        "3",
        {
            "failure_flag": True,
            "failure_stage": "llm_ranking",
            "failure_reason": "llm_transport_error",
            "exclude_from_ranking_eval": True,
            "error": "upstream connect error or disconnect/reset before headers",
        },
    )

    assert rows == [
        {
            "api_id": "",
            "mode_rank": None,
            "retrieved_rank": None,
            "rag_score": None,
            "reason": "llm_transport_error",
            "service": {},
            "failure_flag": True,
            "failure_stage": "llm_ranking",
            "failure_reason": "llm_transport_error",
            "exclude_from_ranking_eval": True,
            "error": "upstream connect error or disconnect/reset before headers",
            "ranking_anomaly": True,
            "ranking_anomaly_reason": "llm_transport_error",
            "ranking_anomaly_stage": "llm_ranking",
            "invalid_output_row": True,
        }
    ]


def test_planner_precheck_skips_mode_when_selected_empty_due_to_ranking_failure(tmp_path):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("planner LLM should not be called")

    ranked_full = {
        "3": [
            {
                "api_id": "",
                "failure_flag": True,
                "invalid_output_row": True,
                "failure_reason": "llm_transport_error",
            }
        ]
    }

    try:
        _deterministic_select_and_plan(
            "qos_pure_llm",
            [{"id": 3, "description": "send the digest by SMS"}],
            ranked_full,
            {"3": 1},
            fail_if_called,
            "Build a news SMS digest.",
            tmp_path,
            "prompts/planner_qos.md",
        )
    except PlannerPrecheckFailure as exc:
        payload = exc.payload
    else:
        raise AssertionError("expected PlannerPrecheckFailure")

    assert payload["failure_stage"] == "planner_precheck"
    assert payload["failure_reason"] == "missing_selected_apis_due_to_ranking_failure"
    assert payload["missing_subtask_ids"] == ["3"]
    assert payload["subtask_failure_reasons"] == {"3": "llm_transport_error"}
    assert payload["planner_called"] is False
    assert (tmp_path / "qos_pure_llm" / "planner_error.txt").exists()
    assert (tmp_path / "qos_pure_llm" / "planner_failure.json").exists()


def _prompt_file(tmp_path):
    path = tmp_path / "planner_prompt.md"
    path.write_text(
        "Goal: {user_goal}\nSubtasks: {subtasks_json}\nCandidate APIs:\n{selected_candidates_json}\n",
        encoding="utf-8",
    )
    return str(path)


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


def test_selection_passes_exactly_one_api_per_subtask_to_planner(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CONFIG", replace(pipeline.CONFIG, planner_candidate_mode="fixed_one"))
    prompts = []

    def fake_llm(_role, _system, prompt):
        prompts.append(prompt)
        return json.dumps(_planner_payload([("1", "api_a"), ("2", "api_b")]))

    ranked_full = {
        "1": [
            {"api_id": "api_a", "mode_rank": 1, "functional_match_label": 1},
            {"api_id": "api_a_alt", "mode_rank": 2, "functional_match_label": 1},
        ],
        "2": [
            {"api_id": "api_b", "mode_rank": 1, "functional_match_label": 1},
            {"api_id": "api_b_alt", "mode_rank": 2, "functional_match_label": 1},
        ],
    }

    result = _deterministic_select_and_plan(
        "qos_topsis",
        [{"id": 1, "description": "first"}, {"id": 2, "description": "second"}],
        ranked_full,
        {"1": 2, "2": 2},
        fake_llm,
        "Compose two APIs.",
        tmp_path,
        _prompt_file(tmp_path),
    )

    assert [row["api_id"] for row in result["selected"]] == ["api_a", "api_b"]
    assert len(json.loads((tmp_path / "qos_topsis" / "3_selected_s1.json").read_text(encoding="utf-8"))) == 1
    assert len(json.loads((tmp_path / "qos_topsis" / "3_selected_s2.json").read_text(encoding="utf-8"))) == 1
    assert "api_a_alt" not in prompts[0]
    assert "api_b_alt" not in prompts[0]
    assert result["planner"]["planner_provenance"][0]["selected_by"] == "ranking_mode"


def test_qos_hybrid_workflow_selection_blocks_rank_override(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CONFIG", replace(pipeline.CONFIG, planner_candidate_mode="fixed_one"))
    prompts = []
    responses = [
        _planner_payload([("1", "slow_rank1"), ("2", "stable_api")]),
        _planner_payload([("1", "fast_rank2"), ("2", "stable_api")]),
    ]

    def fake_llm(_role, _system, prompt):
        prompts.append(prompt)
        return json.dumps(responses.pop(0))

    ranked_full = {
        "1": [
            {
                "api_id": "slow_rank1",
                "mode_rank": 1,
                "functional_match_label": 1,
                "topsis_rank": 1,
                "topsis_score": 0.99,
                "service": {"qos": {"rt_s": 10.0, "tp_kbps": 1.0, "availability": 0.90}},
            },
            {
                "api_id": "fast_rank2",
                "mode_rank": 2,
                "functional_match_label": 1,
                "topsis_rank": 2,
                "topsis_score": 0.95,
                "service": {"qos": {"rt_s": 1.0, "tp_kbps": 10.0, "availability": 0.99}},
            },
        ],
        "2": [
            {
                "api_id": "stable_api",
                "mode_rank": 1,
                "functional_match_label": 1,
                "topsis_rank": 1,
                "topsis_score": 0.9,
                "service": {"qos": {"rt_s": 1.0, "tp_kbps": 10.0, "availability": 0.99}},
            }
        ],
    }

    result = _deterministic_select_and_plan(
        "qos_hybrid",
        [{"id": 1, "description": "first"}, {"id": 2, "description": "second"}],
        ranked_full,
        {"1": 2, "2": 1},
        fake_llm,
        "Compose two APIs.",
        tmp_path,
        _prompt_file(tmp_path),
    )

    assert [row["api_id"] for row in result["selected"]] == ["fast_rank2", "stable_api"]
    assert "fast_rank2" in prompts[0]
    assert "slow_rank1" not in prompts[0]
    assert len(prompts) == 2
    assert result["planner"]["planner_override_attempted"] is True
    assert {step["api_id"] for step in result["planner"]["primary_plan"]["steps"]} == {"fast_rank2", "stable_api"}
    assert all(item["selected_by"] == "workflow_hybrid_selector" for item in result["planner"]["planner_provenance"])


def test_qos_hybrid_caps_workflow_combinations_deterministically(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "CONFIG",
        replace(pipeline.CONFIG, planner_candidate_mode="fixed_one", selector_top_n=4, hybrid_max_workflow_combinations=8),
    )

    def make_pool(subtask_id):
        rows = []
        for idx, (rt_s, tp_kbps, availability) in enumerate(
            [(1.0, 1.0, 0.70), (2.0, 2.0, 0.80), (3.0, 3.0, 0.90), (4.0, 4.0, 0.99)],
            start=1,
        ):
            rows.append(
                {
                    "api_id": f"s{subtask_id}_api_{idx}",
                    "mode_rank": idx,
                    "functional_match_label": 1,
                    "topsis_rank": idx,
                    "topsis_score": 1.0 - (idx * 0.1),
                    "service": {"qos": {"rt_s": rt_s, "tp_kbps": tp_kbps, "availability": availability}},
                }
            )
        return rows

    subtasks = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    ranked_full = {str(idx): make_pool(idx) for idx in range(1, 4)}

    first_selected, first_trace, first_missing, first_reasons = pipeline._select_workflow_hybrid(
        subtasks=subtasks,
        ranked_full=ranked_full,
        functional_match_map={},
        run_label="synthetic_cap_test",
        planner_candidate_mode="fixed_one",
        planner_top_n={},
    )
    second_selected, second_trace, second_missing, second_reasons = pipeline._select_workflow_hybrid(
        subtasks=subtasks,
        ranked_full=ranked_full,
        functional_match_map={},
        run_label="synthetic_cap_test",
        planner_candidate_mode="fixed_one",
        planner_top_n={},
    )

    assert first_missing == second_missing == []
    assert first_reasons == second_reasons == {}
    assert len(first_selected) == 3
    assert [row["subtask_id"] for row in first_selected] == ["1", "2", "3"]
    assert all(row["selected_by"] == "workflow_hybrid_selector" for row in first_selected)
    assert [row["api_id"] for row in first_selected] == [row["api_id"] for row in second_selected]

    trace = first_trace["1"]
    assert trace["hybrid_total_combinations_before_cap"] == 64
    assert trace["hybrid_total_combinations_after_cap"] == 8
    assert trace["hybrid_max_workflow_combinations"] == 8
    assert trace["hybrid_pool_trimmed"] is True
    assert trace["hybrid_pool_sizes_before_cap"] == {"1": 4, "2": 4, "3": 4}
    assert trace["hybrid_pool_sizes_after_cap"] == {"1": 2, "2": 2, "3": 2}
    assert all(size >= 1 for size in trace["hybrid_pool_sizes_after_cap"].values())
    assert first_trace == second_trace
    assert first_selected[0]["selection_provenance"]["hybrid_total_combinations_after_cap"] == 8


def test_qos_hybrid_workflow_topsis_selects_closest_ideal_and_preserves_api_topsis(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "CONFIG",
        replace(
            pipeline.CONFIG,
            planner_candidate_mode="fixed_one",
            selector_top_n=2,
            hybrid_workflow_selector="workflow_topsis",
        ),
    )

    ranked_full = {
        "1": [
            {
                "api_id": "s1_fast_tp",
                "mode_rank": 1,
                "functional_match_label": 1,
                "topsis_rank": 1,
                "topsis_score": 0.91,
                "service": {"qos": {"rt_s": 1.0, "tp_kbps": 100.0, "availability": 0.90}},
            },
            {
                "api_id": "s1_available",
                "mode_rank": 2,
                "functional_match_label": 1,
                "topsis_rank": 2,
                "topsis_score": 0.82,
                "service": {"qos": {"rt_s": 4.0, "tp_kbps": 40.0, "availability": 0.99}},
            },
        ],
        "2": [
            {
                "api_id": "s2_fast_tp",
                "mode_rank": 1,
                "functional_match_label": 1,
                "topsis_rank": 1,
                "topsis_score": 0.93,
                "service": {"qos": {"rt_s": 1.0, "tp_kbps": 100.0, "availability": 0.90}},
            },
            {
                "api_id": "s2_available",
                "mode_rank": 2,
                "functional_match_label": 1,
                "topsis_rank": 2,
                "topsis_score": 0.84,
                "service": {"qos": {"rt_s": 4.0, "tp_kbps": 40.0, "availability": 0.99}},
            },
        ],
    }

    selected, trace, missing, reasons = pipeline._select_workflow_hybrid(
        subtasks=[{"id": "1"}, {"id": "2"}],
        ranked_full=ranked_full,
        functional_match_map={},
        run_label="synthetic_workflow_topsis",
        planner_candidate_mode="fixed_one",
        planner_top_n={},
    )

    assert missing == []
    assert reasons == {}
    assert [row["api_id"] for row in selected] == ["s1_fast_tp", "s2_fast_tp"]
    assert len(selected) == 2
    assert all(row["selected_rank"] == 1 for row in selected)
    assert all(item["planner_candidates_per_subtask"] == 1 for item in trace.values())
    assert all(item["selected_count"] == 1 for item in trace.values())

    first = selected[0]
    assert first["topsis_rank"] == 1
    assert first["topsis_score"] == 0.91
    assert first["workflow_topsis_rank"] == 1
    assert first["workflow_topsis_score"] > 0.9
    assert first["workflow_total_response_time"] == 2.0
    assert first["workflow_bottleneck_throughput"] == 100.0
    assert first["workflow_average_availability"] == 0.9
    assert first["hybrid_workflow_selector"] == "workflow_topsis"
    assert first["workflow_selector_fallback_used"] is False
    assert first["workflow_topsis_weights"] == {
        "response_time": 1.0 / 3.0,
        "throughput": 1.0 / 3.0,
        "availability": 1.0 / 3.0,
    }
    assert first["selection_provenance"]["topsis_rank"] == 1
    assert first["selection_provenance"]["workflow_topsis_rank"] == 1
    assert first["selection_provenance"]["hybrid_workflow_selector"] == "workflow_topsis"

    assert trace["1"]["hybrid_selector_objective"] == "workflow_topsis"
    assert trace["1"]["workflow_topsis_rank"] == 1
    assert trace["1"]["workflow_selector_fallback_used"] is False
    assert trace["1"]["hybrid_total_combinations_after_cap"] == 4


def test_qos_hybrid_relative_to_best_selector_can_be_configured(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "CONFIG",
        replace(
            pipeline.CONFIG,
            planner_candidate_mode="fixed_one",
            selector_top_n=2,
            hybrid_workflow_selector="relative_to_best",
        ),
    )

    ranked_full = {
        "1": [
            {
                "api_id": "low_latency",
                "mode_rank": 1,
                "functional_match_label": 1,
                "topsis_rank": 1,
                "topsis_score": 0.9,
                "service": {"qos": {"rt_s": 1.0, "tp_kbps": 80.0, "availability": 0.95}},
            },
            {
                "api_id": "slow_available",
                "mode_rank": 2,
                "functional_match_label": 1,
                "topsis_rank": 2,
                "topsis_score": 0.8,
                "service": {"qos": {"rt_s": 4.0, "tp_kbps": 70.0, "availability": 0.99}},
            },
        ],
        "2": [
            {
                "api_id": "steady_partner",
                "mode_rank": 1,
                "functional_match_label": 1,
                "topsis_rank": 1,
                "topsis_score": 0.88,
                "service": {"qos": {"rt_s": 1.0, "tp_kbps": 80.0, "availability": 0.95}},
            }
        ],
    }

    selected, trace, missing, reasons = pipeline._select_workflow_hybrid(
        subtasks=[{"id": "1"}, {"id": "2"}],
        ranked_full=ranked_full,
        functional_match_map={},
        run_label="synthetic_relative_to_best",
        planner_candidate_mode="fixed_one",
        planner_top_n={},
    )

    assert missing == []
    assert reasons == {}
    assert [row["api_id"] for row in selected] == ["low_latency", "steady_partner"]
    assert selected[0]["hybrid_workflow_selector"] == "relative_to_best"
    assert selected[0]["hybrid_selector_objective"] == "final_score_aligned_relative_to_best"
    assert selected[0]["workflow_topsis_score"] is None
    assert selected[0]["workflow_selector_fallback_used"] is False
    assert trace["1"]["hybrid_workflow_selector"] == "relative_to_best"
    assert trace["1"]["hybrid_selector_objective"] == "final_score_aligned_relative_to_best"
    assert trace["1"]["workflow_topsis_score"] is None
    assert trace["1"]["workflow_selector_fallback_used"] is False


def test_qos_hybrid_workflow_topsis_falls_back_on_invalid_workflow_metrics(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "CONFIG",
        replace(
            pipeline.CONFIG,
            planner_candidate_mode="fixed_one",
            selector_top_n=2,
            hybrid_workflow_selector="workflow_topsis",
        ),
    )

    ranked_full = {
        "1": [
            {
                "api_id": "missing_availability",
                "mode_rank": 1,
                "functional_match_label": 1,
                "topsis_rank": 1,
                "topsis_score": 0.95,
                "service": {"qos": {"rt_s": 1.0, "tp_kbps": 10.0}},
            },
            {
                "api_id": "complete_qos",
                "mode_rank": 2,
                "functional_match_label": 1,
                "topsis_rank": 2,
                "topsis_score": 0.85,
                "service": {"qos": {"rt_s": 2.0, "tp_kbps": 10.0, "availability": 0.99}},
            },
        ],
        "2": [
            {
                "api_id": "partner",
                "mode_rank": 1,
                "functional_match_label": 1,
                "topsis_rank": 1,
                "topsis_score": 0.90,
                "service": {"qos": {"rt_s": 1.0, "tp_kbps": 10.0, "availability": 0.99}},
            }
        ],
    }

    selected, trace, missing, reasons = pipeline._select_workflow_hybrid(
        subtasks=[{"id": "1"}, {"id": "2"}],
        ranked_full=ranked_full,
        functional_match_map={},
        run_label="synthetic_workflow_topsis_fallback",
        planner_candidate_mode="fixed_one",
        planner_top_n={},
    )

    assert missing == []
    assert reasons == {}
    assert [row["api_id"] for row in selected] == ["complete_qos", "partner"]
    assert selected[0]["hybrid_workflow_selector"] == "workflow_topsis"
    assert selected[0]["hybrid_selector_objective"] == "final_score_aligned_relative_to_best"
    assert selected[0]["workflow_topsis_score"] is None
    assert selected[0]["workflow_selector_fallback_used"] is True
    assert trace["1"]["workflow_selector_fallback_used"] is True
