from src.agents.planner import _repair_planner_payload
from src.agents.ranker import _exception_reason, _failure_metadata
from src.core.retry import classify_retryable_error
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
