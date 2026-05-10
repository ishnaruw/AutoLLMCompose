from __future__ import annotations

import math
from typing import Any, Dict, List, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic_core import PydanticCustomError


class _SchemaModel(BaseModel):
    model_config = ConfigDict(extra="allow")


def _strip_optional_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, (dict, list, tuple, set)):
        raise PydanticCustomError("invalid_field_type", "identifier must be a string")
    text = str(value).strip()
    return text or None


class _IdentifiedItem(_SchemaModel):
    api_id: str | None = None
    candidate_id: str | None = None

    @field_validator("api_id", "candidate_id", mode="before")
    @classmethod
    def _validate_id(cls, value: Any) -> str | None:
        return _strip_optional_id(value)

    @model_validator(mode="after")
    def _require_identifier(self):
        if not self.api_id and not self.candidate_id:
            raise PydanticCustomError(
                "missing_required_key",
                "item must include api_id or candidate_id",
            )
        return self


class RankedCandidateItem(_IdentifiedItem):
    rank: Any | None = None
    reason: str | None = None
    explanation: str | None = None
    functional_reason: str | None = None
    qos_reason: str | None = None


class RankedCandidatesOutput(_SchemaModel):
    ranked: List[RankedCandidateItem]


def _coerce_score(value: Any) -> float:
    if value is None:
        raise PydanticCustomError("missing_score", "score is required")
    if isinstance(value, bool):
        raise PydanticCustomError("invalid_score_value", "score must be numeric")
    try:
        score = float(value)
    except Exception as exc:
        raise PydanticCustomError("invalid_score_value", "score must be numeric") from exc
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise PydanticCustomError("invalid_score_range", "score must be a finite number from 0.0 to 1.0")
    return score


class QoSScoreItem(_IdentifiedItem):
    score: float | None = None
    qos_score: float | None = None
    reason: str | None = None
    explanation: str | None = None

    @field_validator("score", "qos_score", mode="before")
    @classmethod
    def _validate_score(cls, value: Any) -> float | None:
        if value is None:
            return None
        return _coerce_score(value)

    @model_validator(mode="after")
    def _require_score(self):
        if self.score is None and self.qos_score is None:
            raise PydanticCustomError("missing_score", "score is required")
        return self


class QoSScoreOutput(_SchemaModel):
    scores: List[QoSScoreItem]


def _coerce_label(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if value in (0, 1):
        return int(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "relevant"}:
            return 1
        if text in {"0", "false", "no", "irrelevant", "not relevant"}:
            return 0
        if not text:
            raise PydanticCustomError("missing_label", "label is required")
    if value is None:
        raise PydanticCustomError("missing_label", "label is required")
    raise PydanticCustomError("invalid_label_value", "label must be 0 or 1")


class FunctionalMatchItem(_IdentifiedItem):
    label: int | None = None
    functional_match: int | None = None
    relevant: int | None = None
    comment: str | None = None
    reason: str | None = None
    explanation: str | None = None

    @field_validator("label", "functional_match", "relevant", mode="before")
    @classmethod
    def _validate_label(cls, value: Any) -> int | None:
        if value is None:
            return None
        return _coerce_label(value)

    @model_validator(mode="after")
    def _require_label(self):
        if self.label is None and self.functional_match is None and self.relevant is None:
            raise PydanticCustomError("missing_label", "label is required")
        return self


class FunctionalMatchOutput(_SchemaModel):
    matches: List[FunctionalMatchItem]


class DecompositionItem(_SchemaModel):
    id: Any | None = None
    description: str | None = None
    goal: str | None = None


SubtaskItem = DecompositionItem


class DecompositionOutput(_SchemaModel):
    subtasks: List[DecompositionItem] = Field(default_factory=list)


class PlannerStepItem(_SchemaModel):
    step: Any | None = None
    api_id: str | None = None
    subtask_id: Any | None = None
    action: str | None = None
    why: str | None = None
    score: Any | None = None
    qos: Any | None = None


class PlannerPathItem(_SchemaModel):
    path_id: Any | None = None
    path_score: Any | None = None
    summary: str | None = None
    steps: List[PlannerStepItem] = Field(default_factory=list)
    subtask_coverage: List[Dict[str, Any]] = Field(default_factory=list)


class PlannerOutput(_SchemaModel):
    paths: List[PlannerPathItem] = Field(default_factory=list)
    selected_api_ids: List[Any] = Field(default_factory=list)
    overall_rationale: str | None = None


_T = TypeVar("_T", bound=BaseModel)


def validation_issue(exc: ValidationError) -> Dict[str, Any]:
    errors = exc.errors()
    first = errors[0] if errors else {}
    reason = str(first.get("type") or "schema_validation_error")
    if reason == "missing":
        reason = "missing_required_key"
    elif reason in {"model_type", "dict_type"}:
        reason = "wrong_json_type"
    elif reason == "list_type":
        reason = "invalid_field_type"

    return {
        "reason": reason,
        "schema_error_category": "schema_validation_error",
        "schema_errors": [
            {
                "loc": list(error.get("loc", ())),
                "type": error.get("type"),
                "message": error.get("msg"),
            }
            for error in errors
        ],
    }


def validate_output_schema(model: Type[_T], payload: Any) -> tuple[_T | None, Dict[str, Any] | None]:
    try:
        return model.model_validate(payload), None
    except ValidationError as exc:
        return None, validation_issue(exc)
