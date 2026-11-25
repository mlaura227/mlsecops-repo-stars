"""Strict HTTP request and response contracts for the GitHub stars service."""

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Only the eight input features; the target cannot enter a prediction."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    language: str = Field(min_length=1, max_length=80)
    forks: int = Field(ge=0, strict=True)
    open_issues: int = Field(ge=0, strict=True)
    age_days: int = Field(ge=0, strict=True)
    has_coc: bool = Field(strict=True)
    has_contributing: bool = Field(strict=True)
    has_workflows: bool = Field(strict=True)
    has_readme: bool = Field(strict=True)


class PredictResponse(BaseModel):
    """Prediction and auditable MLflow request run."""

    request_id: str
    prediction: float
    model_name: str
    model_version: int
    audit_run_id: str


class TrainRequest(BaseModel):
    """Training configuration; data locations are server-controlled."""

    model_config = ConfigDict(extra="forbid")

    auto_promote: bool = Field(default=True, strict=True)
    min_mae_improvement: float = Field(default=0.01, ge=0.0, lt=1.0, allow_inf_nan=False)
    random_state: int = Field(default=42, ge=0, le=4_294_967_295, strict=True)


class TrainResponse(BaseModel):
    """Training and promotion results, including the audit run identifier."""

    baseline_run_id: str
    challenger_run_id: str
    baseline_metrics: dict[str, float]
    challenger_metrics: dict[str, float]
    registered_model_version: int
    promotion_accepted: bool | None
    promotion_reason: str | None
    audit_run_id: str


class ModelStatusResponse(BaseModel):
    """Active serving model and its separation from the training registry."""

    serving_model_name: str
    serving_model_version: int
    training_model_name: str
    validation_protocol_id: str | None
