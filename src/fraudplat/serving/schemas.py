"""Request/response contracts for the scoring API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from fraudplat.data.schema import Channel, EntryMode


class ScoreRequest(BaseModel):
    transaction_id: str
    event_time: datetime | None = None
    card_id: str
    customer_id: str | None = None
    merchant_id: str
    merchant_category: str
    merchant_country: str = "US"
    amount: float = Field(gt=0)
    currency: str = "USD"
    channel: Channel = "pos"
    entry_mode: EntryMode = "chip"
    device_id: str | None = None
    ip_prefix: str | None = None
    issuer_country: str = "US"
    explain: bool = Field(
        default=False,
        description="Attach SHAP reason codes. Adds ~1-3ms; always forced on for declines.",
    )
    # False for a pre-auth quote: scores the transaction without letting it
    # advance the cardholder's behavioural baseline.
    commit: bool = Field(default=True, description="Persist the updated card state.")


class ScoreResponse(BaseModel):
    transaction_id: str
    action: str
    risk_score: float
    supervised_score: float
    anomaly_score: float
    sequence_score: float
    reasons: list[str] = []
    triggered_rules: list[str] = []
    model_version: str
    latency_ms: float
    feature_snapshot: dict[str, float] | None = None


class BatchScoreRequest(BaseModel):
    transactions: list[ScoreRequest]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    online_store: dict[str, Any]
    inference_backend: str
    uptime_seconds: float
