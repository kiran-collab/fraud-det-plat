"""Central configuration.

Every knob is overridable through the environment so the same image runs in
local dev, CI and EKS without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    data: Path = field(default_factory=lambda: Path(_env("FP_DATA_DIR", str(REPO_ROOT / "artifacts" / "data"))))
    models: Path = field(default_factory=lambda: Path(_env("FP_MODEL_DIR", str(REPO_ROOT / "artifacts" / "models"))))
    reports: Path = field(default_factory=lambda: Path(_env("FP_REPORT_DIR", str(REPO_ROOT / "artifacts" / "reports"))))
    index: Path = field(default_factory=lambda: Path(_env("FP_INDEX_DIR", str(REPO_ROOT / "artifacts" / "index"))))

    def ensure(self) -> Paths:
        for p in (self.data, self.models, self.reports, self.index):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str = field(default_factory=lambda: _env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    transactions_topic: str = field(default_factory=lambda: _env("KAFKA_TXN_TOPIC", "transactions.raw"))
    decisions_topic: str = field(default_factory=lambda: _env("KAFKA_DECISION_TOPIC", "fraud.decisions"))
    consumer_group: str = field(default_factory=lambda: _env("KAFKA_GROUP", "fraudplat-feature-writer"))


@dataclass(frozen=True)
class OnlineStoreConfig:
    """Redis-backed online store; falls back to an in-process dict when Redis
    is unavailable so the service is runnable on a laptop."""

    url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    key_prefix: str = field(default_factory=lambda: _env("FP_ONLINE_PREFIX", "fp"))
    ttl_seconds: int = field(default_factory=lambda: _env_int("FP_ONLINE_TTL", 60 * 60 * 24 * 8))


@dataclass(frozen=True)
class DecisionConfig:
    """Score thresholds for the three-way decision. Tuned on the validation
    split by ``scripts/tune_thresholds.py`` and pinned here for reproducibility."""

    decline_at: float = field(default_factory=lambda: _env_float("FP_DECLINE_AT", 0.86))
    review_at: float = field(default_factory=lambda: _env_float("FP_REVIEW_AT", 0.55))
    step_up_at: float = field(default_factory=lambda: _env_float("FP_STEP_UP_AT", 0.35))
    high_value_amount: float = field(default_factory=lambda: _env_float("FP_HIGH_VALUE", 2500.0))


@dataclass(frozen=True)
class EnsembleConfig:
    """Blending coefficients, in **log-odds per unit of anomaly score**.

    The supervised model provides the calibrated base probability; these two
    coefficients say how many log-odds a fully-anomalous transaction may add on
    top of it. See ``models/ensemble.py`` for why this is not a weighted
    average over probabilities.
    """

    iforest_logit_weight: float = field(default_factory=lambda: _env_float("FP_W_IFOREST", 1.6))
    sequence_logit_weight: float = field(default_factory=lambda: _env_float("FP_W_SEQUENCE", 1.1))


@dataclass(frozen=True)
class GenAIConfig:
    """Investigation assistant settings."""

    model: str = field(default_factory=lambda: _env("FP_LLM_MODEL", "claude-opus-5"))
    max_tokens: int = field(default_factory=lambda: _env_int("FP_LLM_MAX_TOKENS", 4096))
    effort: str = field(default_factory=lambda: _env("FP_LLM_EFFORT", "medium"))
    vector_backend: str = field(default_factory=lambda: _env("FP_VECTOR_BACKEND", "local"))  # local | pinecone
    pinecone_index: str = field(default_factory=lambda: _env("PINECONE_INDEX", "fraud-cases"))
    embedding_dim: int = field(default_factory=lambda: _env_int("FP_EMBED_DIM", 512))
    top_k: int = field(default_factory=lambda: _env_int("FP_RAG_TOP_K", 6))


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    online_store: OnlineStoreConfig = field(default_factory=OnlineStoreConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    genai: GenAIConfig = field(default_factory=GenAIConfig)
    random_seed: int = field(default_factory=lambda: _env_int("FP_SEED", 17))


SETTINGS = Settings()
