"""Scoring API tests.

These need a registered model, so they train a deliberately small one once per
session rather than depending on whatever happens to be in the registry - a
test that passes only after someone ran the training script by hand is not a
test.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraudplat.data.generator import time_split
from fraudplat.features.transforms import FeatureEngine, MerchantProfile, compute_batch
from fraudplat.models.ensemble import EnsembleScorer
from fraudplat.models.iforest import IsolationForestScorer
from fraudplat.models.lgbm import SupervisedModel
from fraudplat.models.registry import ModelBundle, ModelRegistry
from fraudplat.models.transformer import SequenceAnomalyModel


@pytest.fixture(scope="session")
def trained_bundle(transactions):
    """A small but genuine bundle - same code path as scripts/train.py."""
    train_df, valid_df, _ = time_split(transactions)
    profile = MerchantProfile.fit(train_df)
    engine = FeatureEngine(profile)
    x_train = compute_batch(train_df, engine=engine).to_numpy(dtype=np.float32)
    x_valid = compute_batch(valid_df, engine=engine).to_numpy(dtype=np.float32)
    y_train = train_df["is_fraud"].to_numpy()
    y_valid = valid_df["is_fraud"].to_numpy()

    supervised = SupervisedModel(num_boost_round=60).fit(x_train, y_train, x_valid, y_valid)
    iforest = IsolationForestScorer(n_estimators=40).fit(x_train, y_train)
    sequence = SequenceAnomalyModel(epochs=1).fit(x_train, train_df["card_id"].to_numpy(), y_train)
    ensemble = EnsembleScorer().fit(
        supervised.predict_proba(x_valid), y_valid,
        iforest.score(x_valid), sequence.score(x_valid, valid_df["card_id"].to_numpy()),
    )
    return ModelBundle(supervised, iforest, sequence, ensemble, profile, {"version": "test"})


@pytest.fixture(scope="session")
def client(trained_bundle, tmp_path_factory):
    from fastapi.testclient import TestClient

    from fraudplat.serving import app as app_module
    from fraudplat.serving.scorer import TransactionScorer

    registry = ModelRegistry(tmp_path_factory.mktemp("registry"))
    registry.save(trained_bundle, version="v-test", promote=True)

    scorer = TransactionScorer(bundle=registry.load(), use_onnx=False)
    app_module._state["scorer"] = scorer
    return TestClient(app_module.app)


@pytest.fixture
def payload():
    return {
        "transaction_id": "txn_api_1",
        "card_id": "card_0000001",
        "merchant_id": "mch_000001",
        "merchant_category": "electronics",
        "merchant_country": "US",
        "amount": 150.0,
        "channel": "ecom",
        "entry_mode": "keyed",
        "device_id": "dev_0000001",
    }


def test_health_reports_backends(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["online_store"]["backend"] in {"redis", "memory"}
    assert body["inference_backend"] in {"python", "onnxruntime"}


def test_ready_probe(client):
    assert client.get("/ready").status_code == 200


def test_score_returns_a_complete_decision(client, payload):
    body = client.post("/score", json=payload).json()
    assert body["action"] in {"approve", "step_up", "review", "decline"}
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["model_version"] == "v-test"
    assert body["latency_ms"] > 0


def test_score_is_within_the_latency_budget(client, payload):
    """Not a benchmark - a regression guard. A change that makes scoring an
    order of magnitude slower should fail here rather than in production."""
    for i in range(20):
        body = client.post("/score", json={**payload, "transaction_id": f"t{i}"}).json()
        assert body["latency_ms"] < 200


def test_explain_attaches_reasons_and_features(client, payload):
    body = client.post("/score", json={**payload, "explain": True}).json()
    assert body["reasons"], "explain=true returned no reason codes"
    assert body["feature_snapshot"]
    assert all(isinstance(r, str) for r in body["reasons"])


def test_reasons_omitted_by_default(client, payload):
    body = client.post("/score", json={**payload, "transaction_id": "quiet"}).json()
    assert body["feature_snapshot"] is None


def test_commit_false_does_not_advance_card_state(client, payload):
    """A pre-auth quote must be repeatable: scoring the same transaction twice
    without committing must give the same answer."""
    quote = {**payload, "transaction_id": "quote_1", "commit": False}
    first = client.post("/score", json=quote).json()
    second = client.post("/score", json=quote).json()
    assert first["risk_score"] == pytest.approx(second["risk_score"], abs=1e-9)


def test_commit_true_does_advance_card_state(client, payload):
    """Velocity must actually accumulate, or the online store is inert."""
    card = "card_velocity_probe"
    scores = []
    for i in range(4):
        body = client.post("/score", json={
            **payload, "transaction_id": f"v{i}", "card_id": card,
            "explain": True, "amount": 5.0,
        }).json()
        scores.append(body["feature_snapshot"]["card_txn_count_24h"])
    assert scores == [0.0, 1.0, 2.0, 3.0]


def test_invalid_amount_is_rejected(client, payload):
    assert client.post("/score", json={**payload, "amount": -5}).status_code == 422


def test_missing_required_field_is_rejected(client, payload):
    del payload["merchant_id"]
    assert client.post("/score", json=payload).status_code == 422


def test_batch_endpoint(client, payload):
    body = client.post("/score/batch", json={
        "transactions": [{**payload, "transaction_id": f"b{i}"} for i in range(5)]
    }).json()
    assert body["count"] == 5
    assert len(body["results"]) == 5


def test_model_metadata_endpoints(client):
    assert client.get("/model").status_code == 200
    importance = client.get("/model/importance").json()
    assert importance and abs(sum(importance.values()) - 1.0) < 0.01


def test_metrics_endpoint_exposes_prometheus_format(client, payload):
    client.post("/score", json=payload)
    text = client.get("/metrics").text
    assert "fraudplat_transactions_scored_total" in text
    assert "fraudplat_scoring_latency_seconds" in text
