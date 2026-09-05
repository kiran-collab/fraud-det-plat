"""Scoring API tests.

These need a registered model, so they train a deliberately small one once per
session rather than depending on whatever happens to be in the registry - a
test that passes only after someone ran the training script by hand is not a
test.
"""

from __future__ import annotations

import os

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
        supervised.predict_proba(x_valid),
        y_valid,
        iforest.score(x_valid),
        sequence.score(x_valid, valid_df["card_id"].to_numpy()),
    )
    return ModelBundle(supervised, iforest, sequence, ensemble, profile, {"version": "test"})


TEST_API_KEY = "test-key-primary"


@pytest.fixture(scope="session")
def client(trained_bundle, tmp_path_factory):
    """Authenticated client. The key is set for the whole session, so the
    default posture under test is the secure one."""
    from fastapi.testclient import TestClient

    from fraudplat.serving import app as app_module
    from fraudplat.serving.scorer import TransactionScorer

    os.environ["FP_API_KEYS"] = f"{TEST_API_KEY},test-key-secondary"
    os.environ.pop("FP_ALLOW_ANONYMOUS", None)

    registry = ModelRegistry(tmp_path_factory.mktemp("registry"))
    registry.save(trained_bundle, version="v-test", promote=True)

    scorer = TransactionScorer(bundle=registry.load(), use_onnx=False)
    app_module._state["scorer"] = scorer
    client = TestClient(app_module.app)
    client.headers.update({"X-API-Key": TEST_API_KEY})
    yield client
    os.environ.pop("FP_API_KEYS", None)


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
        body = client.post(
            "/score",
            json={
                **payload,
                "transaction_id": f"v{i}",
                "card_id": card,
                "explain": True,
                "amount": 5.0,
            },
        ).json()
        scores.append(body["feature_snapshot"]["card_txn_count_24h"])
    assert scores == [0.0, 1.0, 2.0, 3.0]


def test_invalid_amount_is_rejected(client, payload):
    assert client.post("/score", json={**payload, "amount": -5}).status_code == 422


def test_missing_required_field_is_rejected(client, payload):
    del payload["merchant_id"]
    assert client.post("/score", json=payload).status_code == 422


def test_batch_endpoint(client, payload):
    body = client.post(
        "/score/batch",
        json={"transactions": [{**payload, "transaction_id": f"b{i}"} for i in range(5)]},
    ).json()
    assert body["count"] == 5
    assert len(body["results"]) == 5


def test_model_metadata_endpoints(client):
    assert client.get("/model").status_code == 200
    importance = client.get("/model/importance").json()
    assert importance and abs(sum(importance.values()) - 1.0) < 0.01


# --- audit trail ---------------------------------------------------------
def _scorer_with_audit(bundle, tmp_path):
    from fraudplat.governance.audit import AuditLog
    from fraudplat.serving.scorer import TransactionScorer

    return TransactionScorer(
        bundle=bundle,
        use_onnx=False,
        audit_log=AuditLog(tmp_path / "decisions.jsonl"),
    )


def test_every_non_approval_writes_an_audit_record(trained_bundle, tmp_path):
    """Regression: the audit log existed and was tested, but was never called
    from the scoring path - so the claim that decisions could be reconstructed
    was false in the only place it mattered."""
    scorer = _scorer_with_audit(trained_bundle, tmp_path)

    written = 0
    for i in range(40):
        result = scorer.score(
            {
                "transaction_id": f"audit_{i}",
                "card_id": f"card_audit_{i % 3}",
                "merchant_id": "mch_000777",
                "merchant_category": "gambling",
                "merchant_country": "RO",
                "amount": 50.0 * (i + 1),
                "channel": "ecom",
                "entry_mode": "swipe",
                "device_id": f"dev_new_{i}",
            }
        )
        if result.action != "approve":
            written += 1

    records = list(scorer.audit_log.read())
    assert written > 0, "test produced no non-approvals; it asserts nothing"
    assert len(records) == written
    assert scorer.audit_failures == 0


def test_audit_record_captures_what_a_reconstruction_needs(trained_bundle, tmp_path):
    scorer = _scorer_with_audit(trained_bundle, tmp_path)
    scorer.score(
        {
            "transaction_id": "audit_detail",
            "card_id": "card_detail_1",
            "merchant_id": "mch_000777",
            "merchant_category": "gambling",
            "merchant_country": "RO",
            "amount": 9000.0,
            "channel": "ecom",
            "entry_mode": "swipe",
            "device_id": "dev_unknown",
        }
    )
    record = list(scorer.audit_log.read())[-1]

    for key in (
        "timestamp",
        "transaction_id",
        "card_hash",
        "model_version",
        "action",
        "risk_score",
        "reason_codes",
        "triggered_rules",
        "feature_snapshot",
        "prev_hash",
        "record_hash",
    ):
        assert key in record, f"audit record missing {key}"
    # The feature values are what make a decision reconstructable - the card's
    # state at that instant is gone afterwards.
    assert record["feature_snapshot"]
    assert record["model_version"] == scorer.model_version


def test_audit_log_never_stores_the_raw_card_id(trained_bundle, tmp_path):
    scorer = _scorer_with_audit(trained_bundle, tmp_path)
    scorer.score(
        {
            "transaction_id": "audit_pii",
            "card_id": "card_secret_9999",
            "merchant_id": "mch_000777",
            "merchant_category": "gambling",
            "merchant_country": "RO",
            "amount": 8000.0,
            "channel": "ecom",
            "entry_mode": "swipe",
            "device_id": "dev_x",
        }
    )
    assert "card_secret_9999" not in (tmp_path / "decisions.jsonl").read_text()


def test_audit_chain_survives_a_run_of_decisions(trained_bundle, tmp_path):
    scorer = _scorer_with_audit(trained_bundle, tmp_path)
    for i in range(25):
        scorer.score(
            {
                "transaction_id": f"chain_{i}",
                "card_id": "card_chain",
                "merchant_id": "mch_000777",
                "merchant_category": "gambling",
                "merchant_country": "RO",
                "amount": 3000.0 + i,
                "channel": "ecom",
                "entry_mode": "swipe",
                "device_id": f"dev_{i}",
            }
        )
    ok, broken = scorer.audit_log.verify_chain()
    assert ok and broken is None


def test_audit_failure_does_not_fail_the_transaction(trained_bundle, tmp_path):
    """An authorization service must not decline live traffic because a log
    file could not be written."""

    class BrokenAuditLog:
        path = tmp_path / "unwritable.jsonl"

        def append(self, **_):
            raise OSError("disk full")

    scorer = _scorer_with_audit(trained_bundle, tmp_path)
    scorer.audit_log = BrokenAuditLog()

    result = scorer.score(
        {
            "transaction_id": "audit_broken",
            "card_id": "card_broken",
            "merchant_id": "mch_000777",
            "merchant_category": "gambling",
            "merchant_country": "RO",
            "amount": 7500.0,
            "channel": "ecom",
            "entry_mode": "swipe",
            "device_id": "dev_y",
        }
    )
    assert result.action in {"approve", "step_up", "review", "decline"}
    assert scorer.audit_failures == 1, "audit failure was not counted for alerting"


def test_approvals_are_not_audited_by_default(trained_bundle, tmp_path):
    scorer = _scorer_with_audit(trained_bundle, tmp_path)
    result = scorer.score(
        {
            "transaction_id": "audit_approve",
            "card_id": "card_quiet",
            "merchant_id": "mch_000001",
            "merchant_category": "grocery",
            "merchant_country": "US",
            "amount": 12.0,
            "channel": "pos",
            "entry_mode": "chip",
            "device_id": "dev_known",
        }
    )
    if result.action == "approve":
        assert list(scorer.audit_log.read()) == []


def test_metrics_endpoint_exposes_prometheus_format(client, payload):
    client.post("/score", json=payload)
    text = client.get("/metrics").text
    assert "fraudplat_transactions_scored_total" in text
    assert "fraudplat_scoring_latency_seconds" in text


# --- authentication ------------------------------------------------------
def test_scoring_requires_a_key(client, payload):
    """The service decides whether to decline a customer's payment; an open
    endpoint is not an acceptable default."""
    response = client.post("/score", json=payload, headers={"X-API-Key": ""})
    assert response.status_code == 401


def test_wrong_key_is_rejected(client, payload):
    response = client.post("/score", json=payload, headers={"X-API-Key": "not-the-key"})
    assert response.status_code == 401
    assert "invalid" in response.json()["detail"].lower()


def test_any_configured_key_is_accepted(client, payload):
    response = client.post("/score", json=payload, headers={"X-API-Key": "test-key-secondary"})
    assert response.status_code == 200


@pytest.mark.parametrize("path", ["/score", "/score/batch", "/model", "/model/importance"])
def test_every_sensitive_endpoint_is_guarded(client, path):
    """A new endpoint that forgets the dependency is the realistic regression."""
    headers = {"X-API-Key": "wrong"}
    if path.startswith("/score"):
        body = {"transactions": []} if path == "/score/batch" else {}
        response = client.post(path, json=body, headers=headers)
    else:
        response = client.get(path, headers=headers)
    assert response.status_code == 401, f"{path} is not protected"


@pytest.mark.parametrize("path", ["/health", "/ready", "/metrics"])
def test_probe_endpoints_stay_open(client, path):
    """kubelet cannot present a key, and a closed /metrics is a monitoring
    outage rather than a security win."""
    assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 200


def test_health_reports_auth_posture(client):
    auth = client.get("/health").json()["auth"]
    assert auth["mode"] == "api_key"
    assert auth["keys_configured"] == 2
    # Fingerprints, never the keys themselves - /health is widely readable.
    assert TEST_API_KEY not in str(auth)


def test_health_reports_audit_state(client):
    audit = client.get("/health").json()["audit"]
    assert audit["enabled"] is True
    assert audit["failures"] == 0


def test_service_fails_closed_when_no_keys_are_configured(client, payload, monkeypatch):
    """Serving openly because configuration is missing is how an
    unauthenticated production service happens."""
    monkeypatch.delenv("FP_API_KEYS", raising=False)
    monkeypatch.delenv("FP_ALLOW_ANONYMOUS", raising=False)
    response = client.post("/score", json=payload)
    assert response.status_code == 503
    assert "FP_API_KEYS" in response.json()["detail"]


def test_anonymous_mode_is_explicit_and_visible(client, payload, monkeypatch):
    monkeypatch.delenv("FP_API_KEYS", raising=False)
    monkeypatch.setenv("FP_ALLOW_ANONYMOUS", "true")
    assert client.post("/score", json=payload).status_code == 200
    assert client.get("/health").json()["auth"]["mode"] == "anonymous"
