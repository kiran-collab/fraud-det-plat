"""FastAPI scoring service.

Run locally:

    uvicorn fraudplat.serving.app:app --port 8080

The scorer is built once at startup and shared. Model load is ~1s and
TreeExplainer construction is slower still, so doing either per request would
blow the latency budget by two orders of magnitude.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from fraudplat.serving.schemas import (
    BatchScoreRequest,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)
from fraudplat.serving.scorer import TransactionScorer

log = logging.getLogger(__name__)
STARTED_AT = time.time()

_state: dict[str, Any] = {"scorer": None}


# --- metrics ------------------------------------------------------------
def _build_metrics():
    from prometheus_client import Counter, Histogram

    return {
        "scored": Counter("fraudplat_transactions_scored_total", "Transactions scored", ["action"]),
        # Buckets are dense below 50ms because that is the SLO; anything above
        # 100ms is a single "too slow" bucket, since the exact value stops
        # mattering once the budget is blown.
        "latency": Histogram(
            "fraudplat_scoring_latency_seconds",
            "End-to-end scoring latency",
            buckets=(0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 1.0),
        ),
        "errors": Counter("fraudplat_scoring_errors_total", "Scoring errors"),
    }


METRICS = _build_metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["scorer"] = TransactionScorer()
        log.info(
            "loaded model %s (backend=%s, online store=%s)",
            _state["scorer"].model_version,
            _state["scorer"].inference_backend,
            _state["scorer"].store.backend,
        )
    except FileNotFoundError as exc:
        # Start anyway so /health can report *why* the pod is not ready,
        # instead of crash-looping with the reason buried in pod logs.
        log.error("no model available: %s", exc)
        _state["scorer"] = None
    yield
    _state.clear()


app = FastAPI(
    title="Fraud Detection Platform",
    version=os.environ.get("FP_VERSION", "0.1.0"),
    description="Real-time transaction risk scoring and decisioning.",
    lifespan=lifespan,
)


def _scorer() -> TransactionScorer:
    scorer = _state.get("scorer")
    if scorer is None:
        raise HTTPException(status_code=503, detail="model not loaded; see /health")
    return scorer


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    scorer = _state.get("scorer")
    return HealthResponse(
        status="ok" if scorer is not None else "degraded",
        model_version=scorer.model_version if scorer else "none",
        online_store=scorer.store.health() if scorer else {"backend": "none", "ok": False},
        inference_backend=scorer.inference_backend if scorer else "none",
        uptime_seconds=round(time.time() - STARTED_AT, 1),
    )


@app.get("/ready")
def ready() -> Response:
    """Kubernetes readiness probe - distinct from /health.

    A pod with no model must fail readiness so it is pulled from the Service
    endpoints, while /health stays available for humans to see the reason.
    """
    if _state.get("scorer") is None:
        return Response(status_code=503, content="model not loaded")
    return Response(status_code=200, content="ready")


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    scorer = _scorer()
    try:
        with METRICS["latency"].time():
            result = scorer.score(
                request.model_dump(exclude={"explain", "commit"}),
                explain=request.explain,
                commit=request.commit,
            )
    except Exception as exc:
        METRICS["errors"].inc()
        log.exception("scoring failed for %s", request.transaction_id)
        raise HTTPException(status_code=500, detail=f"scoring failed: {exc}") from exc

    METRICS["scored"].labels(action=result.action).inc()
    return ScoreResponse(
        transaction_id=request.transaction_id,
        action=result.action,
        risk_score=result.risk_score,
        supervised_score=result.supervised_score,
        anomaly_score=result.anomaly_score,
        sequence_score=result.sequence_score,
        reasons=result.reasons,
        triggered_rules=result.triggered_rules,
        model_version=scorer.model_version,
        latency_ms=result.latency_ms,
        feature_snapshot=result.features if request.explain else None,
    )


@app.post("/score/batch")
def score_batch(request: BatchScoreRequest) -> dict[str, Any]:
    """Offline replay / backtesting endpoint.

    Deliberately separate from ``/score``: batch callers tolerate seconds of
    latency, and mixing them into the authorization path would pollute the
    latency histogram the SLO is measured against.
    """
    scorer = _scorer()
    results = []
    for txn in request.transactions:
        result = scorer.score(
            txn.model_dump(exclude={"explain", "commit"}),
            explain=txn.explain,
            commit=txn.commit,
        )
        results.append({"transaction_id": txn.transaction_id, **result.__dict__})
    return {"count": len(results), "results": results}


@app.get("/model")
def model_info() -> dict[str, Any]:
    """Manifest of the loaded model - what governance and on-call both ask for."""
    scorer = _scorer()
    m = dict(scorer.bundle.manifest)
    m.pop("feature_importance", None)  # large; served by /model/importance
    return m


@app.get("/model/importance")
def model_importance() -> dict[str, float]:
    return _scorer().bundle.supervised.feature_importance()


@app.get("/metrics")
def metrics() -> Response:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def unhandled(request, exc):  # pragma: no cover - safety net
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})
