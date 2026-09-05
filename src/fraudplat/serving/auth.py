"""API-key authentication for the scoring service.

The service decides whether to decline a customer's payment. Anything able to
reach it can score transactions, read the model manifest, and infer feature
values from the responses. Network policy is the first control, but "it's only
reachable inside the cluster" is not an access-control statement - it is an
assumption about every other workload in the cluster.

**Secure by default.** With no keys configured, the protected endpoints return
503 rather than serving traffic openly. That is deliberate: a service whose
auth silently switches off when a config value is missing will eventually run
that way in production and nobody will notice, because the symptom is *things
working*. Local development and tests opt out explicitly with
``FP_ALLOW_ANONYMOUS=true``, which is visible in the process environment and
reported on ``/health``.

Keys are compared with ``secrets.compare_digest``. A plain ``==`` on a secret
returns as soon as it hits a differing byte, and that timing difference is
enough to recover a key one byte at a time given enough requests.

This is deliberately a simple shared-secret scheme, appropriate for
service-to-service calls from an authorization gateway inside the perimeter.
It is not a user-facing identity system: no per-user identity, no scopes, no
expiry. Anything customer-facing belongs behind a real OIDC/mTLS boundary, and
the trade is documented rather than implied.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets

from fastapi import Header, HTTPException, status

log = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"


def _configured_keys() -> set[str]:
    raw = os.environ.get("FP_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def _anonymous_allowed() -> bool:
    return os.environ.get("FP_ALLOW_ANONYMOUS", "").lower() == "true"


def auth_status() -> dict[str, object]:
    """Auth posture, surfaced on ``/health`` so a misconfiguration is visible.

    Reports the number of keys and a short fingerprint of each - never the keys
    themselves, since /health is typically readable by anything that can reach
    the pod.
    """
    keys = _configured_keys()
    if keys:
        return {
            "mode": "api_key",
            "keys_configured": len(keys),
            "key_fingerprints": sorted(key_fingerprint(k) for k in keys),
        }
    if _anonymous_allowed():
        return {"mode": "anonymous", "keys_configured": 0, "warning": "FP_ALLOW_ANONYMOUS=true"}
    return {"mode": "locked", "keys_configured": 0, "detail": "no keys configured; scoring refused"}


def key_fingerprint(key: str) -> str:
    """First 8 hex chars of the SHA-256. Enough to confirm *which* key is
    deployed when debugging a 401, useless for reconstructing it."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency guarding the scoring and model endpoints.

    Probe and metrics endpoints are intentionally left open: kubelet cannot
    present a key, and Prometheus scraping a closed /metrics is a
    monitoring outage rather than a security win.
    """
    keys = _configured_keys()

    if not keys:
        if _anonymous_allowed():
            return "anonymous"
        # Fail closed. Serving openly because configuration is missing is how
        # an unauthenticated production service happens.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "authentication is not configured: set FP_API_KEYS, or "
                "FP_ALLOW_ANONYMOUS=true for local development"
            ),
        )

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"missing {API_KEY_HEADER} header",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    # Compare against every key rather than returning on the first match, so
    # the work done does not depend on which key was presented.
    matched = False
    for candidate in keys:
        if secrets.compare_digest(x_api_key, candidate):
            matched = True

    if not matched:
        log.warning("rejected request with unknown API key (fp=%s)", key_fingerprint(x_api_key))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    return key_fingerprint(x_api_key)
