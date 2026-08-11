"""Online feature store client.

The scoring API needs a card's recent history in single-digit milliseconds.
That history is written by the Kafka consumer
(``features/streaming/consumer.py``) and read here.

What is stored is exactly ``FeatureEngine``'s serialised card state - not a
second, parallel representation. Defining a separate "online schema" is the
usual way training/serving skew creeps back in after you have carefully shared
the feature *code*: the two schemas drift, someone adds a field to one, and the
online vector quietly stops matching the offline one.

Redis is the backing store, with an in-process fallback so the service runs on
a laptop and in unit tests without infrastructure. The fallback is explicitly
*not* production-safe - it is per-process, so with more than one worker each
would see a different card history - and ``backend`` is surfaced on ``/health``
so a misconfigured deployment is visible rather than silently degraded.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fraudplat.config import OnlineStoreConfig


class OnlineFeatureStore:
    def __init__(self, config: OnlineStoreConfig | None = None) -> None:
        self.config = config or OnlineStoreConfig()
        self._redis = None
        self._memory: dict[str, str] = {}
        self.backend = "memory"
        self._connect()

    def _connect(self) -> None:
        try:
            import redis

            client = redis.Redis.from_url(self.config.url, socket_timeout=0.25)
            client.ping()
            self._redis = client
            self.backend = "redis"
        except Exception:
            # Not fatal: fall back to process memory and report it on /health.
            self._redis = None
            self.backend = "memory"

    def _key(self, card_id: str) -> str:
        return f"{self.config.key_prefix}:card:{card_id}"

    # -- read/write ------------------------------------------------------
    def get_card_state(self, card_id: str) -> dict[str, Any] | None:
        key = self._key(card_id)
        raw = self._redis.get(key) if self._redis is not None else self._memory.get(key)
        return json.loads(raw) if raw else None

    def put_card_state(self, card_id: str, state: dict[str, Any]) -> None:
        key = self._key(card_id)
        payload = json.dumps(state, separators=(",", ":"))
        if self._redis is not None:
            # TTL matches the longest feature window (7d) plus slack, so a
            # dormant card's state expires instead of accumulating forever.
            self._redis.setex(key, self.config.ttl_seconds, payload)
        else:
            self._memory[key] = payload

    def health(self) -> dict[str, Any]:
        latency_ms, ok = None, True
        if self._redis is not None:
            t0 = time.perf_counter()
            try:
                self._redis.ping()
                latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            except Exception:
                ok = False
        return {"backend": self.backend, "ok": ok, "ping_ms": latency_ms}
