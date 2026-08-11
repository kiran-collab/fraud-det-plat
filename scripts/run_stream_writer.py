#!/usr/bin/env python
"""Entrypoint for the Kafka feature-writer pod.

    python scripts/run_stream_writer.py

Loads the current model's merchant profile (so the streaming path encodes
merchant risk exactly as training did), then consumes until SIGTERM.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudplat.features.store.online import OnlineFeatureStore  # noqa: E402
from fraudplat.features.streaming.consumer import FeatureStreamWriter  # noqa: E402
from fraudplat.models.registry import ModelRegistry  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
log = logging.getLogger("stream_writer")


def main() -> int:
    try:
        profile = ModelRegistry().load().merchant_profile
        log.info("loaded merchant profile with %s merchants", len(profile.total_counts))
    except FileNotFoundError:
        # The writer's job is to maintain velocity state, which does not depend
        # on the merchant encoding. Running without a model degrades one
        # feature rather than stopping the whole stream.
        log.warning("no registered model; merchant risk will fall back to the prior")
        profile = None

    store = OnlineFeatureStore()
    if store.backend != "redis":
        # In a pod this is fatal: an in-process store means every replica has a
        # different view of each card and velocity features are meaningless.
        log.error("online store backend is %r, expected redis - refusing to start", store.backend)
        return 1

    writer = FeatureStreamWriter(profile, store)
    log.info("consuming from %s", writer.kafka.transactions_topic)
    stats = writer.run()
    log.info("stopped after %s messages (%s errors)", stats.messages, stats.errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
