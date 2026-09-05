"""Concurrent embedding client for building the case index.

The local ``HashingEmbedding`` needs none of this - it is pure CPU. This is the
path for a *hosted* embedding API, where indexing 50,000 historical fraud cases
means thousands of network calls against a provider that enforces limits.

Composition of the three controls from ``concurrency.py``:

    texts ─► chunk into batches ─► semaphore (in-flight cap)
                                     └─► token bucket (requests/sec)
                                           └─► retry w/ backoff + shared budget

Ordering is preserved: batch *k* always lands at the same offset in the output
regardless of which batches finished first. Callers embed documents and
vectors positionally, so a reordered result would silently attach every case's
text to another case's vector - a corruption that no exception would surface
and that retrieval quality would only vaguely reflect.

Partial failure is explicit. ``allow_partial`` decides whether a batch that
exhausts its retries fails the whole run or is recorded in ``failed_batches``
for the caller to handle, because "index 49,900 of 50,000 cases and tell me
which ones are missing" is often the right outcome for a backfill.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from fraudplat.genai.concurrency import (
    Clock,
    RealClock,
    RetryBudget,
    RetryPolicy,
    RetryStats,
    TokenBucket,
    call_with_retry,
    chunked,
    gather_bounded,
)

log = logging.getLogger(__name__)

# A coroutine that turns a batch of texts into a batch of vectors. Whatever
# provider SDK sits behind this is the caller's business.
EmbedFn = Callable[[Sequence[str]], Awaitable[Sequence[Sequence[float]]]]


@dataclass
class EmbeddingRunReport:
    total_texts: int = 0
    batches: int = 0
    succeeded: int = 0
    failed_batches: list[int] = field(default_factory=list)
    stats: RetryStats = field(default_factory=RetryStats)
    limiter_waits: int = 0
    limiter_pauses: int = 0
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failed_batches

    def summary(self) -> str:
        return (
            f"{self.succeeded}/{self.batches} batches "
            f"({self.total_texts:,} texts) in {self.elapsed_seconds:.1f}s | "
            f"attempts={self.stats.attempts} retries={self.stats.retries} "
            f"429s={self.stats.rate_limited} "
            f"limiter_waits={self.limiter_waits} pauses={self.limiter_pauses}"
            + (f" | FAILED batches: {self.failed_batches}" if self.failed_batches else "")
        )


class AsyncEmbeddingClient:
    """Rate-limited, retrying, bounded-concurrency batch embedder."""

    def __init__(
        self,
        embed_fn: EmbedFn,
        dim: int,
        requests_per_second: float = 10.0,
        max_concurrency: int = 8,
        batch_size: int = 64,
        retry: RetryPolicy | None = None,
        retry_budget: int | None = None,
        clock: Clock | None = None,
        burst_capacity: float | None = None,
    ) -> None:
        self.embed_fn = embed_fn
        self.dim = dim
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.clock = clock or RealClock()
        self.retry = retry or RetryPolicy()
        self.limiter = TokenBucket(
            rate_per_sec=requests_per_second,
            capacity=burst_capacity if burst_capacity is not None else requests_per_second,
            clock=self.clock,
        )
        self._retry_budget_size = retry_budget

    async def embed(
        self, texts: Sequence[str], allow_partial: bool = False
    ) -> tuple[np.ndarray, EmbeddingRunReport]:
        """Embed ``texts``, returning a ``(len(texts), dim)`` array in input order.

        Rows belonging to a failed batch are left as zeros when
        ``allow_partial`` is set; ``report.failed_batches`` names them.
        """
        report = EmbeddingRunReport(total_texts=len(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        if not texts:
            return out, report

        batches = chunked(list(texts), self.batch_size)
        report.batches = len(batches)
        # Default budget scales with the workload: enough headroom for normal
        # transient failures, not enough to grind through a total outage.
        budget = RetryBudget(
            self._retry_budget_size
            if self._retry_budget_size is not None
            else max(10, len(batches))
        )

        offsets, cursor = [], 0
        for batch in batches:
            offsets.append(cursor)
            cursor += len(batch)

        started = self.clock.now()

        def make(index: int, batch: Sequence[str]):
            async def run() -> tuple[int, Sequence[Sequence[float]]]:
                vectors = await call_with_retry(
                    lambda: self.embed_fn(batch),
                    policy=self.retry,
                    clock=self.clock,
                    limiter=self.limiter,
                    budget=budget,
                    stats=report.stats,
                )
                if len(vectors) != len(batch):
                    # A provider returning a different count is a silent
                    # misalignment if we just write what we got.
                    raise ValueError(
                        f"batch {index}: expected {len(batch)} vectors, got {len(vectors)}"
                    )
                return index, vectors

            return run

        results = await gather_bounded(
            [make(i, b) for i, b in enumerate(batches)],
            limit=self.max_concurrency,
            return_exceptions=allow_partial,
        )

        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                report.failed_batches.append(index)
                log.error("batch %s failed after retries: %s", index, result)
                continue
            batch_index, vectors = result
            start = offsets[batch_index]
            out[start:start + len(vectors)] = np.asarray(vectors, dtype=np.float32)
            report.succeeded += 1

        report.elapsed_seconds = self.clock.now() - started
        report.limiter_waits = self.limiter.waits
        report.limiter_pauses = self.limiter.pauses
        log.info("embedding run: %s", report.summary())
        return out, report
