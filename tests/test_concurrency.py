"""Concurrency, rate-limiting and retry tests.

Every test runs under ``FakeClock``, so behaviour that takes minutes of
wall-clock backoff is asserted in milliseconds and deterministically. Retry
code tested against the real clock is either slow or (more often) not really
tested - the assertions get weakened until the suite is fast, and by then they
no longer check the timing that was the entire point.

The fake provider below is the other half: it can be told to fail a set number
of times, return 429 with a ``Retry-After``, or record its own arrival times so
tests can assert on the *rate* the limiter actually produced.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from fraudplat.genai.concurrency import (
    FakeClock,
    PermanentError,
    RateLimited,
    RetryBudget,
    RetryPolicy,
    RetryStats,
    TokenBucket,
    TransientError,
    call_with_retry,
    chunked,
    default_is_retryable,
    gather_bounded,
)
from fraudplat.genai.embedding_client import AsyncEmbeddingClient


def simulate(coro_fn):
    """Run ``coro_fn(clock)`` under virtual time.

    The clock is created here and passed in, so the coroutine sleeps on exactly
    the clock the pump advances. Creating a FakeClock *inside* the coroutine
    while pumping another one deadlocks - which is why FakeClock now detects
    that state and raises instead of hanging.
    """
    async def outer():
        clock = FakeClock()
        return await clock.run(coro_fn(clock))

    return asyncio.run(outer())


def run(coro):
    """For coroutines that never sleep and so need no clock pump."""
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Fake provider
# --------------------------------------------------------------------------
class FakeProvider:
    """Stands in for a hosted embedding API."""

    def __init__(
        self,
        clock: FakeClock,
        dim: int = 4,
        fail_times: int = 0,
        error=None,
        latency: float = 0.0,
        fail_batches: set[int] | None = None,
    ) -> None:
        self.clock = clock
        self.dim = dim
        self.fail_times = fail_times
        self.error = error or TransientError("upstream blip")
        self.latency = latency
        self.fail_batches = fail_batches or set()
        self.calls = 0
        self.call_times: list[float] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._batch_seen: dict[str, int] = {}

    async def embed(self, texts):
        self.calls += 1
        self.call_times.append(self.clock.now())
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.fail_times > 0:
                self.fail_times -= 1
                raise self.error
            key = texts[0] if texts else ""
            if key in self.fail_batches:
                raise TransientError(f"permanently failing batch {key}")
            if self.latency:
                await self.clock.sleep(self.latency)
            # Encode the text's index in the vector so ordering is checkable.
            return [[float(t.split("-")[-1])] * self.dim for t in texts]
        finally:
            self.concurrent -= 1


# --------------------------------------------------------------------------
# Token bucket
# --------------------------------------------------------------------------
def test_bucket_allows_an_initial_burst_up_to_capacity():
    async def main(clock):
        bucket = TokenBucket(rate_per_sec=5, capacity=5, clock=clock)
        for _ in range(5):
            await bucket.acquire()
        return clock.now(), bucket.waits

    elapsed, waits = simulate(main)
    assert elapsed == 0.0, "a full bucket must not delay the first burst"
    assert waits == 0


def test_bucket_throttles_beyond_capacity():
    """5 burst + 5 more at 5/sec must take about one second, not zero."""

    async def main(clock):
        bucket = TokenBucket(rate_per_sec=5, capacity=5, clock=clock)
        for _ in range(10):
            await bucket.acquire()
        return clock.now()

    elapsed = simulate(main)
    assert 0.9 <= elapsed <= 1.2, f"expected ~1s of throttling, got {elapsed}"


def test_bucket_refills_over_time():
    async def main(clock):
        bucket = TokenBucket(rate_per_sec=10, capacity=10, clock=clock)
        for _ in range(10):
            await bucket.acquire()
        await clock.sleep(1.0)          # refills 10 tokens
        before = clock.now()
        for _ in range(10):
            await bucket.acquire()
        return clock.now() - before

    assert simulate(main) == pytest.approx(0.0, abs=1e-9)


def test_pause_backs_off_every_worker_not_just_the_rejected_one():
    """The behaviour this whole module exists for.

    A per-caller sleep leaves the rest of the pool hammering an endpoint that
    just asked for a break, which is how a soft throttle becomes a hard ban.
    """

    async def main(clock):
        bucket = TokenBucket(rate_per_sec=100, capacity=100, clock=clock)
        bucket.pause(2.0)               # as if one worker saw Retry-After: 2
        start = clock.now()
        # A different worker, with plenty of tokens available, still waits.
        await bucket.acquire()
        return clock.now() - start

    assert simulate(main) >= 2.0


def test_bucket_does_not_spin_on_floating_point_dust():
    """Regression: the limiter used to livelock on rounding error.

    Sleeping exactly ``deficit / rate`` should refill exactly ``deficit``
    tokens, but that arithmetic does not round-trip in binary floating point,
    so the balance lands a few ulps short. With an exact ``>=`` comparison the
    waiter then computed a deficit of ~1e-17, slept ~2e-18 - below the clock's
    resolution, so ``now + delay == now`` - and re-queued forever.

    Under a real clock this only burns CPU, which is why it would have shipped.
    Rates chosen so ``1/rate`` is not representable in binary.
    """

    async def main(clock):
        bucket = TokenBucket(rate_per_sec=3, capacity=1, clock=clock)
        for _ in range(7):
            await bucket.acquire()
        return clock.now()

    elapsed = simulate(main)
    # 1 free from the burst, then 6 more at 3/sec = 2 seconds.
    assert elapsed == pytest.approx(2.0, abs=0.05)


def test_bucket_survives_concurrent_contention_without_stalling():
    """The same livelock, reached through the concurrent path instead."""

    async def main(clock):
        bucket = TokenBucket(rate_per_sec=5, capacity=5, clock=clock)
        completed = []

        async def worker(i):
            await bucket.acquire()
            completed.append(i)

        await gather_bounded([(lambda i=i: worker(i)) for i in range(20)], limit=10)
        return clock.now(), len(completed)

    elapsed, completed = simulate(main)
    assert completed == 20
    assert elapsed == pytest.approx(3.0, abs=0.1)  # 5 burst + 15 at 5/sec


def test_acquiring_more_than_capacity_fails_fast():
    async def main():
        bucket = TokenBucket(rate_per_sec=1, capacity=1, clock=FakeClock())
        with pytest.raises(ValueError):
            await bucket.acquire(5)

    run(main())


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------
def test_backoff_grows_exponentially_and_is_capped():
    policy = RetryPolicy(base_delay=1.0, multiplier=2.0, max_delay=10.0, jitter=False)
    assert [policy.delay_for(i) for i in range(1, 6)] == [1.0, 2.0, 4.0, 8.0, 10.0]


def test_full_jitter_spreads_retries_across_the_whole_interval():
    """Without jitter, N workers failing together retry together and rebuild
    the burst that caused the failure."""
    policy = RetryPolicy(base_delay=8.0, multiplier=1.0, max_delay=8.0, jitter=True)
    samples = [policy.delay_for(1) for _ in range(300)]
    assert all(0.0 <= s <= 8.0 for s in samples)
    assert len(set(samples)) > 100, "delays are not actually randomised"
    assert min(samples) < 2.0 and max(samples) > 6.0, "jitter does not span the interval"


def test_server_retry_after_overrides_computed_backoff():
    policy = RetryPolicy(base_delay=0.5, max_delay=60.0, jitter=False)
    assert policy.delay_for(1, retry_after=7.5) == 7.5


def test_retry_after_is_still_capped():
    """A hostile or buggy `Retry-After: 86400` must not wedge a worker."""
    policy = RetryPolicy(max_delay=30.0)
    assert policy.delay_for(1, retry_after=86_400) == 30.0


# --------------------------------------------------------------------------
# call_with_retry
# --------------------------------------------------------------------------
def test_succeeds_after_transient_failures():
    async def main(clock):
        provider = FakeProvider(clock, fail_times=2)
        stats = RetryStats()
        result = await call_with_retry(
            lambda: provider.embed(["case-1"]),
            policy=RetryPolicy(base_delay=1.0, jitter=False),
            clock=clock,
            stats=stats,
        )
        return result, stats, provider.calls

    result, stats, calls = simulate(main)
    assert len(result) == 1
    assert calls == 3
    assert stats.attempts == 3 and stats.retries == 2


def test_permanent_error_is_not_retried():
    async def main(clock):
        stats = RetryStats()
        calls = {"n": 0}

        async def fn():
            calls["n"] += 1
            raise PermanentError("401 unauthorized")

        with pytest.raises(PermanentError):
            await call_with_retry(fn, clock=clock, stats=stats)
        return calls["n"], stats

    calls, stats = simulate(main)
    assert calls == 1, "a 401 was retried; that is wasted time and load"
    assert stats.retries == 0


def test_gives_up_after_max_attempts():
    async def main(clock):
        provider = FakeProvider(clock, fail_times=99)
        with pytest.raises(TransientError):
            await call_with_retry(
                lambda: provider.embed(["case-1"]),
                policy=RetryPolicy(max_attempts=4, base_delay=0.1, jitter=False),
                clock=clock,
            )
        return provider.calls

    assert simulate(main) == 4


def test_rate_limited_response_pauses_the_shared_limiter():
    async def main(clock):
        bucket = TokenBucket(rate_per_sec=100, capacity=100, clock=clock)
        provider = FakeProvider(clock, fail_times=1, error=RateLimited(retry_after=5.0))
        stats = RetryStats()
        await call_with_retry(
            lambda: provider.embed(["case-1"]),
            policy=RetryPolicy(base_delay=0.01, jitter=False),
            clock=clock,
            limiter=bucket,
            stats=stats,
        )
        return stats, bucket.pauses, clock.now()

    stats, pauses, elapsed = simulate(main)
    assert stats.rate_limited == 1
    assert pauses == 1, "429 did not pause the shared limiter"
    assert elapsed >= 5.0, "Retry-After was not honoured"


def test_shared_budget_stops_a_doomed_batch_early():
    """With a dead provider, per-call limits still permit attempts x items
    requests. The shared budget is what makes the run fail fast instead."""

    async def main(clock):
        budget = RetryBudget(max_retries=3)
        provider = FakeProvider(clock, fail_times=10_000)
        stats = RetryStats()
        failures = 0
        for i in range(5):
            try:
                await call_with_retry(
                    lambda i=i: provider.embed([f"case-{i}"]),  # bind, don't capture
                    policy=RetryPolicy(max_attempts=5, base_delay=0.1, jitter=False),
                    clock=clock,
                    budget=budget,
                    stats=stats,
                )
            except TransientError:
                failures += 1
        return failures, stats, provider.calls, budget.used

    failures, stats, calls, used = simulate(main)
    assert failures == 5
    assert used == 3, "budget was not enforced"
    # Without the budget this would be 5 items x 5 attempts = 25 calls.
    assert calls == 8, f"expected 5 first-attempts + 3 budgeted retries, got {calls}"
    assert stats.budget_denied >= 1


def test_cancellation_is_not_swallowed():
    """A retry loop that catches CancelledError makes shutdown hang."""

    async def main():
        async def fn():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await call_with_retry(fn, clock=FakeClock(), policy=RetryPolicy(max_attempts=3))

    run(main())


# --------------------------------------------------------------------------
# Bounded fan-out
# --------------------------------------------------------------------------
def test_gather_bounded_caps_in_flight_work():
    async def main(clock):
        provider = FakeProvider(clock, latency=0.5)
        await gather_bounded(
            [(lambda i=i: provider.embed([f"case-{i}"])) for i in range(20)],
            limit=3,
        )
        return provider.max_concurrent

    assert simulate(main) <= 3


def test_gather_bounded_preserves_order():
    """Callers zip vectors against documents positionally - a reordered result
    silently attaches each case's text to another case's vector."""

    async def main(clock):

        async def slow(i):
            await clock.sleep(1.0 if i % 2 == 0 else 0.1)  # finish out of order
            return i

        return await gather_bounded([(lambda i=i: slow(i)) for i in range(12)], limit=6)

    assert simulate(main) == list(range(12))


def test_gather_bounded_rejects_a_zero_limit():
    async def main():
        with pytest.raises(ValueError):
            await gather_bounded([], limit=0)

    run(main())


def test_chunked_splits_without_loss():
    assert chunked(list(range(10)), 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


# --------------------------------------------------------------------------
# Retryability classification
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc,expected",
    [
        (RateLimited(), True),
        (TransientError(), True),
        (TimeoutError(), True),
        (ConnectionError(), True),
        (PermanentError(), False),
        (ValueError("bad input"), False),
    ],
)
def test_retryability_classification(exc, expected):
    assert default_is_retryable(exc) is expected


def test_http_status_codes_are_classified_by_duck_typing():
    class HttpError(Exception):
        def __init__(self, status_code):
            super().__init__(str(status_code))
            self.status_code = status_code

    assert default_is_retryable(HttpError(429)) is True
    assert default_is_retryable(HttpError(503)) is True
    assert default_is_retryable(HttpError(400)) is False
    assert default_is_retryable(HttpError(404)) is False


# --------------------------------------------------------------------------
# End-to-end client
# --------------------------------------------------------------------------
def test_client_embeds_everything_in_input_order():
    async def main(clock):
        provider = FakeProvider(clock, dim=4)
        client = AsyncEmbeddingClient(
            provider.embed, dim=4, requests_per_second=1000,
            max_concurrency=4, batch_size=10, clock=clock,
        )
        texts = [f"case-{i}" for i in range(95)]
        return await client.embed(texts)

    vectors, report = simulate(main)
    assert vectors.shape == (95, 4)
    assert report.ok and report.batches == 10 and report.succeeded == 10
    # The provider encodes each text's index into its vector.
    np.testing.assert_allclose(vectors[:, 0], np.arange(95, dtype=np.float32))


def test_client_respects_the_rate_limit():
    async def main(clock):
        provider = FakeProvider(clock)
        client = AsyncEmbeddingClient(
            provider.embed, dim=4, requests_per_second=5, burst_capacity=5,
            max_concurrency=10, batch_size=1, clock=clock,
        )
        await client.embed([f"case-{i}" for i in range(20)])
        return clock.now()

    # 5 free from the burst, 15 more at 5/sec -> ~3 seconds.
    elapsed = simulate(main)
    assert 2.5 <= elapsed <= 3.5, f"rate limit not enforced end to end ({elapsed}s)"


def test_client_caps_concurrency_end_to_end():
    async def main(clock):
        provider = FakeProvider(clock, latency=0.2)
        client = AsyncEmbeddingClient(
            provider.embed, dim=4, requests_per_second=10_000,
            max_concurrency=3, batch_size=1, clock=clock,
        )
        await client.embed([f"case-{i}" for i in range(30)])
        return provider.max_concurrent

    assert simulate(main) <= 3


def test_client_recovers_from_transient_failures():
    async def main(clock):
        provider = FakeProvider(clock, fail_times=3)
        client = AsyncEmbeddingClient(
            provider.embed, dim=4, requests_per_second=1000, batch_size=5,
            retry=RetryPolicy(base_delay=0.1, jitter=False), clock=clock,
        )
        return await client.embed([f"case-{i}" for i in range(20)])

    vectors, report = simulate(main)
    assert report.ok
    assert report.stats.retries == 3
    np.testing.assert_allclose(vectors[:, 0], np.arange(20, dtype=np.float32))


def test_partial_failure_is_reported_not_hidden():
    """A 50,000-case backfill that loses one batch should say which one, not
    fail wholesale or silently write zeros."""

    async def main(clock):
        provider = FakeProvider(clock, fail_batches={"case-10"})
        client = AsyncEmbeddingClient(
            provider.embed, dim=4, requests_per_second=1000, batch_size=5,
            retry=RetryPolicy(max_attempts=2, base_delay=0.01, jitter=False),
            clock=clock,
        )
        return await client.embed([f"case-{i}" for i in range(25)], allow_partial=True)

    vectors, report = simulate(main)
    assert not report.ok
    assert report.failed_batches == [2]          # texts 10-14
    assert report.succeeded == 4
    assert np.all(vectors[10:15] == 0.0)         # failed rows left as zeros
    np.testing.assert_allclose(vectors[:10, 0], np.arange(10, dtype=np.float32))


def test_strict_mode_raises_on_batch_failure():
    async def main(clock):
        provider = FakeProvider(clock, fail_batches={"case-0"})
        client = AsyncEmbeddingClient(
            provider.embed, dim=4, requests_per_second=1000, batch_size=5,
            retry=RetryPolicy(max_attempts=2, base_delay=0.01, jitter=False),
            clock=clock,
        )
        with pytest.raises(TransientError):
            await client.embed([f"case-{i}" for i in range(10)], allow_partial=False)

    simulate(main)


def test_provider_returning_wrong_vector_count_is_rejected():
    """Silently writing a short batch would misalign every later row."""

    async def main(clock):

        async def bad_embed(texts):
            return [[1.0] * 4 for _ in texts[:-1]]  # one short

        client = AsyncEmbeddingClient(
            bad_embed, dim=4, requests_per_second=1000, batch_size=5,
            retry=RetryPolicy(max_attempts=1), clock=clock,
        )
        with pytest.raises(ValueError, match="expected 5 vectors, got 4"):
            await client.embed([f"case-{i}" for i in range(5)])

    simulate(main)


def test_empty_input_is_a_no_op():
    async def main():
        client = AsyncEmbeddingClient(
            FakeProvider(FakeClock()).embed, dim=4, clock=FakeClock()
        )
        return await client.embed([])

    vectors, report = run(main())
    assert vectors.shape == (0, 4) and report.batches == 0 and report.ok
