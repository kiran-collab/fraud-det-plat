"""Async concurrency primitives for calling rate-limited external APIs.

Building the case index means thousands of embedding calls against a hosted
provider. Issued naively that is slow (serial) or abusive (unbounded fan-out);
either way it fails against a provider that enforces limits. Three *different*
constraints have to be respected at once, and conflating them is the usual bug:

  * **concurrency** - how many requests may be in flight  -> ``Semaphore``
  * **rate**        - how many may start per second       -> ``TokenBucket``
  * **failure**     - what to do when one is rejected     -> ``RetryPolicy``

A semaphore alone does not bound rate: 8 concurrent workers each taking 10ms
issue 800 requests/second. A rate limiter alone does not bound concurrency: 500
slow requests can all be in flight at once and exhaust sockets. Both are needed.

Two design decisions worth knowing:

**A 429 pauses the shared limiter, not just the caller.** The obvious retry
sleeps the one coroutine that was rejected while the other fifteen keep
hammering the endpoint. That converts a soft throttle into a hard ban. Here,
``Retry-After`` pauses the bucket every worker draws from, so the whole pool
backs off together.

**Time is injected.** Everything takes a ``Clock``. With the real clock this is
ordinary ``asyncio.sleep``; with ``FakeClock`` the tests exercise minutes of
backoff in milliseconds, deterministically. Retry code whose tests have to
actually sleep does not get tested properly.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

T = TypeVar("T")

# Tolerance for token comparisons, and a floor on how short a wait may be.
# Both exist to stop floating-point dust from turning into a spin loop - see
# TokenBucket.acquire.
_TOKEN_EPSILON = 1e-9
_MIN_SLEEP = 1e-6


# --------------------------------------------------------------------------
# Clock
# --------------------------------------------------------------------------
class Clock(Protocol):
    def now(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    """Monotonic time. Never ``time.time()`` - a wall-clock step (NTP, DST)
    would make an in-flight backoff jump or hang."""

    def now(self) -> float:
        return asyncio.get_running_loop().time()

    async def sleep(self, seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds)


class FakeClock:
    """Virtual time for tests.

    Sleeping tasks register a wake-up and block. A pump advances time to the
    earliest deadline whenever every task is blocked, so a 30-second backoff
    costs no real time and the ordering is deterministic.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._waiters: list[tuple[float, int, asyncio.Event]] = []
        self._seq = itertools.count()
        self.slept_total = 0.0

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        self.slept_total += seconds
        event = asyncio.Event()
        heapq.heappush(self._waiters, (self._now + seconds, next(self._seq), event))
        await event.wait()

    async def run(self, coro: Awaitable[T]) -> T:
        """Drive ``coro`` to completion under virtual time.

        Waits on the workload *and* the pump. Awaiting only the workload means
        that if the pump ever dies, nothing advances time and the await blocks
        forever - the pump's exception is swallowed and the symptom is an
        indefinite hang rather than the error explaining it.
        """
        task = asyncio.ensure_future(coro)
        pump = asyncio.ensure_future(self._pump(task))
        try:
            done, _ = await asyncio.wait({task, pump}, return_when=asyncio.FIRST_COMPLETED)
            if pump in done and not pump.cancelled() and pump.exception() is not None:
                task.cancel()
                raise pump.exception()
            return await task
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)

    async def _pump(self, task: asyncio.Future, max_idle_spins: int = 1000) -> None:
        idle_spins = 0
        while not task.done():
            # Let every currently-runnable callback progress before deciding
            # that the system is idle and time should advance.
            for _ in range(6):
                await asyncio.sleep(0)
            if task.done():
                return
            if self._waiters:
                idle_spins = 0
                when, _, event = heapq.heappop(self._waiters)
                self._now = max(self._now, when)
                event.set()
            else:
                # Nothing runnable and nothing sleeping on *this* clock: time
                # can never advance, so the workload cannot finish. Fail loudly
                # instead of spinning forever - a hung suite is far harder to
                # diagnose than an exception, and the usual cause is a test
                # that builds a second FakeClock inside the coroutine while
                # pumping a different one.
                idle_spins += 1
                if idle_spins >= max_idle_spins:
                    task.cancel()
                    raise RuntimeError(
                        "FakeClock deadlock: no runnable task and no pending sleep on "
                        "this clock. Is the coroutine sleeping on a different Clock "
                        "instance than the one being pumped?"
                    )
                await asyncio.sleep(0)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class TransientError(Exception):
    """Worth retrying: timeout, connection reset, 5xx."""


class RateLimited(TransientError):
    """429. ``retry_after`` is the server's instruction, in seconds."""

    def __init__(self, message: str = "rate limited", retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(Exception):
    """Not worth retrying: 400, 401, 403, malformed input.

    Separated deliberately. Retrying a 401 five times with exponential backoff
    wastes 30 seconds to arrive at the same answer, and under a batch of 5,000
    it turns one misconfiguration into a self-inflicted outage.
    """


def default_is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, PermanentError):
        return False
    if isinstance(exc, (TransientError, asyncio.TimeoutError, ConnectionError, OSError)):
        return True
    # Duck-type HTTP-ish clients that raise their own exception types.
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    return False


def retry_after_of(exc: BaseException) -> float | None:
    value = getattr(exc, "retry_after", None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
class TokenBucket:
    """Token-bucket limiter shared by every worker.

    Chosen over a fixed window because a fixed window lets the whole pool fire
    the instant the window rolls over - a self-inflicted thundering herd that
    reliably trips the very limit it was meant to respect. A bucket refills
    continuously, so requests spread out, while ``capacity`` still allows a
    controlled burst after an idle period.
    """

    def __init__(
        self,
        rate_per_sec: float,
        capacity: float | None = None,
        clock: Clock | None = None,
    ) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity if capacity is not None else max(rate_per_sec, 1.0))
        self.clock = clock or RealClock()
        self._tokens = self.capacity
        self._updated = self.clock.now()
        self._paused_until = 0.0
        # Serialises the check-and-take so waiters are served roughly FIFO
        # instead of racing and starving each other.
        self._lock = asyncio.Lock()
        self.waits = 0
        self.pauses = 0

    def _refill(self) -> None:
        now = self.clock.now()
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def pause(self, seconds: float) -> None:
        """Back the entire pool off - used when the server says 429.

        Global rather than per-caller: pausing only the rejected coroutine
        leaves every other worker hammering an endpoint that has just asked for
        a break.
        """
        if seconds <= 0:
            return
        until = self.clock.now() + seconds
        if until > self._paused_until:
            self._paused_until = until
            self.pauses += 1

    async def acquire(self, tokens: float = 1.0) -> None:
        if tokens > self.capacity:
            raise ValueError(f"cannot acquire {tokens} tokens from a bucket of {self.capacity}")
        async with self._lock:
            while True:
                pause_left = self._paused_until - self.clock.now()
                if pause_left > 0:
                    self.waits += 1
                    await self.clock.sleep(pause_left)
                    continue
                self._refill()
                # Compare with a tolerance. Sleeping exactly `deficit / rate`
                # should refill exactly `deficit` tokens, but the multiply and
                # the subtract do not round-trip in binary floating point, so
                # the balance often lands a few ulps short. An exact `>=` then
                # computes a new deficit of ~1e-17, sleeps ~2e-18 - which is
                # below the resolution of the clock, so `now + delay == now` -
                # and the waiter re-queues forever without time advancing.
                # Under a real clock this merely busy-spins; under virtual time
                # it deadlocks outright, which is how it was caught.
                if self._tokens >= tokens - _TOKEN_EPSILON:
                    self._tokens = max(0.0, self._tokens - tokens)
                    return
                deficit = tokens - self._tokens
                self.waits += 1
                # Floor the sleep so a pathological rate cannot spin either.
                await self.clock.sleep(max(deficit / self.rate, _MIN_SLEEP))


# --------------------------------------------------------------------------
# Retry
# --------------------------------------------------------------------------
@dataclass
class RetryPolicy:
    """Exponential backoff with full jitter.

    Full jitter - ``uniform(0, cap)`` rather than ``cap`` or ``cap/2`` - because
    N workers failing on the same upstream blip would otherwise retry in
    lockstep and reproduce the burst that caused the failure. Randomising the
    whole interval spreads them out; it is the AWS Architecture Blog result and
    it matters more the wider the fan-out.
    """

    max_attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: bool = True
    rng: random.Random = field(default_factory=random.Random)

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait before ``attempt`` (1-based retry number)."""
        if retry_after is not None:
            # The server knows better than our exponent. Still capped, so a
            # hostile or buggy `Retry-After: 86400` cannot wedge the worker.
            return min(float(retry_after), self.max_delay)
        cap = min(self.max_delay, self.base_delay * (self.multiplier ** max(0, attempt - 1)))
        return self.rng.uniform(0.0, cap) if self.jitter else cap


class RetryBudget:
    """Cap on total retries across a whole run.

    Per-call limits are not enough. With 5,000 items and 5 attempts each, a
    provider that is simply *down* produces 25,000 doomed requests and a very
    long, very expensive failure. A shared budget makes the batch give up
    quickly once retrying is clearly not working - the cheap half of a circuit
    breaker, without the state machine.
    """

    def __init__(self, max_retries: int) -> None:
        self.max_retries = max_retries
        self.used = 0

    def try_consume(self) -> bool:
        if self.used >= self.max_retries:
            return False
        self.used += 1
        return True

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_retries


@dataclass
class RetryStats:
    attempts: int = 0
    retries: int = 0
    rate_limited: int = 0
    budget_denied: int = 0


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
    clock: Clock | None = None,
    limiter: TokenBucket | None = None,
    budget: RetryBudget | None = None,
    stats: RetryStats | None = None,
    is_retryable: Callable[[BaseException], bool] = default_is_retryable,
) -> T:
    """Invoke ``fn`` with backoff, honouring the limiter and shared budget."""
    policy = policy or RetryPolicy()
    clock = clock or RealClock()
    stats = stats or RetryStats()

    last: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        if limiter is not None:
            await limiter.acquire()
        stats.attempts += 1
        try:
            return await fn()
        except asyncio.CancelledError:
            raise  # never swallow cancellation - that is how shutdown hangs
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            last = exc
            if not is_retryable(exc) or attempt == policy.max_attempts:
                raise
            if budget is not None and not budget.try_consume():
                stats.budget_denied += 1
                raise
            retry_after = retry_after_of(exc)
            if isinstance(exc, RateLimited) or retry_after is not None:
                stats.rate_limited += 1
                if limiter is not None and retry_after is not None:
                    limiter.pause(retry_after)
            stats.retries += 1
            await clock.sleep(policy.delay_for(attempt, retry_after))

    assert last is not None  # pragma: no cover - loop always raises or returns
    raise last


# --------------------------------------------------------------------------
# Bounded fan-out
# --------------------------------------------------------------------------
async def gather_bounded(
    factories: Sequence[Callable[[], Awaitable[T]]],
    limit: int,
    return_exceptions: bool = False,
) -> list[T | BaseException]:
    """Run at most ``limit`` awaitables concurrently, preserving input order.

    Takes zero-argument *factories*, not coroutine objects. Passing coroutines
    would construct all N up front - so a 50,000-item batch allocates 50,000
    coroutine frames before the semaphore throttles anything, and any that are
    never awaited emit "coroutine was never awaited" warnings on cancellation.
    A factory is not called until a slot is free.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    semaphore = asyncio.Semaphore(limit)

    async def run(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await factory()

    return await asyncio.gather(*(run(f) for f in factories), return_exceptions=return_exceptions)


def chunked(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    if size < 1:
        raise ValueError("size must be >= 1")
    return [items[i : i + size] for i in range(0, len(items), size)]
