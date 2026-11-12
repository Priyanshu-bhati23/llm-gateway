"""Per-client token-bucket rate limiting and per-upstream circuit breakers."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


class RateLimiter:
    """Token bucket: `rpm` tokens refill continuously over a minute, bursts up to `rpm`."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # client -> (tokens, last refill time)

    def allow(self, client: str, rpm: int) -> tuple[bool, float]:
        """Returns (allowed, seconds until the next token is available)."""
        now = self._clock()
        tokens, last = self._buckets.get(client, (float(rpm), now))
        tokens = min(float(rpm), tokens + (now - last) * rpm / 60.0)
        if tokens >= 1.0:
            self._buckets[client] = (tokens - 1.0, now)
            return True, 0.0
        self._buckets[client] = (tokens, now)
        return False, (1.0 - tokens) * 60.0 / rpm


@dataclass
class _State:
    failures: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    """Stops sending traffic to an upstream that keeps failing.

    closed -> (N consecutive failures) -> open -> (cooldown) -> half-open: the next
    request is a probe; success closes the circuit, failure re-opens it.
    """

    def __init__(
        self, failure_threshold: int = 3, cooldown_seconds: float = 30, clock: Callable[[], float] = time.monotonic
    ):
        self._threshold, self._cooldown, self._clock = failure_threshold, cooldown_seconds, clock
        self._states: dict[str, _State] = {}

    def available(self, upstream: str) -> bool:
        state = self._states.get(upstream)
        if state is None or state.opened_at is None:
            return True
        return self._clock() - state.opened_at >= self._cooldown  # half-open: allow a probe

    def record_success(self, upstream: str) -> None:
        self._states[upstream] = _State()

    def record_failure(self, upstream: str) -> None:
        state = self._states.setdefault(upstream, _State())
        state.failures += 1
        if state.failures >= self._threshold:
            state.opened_at = self._clock()  # (re)start the cooldown

    def state(self, upstream: str) -> str:
        state = self._states.get(upstream)
        if state is None or state.opened_at is None:
            return "closed"
        return "half-open" if self.available(upstream) else "open"
