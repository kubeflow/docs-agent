"""
Exponential backoff retry with jitter and graceful degradation.

Implements the retry and fault-tolerance requirements from the GSoC 2026
Agentic RAG spec (Hardened System Considerations, Requirement #5):

    "Robust retry logic is a must for all tools. The agent implements
    exponential backoff with jitter for Vector DB retrievals and LLM API
    timeouts. If tools strictly fail, the agent is configured to
    transparently degrade, informing the user that 'Live code context is
    currently unreachable.'"
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Callable, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

# Sentinel returned by execute_tool when all retries are exhausted, so the
# LLM receives an explicit degradation message instead of an empty result set.
DEGRADED_RESULT = (
    "The documentation search service is temporarily unreachable after "
    "multiple retries. Please try again in a moment. If the problem "
    "persists, the vector database or embedding service may be offline."
)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries a sync or async callable with exponential backoff.

    Args:
        max_attempts: Total number of attempts (first try + retries).
        base_delay: Initial sleep duration in seconds before the first retry.
        max_delay: Upper bound on the computed sleep duration.
        backoff_factor: Multiplier applied to the delay after each failure.
        jitter: When True, adds ±50 % uniform noise to prevent thundering herd.
        exceptions: Tuple of exception types that trigger a retry.  Other
            exceptions propagate immediately.

    Usage (sync)::

        @with_retry(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
        def milvus_search(query: str) -> dict: ...

    Usage (async)::

        @with_retry(max_attempts=3, base_delay=0.5)
        async def call_kserve(payload: dict) -> dict: ...
    """

    def _compute_delay(attempt: int) -> float:
        delay = min(base_delay * (backoff_factor**attempt), max_delay)
        if jitter:
            # Full-jitter strategy: uniform in [0, delay] avoids correlation
            # between retrying clients (see AWS "Exponential Backoff and Jitter").
            delay = random.uniform(0, delay)
        return delay

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exc: Exception = RuntimeError("unreachable")
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as exc:
                        last_exc = exc
                        if attempt == max_attempts - 1:
                            break
                        delay = _compute_delay(attempt)
                        logger.warning(
                            "[RETRY] %s attempt %d/%d failed: %s. "
                            "Retrying in %.2fs...",
                            func.__name__,
                            attempt + 1,
                            max_attempts,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)

                logger.error(
                    "[RETRY] %s exhausted all %d attempts. Last error: %s",
                    func.__name__,
                    max_attempts,
                    last_exc,
                )
                raise last_exc

            return async_wrapper  # type: ignore[return-value]

        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exc: Exception = RuntimeError("unreachable")
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as exc:
                        last_exc = exc
                        if attempt == max_attempts - 1:
                            break
                        delay = _compute_delay(attempt)
                        logger.warning(
                            "[RETRY] %s attempt %d/%d failed: %s. "
                            "Retrying in %.2fs...",
                            func.__name__,
                            attempt + 1,
                            max_attempts,
                            exc,
                            delay,
                        )
                        time.sleep(delay)

                logger.error(
                    "[RETRY] %s exhausted all %d attempts. Last error: %s",
                    func.__name__,
                    max_attempts,
                    last_exc,
                )
                raise last_exc

            return sync_wrapper  # type: ignore[return-value]

    return decorator  # type: ignore[return-value]
