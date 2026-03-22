import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class RetryableError(Exception):
    pass


def is_retryable_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    retryable_tokens = ["timeout", "connect", "connection", "tempor", "503", "502", "429"]
    return any(t in name or t in msg for t in retryable_tokens)


async def run_with_retries(fn: Callable[[], Awaitable[T]], *, retries: int = 2, base_delay: float = 0.25) -> T:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: PERF203
            last_exc = exc
            if attempt >= retries or not is_retryable_error(exc):
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, 0.1)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
