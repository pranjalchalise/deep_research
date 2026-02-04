"""
LLM wrappers with retry logic.

API calls fail all the time -- rate limits, 502s, random timeouts. We wrap
every LLM call with exponential backoff so the pipeline doesn't just die
on the first hiccup.
"""
from __future__ import annotations

import time
import random
from typing import Any, Callable, List, Optional, Type, TypeVar, Union

# We match on class names (not instances) because the actual exception types
# live in different packages depending on which LLM provider is in use.
RETRYABLE_EXCEPTIONS = (
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
    "Timeout",
    "ConnectionError",
)

T = TypeVar("T")


def is_retryable_error(exc: Exception) -> bool:
    """Check if this error is transient (worth retrying) vs a real bug."""
    exc_name = type(exc).__name__
    exc_str = str(exc).lower()

    if exc_name in RETRYABLE_EXCEPTIONS:
        return True

    # Fallback: scan the error message for known transient patterns.
    # Not elegant, but catches edge cases where providers wrap errors differently.
    retryable_patterns = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "overloaded",
    ]

    return any(pattern in exc_str for pattern in retryable_patterns)


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> T:
    """
    Call func() and retry on transient failures with exponential backoff.

    Jitter is on by default so parallel workers don't all retry at the
    exact same moment (thundering herd problem).
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if not is_retryable_error(e):
                raise

            if attempt >= max_retries:
                raise

            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random())

            if on_retry:
                on_retry(e, attempt + 1, delay)

            time.sleep(delay)

    # Shouldn't get here, but keeps mypy happy
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error")


class LLMWrapper:
    """Wraps a LangChain chat model so every call retries on transient API errors."""

    def __init__(
        self,
        llm: Any,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        verbose: bool = False,
    ):
        self.llm = llm
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.verbose = verbose

    def _on_retry(self, exc: Exception, attempt: int, delay: float) -> None:
        if self.verbose:
            print(f"[LLM] Retry {attempt}/{self.max_retries} after {delay:.1f}s: {type(exc).__name__}")

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return retry_with_backoff(
            func=lambda: self.llm.invoke(*args, **kwargs),
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            on_retry=self._on_retry if self.verbose else None,
        )

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        return retry_with_backoff(
            func=lambda: self.llm.batch(*args, **kwargs),
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            on_retry=self._on_retry if self.verbose else None,
        )

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        # TODO: the retry loop is synchronous -- good enough for now but
        # we should make this properly async eventually
        return retry_with_backoff(
            func=lambda: self.llm.ainvoke(*args, **kwargs),
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            on_retry=self._on_retry if self.verbose else None,
        )


def create_chat_model(
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_retries: int = 3,
    verbose: bool = False,
    **kwargs: Any,
) -> LLMWrapper:
    """Spin up a ChatOpenAI instance wrapped with our retry logic."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=temperature, **kwargs)
    return LLMWrapper(llm, max_retries=max_retries, verbose=verbose)


def create_model_for_node(
    node_name: str,
    temperature: float = 0.2,
    max_retries: int = 3,
    verbose: bool = False,
    **kwargs: Any,
) -> LLMWrapper:
    """
    Pick the right model for a pipeline node based on ResearchConfig.

    This is how we do model tiering -- cheap models for grunt work like
    extraction, smarter models for planning and writing.
    """
    from src.advanced.config import ResearchConfig

    cfg = ResearchConfig()
    model = cfg.get_model_for_node(node_name)

    return create_chat_model(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        verbose=verbose,
        **kwargs,
    )


__all__ = [
    "retry_with_backoff",
    "is_retryable_error",
    "LLMWrapper",
    "create_chat_model",
    "create_model_for_node",
]
