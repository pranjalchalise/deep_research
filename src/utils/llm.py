# src/utils/llm.py
"""
LLM utilities with retry logic for handling transient API failures.
"""
from __future__ import annotations

import time
import random
from typing import Any, Callable, List, Optional, Type, TypeVar, Union

# Common transient error types that should be retried
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
    """Check if an exception is a retryable transient error."""
    exc_name = type(exc).__name__
    exc_str = str(exc).lower()

    # Check exception type name
    if exc_name in RETRYABLE_EXCEPTIONS:
        return True

    # Check for common retryable patterns in message
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
    Execute a function with exponential backoff retry on transient failures.

    Args:
        func: Function to execute (should take no arguments, use lambda/closure)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff (delay = base_delay * base^attempt)
        jitter: Add random jitter to prevent thundering herd
        on_retry: Optional callback(exception, attempt, delay) called before each retry

    Returns:
        Result of func() on success

    Raises:
        Last exception if all retries exhausted
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            # Don't retry on non-retryable errors
            if not is_retryable_error(e):
                raise

            # Don't retry if we've exhausted attempts
            if attempt >= max_retries:
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)

            # Add jitter (0.5x to 1.5x)
            if jitter:
                delay = delay * (0.5 + random.random())

            # Call retry callback if provided
            if on_retry:
                on_retry(e, attempt + 1, delay)

            time.sleep(delay)

    # Should never reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error")


class LLMWrapper:
    """
    Wrapper for LangChain LLM/Chat models with automatic retry.

    Usage:
        llm = ChatOpenAI(model="gpt-4o-mini")
        wrapped = LLMWrapper(llm, max_retries=3)
        response = wrapped.invoke(messages)
    """

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
        """Invoke the LLM with retry logic."""
        return retry_with_backoff(
            func=lambda: self.llm.invoke(*args, **kwargs),
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            on_retry=self._on_retry if self.verbose else None,
        )

    def batch(self, *args: Any, **kwargs: Any) -> Any:
        """Batch invoke the LLM with retry logic."""
        return retry_with_backoff(
            func=lambda: self.llm.batch(*args, **kwargs),
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            on_retry=self._on_retry if self.verbose else None,
        )

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Async invoke - note: retry is synchronous, use for simple cases."""
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
    """
    Create a ChatOpenAI model with retry wrapper.

    Args:
        model: Model name (default: gpt-4o-mini)
        temperature: Temperature setting
        max_retries: Number of retries on transient failures
        verbose: Print retry messages
        **kwargs: Additional arguments passed to ChatOpenAI

    Returns:
        LLMWrapper instance with retry logic
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=temperature, **kwargs)
    return LLMWrapper(llm, max_retries=max_retries, verbose=verbose)


__all__ = [
    "retry_with_backoff",
    "is_retryable_error",
    "LLMWrapper",
    "create_chat_model",
]
