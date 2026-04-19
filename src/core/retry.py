import random
import time


def _is_retryable_error(exc: Exception) -> tuple[bool, str]:
    """
    Return (should_retry, reason).
    Handles rate limits and transient provider/backend failures.
    """
    msg = str(exc).lower()

    retry_signals = {
        "rate_limit": [
            "429",
            "rate limit",
            "rate_limited",
            "too many requests",
        ],
        "backend_unavailable": [
            "503",
            "unreachable_backend",
            "internal server error",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "502",
            "504",
        ],
        "network_transient": [
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "server disconnected",
            "remote protocol error",
        ],
    }

    for reason, signals in retry_signals.items():
        if any(signal in msg for signal in signals):
            return True, reason

    return False, ""


def call_with_backoff(fn, *, max_retries=8, base=2.0, cap=32.0, name="llm"):
    """
    Retry wrapper with exponential backoff + jitter.

    Retries:
    - rate limits (429)
    - transient provider failures (503, unreachable backend, internal server errors)
    - common transient network/time-out failures
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            should_retry, reason = _is_retryable_error(e)

            if not should_retry:
                raise

            if attempt >= max_retries:
                print(f"[{name}] giving up after {attempt + 1} attempts: {e}")
                raise

            sleep_s = min(cap, base * (2 ** attempt)) + random.uniform(0, 0.75)
            print(
                f"[{name}] retryable error ({reason}), sleeping {sleep_s:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(sleep_s)