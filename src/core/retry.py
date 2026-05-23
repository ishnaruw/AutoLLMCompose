import re
import random
import time

from src.core.run_logging import log_line


_RETRY_AFTER_RE = re.compile(
    r"try again in\s*"
    r"(?:(?P<hours>\d+(?:\.\d+)?)h)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)m)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)s)?",
    re.IGNORECASE,
)


_REQUEST_TOO_LARGE_SIGNALS = [
    "request too large for model",
    "please reduce your message size",
    "maximum context length",
    "context length exceeded",
]


def _is_request_too_large_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(signal in msg for signal in _REQUEST_TOO_LARGE_SIGNALS)


def _is_retryable_error(exc: Exception) -> tuple[bool, str]:
    """
    Return (should_retry, reason).
    Handles rate limits and transient provider/backend failures.
    """
    if _is_request_too_large_error(exc):
        return False, "request_too_large"

    msg = str(exc).lower()

    retry_signals = {
        "rate_limit": [
            "429",
            "rate limit",
            "rate_limit_exceeded",
            "rate_limited",
            "too many requests",
            "tokens per minute",
            "requests per minute",
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
            "upstream connect error",
            "disconnect/reset before headers",
            "reset reason",
            "connection termination",
            "request timed out",
            "read timeout",
            "connect timeout",
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


def classify_retryable_error(exc: Exception) -> tuple[bool, str]:
    return _is_retryable_error(exc)


def is_request_too_large_error(exc: Exception) -> bool:
    return _is_request_too_large_error(exc)


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    """
    Parse provider hints like "Please try again in 14m24.864s".
    Returns None when no concrete wait time is present.
    """
    match = _RETRY_AFTER_RE.search(str(exc))
    if not match:
        return None

    hours = float(match.group("hours") or 0.0)
    minutes = float(match.group("minutes") or 0.0)
    seconds = float(match.group("seconds") or 0.0)
    total = (hours * 3600.0) + (minutes * 60.0) + seconds
    return total if total > 0 else None


def extract_retry_after_seconds(exc: Exception) -> float | None:
    return _extract_retry_after_seconds(exc)


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
                log_line(f"[{name}] giving up after {attempt + 1} attempts: {e}")
                raise

            retry_after_s = _extract_retry_after_seconds(e)
            if retry_after_s is not None:
                sleep_s = retry_after_s + random.uniform(0, 0.75)
                detail = f"provider requested {retry_after_s:.1f}s"
            else:
                sleep_s = min(cap, base * (2 ** attempt)) + random.uniform(0, 0.75)
                detail = f"exponential backoff up to {cap:.1f}s"
            log_line(
                f"[{name}] retryable error ({reason}), sleeping {sleep_s:.1f}s "
                f"(attempt {attempt + 1}/{max_retries}; {detail})"
            )
            time.sleep(sleep_s)
