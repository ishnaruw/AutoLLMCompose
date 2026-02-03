import time
import random

def call_with_backoff(fn, *, max_retries=6, base=1.0, cap=30.0, name="llm"):
    """
    Retry wrapper with exponential backoff + jitter.
    Works for rate limits (429) and transient provider errors.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            is_rate = ("429" in msg) or ("rate limit" in msg) or ("rate_limited" in msg) or ("too many requests" in msg)

            if not is_rate:
                raise

            if attempt >= max_retries:
                raise

            sleep_s = min(cap, base * (2 ** attempt)) + random.uniform(0, 0.5)
            print(f"[{name}] rate-limited, sleeping {sleep_s:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(sleep_s)
