# src/agents/retriever.py
import json
import re
from pathlib import Path

from src.tools.fetch_services import compress_service


def _coerce_json(s: str) -> str:
    """Return a valid JSON string or {} if we can't parse."""
    s = (s or "").strip()
    if not s:
        return "{}"
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    # Fallback: try to extract the largest {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else "{}"


def retriever_call(llm_call, prompt: str, debug_path: str | None = None):
    """
    Call the LLM with the retriever prompt.
    Expect strict JSON: {"keep":[{"api_id":"...", "reason":"..."}]}
    """
    resp_raw = llm_call(prompt)

    if debug_path:
        Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
        Path(debug_path).write_text(resp_raw or "", encoding="utf-8")

    resp = _coerce_json(resp_raw)
    data = json.loads(resp)

    keep = data.get("keep", [])
    out = []
    for k in keep:
        api_id = k.get("api_id")
        if api_id:
            out.append({"api_id": api_id, "reason": k.get("reason", "")})
    return out


def collect_candidates(
    llm_call,
    user_goal: str,
    fetch_fn,
    category: str | None,
    with_qos: bool,
    max_batches: int = 5,
    debug_dir: str | None = None,
):
    """
    Iterate through catalog batches, ask the LLM to keep relevant APIs,
    and return up to 8–12 unique candidates.

    If category is None, fetch_fn returns entries from all categories.
    """
    keep: dict[str, str] = {}
    offset = 0
    limit = 200  # keep small to reduce prompt size and provider TPM issues

    prompt_tmpl = Path("prompts/retriever.md").read_text(encoding="utf-8")

    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    for batch_idx in range(max_batches):
        batch = fetch_fn(category=category, offset=offset, limit=limit, with_qos=with_qos)
        batch = [compress_service(b) for b in batch]
        if not batch:
            break

        prompt = (
            prompt_tmpl
            .replace("{user_goal}", user_goal)
            .replace("{batch_json}", json.dumps(batch, ensure_ascii=False))
        )

        debug_path = None
        if debug_dir:
            debug_path = str(Path(debug_dir) / f"debug_retriever_batch{batch_idx}.txt")

        picks = retriever_call(
            llm_call,
            prompt,
            debug_path=debug_path,
        )

        for p in picks:
            api_id = p.get("api_id")
            if api_id:
                keep[api_id] = p.get("reason", "")

        if len(keep) >= 12:
            break

        offset += limit  # consistent pagination

    items = list(keep.items())[:12]
    return [{"api_id": k, "reason": v} for k, v in items]
