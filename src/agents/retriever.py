# src/agents/retriever.py
import json
import os
import re
from pathlib import Path

from src.tools.fetch_services import compress_service

# RAG
try:
    from src.rag.retriever import FaissServiceRetriever
except Exception:
    FaissServiceRetriever = None  # type: ignore


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
    Retrieval modes:

    1) LLM batch scanning (existing behavior)
       - set MAOF_RETRIEVER_MODE=llm_batch (default)

    2) RAG-only
       - set MAOF_RETRIEVER_MODE=rag_only
       - candidates are the top embedding matches (no LLM filtering)

    3) RAG + LLM filter
       - set MAOF_RETRIEVER_MODE=rag_llm_filter
       - FAISS topK -> LLM selects 8–12 and writes reasons

    If category is None, fetch_fn returns entries from all categories.
    """
    mode = (os.getenv("MAOF_RETRIEVER_MODE") or "llm_batch").strip().lower()
    rag_index_dir = os.getenv("MAOF_RAG_INDEX_DIR", "data/index/maof_v1/with_qos" if with_qos else "data/index/maof_v1/no_qos")
    rag_topk = int(os.getenv("MAOF_RAG_TOPK", "60"))
    keep_min = int(os.getenv("MAOF_RAG_KEEP_MIN", "8"))
    keep_max = int(os.getenv("MAOF_RAG_KEEP_MAX", "12"))

    keep: dict[str, str] = {}
    offset = 0

    # Keep this genuinely small to avoid prompt size / TPM / context errors.
    # 10–25 is a good range for most providers.
    limit = 20

    prompt_tmpl = Path("prompts/retriever.md").read_text(encoding="utf-8")

    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

    # ---------- RAG path ----------
    if mode in {"rag_only", "rag_llm_filter"}:
        if FaissServiceRetriever is None:
            raise RuntimeError(
                "RAG mode requested but src.rag.retriever could not be imported. "
                "Install dependencies: faiss-cpu and sentence-transformers."
            )

        rag = FaissServiceRetriever(rag_index_dir)

        # Retrieve topK
        retrieved = rag.query(user_goal, top_k=rag_topk)

        if debug_dir:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            (Path(debug_dir) / "retrieved.json").write_text(
                json.dumps(
                    [
                        {
                            "api_id": c.api_id,
                            "score": c.score,
                            "category": c.category,
                            "service": c.compressed,
                        }
                        for c in retrieved
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        # RAG-only: just take the top N
        if mode == "rag_only":
            n = min(keep_max, len(retrieved))
            n = max(min(n, keep_max), min(keep_min, n))
            out = []
            for c in retrieved[:n]:
                out.append({"api_id": c.api_id, "reason": "High embedding similarity to subtask."})

            if debug_dir:
                (Path(debug_dir) / "keep.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            return out

        # RAG + LLM filter: call the LLM once over topK candidates
        candidates = [c.compressed for c in retrieved]
        prompt = (
            "You are a retrieval filter. Select the most functionally relevant APIs.\n"
            "Return STRICT JSON ONLY with this schema: {\"keep\":[{\"api_id\":\"...\",\"reason\":\"...\"}]}.\n\n"
            f"Subtask goal:\n{user_goal}\n\n"
            f"Candidates (JSON list):\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
            f"Rules:\n"
            f"1) Keep {keep_min} to {keep_max} candidates.\n"
            "2) Prefer direct functional match to the goal.\n"
            "3) Do not invent api_ids. Use only api_id values present in candidates.\n"
            "4) If nothing matches, return {\"keep\":[]}.\n"
        )

        debug_path = str(Path(debug_dir) / "debug_rag_filter_llm_raw.txt") if debug_dir else None
        picks = retriever_call(llm_call, prompt, debug_path=debug_path)

        # Fallback if model returns nothing useful
        if not picks:
            n = min(keep_max, len(retrieved))
            picks = [{"api_id": c.api_id, "reason": "Fallback to embedding similarity."} for c in retrieved[:n]]

        if debug_dir:
            (Path(debug_dir) / "keep.json").write_text(json.dumps(picks, indent=2, ensure_ascii=False), encoding="utf-8")
        return picks

    # ---------- Existing LLM batch scan path ----------

    for batch_idx in range(max_batches):
        batch = fetch_fn(category=category, offset=offset, limit=limit, with_qos=with_qos)
        batch = [compress_service(b) for b in batch]
        if not batch:
            break

        prompt = (
            prompt_tmpl
            .replace("{user_goal}", user_goal)
            .replace("{batch_json}", json.dumps(batch, ensure_ascii=False))
            + "\n\nIMPORTANT:\n"
              "If NONE of the APIs in this batch are relevant to the user goal, "
              "return exactly this JSON and nothing else:\n"
              "{\"keep\": []}\n"
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

        # Correct pagination: move by what you requested (or len(batch)).
        offset += limit

    items = list(keep.items())[:12]
    return [{"api_id": k, "reason": v} for k, v in items]

