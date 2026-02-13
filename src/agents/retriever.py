# src/agents/retriever.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.retriever import FaissServiceRetriever


def collect_candidates(
    user_query: str,
    subtask_goal: str,
    *,
    index_dir: str,
    top_k: int = 60,
    debug_dir: str | None = None,
) -> List[Dict[str, Any]]:
    """
    RAG-only retriever.

    For a given subtask string, retrieve top_k candidates from FAISS.
    Returns a list of dicts that includes:
      - api_id
      - rag_score  (FAISS similarity)
      - compressed (compact catalog fields; used by ranker)
    """
    rag = FaissServiceRetriever(index_dir=index_dir)
    retrieved = rag.query(subtask_goal, top_k=top_k)

    out: List[Dict[str, Any]] = []
    for c in retrieved:
        out.append(
            {
                "api_id": c.api_id,
                "rag_score": c.rag_score,
                "compressed": c.compressed,
            }
        )

    if debug_dir:
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "retrieved.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        # helpful context file
        (d / "query_context.json").write_text(
            json.dumps({"user_query": user_query, "subtask_goal": subtask_goal, "top_k": top_k}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return out
