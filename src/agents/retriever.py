# src/agents/retriever.py
from __future__ import annotations

from typing import Any, Dict, List

from src.rag.retriever import FaissServiceRetriever


def collect_candidates(
    subtask_goal: str,
    *,
    index_dir: str,
    top_k: int = 60,
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

    return out
