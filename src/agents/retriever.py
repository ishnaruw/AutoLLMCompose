# src/agents/retriever.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from src.rag.retriever import FaissServiceRetriever


def _normalize_retrieval_query(subtask_goal: str) -> str:
    """Apply narrow deterministic query boosts for known retrieval failure modes."""
    query = re.sub(r"\s+", " ", str(subtask_goal or "").strip())
    if not query:
        return query

    text = query.lower()
    has_email = bool(re.search(r"\be-?mail\b", text))
    has_send_action = bool(re.search(r"\b(?:send|deliver|mail|notify|email)\b", text))
    has_other_channel = bool(
        re.search(r"\b(?:sms|text message|push|whatsapp|slack|phone|voice)\b", text)
    )

    if has_email and has_send_action and not has_other_channel:
        return f"send email email delivery api email notification api {query}"

    return query


@dataclass
class RagRetrieverAgent:
    """
    Deterministic RAG retrieval agent adapter.

    This intentionally does not use an LLM to rewrite or filter the query. It
    applies only narrow deterministic normalization before FAISS retrieval while
    giving the retrieval stage an agent-shaped interface for the AutoGen pipeline.
    """
    index_dir: str
    name: str = "rag_retriever"
    description: str = "Deterministic FAISS-backed API retrieval agent"
    rag: "FaissServiceRetriever" = field(init=False)

    def __post_init__(self) -> None:
        from src.rag.retriever import FaissServiceRetriever

        self.rag = FaissServiceRetriever(index_dir=str(self.index_dir))

    def retrieve(self, subtask_goal: str, *, top_k: int = 60) -> List[Dict[str, Any]]:
        retrieved = self.rag.query(_normalize_retrieval_query(subtask_goal), top_k=top_k)

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

    def run(self, task: str, *, top_k: int = 60) -> List[Dict[str, Any]]:
        return self.retrieve(task, top_k=top_k)


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
    return RagRetrieverAgent(index_dir=index_dir).retrieve(subtask_goal, top_k=top_k)
