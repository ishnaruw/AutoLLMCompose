# src/agents/retriever.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.rag.retriever import FaissServiceRetriever


@dataclass
class RagRetrieverAgent:
    """
    Deterministic RAG retrieval agent adapter.

    This intentionally does not use an LLM to rewrite, filter, or expand the
    query. It preserves the existing FAISS retrieval behavior while giving the
    retrieval stage an agent-shaped interface for the AutoGen pipeline.
    """
    index_dir: str
    name: str = "rag_retriever"
    description: str = "Deterministic FAISS-backed API retrieval agent"
    rag: FaissServiceRetriever = field(init=False)

    def __post_init__(self) -> None:
        self.rag = FaissServiceRetriever(index_dir=str(self.index_dir))

    def retrieve(self, subtask_goal: str, *, top_k: int = 60) -> List[Dict[str, Any]]:
        retrieved = self.rag.query(subtask_goal, top_k=top_k)

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
