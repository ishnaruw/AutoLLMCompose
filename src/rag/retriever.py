from __future__ import annotations

"""FAISS-based retriever for MAOF.

Loads the FAISS index built by src.rag.index_build and retrieves top-K APIs
for a given subtask string.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("FAISS is required. Install faiss-cpu or faiss-gpu.") from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError(
        "sentence-transformers is required. pip install sentence-transformers"
    ) from e


@dataclass
class RetrievedCandidate:
    api_id: str
    score: float
    category: Optional[str]
    compressed: Dict[str, Any]


class _Embedder:
    def __init__(self, model_name: str, normalize: bool = True) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)

    def encode_one(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(vec, dtype=np.float32)


class FaissServiceRetriever:
    def __init__(self, index_dir: str) -> None:
        self.index_dir = Path(index_dir)

        self.index = faiss.read_index(str(self.index_dir / "faiss.index"))
        self.meta = self._load_meta(self.index_dir / "meta.jsonl")
        self.config = self._load_config(self.index_dir / "config.json")

        model_name = self.config.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        normalize = bool(self.config.get("normalize", True))
        self.embedder = _Embedder(model_name, normalize=normalize)

        if int(self.index.ntotal) != len(self.meta):
            raise RuntimeError(
                f"Index rows mismatch: faiss.ntotal={self.index.ntotal} meta={len(self.meta)}"
            )

    @staticmethod
    def _load_meta(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def query(self, subtask: str, *, top_k: int = 60) -> List[RetrievedCandidate]:
        vec = self.embedder.encode_one(subtask)
        scores, idxs = self.index.search(vec, int(top_k))

        out: List[RetrievedCandidate] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            row = self.meta[idx]
            out.append(
                RetrievedCandidate(
                    api_id=str(row.get("api_id", "")),
                    score=float(score),
                    category=row.get("category"),
                    compressed=row.get("compressed", {}) or {},
                )
            )
        return out
