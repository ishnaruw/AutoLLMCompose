from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss is required. Install faiss-cpu (recommended) or faiss-gpu.") from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError("sentence-transformers is required. pip install sentence-transformers") from e


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_embed_text(comp: Dict[str, Any], raw: Dict[str, Any]) -> str:
    parts: List[str] = []
    cat = comp.get("category")
    if cat:
        parts.append(f"Category: {cat}")
    tool_name = comp.get("tool_name") or raw.get("tool_name")
    if tool_name:
        parts.append(f"Tool: {tool_name}")
    tool_desc = comp.get("tool_description") or raw.get("tool_description")
    if tool_desc:
        parts.append(f"Tool Description: {tool_desc}")
    name = comp.get("name") or comp.get("operation") or comp.get("title")
    if name:
        parts.append(f"Endpoint: {name}")
    summary = comp.get("summary") or comp.get("description") or comp.get("desc")
    if summary:
        parts.append(f"Endpoint Description: {summary}")
    method = comp.get("method") or raw.get("method")
    if method:
        parts.append(f"Method: {method}")
    return "\n".join(parts).strip()


@dataclass
class BuildConfig:
    index_dir: str
    embed_model: str
    normalize: bool
    fetch_limit: int


class _Embedder:
    def __init__(self, model_name: str, normalize: bool = True) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(vecs, dtype=np.float32)


def _save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Build shared semantic FAISS index for AutoLLMCompose API catalog")
    p.add_argument("--index_dir", type=str, default="data/index/faiss_no_qos")
    p.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--fetch_limit", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    cfg = BuildConfig(
        index_dir=args.index_dir,
        embed_model=args.embed_model,
        normalize=not args.no_normalize,
        fetch_limit=int(args.fetch_limit),
    )

    from src.tools.fetch_services import fetch_services, compress_service

    index_dir = Path(cfg.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    embedder = _Embedder(cfg.embed_model, normalize=cfg.normalize)

    texts: List[str] = []
    meta_rows: List[Dict[str, Any]] = []

    offset = 0
    total = 0
    while True:
        batch = fetch_services(category=None, offset=offset, limit=cfg.fetch_limit, with_qos=False)
        if not batch:
            break
        for raw in batch:
            comp = compress_service(raw)
            txt = default_embed_text(comp, raw)
            api_id = comp.get("api_id") or raw.get("api_id") or raw.get("id")
            meta_rows.append({"api_id": api_id, "category": comp.get("category") or raw.get("category"), "compressed": comp, "embed_text": txt})
            texts.append(txt)
            total += 1
        offset += len(batch)
        if total % 2000 == 0:
            print(f"[index_build] processed {total} services...")

    if not meta_rows:
        raise RuntimeError("No services loaded from catalog. Check fetch_services paths and data.")

    emb = embedder.encode(texts, batch_size=int(args.batch_size))
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)

    faiss.write_index(index, str(index_dir / "faiss.index"))
    _save_jsonl(index_dir / "meta.jsonl", meta_rows)

    config = {**asdict(cfg), "dims": int(d), "count": int(index.ntotal), "created_at": _now_iso()}
    (index_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[index_build] done. count={config['count']} dims={config['dims']} dir={index_dir}")


if __name__ == "__main__":
    main()
