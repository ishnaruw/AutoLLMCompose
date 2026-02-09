from __future__ import annotations

"""Build a reusable FAISS index over the API catalog.

Usage:
  python -m src.rag.index_build --index_dir data/index/maof_v1/with_qos --with_qos
  python -m src.rag.index_build --index_dir data/index/maof_v1/no_qos

This index is intended to be built once, then reused across experiments.
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError(
        "FAISS is required. Install faiss-cpu (recommended) or faiss-gpu."
    ) from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError(
        "sentence-transformers is required. pip install sentence-transformers"
    ) from e


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def build_embed_text(raw: Dict[str, Any], compressed: Dict[str, Any]) -> str:
    """Create an embedding string focused on *functional* relevance.

    We keep QoS out of the main text because it can drown semantic similarity.
    QoS still remains available later for ranker/planner.
    """

    parts: List[str] = []

    api_id = compressed.get("api_id") or raw.get("api_id")
    if api_id:
        parts.append(f"api_id: {api_id}")

    cat = compressed.get("category") or raw.get("category")
    if cat:
        parts.append(f"category: {cat}")

    name = raw.get("name") or compressed.get("name")
    if name:
        parts.append(f"name: {name}")

    desc = raw.get("description") or compressed.get("description")
    if isinstance(desc, str) and desc.strip():
        parts.append("description: " + desc.strip()[:600])

    # Try to pick up common fields used in API catalogs.
    method = raw.get("method") or raw.get("http_method")
    path = raw.get("path") or raw.get("endpoint") or raw.get("url")
    if method or path:
        parts.append(f"endpoint: {method or ''} {path or ''}".strip())

    tags = raw.get("tags") or raw.get("tag")
    if isinstance(tags, list) and tags:
        parts.append("tags: " + ", ".join(str(t) for t in tags[:30]))

    # Parameters commonly show up under: parameters / params / inputs
    params = raw.get("parameters") or raw.get("params") or raw.get("inputs")
    params = _as_list(params)
    if params:
        bits: List[str] = []
        for p in params[:30]:
            if isinstance(p, dict):
                n = p.get("name")
                d = p.get("description") or p.get("desc")
                if n and d:
                    bits.append(f"{n} ({str(d)[:80]})")
                elif n:
                    bits.append(str(n))
            elif isinstance(p, str):
                bits.append(p)
        if bits:
            parts.append("params: " + "; ".join(bits))

    return "\n".join(parts).strip()


@dataclass
class BuildConfig:
    index_dir: str
    embed_model: str
    normalize: bool
    with_qos: bool
    fetch_limit: int
    batch_size: int
    created_at: str


class LocalEmbedder:
    def __init__(self, model_name: str, normalize: bool = True) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        return vecs


def iter_services(fetch_services_fn, *, with_qos: bool, fetch_limit: int) -> Iterable[Dict[str, Any]]:
    offset = 0
    while True:
        batch = fetch_services_fn(category=None, offset=offset, limit=fetch_limit, with_qos=with_qos)
        if not batch:
            break
        for item in batch:
            yield item
        offset += len(batch)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index for MAOF API catalog")
    parser.add_argument("--index_dir", type=str, default="data/index/maof_v1/with_qos")
    parser.add_argument(
        "--embed_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--with_qos", action="store_true")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--fetch_limit", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    from src.tools.fetch_services import fetch_services, compress_service

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    cfg = BuildConfig(
        index_dir=str(index_dir),
        embed_model=args.embed_model,
        normalize=not args.no_normalize,
        with_qos=bool(args.with_qos),
        fetch_limit=int(args.fetch_limit),
        batch_size=int(args.batch_size),
        created_at=_utc_iso(),
    )

    embedder = LocalEmbedder(cfg.embed_model, normalize=cfg.normalize)

    meta_rows: List[Dict[str, Any]] = []
    texts: List[str] = []

    t0 = time.time()
    n = 0
    for raw in iter_services(fetch_services, with_qos=cfg.with_qos, fetch_limit=cfg.fetch_limit):
        comp = compress_service(raw)
        text = build_embed_text(raw, comp)

        meta_rows.append(
            {
                "api_id": comp.get("api_id") or raw.get("api_id"),
                "category": comp.get("category") or raw.get("category"),
                "compressed": comp,
                "embed_text": text,
            }
        )
        texts.append(text)
        n += 1
        if n % 5000 == 0:
            print(f"[index_build] queued {n} services...")

    if not meta_rows:
        raise RuntimeError("No services loaded from fetch_services(). Check catalog paths.")

    print(f"[index_build] embedding {len(texts)} services...")
    vecs = embedder.encode(texts, batch_size=cfg.batch_size)

    if vecs.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape: {vecs.shape}")

    d = int(vecs.shape[1])
    # Use cosine similarity by using inner product on normalized vectors.
    index = faiss.IndexFlatIP(d)
    index.add(vecs)

    print("[index_build] writing artifacts...")
    faiss.write_index(index, str(index_dir / "faiss.index"))
    save_jsonl(index_dir / "meta.jsonl", meta_rows)

    config_out = {
        **asdict(cfg),
        "dims": d,
        "count": int(vecs.shape[0]),
        "build_seconds": round(time.time() - t0, 3),
    }
    (index_dir / "config.json").write_text(json.dumps(config_out, indent=2), encoding="utf-8")

    print(f"[index_build] done: {config_out['count']} docs, dims={d}, dir={index_dir}")


if __name__ == "__main__":
    main()
