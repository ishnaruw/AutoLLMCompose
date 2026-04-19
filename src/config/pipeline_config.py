from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class PipelineConfig:
    queries_path: Path = Path("data/queries/one_user_query.jsonl")
    run_tag: str | None = "run_APR_19_26"

    shared_index_dir: Path = Path("data/index/maof_v3/shared_no_qos")

    rag_top_k: int = 40
    ranker_max_candidates: int = 40
    ranker_pool_n: int = 40
    selector_top_n: int = 5

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}


CONFIG = PipelineConfig()
