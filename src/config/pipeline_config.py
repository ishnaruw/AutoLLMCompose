from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class PipelineConfig:
    queries_path: Path = Path("data/queries/user_queries.jsonl")
    prefix_run_dir_with_query_id: bool = True

    shared_index_dir: Path = Path("data/index/maof_v3/shared_no_qos")

    rag_top_k: int = 40
    ranker_max_candidates: int = 40
    ranker_pool_n: int = 40
    selector_fallback_top_n: int = 5
    qos_metric_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}


CONFIG = PipelineConfig()
QUERIES_PATH = CONFIG.queries_path
PREFIX_RUN_DIR_WITH_QUERY_ID = CONFIG.prefix_run_dir_with_query_id
