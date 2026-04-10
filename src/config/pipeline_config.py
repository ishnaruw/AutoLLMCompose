from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class PipelineConfig:
    queries_path: Path = Path("data/queries/one_user_query.jsonl")
    prefix_run_dir_with_query_id: bool = True

    no_qos_index_dir: Path = Path("data/index/maof_v2/no_qos")
    with_qos_index_dir: Path = Path("data/index/maof_v2/with_qos")

    rag_top_k: int = 40
    ranker_max_candidates: int = 25
    ranker_pool_n: int = 20
    selector_top_n: int = 10
    topsis_top_k: int = 20
    topsis_min_qos_candidates: int = 5

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}


CONFIG = PipelineConfig()
QUERIES_PATH = CONFIG.queries_path
PREFIX_RUN_DIR_WITH_QUERY_ID = CONFIG.prefix_run_dir_with_query_id
