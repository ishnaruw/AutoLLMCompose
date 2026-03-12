from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    rag_index_dir_qos: str = "data/index/maof_v1/with_qos"
    rag_index_dir_no_qos: str = "data/index/maof_v1/no_qos"
    rag_top_k: int = 40
    ranker_max_candidates: int = 25
    ranker_pool_n: int = 20
    selector_top_n: int = 10
    topsis_top_k: int = 20
    topsis_min_qos_candidates: int = 5
    planner_paths: int = 3
    queries_path: str = "data/queries/one_user_query.jsonl"
    results_root: str = "results/logs"


CONFIG = PipelineConfig()


def queries_path() -> Path:
    return Path(CONFIG.queries_path)
