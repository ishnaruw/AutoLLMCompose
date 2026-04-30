from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class PipelineConfig:
    # run_tag: str | None = "RUNS_Groq_Multi"
    run_tag: str | None = "RUNS_APR_30"

    shared_index_dir: Path = Path("data/index/maof_v3/shared_no_qos")
    catalog_no_qos_path: Path = Path("data/processed/api_catalog_sample_balanced/api_repo.no_qos.jsonl")

    rag_top_k: int = 40
    ranker_max_candidates: int = 40
    ranker_pool_n: int = 40
    functional_match_chunk_size: int = 6
    selector_top_n: int = 5
    planner_enabled: bool = False
    use_autogen_agents: bool = True
    lmstudio_timeout_seconds: int = 600
    lmstudio_ranker_max_tokens: int = 2500
    llm_debug_enabled: bool = True

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}


CONFIG = PipelineConfig()
