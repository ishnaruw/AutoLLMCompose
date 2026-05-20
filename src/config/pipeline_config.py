from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class PipelineConfig:
    run_tag: str | None = "RUNS_MAY_20"

    shared_index_dir: Path = Path("data/index/AutoLLMCompose_v3/shared_no_qos")
    catalog_path: Path = Path("data/processed/api_catalog_sample_balanced/api_repo.tooldesc.jsonl")
    catalog_enriched_path: Path = Path("data/processed/api_catalog_sample_balanced/api_repo.enriched.jsonl")
    api_qos_path: Path = Path("data/processed/api_catalog_sample_balanced/api_qos.jsonl")
    catalog_no_qos_path: Path = Path("data/processed/api_catalog_sample_balanced/misc/api_repo.no_qos.tooldesc.jsonl")
    catalog_with_qos_path: Path = Path("data/processed/api_catalog_sample_balanced/misc/api_repo.with_qos.tooldesc.jsonl")

    rag_top_k: int = 40
    ranker_max_candidates: int = 40
    ranker_pool_n: int = 40
    functional_match_chunk_size: int = 20
    selector_top_n: int = 5
    planner_enabled: bool = True
    composition_qos_eval_enabled: bool = True
    use_autogen_agents: bool = True
    lmstudio_timeout_seconds: int = 600
    lmstudio_ranker_max_tokens: int = 2500
    llm_debug_enabled: bool = True
    llm_validation_max_retries: int = 2
    include_llm_reasons: bool = False
    qos_llm_batch_size: int = 0
    qos_llm_validate_formula: bool = False
    qos_llm_formula_audit: bool = False

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}


CONFIG = PipelineConfig()
